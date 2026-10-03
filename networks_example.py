import importlib
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantization.lsqplus_quantize_V1 import (
    QuantAdd,
    QuantConcat,
    QuantConv2d,
    QuantConvTranspose2d,
    QuantDiv,
    QuantLinear,
    QuantMaxPool,
    QuantMultiply,
    QuantSub,
)
from quantization.constants import INIT_STATE_FROZEN
import quantization as quant_pkg  # 仅用于 print 分钟级时间戳 / only for print minute-precision timestamp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datas")
_MODEL_DIR_BASE = os.path.join(BASE_DIR, "model", "example")
MODEL_DIR = _MODEL_DIR_BASE


def _model_dir_for(quant_method=None):
    """按网络名 + 量化后端返回产物目录 / Return artifact directory by network name + quantization backend."""
    if quant_method is None:
        return _MODEL_DIR_BASE
    path = os.path.join(_MODEL_DIR_BASE, quant_method)
    os.makedirs(path, exist_ok=True)
    return path
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_WEIGHT_QUANT_OPS = (QuantConv2d, QuantConvTranspose2d, QuantLinear)
_QUANT_OPS = _WEIGHT_QUANT_OPS + (
    QuantAdd,
    QuantSub,
    QuantMultiply,
    QuantDiv,
    QuantConcat,
    QuantMaxPool,
)


class FloatAdd(nn.Module):
    def forward(self, a, b):
        return a + b


class FloatSub(nn.Module):
    def forward(self, a, b):
        return a - b


class FloatMultiply(nn.Module):
    def forward(self, a, b):
        return a * b


class FloatDiv(nn.Module):
    def forward(self, a, b):
        return a / b


class FloatConcat(nn.Module):
    def forward(self, a, b, dim):
        return torch.cat([a, b], dim=dim)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.skip_add = FloatAdd()

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.skip_add(out, self.shortcut(x))
        return F.relu(out, inplace=True)


class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = QuantConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            a_bits=8, w_bits=8, per_channel=True,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
            a_bits=8, w_bits=8, per_channel=True,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.skip_add = QuantAdd(a_bits=8, quant_inference=True)

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                QuantConv2d(
                    in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False,
                    a_bits=8, w_bits=8, per_channel=True,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.skip_add(out, self.shortcut(x))
        return F.relu(out, inplace=True)


class FloatCifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks=3, stride=2)

        self.up_conv = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.down_conv = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)

        self.add_branch = FloatAdd()
        self.sub_branch = FloatSub()
        self.mul_branch = FloatMultiply()
        self.add_denom = FloatAdd()
        self.div_branch = FloatDiv()
        self.div_bn = nn.BatchNorm2d(64)

        self.concat_branch = FloatConcat()
        self.merge_conv = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.mix_branch = FloatAdd()
        self.residual = FloatAdd()
        self.out_bn = nn.BatchNorm2d(64)

        self.register_buffer("div_eps", torch.full((1, 64, 1, 1), 0.1))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _op_block(self, feat):
        up = self.up_conv(feat)
        down = self.down_conv(up)

        added = self.add_branch(down, feat)
        subbed = self.sub_branch(down, feat)
        gated = self.mul_branch(added, subbed)
        gated = torch.sigmoid(gated)
        divided = self.div_branch(gated, self.add_denom(gated, self.div_eps))
        divided = F.relu(self.div_bn(divided), inplace=True)

        merged = self.merge_conv(self.concat_branch(added, subbed, 1))

        mixed = self.mix_branch(divided, merged)
        return F.relu(self.out_bn(self.residual(mixed, feat)), inplace=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self._op_block(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = QuantConv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False,
            a_bits=8, w_bits=8, per_channel=True,
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(QuantBasicBlock, 16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(QuantBasicBlock, 32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(QuantBasicBlock, 64, num_blocks=3, stride=2)

        self.up_conv = QuantConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False,
            a_bits=8, w_bits=8, per_channel=True,
        )
        self.down_conv = QuantConv2d(
            64, 64, kernel_size=3, stride=2, padding=1, bias=False,
            a_bits=8, w_bits=8, per_channel=True,
        )

        self.add_branch = QuantAdd(a_bits=8, quant_inference=True)
        self.sub_branch = QuantSub(a_bits=8, quant_inference=True)
        self.mul_branch = QuantMultiply(a_bits=8, quant_inference=True)
        self.add_denom = QuantAdd(a_bits=8, quant_inference=True)
        self.div_branch = QuantDiv(a_bits=8, quant_inference=True)
        self.div_bn = nn.BatchNorm2d(64)

        self.concat_branch = QuantConcat(a_bits=8, quant_inference=True)
        self.merge_conv = QuantConv2d(
            128, 64, kernel_size=1, bias=False, a_bits=8, w_bits=8, per_channel=True
        )

        self.mix_branch = QuantAdd(a_bits=8, quant_inference=True)
        self.residual = QuantAdd(a_bits=8, quant_inference=True)
        self.out_bn = nn.BatchNorm2d(64)

        self.register_buffer("div_eps", torch.full((1, 64, 1, 1), 0.1))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = QuantLinear(64, num_classes, a_bits=8, w_bits=8, per_channel=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _op_block(self, feat):
        up = self.up_conv(feat)
        down = self.down_conv(up)

        added = self.add_branch(down, feat)
        subbed = self.sub_branch(down, feat)
        gated = self.mul_branch(added, subbed)
        gated = torch.sigmoid(gated)
        divided = self.div_branch(gated, self.add_denom(gated, self.div_eps))
        divided = F.relu(self.div_bn(divided), inplace=True)

        merged = self.merge_conv(self.concat_branch(added, subbed, 1))

        mixed = self.mix_branch(divided, merged)
        return F.relu(self.out_bn(self.residual(mixed, feat)), inplace=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self._op_block(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


def get_dataloaders(batch_size=128, num_workers=2, calibration=False):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=eval_transform if calibration else train_transform,
    )
    test_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=not calibration,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, max_batches=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


def _maybe_visualize_cifar(images, labels, outputs, viz_dir, viz_prefix, viz_state, viz_max, class_names=None):
    """CIFAR 分类任务可视化：用 matplotlib 拼接网格，显示 GT 与预测类别（带置信度） /
    CIFAR classification visualization: use matplotlib to create a grid showing GT and predicted classes (with confidence).
    """
    bs = images.shape[0]
    n = min(bs, viz_max - viz_state["saved"])
    if n <= 0:
        return
    bi = viz_state["batch"]
    rows = cols = int(math.ceil(math.sqrt(n)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if n == 1:
        axes = [[axes]]
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    _, preds = outputs[:n].max(dim=1)
    confs = outputs[:n].softmax(dim=1).max(dim=1)[0]
    images = images[:n].cpu()
    labels = labels[:n].cpu()
    preds = preds.cpu()
    confs = confs.cpu()

    for i in range(rows * cols):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
        ax.axis("off")
        if i < n:
            img = images[i].permute(1, 2, 0).numpy()
            # 反归一化 / denormalize
            img = img * CIFAR_STD + CIFAR_MEAN
            img = img.clip(0, 1)
            ax.imshow(img)
            gt = class_names[labels[i]] if class_names else int(labels[i])
            pr = class_names[preds[i]] if class_names else int(preds[i])
            ax.set_title(
                f"GT:{gt}  P:{pr} {confs[i]:.2f}",
                color="green" if preds[i] == labels[i] else "red",
                fontsize=9,
            )
    plt.tight_layout()
    os.makedirs(viz_dir, exist_ok=True)
    fig.savefig(os.path.join(viz_dir, f"{viz_prefix}_batch{bi}.jpg"), dpi=150)
    plt.close(fig)
    viz_state["saved"] += n
    viz_state["batch"] += 1


@torch.no_grad()
def evaluate(model, loader, criterion, max_batches=None, viz_dir=None, viz_max=30, viz_prefix="eval", class_names=None):
    """top-1 accuracy（评估态模型输出 logits）。若 viz_dir 不为空，
    额外把前 viz_max 张验证图的 GT/预测类别网格保存到 {viz_dir}/fvisualize/ /
    top-1 accuracy (model outputs logits in eval mode). If viz_dir is set,
    additionally save GT/predicted-class grids of the first viz_max val images into {viz_dir}/fvisualize/."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    # 可视化输出目录与计数器 / visualization output dir and counters
    viz_out = os.path.join(viz_dir, "fvisualize") if viz_dir else None
    viz_state = {"saved": 0, "batch": 0}
    if viz_out:
        os.makedirs(viz_out, exist_ok=True)

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 每次评估都可视化前几批（失败仅告警，绝不影响评估） / visualize first batches on every evaluate (failure only warns, never breaks eval)
        if viz_out and viz_state["saved"] < viz_max:
            try:
                _maybe_visualize_cifar(images, labels, outputs, viz_out, viz_prefix, viz_state, viz_max, class_names)
            except Exception as exc:
                print(f"      [viz] 可视化保存失败（仅告警）: {exc} / visualization save failed (warn only): {exc}")

    return total_loss / total, correct / total


def load_checkpoint(model, path, return_meta=False):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        meta = checkpoint.get("meta", {})
    else:
        state_dict = checkpoint
        meta = {}
    model.load_state_dict(state_dict)
    if return_meta:
        return model, meta
    return model


def save_checkpoint(model, path, **metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if metadata:
        torch.save({"state_dict": model.state_dict(), "meta": metadata}, path)
    else:
        torch.save(model.state_dict(), path)


def copy_float_to_quant(float_model, quant_model):
    float_state = float_model.state_dict()
    quant_state = quant_model.state_dict()

    for key, value in float_state.items():
        if key in quant_state and quant_state[key].shape == value.shape:
            quant_state[key] = value.detach().clone().to(device=quant_state[key].device)

    quant_model.load_state_dict(quant_state)
    return quant_model


def _freeze_batch_init(model):
    for module in model.modules():
        if hasattr(module, "init_state"):
            module.init_state = INIT_STATE_FROZEN


def _dequantized_weight(module):
    with torch.no_grad():
        return module.weight_quantizer(module.weight).detach().clone()


def build_float_model(quant_model, num_classes=10):
    _freeze_batch_init(quant_model)
    quant_model.eval()

    quant_state = quant_model.state_dict()
    for name, module in quant_model.named_modules():
        if isinstance(module, _WEIGHT_QUANT_OPS):
            quant_state[f"{name}.weight"] = _dequantized_weight(module)

    float_model = FloatCifarCNN(num_classes=num_classes)
    float_state = float_model.state_dict()
    missing = []

    for key in float_state:
        if key in quant_state and quant_state[key].shape == float_state[key].shape:
            float_state[key] = quant_state[key].detach().cpu().clone()
        else:
            missing.append(key)

    if missing:
        raise RuntimeError(f"浮点模型缺少对应参数: {missing}")

    float_model.load_state_dict(float_state)
    return float_model


def _append_activation_quantizer(params, module, quantizer_name, tensor_name):
    if not hasattr(module, quantizer_name):
        return

    weighted_module = isinstance(module, _WEIGHT_QUANT_OPS)
    if not weighted_module and not getattr(module, "quant_inference", False):
        return

    quantizer = getattr(module, quantizer_name)
    scale = quantizer.s.detach().cpu().flatten().to(torch.float64)
    beta = quantizer.beta.detach().cpu().flatten().to(torch.float64)
    zero_point = torch.round(-beta / scale).to(torch.int64).clamp(quantizer.Qn, quantizer.Qp)

    if scale.numel() == 1:
        scale_value = float(scale.item())
        zero_point_value = int(zero_point.item())
    else:
        scale_value = scale.tolist()
        zero_point_value = zero_point.tolist()

    params[tensor_name] = {
        "scale": scale_value,
        "zero_point": zero_point_value,
    }


def collect_quant_params(quant_model):
    params = {}

    for name, module in quant_model.named_modules():
        if not isinstance(module, _QUANT_OPS):
            continue

        if isinstance(module, _WEIGHT_QUANT_OPS):
            weight_quantizer = module.weight_quantizer
            scale = weight_quantizer.s.detach().cpu().flatten().to(torch.float64)
            if weight_quantizer.per_channel:
                params[f"{name}.weight"] = {
                    "scale": scale.tolist(),
                    "zero_point": [0] * scale.numel(),
                }
            else:
                params[f"{name}.weight"] = {
                    "scale": float(scale.item()),
                    "zero_point": 0,
                }

        _append_activation_quantizer(
            params,
            module,
            "activation_quantizer",
            f"{name}.input",
        )
        _append_activation_quantizer(
            params,
            module,
            "activation_quantizer0",
            f"{name}.input0",
        )
        _append_activation_quantizer(
            params,
            module,
            "activation_quantizer1",
            f"{name}.input1",
        )

    return params


def export_onnx(float_model, onnx_path, opset=16, imgsz=None):
    """导出 ONNX，输入形状完全固定为 [1, 3, H, W]（无 dynamic_axes，图尺寸清晰可见） /
    Export ONNX with fully static input shape [1, 3, H, W] (no dynamic_axes, graph dimensions clearly visible)."""
    H = W = imgsz if imgsz is not None else 32
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    float_model.eval()
    model_device = next(float_model.parameters()).device
    dummy = torch.randn(1, 3, H, W, device=model_device)

    torch.onnx.export(
        float_model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=["input"],
        output_names=["logits"],
        # 无 dynamic_axes → 输入输出形状完全固定 / No dynamic_axes → all shapes fully fixed
    )
    # onnxsim 简化（若已安装） / onnxsim simplification (if installed)
    try:
        import onnx
        from onnxsim import simplify as onnxsim_simplify

        model = onnx.load(onnx_path)
        model_simplified, check = onnxsim_simplify(model)
        if check:
            onnx.save(model_simplified, onnx_path)
            print(f"      ONNX simplify 成功: {len(model.graph.node)} → {len(model_simplified.graph.node)} nodes")
        else:
            print("      [skip] ONNX simplify check 失败，保留原图")
    except ImportError:
        print("      [skip] 未安装 onnxsim，跳过 simplify")
    except Exception as e:
        print(f"      [skip] ONNX simplify 失败: {e}")


def _try_export_onnx(model, onnx_path):
    """训练保存 best checkpoint 时同步导出 ONNX；失败仅告警，绝不影响训练 / Export ONNX alongside best checkpoint during training; failure only warns and never affects training."""
    try:
        export_onnx(model, onnx_path)
        print(f"      ONNX 已同步导出: {onnx_path}")
    except Exception as e:
        print(f"      [warn] ONNX 导出失败（不影响训练）: {e}")


def verify(float_model, quant_model, onnx_path, quant_params):
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        op_types = {node.op_type for node in onnx_model.graph.node}
        quant_nodes = op_types & {"QuantizeLinear", "DequantizeLinear"}
        assert not quant_nodes, f"ONNX 中仍存在量化节点: {sorted(quant_nodes)}"
        print(f"      ONNX checker 通过，算子集合: {sorted(op_types)}")
    except ImportError:
        print("      [skip] 未安装 onnx，跳过结构检查")

    try:
        import numpy as np

        ort = importlib.import_module("onnxruntime")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        torch.manual_seed(0)
        float_model.cpu().eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y_torch = float_model(x).numpy()
        y_onnx = session.run(["logits"], {"input": x.numpy()})[0]
        max_diff = float(np.abs(y_torch - y_onnx).max())
        print(f"      onnxruntime vs PyTorch 最大绝对误差: {max_diff:.3e}")
        assert max_diff < 1e-4, f"ONNX 数值误差过大: {max_diff}"
    except ImportError:
        print("      [skip] 未安装 onnxruntime，跳过数值比对")

    weighted_layers = [
        (name, module)
        for name, module in quant_model.named_modules()
        if isinstance(module, _WEIGHT_QUANT_OPS)
    ]
    missing = []

    for name, _ in weighted_layers:
        for suffix in ("weight", "input"):
            key = f"{name}.{suffix}"
            if key not in quant_params:
                missing.append(key)

    active_activation_layers = [
        (name, module)
        for name, module in quant_model.named_modules()
        if (
            not isinstance(module, _WEIGHT_QUANT_OPS)
            and hasattr(module, "activation_quantizer")
            and getattr(module, "quant_inference", False)
        )
    ]

    for name, _ in active_activation_layers:
        key = f"{name}.input"
        if key not in quant_params:
            missing.append(key)

    assert not missing, f"以下量化参数不完整: {missing}"
    print(f"      量化参数完整性检查通过（{len(quant_params)} 个张量）")


def save_quant_outputs(quant_model, prefix, num_classes=10, meta=None, model_dir=None):
    _freeze_batch_init(quant_model)
    quant_model.eval()

    if model_dir is None:
        model_dir = _MODEL_DIR_BASE
    quant_checkpoint = os.path.join(model_dir, f"{prefix}_cifar10_cnn.pth")
    save_checkpoint(quant_model, quant_checkpoint, **(meta or {}))

    quant_params = collect_quant_params(quant_model)
    json_path = os.path.join(model_dir, f"{prefix}_quant_params.json")
    pth_path = os.path.join(model_dir, f"{prefix}_quant_params.pth")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(quant_params, file, indent=2, ensure_ascii=False)

    torch.save(
        {
            key: {
                "scale": torch.tensor(value["scale"], dtype=torch.float64),
                "zero_point": torch.tensor(value["zero_point"], dtype=torch.int64),
            }
            for key, value in quant_params.items()
        },
        pth_path,
    )
    print(f"[2/5] 量化参数已写出: {json_path} / {pth_path}")
    print(f"      共 {len(quant_params)} 个张量的 scale/zero_point")

    float_model = build_float_model(quant_model, num_classes=num_classes)
    float_checkpoint = os.path.join(model_dir, f"{prefix}_cifar10_cnn_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 正常 PyTorch 浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(model_dir, f"{prefix}_cifar10_cnn_float.onnx")
    export_onnx(float_model, onnx_path, opset=16)
    print(f"[4/5] 干净浮点 ONNX 已写出: {onnx_path}")

    verify(float_model, quant_model, onnx_path, quant_params)
    print("[5/5] 完成。")
    return quant_checkpoint, json_path, pth_path, float_checkpoint, onnx_path


def float_train(batch_size=128, lr=1e-3, epochs=100, num_classes=10, num_workers=2,
                max_train_batches=None, max_eval_batches=None):
    print("========== Float training ==========")
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    float_model = FloatCifarCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(float_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_epoch = -1
    best_meta = {}
    model_dir = _model_dir_for()
    checkpoint_path = os.path.join(model_dir, "cifar10_cnn.pth")
    last_checkpoint = os.path.join(model_dir, "cifar10_cnn_last.pth")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            float_model,
            train_loader,
            criterion,
            optimizer,
            max_batches=max_train_batches,
        )
        test_loss, test_acc = evaluate(float_model, test_loader, criterion, max_batches=max_eval_batches,
                                       viz_dir=model_dir, viz_prefix=f"float_ep{epoch + 1:03d}")
        scheduler.step()

        epoch_meta = {
            "stage": "float",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        }
        save_checkpoint(float_model, last_checkpoint, **epoch_meta)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_meta = epoch_meta
            save_checkpoint(float_model, checkpoint_path, **best_meta)
            _try_export_onnx(float_model, os.path.splitext(checkpoint_path)[0] + ".onnx")

        print(
            f"[Float] Epoch [{epoch + 1}/{epochs}] | "
            f"Train loss:{train_loss:.4f} acc:{train_acc:.4f} | "
            f"Test loss:{test_loss:.4f} acc:{test_acc:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)
    final_loss, final_acc = evaluate(float_model, test_loader, criterion,
                                     viz_dir=model_dir, viz_prefix="float_best")
    print(f"[Float] Best checkpoint: {checkpoint_path}")
    print(f"[Float] Last checkpoint: {last_checkpoint}")
    print(
        f"[Float] Final validation | "
        f"Best epoch:{best_epoch}/{epochs} | "
        f"Test loss:{final_loss:.4f} acc:{final_acc:.4f} | "
        f"Best epoch acc:{best_acc:.4f}"
    )
    return checkpoint_path, final_acc, best_meta


@torch.no_grad()
def calibrate_quantizer(quant_model, calibration_loader, calibration_batches=20):
    quant_model.eval()
    for batch_index, (images, _) in enumerate(calibration_loader):
        quant_model(images.to(device))
        if batch_index + 1 >= calibration_batches:
            break
    _freeze_batch_init(quant_model)


def PTQ_calibration(batch_size=128, num_classes=10, calibration_batches=20, num_workers=2,
                    max_eval_batches=None):
    print("========== PTQ calibration ==========")
    print(f"Device: {device}")

    calibration_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        calibration=True,
    )

    float_dir = _model_dir_for()
    quant_dir = _model_dir_for("lsqplus_v1")
    float_checkpoint = os.path.join(float_dir, "cifar10_cnn.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatCifarCNN(num_classes=num_classes).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = CifarCNN(num_classes=num_classes).to(device)
    copy_float_to_quant(float_model, ptq_model)

    calibrate_quantizer(
        ptq_model,
        calibration_loader,
        calibration_batches=calibration_batches,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(ptq_model, test_loader, criterion, max_batches=max_eval_batches,
                                   viz_dir=quant_dir, viz_prefix="ptq_lsqplus_v1")
    print(f"[PTQ] Test loss:{test_loss:.4f} acc:{test_acc:.4f}")

    ptq_meta = {
        "stage": "ptq",
        "epoch": 0,
        "total_epochs": 0,
        "lr": None,
        "calibration_batches": calibration_batches,
        "train_loss": None,
        "train_acc": None,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }
    ptq_checkpoint = save_quant_outputs(
        ptq_model, "ptq", num_classes=num_classes, meta=ptq_meta, model_dir=quant_dir
    )[0]

    load_checkpoint(ptq_model, ptq_checkpoint)
    final_loss, final_acc = evaluate(ptq_model, test_loader, criterion, max_batches=max_eval_batches,
                                     viz_dir=quant_dir, viz_prefix="ptq_lsqplus_v1_reload")
    print(
        f"[PTQ] Final validation | "
        f"Test loss:{final_loss:.4f} acc:{final_acc:.4f} | "
        f"Checkpoint: {ptq_checkpoint}"
    )

    return ptq_checkpoint


def QAT_training(batch_size=128, lr=1e-4, epochs=20, num_classes=10, num_workers=2,
                 max_train_batches=None, max_eval_batches=None):
    print("========== QAT training ==========")
    print(f"Device: {device}")
    print(f"QAT LR: {lr:g} (float LR x 0.1)")

    train_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    float_dir = _model_dir_for()
    quant_dir = _model_dir_for("lsqplus_v1")
    ptq_checkpoint = os.path.join(quant_dir, "ptq_cifar10_cnn.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(f"找不到 PTQ 权重，请先运行 PTQ_calibration(): {ptq_checkpoint}")

    qat_model = CifarCNN(num_classes=num_classes).to(device)
    load_checkpoint(qat_model, ptq_checkpoint)
    _freeze_batch_init(qat_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(qat_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_epoch = -1
    best_meta = {}
    best_checkpoint = os.path.join(quant_dir, "qat_cifar10_cnn_best.pth")
    last_checkpoint = os.path.join(quant_dir, "qat_cifar10_cnn_last.pth")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            qat_model,
            train_loader,
            criterion,
            optimizer,
            max_batches=max_train_batches,
        )
        test_loss, test_acc = evaluate(qat_model, test_loader, criterion, max_batches=max_eval_batches,
                                       viz_dir=quant_dir, viz_prefix=f"qat_lsqplus_v1_ep{epoch + 1:03d}")
        scheduler.step()

        epoch_meta = {
            "stage": "qat",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        }
        save_checkpoint(qat_model, last_checkpoint, **epoch_meta)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_meta = epoch_meta
            save_checkpoint(qat_model, best_checkpoint, **best_meta)
            _try_export_onnx(qat_model, os.path.splitext(best_checkpoint)[0] + ".onnx")

        print(
            f"[QAT] Epoch [{epoch + 1}/{epochs}] | "
            f"Train loss:{train_loss:.4f} acc:{train_acc:.4f} | "
            f"Test loss:{test_loss:.4f} acc:{test_acc:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    final_loss, final_acc = evaluate(qat_model, test_loader, criterion, max_batches=max_eval_batches,
                                     viz_dir=quant_dir, viz_prefix="qat_lsqplus_v1_best")
    print(
        f"[QAT] Final validation | "
        f"Best epoch:{best_epoch}/{epochs} | "
        f"Test loss:{final_loss:.4f} acc:{final_acc:.4f} | "
        f"Best epoch acc:{best_acc:.4f} | "
        f"Best: {best_checkpoint} | Last: {last_checkpoint}"
    )
    qat_checkpoint = save_quant_outputs(
        qat_model, "qat", num_classes=num_classes, meta=best_meta, model_dir=quant_dir
    )[0]
    return qat_checkpoint, final_acc, best_meta


def compare_precision(batch_size=128, num_classes=10, num_workers=2, max_eval_batches=None):
    print("========== Float vs QAT precision ==========")
    print(f"Device: {device}")

    float_dir = _model_dir_for()
    quant_dir = _model_dir_for("lsqplus_v1")
    # 优先 _best.pth，fallback 到无后缀 / Prefer _best.pth, fallback to no suffix
    float_checkpoint = os.path.join(float_dir, "cifar10_cnn_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, "cifar10_cnn.pth")
    qat_checkpoint = os.path.join(quant_dir, "qat_cifar10_cnn_best.pth")
    if not os.path.exists(qat_checkpoint):
        qat_checkpoint = os.path.join(quant_dir, "qat_cifar10_cnn.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(f"找不到 QAT 权重，请先运行 QAT_training(): {qat_checkpoint}")

    _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )
    criterion = nn.CrossEntropyLoss()

    float_model = FloatCifarCNN(num_classes=num_classes).to(device)
    float_model, float_meta = load_checkpoint(
        float_model, float_checkpoint, return_meta=True
    )
    float_loss, float_acc = evaluate(float_model, test_loader, criterion, max_batches=max_eval_batches,
                                     viz_dir=quant_dir, viz_prefix="compare_float")

    qat_model = CifarCNN(num_classes=num_classes).to(device)
    qat_model, qat_meta = load_checkpoint(
        qat_model, qat_checkpoint, return_meta=True
    )
    _freeze_batch_init(qat_model)
    qat_loss, qat_acc = evaluate(qat_model, test_loader, criterion, max_batches=max_eval_batches,
                                 viz_dir=quant_dir, viz_prefix="compare_qat")

    delta = qat_acc - float_acc
    print(
        f"[Compare] Float | best epoch:{float_meta.get('epoch', '-')}/"
        f"{float_meta.get('total_epochs', '-')} lr:{float_meta.get('lr', '-')} | "
        f"Test loss:{float_loss:.4f} acc:{float_acc:.4f}"
    )
    print(
        f"[Compare] QAT   | best epoch:{qat_meta.get('epoch', '-')}/"
        f"{qat_meta.get('total_epochs', '-')} lr:{qat_meta.get('lr', '-')} | "
        f"Test loss:{qat_loss:.4f} acc:{qat_acc:.4f}"
    )
    print(f"[Compare] QAT - Float acc delta: {delta:+.4f}")

    return {
        "float_acc": float_acc,
        "qat_acc": qat_acc,
        "delta": delta,
        "float_loss": float_loss,
        "qat_loss": qat_loss,
        "float_meta": float_meta,
        "qat_meta": qat_meta,
    }


if __name__ == "__main__":
    quant_pkg.install_print_timestamp()  # print 加分钟级时间戳 / minute-precision timestamp for print
    float_lr = 1e-3
    float_epochs = 100
    qat_lr = float_lr * 0.1
    qat_epochs = max(1, int(round(float_epochs * 0.2)))

    float_train(lr=float_lr, epochs=float_epochs)
    PTQ_calibration()
    QAT_training(lr=qat_lr, epochs=qat_epochs)
    compare_precision()
