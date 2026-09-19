import argparse
import importlib
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import quantization as quant_pkg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datas")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 量化后端可切换：dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact
# 每个后端文件都实现了同一套算子；set_quant_method() 切换后，
# QuantBasicBlock / CifarCNN 再实例化时使用的就是新后端的算子类。
# ---------------------------------------------------------------------------
DEFAULT_QUANT_METHOD = "minmax"
QUANT_METHOD = DEFAULT_QUANT_METHOD
Q = quant_pkg.load_quant_backend(DEFAULT_QUANT_METHOD)

QuantAdd = Q.QuantAdd
QuantConcat = Q.QuantConcat
QuantConv2d = Q.QuantConv2d
QuantConvTranspose2d = Q.QuantConvTranspose2d
QuantDiv = Q.QuantDiv
QuantLinear = Q.QuantLinear
QuantMaxPool = Q.QuantMaxPool
QuantMultiply = Q.QuantMultiply
QuantSub = Q.QuantSub


def set_quant_method(method):
    """切换量化后端（必须在构建 CifarCNN 之前调用）。"""
    global Q, QUANT_METHOD
    global QuantAdd, QuantConcat, QuantConv2d, QuantConvTranspose2d
    global QuantDiv, QuantLinear, QuantMaxPool, QuantMultiply, QuantSub

    Q = quant_pkg.load_quant_backend(method)
    QUANT_METHOD = method
    QuantAdd = Q.QuantAdd
    QuantConcat = Q.QuantConcat
    QuantConv2d = Q.QuantConv2d
    QuantConvTranspose2d = Q.QuantConvTranspose2d
    QuantDiv = Q.QuantDiv
    QuantLinear = Q.QuantLinear
    QuantMaxPool = Q.QuantMaxPool
    QuantMultiply = Q.QuantMultiply
    QuantSub = Q.QuantSub
    return Q


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


@torch.no_grad()
def evaluate(model, loader, criterion, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

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
    # 灌入新权重后，让各后端的量化器从第 0 批重新统计
    # （lsq/lsq+ 重估 s/beta；minmax/pact 重新取首批 min/max、alpha）
    quant_pkg.reset_quantizer_states(quant_model)
    return quant_model


def build_float_model(quant_model, num_classes=10):
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_state = quant_model.state_dict()
    for name, module in quant_model.named_modules():
        if quant_pkg.is_weight_quant_module(module):
            quant_state[f"{name}.weight"] = quant_pkg.dequantized_weight(module)

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


def collect_quant_params(quant_model):
    return quant_pkg.collect_quant_params(quant_model)


def export_onnx(float_model, onnx_path, opset=16):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    float_model.eval()
    model_device = next(float_model.parameters()).device
    dummy = torch.randn(1, 3, 32, 32, device=model_device)

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
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


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
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            y_torch = float_model(x).numpy()
        y_onnx = session.run(["logits"], {"input": x.numpy()})[0]
        max_diff = float(np.abs(y_torch - y_onnx).max())
        print(f"      onnxruntime vs PyTorch 最大绝对误差: {max_diff:.3e}")
        assert max_diff < 1e-4, f"ONNX 数值误差过大: {max_diff}"
    except ImportError:
        print("      [skip] 未安装 onnxruntime，跳过数值比对")

    required_keys = quant_pkg.required_quant_keys(quant_model)
    missing = [key for key in required_keys if key not in quant_params]

    assert not missing, f"以下量化参数不完整: {missing}"
    print(f"      量化参数完整性检查通过（{len(quant_params)} 个张量）")


def save_quant_outputs(quant_model, prefix, num_classes=10, meta=None):
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_cifar10_cnn.pth")
    save_checkpoint(quant_model, quant_checkpoint, **(meta or {}))

    quant_params = collect_quant_params(quant_model)
    json_path = os.path.join(MODEL_DIR, f"{prefix}_quant_params.json")
    pth_path = os.path.join(MODEL_DIR, f"{prefix}_quant_params.pth")

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
    float_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_cifar10_cnn_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 正常 PyTorch 浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(MODEL_DIR, f"{prefix}_cifar10_cnn_float.onnx")
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
    checkpoint_path = os.path.join(MODEL_DIR, "cifar10_cnn.pth")
    last_checkpoint = os.path.join(MODEL_DIR, "cifar10_cnn_last.pth")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            float_model,
            train_loader,
            criterion,
            optimizer,
            max_batches=max_train_batches,
        )
        test_loss, test_acc = evaluate(
            float_model, test_loader, criterion, max_batches=max_eval_batches
        )
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

        print(
            f"[Float] Epoch [{epoch + 1}/{epochs}] | "
            f"Train loss:{train_loss:.4f} acc:{train_acc:.4f} | "
            f"Test loss:{test_loss:.4f} acc:{test_acc:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)
    final_loss, final_acc = evaluate(
        float_model, test_loader, criterion, max_batches=max_eval_batches
    )
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
    quant_pkg.freeze_batch_init(quant_model)


def PTQ_calibration(quant_method=DEFAULT_QUANT_METHOD, batch_size=128, num_classes=10,
                    calibration_batches=20, num_workers=2, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== PTQ calibration [{tag}] ==========")
    print(f"Device: {device}")

    calibration_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        calibration=True,
    )

    float_checkpoint = os.path.join(MODEL_DIR, "cifar10_cnn.pth")
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
    test_loss, test_acc = evaluate(
        ptq_model, test_loader, criterion, max_batches=max_eval_batches
    )
    print(f"[PTQ-{tag}] Test loss:{test_loss:.4f} acc:{test_acc:.4f}")

    ptq_meta = {
        "stage": "ptq",
        "quant_method": tag,
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
        ptq_model, f"ptq_{tag}", num_classes=num_classes, meta=ptq_meta
    )[0]

    load_checkpoint(ptq_model, ptq_checkpoint)
    final_loss, final_acc = evaluate(
        ptq_model, test_loader, criterion, max_batches=max_eval_batches
    )
    print(
        f"[PTQ] Final validation | "
        f"Test loss:{final_loss:.4f} acc:{final_acc:.4f} | "
        f"Checkpoint: {ptq_checkpoint}"
    )

    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=128, lr=1e-4, epochs=20,
                 num_classes=10, num_workers=2,
                 max_train_batches=None, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training [{tag}] ==========")
    print(f"Device: {device}")
    print(f"QAT LR: {lr:g} (float LR x 0.1)")

    train_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    ptq_checkpoint = os.path.join(MODEL_DIR, f"ptq_{tag}_cifar10_cnn.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = CifarCNN(num_classes=num_classes).to(device)
    _, ptq_meta = load_checkpoint(qat_model, ptq_checkpoint, return_meta=True)
    saved_method = ptq_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] PTQ checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    quant_pkg.freeze_batch_init(qat_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(qat_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_epoch = -1
    best_meta = {}
    best_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_cifar10_cnn_best.pth")
    last_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_cifar10_cnn_last.pth")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            qat_model,
            train_loader,
            criterion,
            optimizer,
            max_batches=max_train_batches,
        )
        test_loss, test_acc = evaluate(
            qat_model, test_loader, criterion, max_batches=max_eval_batches
        )
        scheduler.step()

        epoch_meta = {
            "stage": "qat",
            "quant_method": tag,
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

        print(
            f"[QAT-{tag}] Epoch [{epoch + 1}/{epochs}] | "
            f"Train loss:{train_loss:.4f} acc:{train_acc:.4f} | "
            f"Test loss:{test_loss:.4f} acc:{test_acc:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    final_loss, final_acc = evaluate(
        qat_model, test_loader, criterion, max_batches=max_eval_batches
    )
    print(
        f"[QAT-{tag}] Final validation | "
        f"Best epoch:{best_epoch}/{epochs} | "
        f"Test loss:{final_loss:.4f} acc:{final_acc:.4f} | "
        f"Best epoch acc:{best_acc:.4f} | "
        f"Best: {best_checkpoint} | Last: {last_checkpoint}"
    )
    qat_checkpoint = save_quant_outputs(
        qat_model, f"qat_{tag}", num_classes=num_classes, meta=best_meta
    )[0]
    return qat_checkpoint, final_acc, best_meta


def compare_precision(quant_method=DEFAULT_QUANT_METHOD, batch_size=128, num_classes=10,
                      num_workers=2, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== Float vs QAT precision [{tag}] ==========")
    print(f"Device: {device}")

    float_checkpoint = os.path.join(MODEL_DIR, "cifar10_cnn.pth")
    qat_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_cifar10_cnn.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )
    criterion = nn.CrossEntropyLoss()

    float_model = FloatCifarCNN(num_classes=num_classes).to(device)
    float_model, float_meta = load_checkpoint(
        float_model, float_checkpoint, return_meta=True
    )
    float_loss, float_acc = evaluate(
        float_model, test_loader, criterion, max_batches=max_eval_batches
    )

    qat_model = CifarCNN(num_classes=num_classes).to(device)
    qat_model, qat_meta = load_checkpoint(
        qat_model, qat_checkpoint, return_meta=True
    )
    saved_method = qat_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] QAT checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    quant_pkg.freeze_batch_init(qat_model)
    qat_loss, qat_acc = evaluate(
        qat_model, test_loader, criterion, max_batches=max_eval_batches
    )

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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="CIFAR-CNN: 浮点训练 -> PTQ -> QAT -> 精度对比，量化方法可选"
    )
    parser.add_argument(
        "--quant",
        choices=quant_pkg.QUANT_METHODS,
        default=DEFAULT_QUANT_METHOD,
        help="量化方法（默认 lsqplus_v1）",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "float", "ptq", "qat", "compare"),
        default="all",
        help="只跑指定阶段（默认全流程）",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--float-epochs", type=int, default=100)
    parser.add_argument("--qat-epochs", type=int, default=None,
                        help="默认 max(1, float-epochs*0.2)")
    parser.add_argument("--float-lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=None,
                        help="默认 float-lr x 0.1")
    parser.add_argument("--calibration-batches", type=int, default=20)
    # 冒烟/快速验证用：每个 epoch / 评估最多跑多少个 batch，默认不限制（完整训练）
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print(f"Python: {sys.executable}")
    print(f"torch: {torch.__version__} | Device: {device}")
    print(f"量化方法: {args.quant} | 阶段: {args.stage}")

    tag = args.quant
    qat_lr = args.qat_lr if args.qat_lr is not None else args.float_lr * 0.1
    qat_epochs = (
        args.qat_epochs
        if args.qat_epochs is not None
        else max(1, int(round(args.float_epochs * 0.2)))
    )

    common = dict(
        batch_size=args.batch_size,
        num_classes=10,
        num_workers=args.num_workers,
    )

    if args.stage in ("all", "float"):
        float_train(
            lr=args.float_lr,
            epochs=args.float_epochs,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            **common,
        )

    if args.stage in ("all", "ptq"):
        PTQ_calibration(
            quant_method=tag,
            calibration_batches=args.calibration_batches,
            max_eval_batches=args.max_eval_batches,
            **common,
        )

    if args.stage in ("all", "qat"):
        QAT_training(
            quant_method=tag,
            lr=qat_lr,
            epochs=qat_epochs,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            **common,
        )

    if args.stage in ("all", "compare"):
        compare_precision(
            quant_method=tag,
            max_eval_batches=args.max_eval_batches,
            **common,
        )
