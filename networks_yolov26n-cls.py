"""YOLO26n-cls 图像分类网络的 aLSQ+ 量化感知训练流程（imagenet10）。

流程：
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练（默认加载 yolo26n-cls.pt 预训练权重）
    PTQ_calibration()                               # 训练后量化校准
    QAT_training(lr=qat_lr, epochs=qat_epochs)      # 量化感知训练
    compare_precision()                             # 浮点 vs QAT 的 top-1/top-5 精度对比

网络结构与 ultralytics/cfg/models/26/yolo26-cls.yaml（scale=n, imgsz=224）逐层对齐：
与检测版 backbone 的差异是【没有 SPPF】，层 0-8 相同，层 9 直接是 C2PSA，
之后接官方分类头 Classify：Conv1x1(256->1280) -> AdaptiveAvgPool -> Dropout -> Linear(1280->nc)。
训练输出原始 logits（CrossEntropyLoss），评估输出 softmax 概率，指标为 top-1/top-5 accuracy。

关键点：
  * 数据管道用 ClassificationDataset（torchvision transforms），输出图像本身就是
    0~1 float，【不能】再 /255（与 detect/seg/pose 的 YOLODataset 不同）；
  * yolo26n-cls.pt 在 ImageNet 上是 nc=1000，微调到 imagenet10(nc=10) 时仅最后的
    Linear(1280*1000) 形状不匹配被跳过（迁移学习），backbone 与 1x1 Conv 全部加载；
  * 全局平均池化保持浮点 nn.AdaptiveAvgPool2d（与 networks_cifarCNN.py 的既有约定一致），
    其余 Conv2d / Linear 参与量化。
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import quantization as quant_pkg

# 复用检测网络的 backbone 组件与训练基础设施
# （文件名 networks_yolov26n-detect.py 含 '-'，不能直接 import，用文件路径加载）
det_spec = importlib.util.spec_from_file_location(
    "networks_yolov26n_detect",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "networks_yolov26n-detect.py"),
)
det = importlib.util.module_from_spec(det_spec)
det_spec.loader.exec_module(det)

# ultralytics 提供分类数据管道
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_cls_dataset
from ultralytics.data import ClassificationDataset, build_dataloader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ULTRA_DIR = os.path.join(BASE_DIR, "ultralytics", "ultralytics")
PRETRAINED_WEIGHTS = os.path.join(ULTRA_DIR, "yolo26n-cls.pt")

DATASET = "imagenet10"   # check_cls_dataset 自动解析/下载（12 train / 12 val, 10 类）
IMGSZ = 224
NUM_CLASSES = 10
CLS_HIDDEN = 1280        # 官方 Classify 头的 EfficientNet-B0 卷积通道数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 量化后端可切换：dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact
# backbone 组件从检测网络复用，因此切换时同步切换检测网络模块内绑定的后端算子。
# ---------------------------------------------------------------------------
DEFAULT_QUANT_METHOD = "lsqplus_v1"
QUANT_METHOD = DEFAULT_QUANT_METHOD
Q = quant_pkg.load_quant_backend(DEFAULT_QUANT_METHOD)

QuantAdd = Q.QuantAdd
QuantCat = Q.QuantConcat
QuantConv2d = Q.QuantConv2d
QuantLinear = Q.QuantLinear
QuantMaxPool = Q.QuantMaxPool


def set_quant_method(method):
    """切换量化后端（必须在构建 QuantYOLO26Cls 之前调用）。"""
    global Q, QUANT_METHOD
    global QuantAdd, QuantCat, QuantConv2d, QuantLinear, QuantMaxPool

    # 复用的 backbone block 类内部引用的是 det 模块的全局算子，必须先同步切换
    det.set_quant_method(method)
    Q = quant_pkg.load_quant_backend(method)
    QUANT_METHOD = method
    QuantAdd = det.QuantAdd
    QuantCat = det.QuantConcat
    QuantConv2d = det.QuantConv2d
    QuantLinear = Q.QuantLinear
    QuantMaxPool = det.QuantMaxPool
    return Q


# ============================== Classify 分类头 ==============================


class Classify(nn.Module):
    """复刻 ultralytics.nn.modules.head.Classify：x(B,256,7,7) -> x(B,nc)。

    conv: Conv1x1(256->1280)（量化）；pool: 全局平均池化（浮点，同 cifar 既有约定）；
    drop: Dropout(0.0)；linear: Linear(1280->nc)（量化）。
    训练返回原始 logits，评估返回 softmax 概率。
    """

    def __init__(self, c1=256, c2=NUM_CLASSES, c_=CLS_HIDDEN, quant=False):
        super().__init__()
        self.conv = det.Conv(c1, c_, k=1, quant=quant)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = (
            QuantLinear(c_, c2, a_bits=8, w_bits=8, per_channel=True)
            if quant else nn.Linear(c_, c2)
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        logits = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return logits if self.training else logits.softmax(1)


# ============================== 网络主体 ==============================


class YOLO26Cls(nn.Module):
    """yolo26n-cls（scale=n, imgsz=224）分类网络。quant=False 浮点模型，True 为伪量化模型。

    层编号 / 命名与 ultralytics 解析 yolo26-cls.yaml 得到的 ClassificationModel 一致
    （0-8 与检测版相同，无 SPPF，9=C2PSA，10=Classify），因此可直接加载
    yolo26n-cls.pt 的 backbone + 1x1 Conv 权重，仅末端 1000 类 Linear 因类别数不同
    （imagenet10 为 10 类）被跳过。
    """

    def __init__(self, nc=NUM_CLASSES, quant=False):
        super().__init__()
        self.quant = quant
        self.nc = nc
        layers = []

        # ---------------- backbone（与 yolo26-cls.yaml 对齐，无 SPPF） ----------------
        layers += [det._tag(det.Conv(3, 16, 3, 2, quant=quant), 0, -1)]       # P1/2
        layers += [det._tag(det.Conv(16, 32, 3, 2, quant=quant), 1, -1)]      # P2/4
        layers += [det._tag(det.C3k2(32, 64, n=1, c3k=False, e=0.25, quant=quant), 2, -1)]
        layers += [det._tag(det.Conv(64, 64, 3, 2, quant=quant), 3, -1)]      # P3/8
        layers += [det._tag(det.C3k2(64, 128, n=1, c3k=False, e=0.25, quant=quant), 4, -1)]
        layers += [det._tag(det.Conv(128, 128, 3, 2, quant=quant), 5, -1)]    # P4/16
        layers += [det._tag(det.C3k2(128, 128, n=1, c3k=True, quant=quant), 6, -1)]
        layers += [det._tag(det.Conv(128, 256, 3, 2, quant=quant), 7, -1)]    # P5/32
        layers += [det._tag(det.C3k2(256, 256, n=1, c3k=True, quant=quant), 8, -1)]
        layers += [det._tag(det.C2PSA(256, 256, n=1, quant=quant), 9, -1)]

        # ---------------- 分类头 ----------------
        layers += [det._tag(Classify(256, nc, quant=quant), 10, -1)]

        self.model = nn.ModuleList(layers)
        self._register_quantizer_buffers()

    def _register_quantizer_buffers(self):
        """把量化器的 init_state 注册为持久化 buffer（与检测/分割/pose 网络同一逻辑）。"""
        for module in self.modules():
            if "init_state" in dict(module.named_buffers()):
                continue
            if hasattr(module, "init_state"):
                value = int(module.init_state)
                del module.init_state
                module.register_buffer("init_state", torch.tensor(value), persistent=True)

    def forward(self, x):
        for module in self.model:
            x = module(x)
        return x


class FloatYOLO26Cls(YOLO26Cls):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__(nc=nc, quant=False)


class QuantYOLO26Cls(YOLO26Cls):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__(nc=nc, quant=True)


# ============================== 数据 / 评估 ==============================


def _make_cfg(num_workers):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = IMGSZ
    cfg.workers = num_workers
    return cfg


def _build_loader(cfg, path, data, batch_size, num_workers, augment, shuffle):
    dataset = ClassificationDataset(
        path, cfg, augment=augment, prefix="train" if augment else "val", names=data["names"]
    )
    return build_dataloader(
        dataset, batch_size, num_workers, shuffle=shuffle, rank=-1, device=device
    )


def get_dataloaders(batch_size=64, num_workers=2, calibration=False):
    """复用 ultralytics 官方 imagenet10 分类数据管道（图像已为 0~1 float）。"""
    data = check_cls_dataset(DATASET)
    cfg = _make_cfg(num_workers)
    if calibration:
        # 校准则用验证集、不做增强
        return _build_loader(cfg, data["val"], data, batch_size, num_workers, False, False), data
    train_loader = _build_loader(cfg, data["train"], data, batch_size, num_workers, True, True)
    val_loader = _build_loader(cfg, data["val"], data, batch_size, num_workers, False, False)
    return train_loader, val_loader, data


@torch.no_grad()
def evaluate(model, val_loader, max_batches=None):
    """top-1 / top-5 accuracy（评估态模型输出 softmax 概率）。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct1, correct5, total = 0.0, 0, 0, 0
    nb = len(val_loader)
    steps = nb if max_batches is None else min(nb, max_batches)

    for batch_index, batch in enumerate(val_loader):
        if batch_index >= steps:
            break
        images = batch["img"].float().to(device)  # 注意：ClassificationDataset 已归一化到 0~1
        targets = batch["cls"].long().to(device)
        outputs = model(images)  # (B, nc) softmax
        loss_sum += float(criterion(outputs, targets)) * images.shape[0]

        top5_pred = outputs.argsort(1, descending=True)[:, :5]
        correct1 += int((top5_pred[:, 0] == targets).sum())
        correct5 += int((top5_pred == targets[:, None]).any(dim=1).sum())
        total += images.shape[0]

    return {
        "test_loss": loss_sum / max(total, 1),
        "top1": correct1 / max(total, 1),
        "top5": correct5 / max(total, 1),
    }


# ============================== checkpoint / 权重复制 / 导出 ==============================


def load_checkpoint(model, path, return_meta=False):
    return det.load_checkpoint(model, path, return_meta=return_meta)


def save_checkpoint(model, path, **metadata):
    det.save_checkpoint(model, path, **metadata)


def load_pretrained(model, path=PRETRAINED_WEIGHTS):
    """加载官方 yolo26n-cls.pt（nc=1000）；末端 Linear 与 imagenet10(nc=10) 不匹配，跳过。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].float().state_dict()
    model_state = model.state_dict()
    # strict=False 仍会在形状不匹配时报错（1000 类 -> 10 类），因此先按形状过滤
    filtered, mismatched = {}, []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape != value.shape:
            mismatched.append(key)
        else:
            filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        print(f"[Pretrain] 警告：{len(unexpected)} 个未识别权重未加载: {unexpected[:3]}")
    print(
        f"[Pretrain] 已加载 {path}（末端分类 Linear 因 1000->{model.nc} 类别数不同跳过 "
        f"{len(mismatched)} 个形状不匹配参数，缺失 {len(missing)} 个）"
    )
    if mismatched:
        print(f"[Pretrain] 跳过的不匹配参数: {mismatched}")
    other_missing = [k for k in missing if "linear" not in k]
    if other_missing:
        print(f"[Pretrain] 非分类头缺失参数: {other_missing[:5]}")
    return model


def copy_float_to_quant(float_model, quant_model):
    return det.copy_float_to_quant(float_model, quant_model)


def build_float_model(quant_model, nc=NUM_CLASSES):
    """把量化模型的反量化权重灌进同结构的干净浮点模型（用于导出纯浮点 ONNX）。"""
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_state = quant_model.state_dict()
    for name, module in quant_model.named_modules():
        if quant_pkg.is_weight_quant_module(module):
            quant_state[f"{name}.weight"] = quant_pkg.dequantized_weight(module)

    float_model = FloatYOLO26Cls(nc=nc)
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
    dummy = torch.randn(1, 3, IMGSZ, IMGSZ, device=model_device)
    torch.onnx.export(
        float_model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=["images"],
        output_names=["preds"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "preds": {0: "batch_size"},
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
        print(f"      ONNX checker 通过，算子数: {len(op_types)}（含 Conv/Gemm 等）")
    except ImportError:
        print("      [skip] 未安装 onnx，跳过结构检查")

    try:
        ort = importlib.import_module("onnxruntime")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        torch.manual_seed(0)
        float_model.cpu().eval()
        x = torch.randn(1, 3, IMGSZ, IMGSZ)
        with torch.no_grad():
            y_torch = float_model(x)
        y_onnx = session.run(["preds"], {"images": x.numpy()})[0]
        max_diff = float(np.abs(y_torch.numpy() - y_onnx).max())
        print(f"      onnxruntime vs PyTorch 最大绝对误差: preds {max_diff:.3e}")
        assert max_diff < 1e-3, "ONNX 数值误差过大"
    except ImportError:
        print("      [skip] 未安装 onnxruntime，跳过数值比对")

    required_keys = quant_pkg.required_quant_keys(quant_model)
    missing = [key for key in required_keys if key not in quant_params]
    assert not missing, f"以下量化参数不完整: {missing[:10]}"
    print(f"      量化参数完整性检查通过（{len(quant_params)} 个张量）")


def save_quant_outputs(quant_model, prefix, nc=NUM_CLASSES, meta=None):
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_yolo26ncls.pth")
    save_checkpoint(quant_model, quant_checkpoint, **(meta or {}))

    quant_params = collect_quant_params(quant_model)
    json_path = os.path.join(MODEL_DIR, f"{prefix}_yolo26ncls_quant_params.json")
    pth_path = os.path.join(MODEL_DIR, f"{prefix}_yolo26ncls_quant_params.pth")
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
    print(f"[2/5] 量化参数已写出: {json_path} / {pth_path}（{len(quant_params)} 个张量）")

    float_model = build_float_model(quant_model, nc=nc)
    float_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_yolo26ncls_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(MODEL_DIR, f"{prefix}_yolo26ncls_float.onnx")
    export_onnx(float_model, onnx_path, opset=16)
    print(f"[4/5] 干净浮点 ONNX 已写出: {onnx_path}")

    verify(float_model, quant_model, onnx_path, quant_params)
    print("[5/5] 完成。")
    return quant_checkpoint, json_path, pth_path, float_checkpoint, onnx_path


# ============================== 训练各阶段 ==============================


def train_one_epoch(model, loader, criterion, optimizer, max_batches=None):
    """简单 Adam 训练循环（同 networks_cifarCNN.py 风格），返回平均 loss。"""
    model.train()
    total_loss, num_steps = 0.0, 0
    nb = len(loader)
    total_steps = nb if max_batches is None else min(nb, max_batches)
    for i, batch in enumerate(loader):
        if i >= total_steps:
            break
        # ClassificationDataset 输出已是 0~1 float，不要再 /255
        images = batch["img"].float().to(device)
        targets = batch["cls"].long().to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        num_steps += 1
    return total_loss / max(num_steps, 1)


def float_train(batch_size=64, lr=1e-3, epochs=100, num_classes=NUM_CLASSES, num_workers=2,
                max_train_batches=None, max_eval_batches=None):
    print("========== Float training (YOLO26n-cls / imagenet10) ==========")
    print(f"Device: {device}")
    print(f"Optimizer: Adam(lr={lr:g}) | loss: CrossEntropy | metric: top-1/top-5")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26Cls(num_classes).to(device)
    load_pretrained(float_model, PRETRAINED_WEIGHTS)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(float_model.parameters(), lr=lr)

    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    checkpoint_path = os.path.join(MODEL_DIR, "yolo26n-cls.pth")
    last_checkpoint = os.path.join(MODEL_DIR, "yolo26n-cls_last.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            float_model, train_loader, criterion, optimizer, max_batches=max_train_batches
        )
        metrics = evaluate(float_model, val_loader, max_batches=max_eval_batches)
        fitness = metrics["top1"]

        epoch_meta = {
            "stage": "float",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            **metrics,
        }
        save_checkpoint(float_model, last_checkpoint, **epoch_meta)

        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch + 1
            best_meta = epoch_meta
            save_checkpoint(float_model, checkpoint_path, **best_meta)

        print(
            f"[Float] Epoch [{epoch + 1}/{epochs}] | train loss:{train_loss:.4f} | "
            f"test loss:{metrics['test_loss']:.4f} top1:{metrics['top1']:.4f} top5:{metrics['top5']:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)
    best_metrics = evaluate(float_model, val_loader, max_batches=max_eval_batches)
    print(f"[Float] Best checkpoint: {checkpoint_path}（epoch {best_epoch}/{epochs}）")
    print(f"[Float] Best top1:{best_metrics['top1']:.4f} top5:{best_metrics['top5']:.4f}")
    return checkpoint_path, best_fitness, best_meta


@torch.no_grad()
def calibrate_quantizer(quant_model, calibration_loader, calibration_batches=20):
    quant_model.eval()
    for batch_index, batch in enumerate(calibration_loader):
        quant_model(batch["img"].float().to(device))
        if batch_index + 1 >= calibration_batches:
            break
    quant_pkg.freeze_batch_init(quant_model)


def PTQ_calibration(quant_method=DEFAULT_QUANT_METHOD, batch_size=64, num_classes=NUM_CLASSES,
                    calibration_batches=20, num_workers=2, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== PTQ calibration (YOLO26n-cls / {tag}) ==========")
    print(f"Device: {device}")

    calibration_loader, data = get_dataloaders(batch_size, num_workers, calibration=True)
    _, val_loader, _ = get_dataloaders(batch_size, num_workers)

    float_checkpoint = os.path.join(MODEL_DIR, "yolo26n-cls.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatYOLO26Cls(num_classes).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = QuantYOLO26Cls(num_classes).to(device)
    copy_float_to_quant(float_model, ptq_model)

    calibrate_quantizer(ptq_model, calibration_loader, calibration_batches)

    metrics = evaluate(ptq_model, val_loader, max_batches=max_eval_batches)
    print(f"[PTQ-{tag}] Test loss:{metrics['test_loss']:.4f} top1:{metrics['top1']:.4f} top5:{metrics['top5']:.4f}")

    ptq_meta = {
        "stage": "ptq",
        "quant_method": tag,
        "epoch": 0,
        "total_epochs": 0,
        "lr": None,
        "calibration_batches": calibration_batches,
        **metrics,
    }
    ptq_checkpoint = save_quant_outputs(
        ptq_model, f"ptq_{tag}", nc=num_classes, meta=ptq_meta
    )[0]

    load_checkpoint(ptq_model, ptq_checkpoint)
    quant_pkg.freeze_batch_init(ptq_model)
    final_metrics = evaluate(ptq_model, val_loader, max_batches=max_eval_batches)
    print(f"[PTQ] 重载 checkpoint 后 top1:{final_metrics['top1']:.4f} top5:{final_metrics['top5']:.4f}")
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=64, lr=1e-4, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training (YOLO26n-cls / {tag}) ==========")
    print(f"Device: {device}")
    print(f"Optimizer: Adam(lr={lr:g}) | loss: CrossEntropy | metric: top-1/top-5")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)

    ptq_checkpoint = os.path.join(MODEL_DIR, f"ptq_{tag}_yolo26ncls.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = QuantYOLO26Cls(num_classes).to(device)
    _, ptq_meta = load_checkpoint(qat_model, ptq_checkpoint, return_meta=True)
    saved_method = ptq_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] PTQ checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    quant_pkg.freeze_batch_init(qat_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr)

    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    best_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_yolo26ncls_best.pth")
    last_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_yolo26ncls_last.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            qat_model, train_loader, criterion, optimizer, max_batches=max_train_batches
        )
        metrics = evaluate(qat_model, val_loader, max_batches=max_eval_batches)
        fitness = metrics["top1"]

        epoch_meta = {
            "stage": "qat",
            "quant_method": tag,
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            **metrics,
        }
        save_checkpoint(qat_model, last_checkpoint, **epoch_meta)

        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch + 1
            best_meta = epoch_meta
            save_checkpoint(qat_model, best_checkpoint, **best_meta)

        print(
            f"[QAT-{tag}] Epoch [{epoch + 1}/{epochs}] | train loss:{train_loss:.4f} | "
            f"test loss:{metrics['test_loss']:.4f} top1:{metrics['top1']:.4f} top5:{metrics['top5']:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    best_metrics = evaluate(qat_model, val_loader, max_batches=max_eval_batches)
    print(
        f"[QAT-{tag}] Best epoch:{best_epoch}/{epochs} | "
        f"top1:{best_metrics['top1']:.4f} top5:{best_metrics['top5']:.4f} | Best: {best_checkpoint}"
    )
    qat_checkpoint = save_quant_outputs(
        qat_model, f"qat_{tag}", nc=num_classes, meta=best_meta
    )[0]
    return qat_checkpoint, best_fitness, best_meta


def compare_precision(quant_method=DEFAULT_QUANT_METHOD, batch_size=64, num_classes=NUM_CLASSES,
                      num_workers=2, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== Float vs QAT precision (YOLO26n-cls / imagenet10 / {tag}) ==========")
    print(f"Device: {device}")

    float_checkpoint = os.path.join(MODEL_DIR, "yolo26n-cls.pth")
    qat_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_yolo26ncls.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26Cls(num_classes).to(device)
    float_model, float_meta = load_checkpoint(float_model, float_checkpoint, return_meta=True)
    float_metrics = evaluate(float_model, val_loader, max_batches=max_eval_batches)

    qat_model = QuantYOLO26Cls(num_classes).to(device)
    qat_model, qat_meta = load_checkpoint(qat_model, qat_checkpoint, return_meta=True)
    saved_method = qat_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] QAT checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    quant_pkg.freeze_batch_init(qat_model)
    qat_metrics = evaluate(qat_model, val_loader, max_batches=max_eval_batches)

    print(
        f"[Compare] Float | best epoch:{float_meta.get('epoch', '-')}/"
        f"{float_meta.get('total_epochs', '-')} | "
        f"test loss:{float_metrics['test_loss']:.4f} top1:{float_metrics['top1']:.4f} "
        f"top5:{float_metrics['top5']:.4f}"
    )
    print(
        f"[Compare] QAT   | best epoch:{qat_meta.get('epoch', '-')}/"
        f"{qat_meta.get('total_epochs', '-')} | "
        f"test loss:{qat_metrics['test_loss']:.4f} top1:{qat_metrics['top1']:.4f} "
        f"top5:{qat_metrics['top5']:.4f}"
    )
    print(
        f"[Compare] delta | top1:{qat_metrics['top1'] - float_metrics['top1']:+.4f} "
        f"top5:{qat_metrics['top5'] - float_metrics['top5']:+.4f}"
    )
    return {"float": float_metrics, "qat": qat_metrics, "float_meta": float_meta, "qat_meta": qat_meta}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="YOLO26n-cls: 浮点训练(加载 yolo26n-cls.pt) -> PTQ -> QAT -> top1/top5 对比，量化方法可选"
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
    parser.add_argument("--float-batch-size", type=int, default=64)
    parser.add_argument("--qat-batch-size", type=int, default=64)
    parser.add_argument("--ptq-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--float-epochs", type=int, default=100)
    parser.add_argument("--qat-epochs", type=int, default=None,
                        help="默认 max(1, float-epochs*0.2)")
    parser.add_argument("--float-lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument("--calibration-batches", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print(f"Python: {sys.executable}")
    print(f"torch: {torch.__version__} | Device: {device}")
    print(f"量化方法: {args.quant} | 阶段: {args.stage}")

    tag = args.quant
    qat_epochs = (
        args.qat_epochs
        if args.qat_epochs is not None
        else max(1, int(round(args.float_epochs * 0.2)))
    )

    if args.stage in ("all", "float"):
        float_train(
            batch_size=args.float_batch_size,
            lr=args.float_lr,
            epochs=args.float_epochs,
            num_classes=NUM_CLASSES,
            num_workers=args.num_workers,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
        )

    if args.stage in ("all", "ptq"):
        PTQ_calibration(
            quant_method=tag,
            batch_size=args.ptq_batch_size,
            num_classes=NUM_CLASSES,
            calibration_batches=args.calibration_batches,
            num_workers=args.num_workers,
            max_eval_batches=args.max_eval_batches,
        )

    if args.stage in ("all", "qat"):
        QAT_training(
            quant_method=tag,
            batch_size=args.qat_batch_size,
            lr=args.qat_lr,
            epochs=qat_epochs,
            num_classes=NUM_CLASSES,
            num_workers=args.num_workers,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
        )

    if args.stage in ("all", "compare"):
        compare_precision(
            quant_method=tag,
            batch_size=args.ptq_batch_size,
            num_classes=NUM_CLASSES,
            num_workers=args.num_workers,
            max_eval_batches=args.max_eval_batches,
        )
