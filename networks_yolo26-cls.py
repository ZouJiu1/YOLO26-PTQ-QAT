"""YOLO26n-cls 图像分类网络的 aLSQ+ 量化感知训练流程（imagenet10） / aLSQ+ quantization-aware training pipeline for YOLO26n-cls image classification network (imagenet10).

流程 / Pipeline:
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练（默认加载 yolo26n-cls.pt 预训练权重） / float training (default loads yolo26n-cls.pt pretrained weights)
    PTQ_calibration()                               # 训练后量化校准 / post-training quantization calibration
    QAT_training(lr=qat_lr, epochs=qat_epochs)      # 量化感知训练 / quantization-aware training
    compare_precision()                             # 浮点 vs QAT 的 top-1/top-5 精度对比 / top-1/top-5 accuracy comparison between float and QAT

网络结构与 ultralytics/cfg/models/26/yolo26-cls.yaml（scale=n, imgsz=224）逐层对齐：
与检测版 backbone 的差异是【没有 SPPF】，层 0-8 相同，层 9 直接是 C2PSA，
之后接官方分类头 Classify：Conv1x1(256->1280) -> AdaptiveAvgPool -> Dropout -> Linear(1280->nc)。
训练输出原始 logits（CrossEntropyLoss），评估输出 softmax 概率，指标为 top-1/top-5 accuracy.
/ Network structure is aligned layer-by-layer with ultralytics/cfg/models/26/yolo26-cls.yaml (scale=n, imgsz=224);
the difference from the detection backbone is 【no SPPF】, layers 0-8 are the same, layer 9 is directly C2PSA,
followed by the official Classify head: Conv1x1(256->1280) -> AdaptiveAvgPool -> Dropout -> Linear(1280->nc).
Training outputs raw logits (CrossEntropyLoss), evaluation outputs softmax probabilities, metrics are top-1/top-5 accuracy.

关键点 / Key points:
  * 数据管道用 ClassificationDataset（torchvision transforms），输出图像本身就是
    0~1 float，【不能】再 /255（与 detect/seg/pose 的 YOLODataset 不同）；
    / data pipeline uses ClassificationDataset (torchvision transforms), output images are already
    0~1 float, 【must not】 divide by 255 again (unlike YOLODataset for detect/seg/pose);
  * yolo26n-cls.pt 在 ImageNet 上是 nc=1000，微调到 imagenet10(nc=10) 时仅最后的
    Linear(1280*1000) 形状不匹配被跳过（迁移学习），backbone 与 1x1 Conv 全部加载；
    / yolo26n-cls.pt has nc=1000 on ImageNet; when fine-tuning to imagenet10 (nc=10), only the final
    Linear(1280*1000) with mismatched shape is skipped (transfer learning), backbone and 1x1 Conv are all loaded;
  * 全局平均池化保持浮点 nn.AdaptiveAvgPool2d（与 networks_cifarCNN.py 的既有约定一致），
    其余 Conv2d / Linear 参与量化。
    / global average pooling stays as float nn.AdaptiveAvgPool2d (consistent with existing convention in networks_cifarCNN.py),
    other Conv2d / Linear layers participate in quantization.
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

# 复用检测网络的 backbone 组件与训练基础设施 / Reuse backbone components and training infrastructure from the detection network
# （文件名 networks_yolo26-detect.py 含 '-'，不能直接 import，用文件路径加载） / (filename networks_yolo26-detect.py contains '-', cannot import directly, load via file path)
det_spec = importlib.util.spec_from_file_location(
    "networks_yolo26_detect",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "networks_yolo26-detect.py"),
)
det = importlib.util.module_from_spec(det_spec)
det_spec.loader.exec_module(det)

# ultralytics 提供分类数据管道 / Classification data pipeline provided by ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_cls_dataset
from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.utils.torch_utils import model_info
from ultralytics.utils.plotting import plot_images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR_BASE = os.path.join(BASE_DIR, "model", "yolo26-cls")
MODEL_DIR = _MODEL_DIR_BASE


def _model_dir_for(scale=None, quant_method=None):
    """按网络名 + 尺度 + 量化后端(+非默认量化配置标签)返回产物目录 /
    Return artifact directory by net name + scale + quant backend (+ non-default quant config tag)."""
    path = _MODEL_DIR_BASE
    if scale is not None:
        path = os.path.join(path, scale)
    if quant_method is not None:
        path = os.path.join(path, quant_method + det._quant_cfg_tag())
    os.makedirs(path, exist_ok=True)
    return path
ULTRA_DIR = os.path.join(BASE_DIR, "ultralytics", "ultralytics")
PRETRAINED_WEIGHTS = os.path.join(ULTRA_DIR, "yolo26n-cls.pt")

DATASET = "imagenet10"   # check_cls_dataset 自动解析/下载（12 train / 12 val, 10 类） / auto-resolve/download (12 train / 12 val, 10 classes)
IMGSZ = 224
NUM_CLASSES = 10
CLS_HIDDEN = 1280        # 官方 Classify 头的 EfficientNet-B0 卷积通道数 / EfficientNet-B0 conv channels of the official Classify head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 量化后端可切换：dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact / Quantization backend is switchable: dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact
# backbone 组件从检测网络复用，因此切换时同步切换检测网络模块内绑定的后端算子。 / Backbone components are reused from the detection network, so switching must also switch the backend ops bound in the detection module.
# ---------------------------------------------------------------------------
DEFAULT_QUANT_METHOD = "lsqplus_v1"
QUANT_METHOD = DEFAULT_QUANT_METHOD
Q = quant_pkg.load_quant_backend(DEFAULT_QUANT_METHOD)

QuantAdd = Q.QuantAdd
QuantCat = Q.QuantConcat
QuantConv2d = Q.QuantConv2d
QuantLinear = Q.QuantLinear
QuantMaxPool = Q.QuantMaxPool
QuantSiLU = getattr(Q, 'QuantSiLU', None)
QuantSigmoid = getattr(Q, 'QuantSigmoid', None)
QuantMatMul = getattr(Q, 'QuantMatMul', None)


def set_quant_method(method):
    """切换量化后端（必须在构建 QuantYOLO26Cls 之前调用）。 / Switch quantization backend (must be called before constructing QuantYOLO26Cls)."""
    global Q, QUANT_METHOD
    global QuantAdd, QuantCat, QuantConv2d, QuantLinear, QuantMaxPool, QuantSiLU, QuantSigmoid, QuantMatMul

    # 复用的 backbone block 类内部引用的是 det 模块的全局算子，必须先同步切换 / Reused backbone block classes reference the det module's global ops internally, must switch them first
    det.set_quant_method(method)
    Q = quant_pkg.load_quant_backend(method)
    QUANT_METHOD = method
    QuantAdd = det.QuantAdd
    QuantCat = det.QuantConcat
    QuantConv2d = det.QuantConv2d
    QuantLinear = Q.QuantLinear
    QuantMaxPool = det.QuantMaxPool
    QuantSiLU = getattr(Q, 'QuantSiLU', None)
    QuantSigmoid = getattr(Q, 'QuantSigmoid', None)
    QuantMatMul = getattr(Q, 'QuantMatMul', None)
    return Q


# ============================== Classify 分类头 / Classification head ==============================


class Classify(nn.Module):
    """复刻 ultralytics.nn.modules.head.Classify：x(B,256,7,7) -> x(B,nc)。 / Replicate ultralytics.nn.modules.head.Classify: x(B,256,7,7) -> x(B,nc).

    conv: Conv1x1(256->1280)（量化）；pool: 全局平均池化（浮点，同 cifar 既有约定）；
    / conv: Conv1x1(256->1280) (quantized); pool: global average pooling (float, same as cifar convention);
    drop: Dropout(0.0)；linear: Linear(1280->nc)（量化）。 / drop: Dropout(0.0); linear: Linear(1280->nc) (quantized).
    训练返回原始 logits，评估返回 softmax 概率。 / Training returns raw logits, evaluation returns softmax probabilities.
    """

    def __init__(self, c1=256, c2=NUM_CLASSES, c_=CLS_HIDDEN, quant=False):
        super().__init__()
        # 混合量化时分类头保持 FP32 / Under mixed quant the classification head stays FP32
        hq = quant and not det.QUANT_CFG.mixed_quant
        self.conv = det.Conv(c1, c_, k=1, quant=hq)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = (
            QuantLinear(c_, c2, **det._quant_layer_kwargs())
            if hq else nn.Linear(c_, c2)
        )

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        logits = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return logits if self.training else logits.softmax(1)


# ============================== 网络主体 / Network body ==============================


class YOLO26Cls(nn.Module):
    """yolo26n-cls（scale=n, imgsz=224）分类网络。quant=False 浮点模型，True 为伪量化模型。 / yolo26n-cls (scale=n, imgsz=224) classification network. quant=False is float model, True is pseudo-quantized model.

    层编号 / 命名与 ultralytics 解析 yolo26-cls.yaml 得到的 ClassificationModel 一致
    （0-8 与检测版相同，无 SPPF，9=C2PSA，10=Classify），因此可直接加载
    yolo26n-cls.pt 的 backbone + 1x1 Conv 权重，仅末端 1000 类 Linear 因类别数不同
    （imagenet10 为 10 类）被跳过。
    / Layer numbering / naming matches the ClassificationModel parsed from yolo26-cls.yaml by ultralytics
    (0-8 same as detection version, no SPPF, 9=C2PSA, 10=Classify), so yolo26n-cls.pt
    backbone + 1x1 Conv weights can be loaded directly; only the final 1000-class Linear
    is skipped due to different class count (imagenet10 has 10 classes).
    """

    def __init__(self, nc=NUM_CLASSES, quant=False, scale=det.DEFAULT_SCALE):
        super().__init__()
        self.quant = quant
        self.nc = nc
        self.scale = det.get_scale(scale)
        s = self.scale
        C = lambda c: det.scaled_channels(c, s)   # 通道缩放 / Channel scaling
        N = lambda r: det.scaled_repeats(r, s)    # 层重复次数缩放 / Layer repeat count scaling
        # ultralytics parse_model 特殊规则：M/L/X 尺度所有 C3k2 强制 c3k=True / ultralytics parse_model special rule: for M/L/X scales all C3k2 force c3k=True
        c3k_all = s in ("m", "l", "x")
        layers = []

        # ---------------- backbone（与 yolo26-cls.yaml 对齐，无 SPPF） / backbone (aligned with yolo26-cls.yaml, no SPPF) ----------------
        # 混合量化：首层 stem（3→64）保持 FP32 / Mixed quant: stem (3→64) stays FP32
        stem_quant = quant and not det.QUANT_CFG.mixed_quant
        layers += [det._tag(det.Conv(3, C(64), 3, 2, quant=stem_quant), 0, -1)]               # P1/2
        layers += [det._tag(det.Conv(C(64), C(128), 3, 2, quant=quant), 1, -1)]          # P2/4
        layers += [det._tag(det.C3k2(C(128), C(256), n=N(2), c3k=c3k_all, e=0.25, quant=quant), 2, -1)]
        layers += [det._tag(det.Conv(C(256), C(256), 3, 2, quant=quant), 3, -1)]         # P3/8
        layers += [det._tag(det.C3k2(C(256), C(512), n=N(2), c3k=c3k_all, e=0.25, quant=quant), 4, -1)]
        layers += [det._tag(det.Conv(C(512), C(512), 3, 2, quant=quant), 5, -1)]          # P4/16
        layers += [det._tag(det.C3k2(C(512), C(512), n=N(2), c3k=True, quant=quant), 6, -1)]
        layers += [det._tag(det.Conv(C(512), C(1024), 3, 2, quant=quant), 7, -1)]         # P5/32
        layers += [det._tag(det.C3k2(C(1024), C(1024), n=N(2), c3k=True, quant=quant), 8, -1)]
        layers += [det._tag(det.C2PSA(C(1024), C(1024), n=N(2), quant=quant), 9, -1)]

        # ---------------- 分类头 / Classification head ----------------
        layers += [det._tag(Classify(C(1024), nc, quant=quant), 10, -1)]

        self.model = nn.ModuleList(layers)
        self._register_quantizer_buffers()

    def _register_quantizer_buffers(self):
        """把量化器的 init_state 注册为持久化 buffer（与检测/分割/pose 网络同一逻辑）。 / Register quantizer init_state as a persistent buffer (same logic as detection/segmentation/pose networks)."""
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
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
        super().__init__(nc=nc, quant=False, scale=scale)


class QuantYOLO26Cls(YOLO26Cls):
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
        super().__init__(nc=nc, quant=True, scale=scale)


# ============================== 数据 / 评估 / Data / Evaluation ==============================


def _make_cfg(num_workers):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = IMGSZ
    cfg.workers = num_workers
    return cfg


def _build_loader(cfg, path, data, batch_size, num_workers, augment, shuffle):
    dataset = ClassificationDataset(
        path, cfg, augment=augment, prefix="train" if augment else "val",
        names=data["names"] if isinstance(data, dict) else getattr(data, "names", None)
    )
    return build_dataloader(
        dataset, batch_size, num_workers, shuffle=shuffle, rank=-1, device=device
    )


def get_dataloaders(batch_size=64, num_workers=2, calibration=False):
    """复用 ultralytics 官方 imagenet10 分类数据管道（图像已为 0~1 float）。 / Reuse ultralytics official imagenet10 classification data pipeline (images are already 0~1 float)."""
    data = check_cls_dataset(DATASET)
    cfg = _make_cfg(num_workers)
    if calibration:
        # 校准则用验证集、不做增强 / Calibration uses validation set, no augmentation
        return _build_loader(cfg, data["val"], data, batch_size, num_workers, False, False), data
    train_loader = _build_loader(cfg, data["train"], data, batch_size, num_workers, True, True)
    val_loader = _build_loader(cfg, data["val"], data, batch_size, num_workers, False, False)
    return train_loader, val_loader, data


def _maybe_visualize_cls(images, outputs, targets, names, viz_dir, viz_prefix, viz_state, viz_max):
    """分类任务可视化：用 ultralytics 官方 plot_images 在拼图上标注 GT 与预测类别（val_batch 风格） /
    Classification visualization: annotate GT and predicted classes on mosaics via official ultralytics plot_images (val_batch style).

    viz_state 跨 batch 记录已保存图片数 / viz_state tracks saved image count across batches.
    """
    os.makedirs(viz_dir, exist_ok=True)
    n = min(images.shape[0], viz_max - viz_state["saved"])
    bi = viz_state["batch"]
    idx = torch.arange(n)
    # GT 拼图 / GT mosaic
    plot_images(
        labels={"cls": targets[:n], "batch_idx": idx},
        images=images[:n],
        fname=os.path.join(viz_dir, f"{viz_prefix}_batch{bi}_labels.jpg"),
        names=names,
        max_subplots=viz_max,
        threaded=False,  # 训练循环内同步执行，避免线程堆积 / run synchronously inside training loop to avoid thread pile-up
    )
    # 预测拼图（带 top-1 置信度）/ prediction mosaic (with top-1 confidence)
    conf, pred = outputs[:n].max(dim=1)
    plot_images(
        labels={"cls": pred, "conf": conf, "batch_idx": idx},
        images=images[:n],
        fname=os.path.join(viz_dir, f"{viz_prefix}_batch{bi}_pred.jpg"),
        names=names,
        max_subplots=viz_max,
        threaded=False,
    )
    viz_state["saved"] += n
    viz_state["batch"] += 1


@torch.no_grad()
def evaluate(model, val_loader, max_batches=None, viz_dir=None, viz_max=30, viz_prefix="eval"):
    """top-1 / top-5 accuracy（评估态模型输出 softmax 概率）。若 viz_dir 不为空，
    额外把前 viz_max 张验证图的 GT/预测类别拼图保存到 {viz_dir}/fvisualize/ /
    top-1 / top-5 accuracy (model in eval mode outputs softmax probabilities). If viz_dir is set,
    additionally save GT/predicted-class mosaics of the first viz_max val images into {viz_dir}/fvisualize/."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct1, correct5, total = 0.0, 0, 0, 0
    nb = len(val_loader)
    steps = nb if max_batches is None else min(nb, max_batches)
    # 可视化输出目录与计数器 / visualization output dir and counters
    viz_out = os.path.join(viz_dir, "fvisualize") if viz_dir else None
    viz_state = {"saved": 0, "batch": 0}
    if viz_out:
        os.makedirs(viz_out, exist_ok=True)
    names = getattr(val_loader.dataset, "names", None)  # 类别名（缺失时显示索引）/ class names (fallback to indices)

    for batch_index, batch in enumerate(val_loader):
        if batch_index >= steps:
            break
        images = batch["img"].float().to(device)  # 注意：ClassificationDataset 已归一化到 0~1 / Note: ClassificationDataset already normalizes to 0~1
        targets = batch["cls"].long().to(device)
        outputs = model(images)  # (B, nc) softmax
        loss_sum += float(criterion(outputs, targets)) * images.shape[0]

        top5_pred = outputs.argsort(1, descending=True)[:, :5]
        correct1 += int((top5_pred[:, 0] == targets).sum())
        correct5 += int((top5_pred == targets[:, None]).any(dim=1).sum())
        total += images.shape[0]

        # 每次评估都可视化前几批（失败仅告警，绝不影响评估） / visualize first batches on every evaluate (failure only warns, never breaks eval)
        if viz_out and viz_state["saved"] < viz_max:
            try:
                _maybe_visualize_cls(images, outputs, targets, names, viz_out,
                                     viz_prefix, viz_state, viz_max)
            except Exception as exc:
                print(f"      [viz] 可视化保存失败（仅告警）: {exc} / visualization save failed (warn only): {exc}")

    return {
        "test_loss": loss_sum / max(total, 1),
        "top1": correct1 / max(total, 1),
        "top5": correct5 / max(total, 1),
    }


# ============================== checkpoint / 权重复制 / 导出 / checkpoint / weight copy / export ==============================


def load_checkpoint(model, path, return_meta=False):
    return det.load_checkpoint(model, path, return_meta=return_meta)


def save_checkpoint(model, path, **metadata):
    det.save_checkpoint(model, path, **metadata)


def load_pretrained(model, path=None, scale=det.DEFAULT_SCALE):
    """加载官方 yolo26{scale}-cls.pt（nc=1000）；末端 Linear 与 imagenet10(nc=10) 不匹配，跳过。 / Load official yolo26{scale}-cls.pt (nc=1000); final Linear does not match imagenet10 (nc=10), skipped.

    本地缺失的官方权重（如 yolo26s-cls.pt）会按 ultralytics 方式自动下载；仅离线或自定义路径缺失时才从头训练。
    / Missing official weights (e.g. yolo26s-cls.pt) are auto-downloaded like ultralytics; random init only when offline or custom path missing.
    """
    if path is None:
        path = os.path.join(ULTRA_DIR, f"{det.base_name(scale, 'cls')}.pt")
    # 官方权重名缺失时自动下载（同 ultralytics）；自定义路径缺失则从头训练 /
    # Auto-download official asset names (like ultralytics); missing custom path -> train from scratch
    path = det.resolve_pretrained_path(path)
    if path is None:
        print(f"[Pretrain] [warn] 未找到预训练权重，{det.model_name(scale)}-cls 将从头训练")
        return model
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].float().state_dict()
    model_state = model.state_dict()
    # strict=False 仍会在形状不匹配时报错（1000 类 -> 10 类），因此先按形状过滤 / strict=False still errors on shape mismatch (1000 classes -> 10 classes), so filter by shape first
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


def build_float_model(quant_model, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE, use_clip=False):
    """把量化模型的反量化权重灌进同结构的干净浮点模型（用于导出纯浮点 ONNX）。 / Pour dequantized weights from quantized model into a clean float model of the same structure (for exporting pure float ONNX)."""
    # 委托通用构建器（烘焙反量化权重；use_clip 控制激活伪量化换恒等或仅截断），nc/scale 仅兼容旧签名 /
    # Delegates to the generic builder (bakes dequantized weights; use_clip selects identity vs clip-only activation replacement); nc/scale kept for signature compatibility.
    return quant_pkg.build_float_model(quant_model, use_clip=use_clip)


def collect_quant_params(quant_model):
    return quant_pkg.collect_quant_params(quant_model)


def export_onnx(float_model, onnx_path, opset=16, imgsz=None):
    """导出 ONNX，输入形状完全固定为 [1, 3, H, W]（无 dynamic_axes，图尺寸清晰可见） /
    Export ONNX with fully static input shape [1, 3, H, W] (no dynamic_axes, graph dimensions clearly visible)."""
    H = W = imgsz if imgsz is not None else IMGSZ
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
        input_names=["images"],
        output_names=["preds"],
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
    """训练保存 best checkpoint 时同步导出 ONNX；失败仅告警，绝不影响训练。 / Synchronously export ONNX when saving best checkpoint during training; failure only warns, never affects training.

    导出后模型被置为 eval，由下一轮 train_one_epoch 的 model.train() 恢复。
    / After export, model is set to eval; it is restored by model.train() in the next train_one_epoch round.
    """
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
        print(f"      ONNX checker 通过，算子数: {len(op_types)}（含 Conv/Gemm 等）")
    except ImportError:
        print("      [skip] 未安装 onnx，跳过结构检查")

    # 不做「导出 ONNX vs PyTorch 浮点」数值比对：导出的 ONNX 不含 scale/zero_point
    # （板端 NPU/TPU 的 PTQ 工具会自行校准），该比对不构成部署口径；量化精度对比由
    # compare 阶段的 QAT-vs-float 指标承担 / No exported-ONNX-vs-float numeric check:
    # the exported ONNX carries no scale/zero_point (board NPU/TPU PTQ tools calibrate
    # them), so the comparison is not deployment-meaningful; quant accuracy is compared
    # by the compare stage's QAT-vs-float metrics.

    required_keys = quant_pkg.required_quant_keys(quant_model)
    missing = [key for key in required_keys if key not in quant_params]
    assert not missing, f"以下量化参数不完整: {missing[:10]}"
    print(f"      量化参数完整性检查通过（{len(quant_params)} 个张量）")


def save_quant_outputs(quant_model, prefix, nc=NUM_CLASSES, meta=None, scale=det.DEFAULT_SCALE,
                      model_dir=None):
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    if model_dir is None:
        model_dir = _MODEL_DIR_BASE
    base = det.base_name(scale, "cls")
    quant_checkpoint = os.path.join(model_dir, f"{prefix}_{base}.pth")
    save_checkpoint(quant_model, quant_checkpoint, **(meta or {}))

    quant_params = collect_quant_params(quant_model)
    json_path = os.path.join(model_dir, f"{prefix}_{base}_quant_params.json")
    pth_path = os.path.join(model_dir, f"{prefix}_{base}_quant_params.pth")
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

    float_model = build_float_model(quant_model, nc=nc, scale=scale)
    float_checkpoint = os.path.join(model_dir, f"{prefix}_{base}_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(model_dir, f"{prefix}_{base}_float.onnx")
    export_onnx(float_model, onnx_path, opset=16)
    print(f"[4/5] 干净浮点 ONNX 已写出: {onnx_path}")

    verify(float_model, quant_model, onnx_path, quant_params)
    print("[5/5] 完成。")
    return quant_checkpoint, json_path, pth_path, float_checkpoint, onnx_path


# ============================== 训练各阶段 / Training stages ==============================


def train_one_epoch(model, loader, criterion, optimizer, max_batches=None):
    """简单 Adam 训练循环（同 networks_cifarCNN.py 风格），返回平均 loss。 / Simple Adam training loop (same style as networks_cifarCNN.py), returns average loss."""
    model.train()
    total_loss, num_steps = 0.0, 0
    nb = len(loader)
    total_steps = nb if max_batches is None else min(nb, max_batches)
    for i, batch in enumerate(loader):
        if i >= total_steps:
            break
        # ClassificationDataset 输出已是 0~1 float，不要再 /255 / ClassificationDataset output is already 0~1 float, do not divide by 255 again
        images = batch["img"].float().to(device)
        targets = batch["cls"].long().to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        num_steps += 1
    return total_loss / max(num_steps, 1)


def float_train(batch_size=64, lr=1e-3, epochs=100, num_classes=NUM_CLASSES, num_workers=2,
                max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                resume=False):
    print(f"========== Float training ({det.model_name(scale)}-cls / imagenet10) ==========")
    print(f"Device: {device}")
    print(f"Optimizer: Adam(lr={lr:g}) | loss: CrossEntropy | metric: top-1/top-5")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26Cls(num_classes, scale=scale).to(device)
    det.model_info(float_model, imgsz=IMGSZ)
    load_pretrained(float_model, scale=scale)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(float_model.parameters(), lr=lr)

    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    model_dir = _model_dir_for(scale=scale)
    checkpoint_path = os.path.join(model_dir, f"{det.base_name(scale, 'cls')}_best.pth")
    last_checkpoint = os.path.join(model_dir, f"{det.base_name(scale, 'cls')}_last.pth")

    # resume：从 _last.pth 恢复，接续训练 / resume: restore from _last.pth and continue training
    start_epoch = 0
    if resume and os.path.exists(last_checkpoint):
        print(f"[Float] Resume from {last_checkpoint}")
        _, meta = det.load_checkpoint(float_model, last_checkpoint, return_meta=True)
        start_epoch = meta.get("epoch", 0)
        best_fitness = meta.get("fitness", -1.0)
        best_meta = meta
        best_epoch = start_epoch
        print(f"[Float] 从 epoch {start_epoch} 接续训练，当前 best top1={meta.get('top1', 0):.4f}")
    elif resume:
        print(f"[Float] Resume 开启但找不到 {last_checkpoint}，从头训练")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            float_model, train_loader, criterion, optimizer, max_batches=max_train_batches
        )
        metrics = evaluate(float_model, val_loader, max_batches=max_eval_batches,
                           viz_dir=model_dir, viz_prefix=f"float_ep{epoch + 1:03d}")
        fitness = metrics["top1"]

        epoch_meta = {
            "stage": "float",
            "scale": det.get_scale(scale),
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "fitness": float(fitness),
            **metrics,
        }
        save_checkpoint(float_model, last_checkpoint, **epoch_meta)

        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch + 1
            best_meta = epoch_meta
            save_checkpoint(float_model, checkpoint_path, **best_meta)
            _try_export_onnx(float_model, os.path.splitext(checkpoint_path)[0] + ".onnx")

        print(
            f"[Float] Epoch [{epoch + 1}/{epochs}] | train loss:{train_loss:.4f} | "
            f"test loss:{metrics['test_loss']:.4f} top1:{metrics['top1']:.4f} top5:{metrics['top5']:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)
    best_metrics = evaluate(float_model, val_loader, max_batches=max_eval_batches,
                            viz_dir=model_dir, viz_prefix="float_best")
    print(f"[Float] Best checkpoint: {checkpoint_path}（epoch {best_epoch}/{epochs}）")
    print(f"[Float] Best top1:{best_metrics['top1']:.4f} top5:{best_metrics['top5']:.4f}")
    return checkpoint_path, best_fitness, best_meta


@torch.no_grad()
def calibrate_quantizer(quant_model, calibration_loader, calibration_batches=20,
                         float_model=None):
    """校准量化器（委托给 detect 模块，注意 cls 不需要 /255 归一化）。 / Calibrate quantizer (delegated to detect module; note cls does not need /255 normalization)."""
    return det.calibrate_quantizer(quant_model, calibration_loader,
                                    calibration_batches, float_model=float_model,
                                    normalize_img=False)


def PTQ_calibration(quant_method=DEFAULT_QUANT_METHOD, batch_size=64, num_classes=NUM_CLASSES,
                    calibration_batches=20, num_workers=2, max_eval_batches=None,
                    scale=det.DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== PTQ calibration ({det.model_name(scale)}-cls / {tag}) ==========")
    print(f"Device: {device}")

    calibration_loader, data = get_dataloaders(batch_size, num_workers, calibration=True)
    _, val_loader, _ = get_dataloaders(batch_size, num_workers)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth（与 QAT 命名一致），fallback 到旧版无后缀 .pth / Prefer _best.pth (consistent with QAT naming), fallback to old version without suffix .pth
    float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'cls')}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'cls')}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatYOLO26Cls(num_classes, scale=scale).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = QuantYOLO26Cls(num_classes, scale=scale).to(device)
    det.model_info(ptq_model, imgsz=IMGSZ)
    copy_float_to_quant(float_model, ptq_model)

    calibrate_quantizer(ptq_model, calibration_loader, calibration_batches,
                         float_model=float_model)

    metrics = evaluate(ptq_model, val_loader, max_batches=max_eval_batches,
                       viz_dir=quant_dir, viz_prefix=f"ptq_{tag}")
    print(f"[PTQ-{tag}] Test loss:{metrics['test_loss']:.4f} top1:{metrics['top1']:.4f} top5:{metrics['top5']:.4f}")

    ptq_meta = {
        "stage": "ptq",
        "scale": det.get_scale(scale),
        "quant_method": tag,
        "epoch": 0,
        "total_epochs": 0,
        "lr": None,
        "calibration_batches": calibration_batches,
        **metrics,
    }
    ptq_checkpoint = save_quant_outputs(
        ptq_model, f"ptq_{tag}", nc=num_classes, meta=ptq_meta, scale=scale,
        model_dir=quant_dir,
    )[0]

    load_checkpoint(ptq_model, ptq_checkpoint)
    quant_pkg.freeze_batch_init(ptq_model)
    final_metrics = evaluate(ptq_model, val_loader, max_batches=max_eval_batches,
                             viz_dir=quant_dir, viz_prefix=f"ptq_{tag}_reload")
    print(f"[PTQ] 重载 checkpoint 后 top1:{final_metrics['top1']:.4f} top5:{final_metrics['top5']:.4f}")
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=64, lr=1e-4, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                 resume=False):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training ({det.model_name(scale)}-cls / {tag}) ==========")
    print(f"Device: {device}")
    print(f"Optimizer: Adam(lr={lr:g}) | loss: CrossEntropy | metric: top-1/top-5")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    ptq_checkpoint = os.path.join(quant_dir, f"ptq_{tag}_{det.base_name(scale, 'cls')}.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = QuantYOLO26Cls(num_classes, scale=scale).to(device)
    det.model_info(qat_model, imgsz=IMGSZ)
    _, ptq_meta = load_checkpoint(qat_model, ptq_checkpoint, return_meta=True)
    saved_method = ptq_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] PTQ checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    saved_scale = ptq_meta.get("scale")
    if saved_scale is not None and saved_scale != det.get_scale(scale):
        print(f"      [warn] PTQ checkpoint 记录的尺度为 yolo26{saved_scale}，当前为 yolo26{det.get_scale(scale)}")
    quant_pkg.freeze_batch_init(qat_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr)

    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    base = det.base_name(scale, "cls")
    best_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{base}_best.pth")
    last_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{base}_last.pth")

    # resume：从 _last.pth 恢复，接续训练 / resume: restore from _last.pth and continue training
    start_epoch = 0
    if resume and os.path.exists(last_checkpoint):
        print(f"[QAT-{tag}] Resume from {last_checkpoint}")
        _, meta = load_checkpoint(qat_model, last_checkpoint, return_meta=True)
        start_epoch = meta.get("epoch", 0)
        best_fitness = meta.get("fitness", -1.0)
        best_meta = meta
        best_epoch = start_epoch
        print(f"[QAT-{tag}] 从 epoch {start_epoch} 接续训练，当前 best top1={meta.get('top1', 0):.4f}")
    elif resume:
        print(f"[QAT-{tag}] Resume 开启但找不到 {last_checkpoint}，从头训练")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            qat_model, train_loader, criterion, optimizer, max_batches=max_train_batches
        )
        metrics = evaluate(qat_model, val_loader, max_batches=max_eval_batches,
                           viz_dir=quant_dir, viz_prefix=f"qat_{tag}_ep{epoch + 1:03d}")
        fitness = metrics["top1"]

        epoch_meta = {
            "stage": "qat",
            "scale": det.get_scale(scale),
            "quant_method": tag,
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "fitness": float(fitness),  # QAT checkpoint 需带 fitness 供接续/对比 / fitness required in QAT checkpoint meta
            **metrics,
        }
        save_checkpoint(qat_model, last_checkpoint, **epoch_meta)

        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch + 1
            best_meta = epoch_meta
            save_checkpoint(qat_model, best_checkpoint, **best_meta)
            _try_export_onnx(qat_model, os.path.splitext(best_checkpoint)[0] + ".onnx")

        print(
            f"[QAT-{tag}] Epoch [{epoch + 1}/{epochs}] | train loss:{train_loss:.4f} | "
            f"test loss:{metrics['test_loss']:.4f} top1:{metrics['top1']:.4f} top5:{metrics['top5']:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    best_metrics = evaluate(qat_model, val_loader, max_batches=max_eval_batches,
                            viz_dir=quant_dir, viz_prefix=f"qat_{tag}_best")
    print(
        f"[QAT-{tag}] Best epoch:{best_epoch}/{epochs} | "
        f"top1:{best_metrics['top1']:.4f} top5:{best_metrics['top5']:.4f} | Best: {best_checkpoint}"
    )
    qat_checkpoint = save_quant_outputs(
        qat_model, f"qat_{tag}", nc=num_classes, meta=best_meta, scale=scale,
        model_dir=quant_dir,
    )[0]
    return qat_checkpoint, best_fitness, best_meta


def compare_precision(quant_method=DEFAULT_QUANT_METHOD, batch_size=64, num_classes=NUM_CLASSES,
                      num_workers=2, max_eval_batches=None, scale=det.DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== Float vs QAT precision ({det.model_name(scale)}-cls / imagenet10 / {tag}) ==========")
    print(f"Device: {device}")

    base = det.base_name(scale, "cls")
    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth，fallback 到无后缀 / Prefer _best.pth, fallback to no suffix
    float_checkpoint = os.path.join(float_dir, f"{base}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{base}.pth")
    qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{base}_best.pth")
    if not os.path.exists(qat_checkpoint):
        qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{base}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26Cls(num_classes, scale=scale).to(device)
    float_model, float_meta = load_checkpoint(float_model, float_checkpoint, return_meta=True)
    float_metrics = evaluate(float_model, val_loader, max_batches=max_eval_batches,
                             viz_dir=quant_dir, viz_prefix="compare_float")

    qat_model = QuantYOLO26Cls(num_classes, scale=scale).to(device)
    qat_model, qat_meta = load_checkpoint(qat_model, qat_checkpoint, return_meta=True)
    saved_method = qat_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] QAT checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    saved_scale = qat_meta.get("scale")
    if saved_scale is not None and saved_scale != det.get_scale(scale):
        print(f"      [warn] QAT checkpoint 记录的尺度为 yolo26{saved_scale}，当前为 yolo26{det.get_scale(scale)}")
    quant_pkg.freeze_batch_init(qat_model)
    qat_metrics = evaluate(qat_model, val_loader, max_batches=max_eval_batches,
                           viz_dir=quant_dir, viz_prefix="compare_qat")

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
        description="YOLO26-cls: 浮点训练(加载 yolo26{scale}-cls.pt) -> PTQ -> QAT -> top1/top5 对比，模型尺度与量化方法可选"
    )
    parser.add_argument(
        "--model",
        choices=det.MODEL_CHOICES,
        default=f"yolo26{det.DEFAULT_SCALE}",
        help="模型尺度（默认 yolo26n；仅 yolo26n 随仓库提供 .pt 预训练权重）",
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
    # 量化超参（不再硬编码）：--a-bits/--w-bits/--per-channel/--all-positive/--mixed-quant / Quant hyperparameters (no longer hardcoded)
    det.add_quant_cfg_args(parser)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--data",
        default=DATASET,
        help="数据集名或 yaml（默认 imagenet10）",
    )
    parser.add_argument("--resume", action="store_true",
                        help="从 _last.pth checkpoint 接续训练")
    return parser


if __name__ == "__main__":
    quant_pkg.install_print_timestamp()  # print 加分钟级时间戳 / minute-precision timestamp for print
    args = build_arg_parser().parse_args()

    # 写入量化超参（det.QUANT_CFG 是所有量化层读取的唯一来源）/ Apply quant hyperparameters (single source read by all quant layers)
    det.apply_quant_cfg_args(args)

    # 根据 --data 动态覆盖全局 DATASET / NUM_CLASSES / Dynamically override global DATASET / NUM_CLASSES based on --data
    DATASET = args.data
    try:
        _tmp = check_cls_dataset(args.data)
        NUM_CLASSES = _tmp["nc"]
    except Exception:
        pass
    print(f"数据集: {args.data} | 类别数: {NUM_CLASSES}")

    scale = det.get_scale(args.model)
    print(f"Python: {sys.executable}")
    print(f"torch: {torch.__version__} | Device: {device}")
    print(f"模型: {det.model_name(scale)}-cls | 量化方法: {args.quant} | 阶段: {args.stage}")

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
            scale=scale,
            resume=args.resume,
        )

    if args.stage in ("all", "ptq"):
        PTQ_calibration(
            quant_method=tag,
            batch_size=args.ptq_batch_size,
            num_classes=NUM_CLASSES,
            calibration_batches=args.calibration_batches,
            num_workers=args.num_workers,
            max_eval_batches=args.max_eval_batches,
            scale=scale,
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
            scale=scale,
            resume=args.resume,
        )

    if args.stage in ("all", "compare"):
        compare_precision(
            quant_method=tag,
            batch_size=args.ptq_batch_size,
            num_classes=NUM_CLASSES,
            num_workers=args.num_workers,
            max_eval_batches=args.max_eval_batches,
            scale=scale,
        )
