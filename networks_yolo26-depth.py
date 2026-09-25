"""YOLO26-depth 单目深度估计网络的量化感知训练流程（depth8，nc=1，尺度可选 n/s/m/l/x） / Quantization-aware training pipeline for YOLO26-depth monocular depth estimation network (depth8, nc=1, scales selectable n/s/m/l/x).

流程 / Pipeline:
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练 / float training（默认加载 / default load yolo26{scale}-depth.pt 预训练权重 / pretrained weights，缺失则从头训练 / train from scratch if missing）
    PTQ_calibration()                               # 训练后量化校准 / post-training quantization calibration
    QAT_training(lr=qat_lr, epochs=qat_epochs)      # 量化感知训练 / quantization-aware training
    compare_precision()                             # 浮点 vs QAT 的 delta1/abs_rel/rmse/silog 对比 / delta1/abs_rel/rmse/silog comparison between float and QAT

模型尺度用 --model yolo26n|yolo26s|yolo26m|yolo26l|yolo26x 选择（默认 yolo26n） / Model scale selected via --model yolo26n|yolo26s|yolo26m|yolo26l|yolo26x (default yolo26n).
backbone / neck（层 0-22）与 networks_yolo26-detect.py 完全一致，直接复用 / backbone/neck (layers 0-22) identical to networks_yolo26-detect.py, directly reused;
深度头为 ultralytics yolo26-depth.yaml 的 Depth（层 23，c_mid 固定 256，不随尺度缩放） / depth head is Depth from ultralytics yolo26-depth.yaml (layer 23, c_mid fixed at 256, not scaled by model scale):
    proj（每尺度 1x1 Conv 投影到 c_mid）+ refine（自顶向下双线性上采样 + 残差融合 + Conv3x3×2） / proj (per-scale 1x1 Conv to c_mid) + refine (top-down bilinear upsample + residual fuse + Conv3x3×2),
    head（Conv3x3 -> ConvTranspose2d 2x 上采样 -> Conv3x3 -> Conv2d 1x1），exp(clamp(-4,5)) 输出 (B,1,H/4,W/4) / head (Conv3x3 -> ConvTranspose2d 2x upsample -> Conv3x3 -> Conv2d 1x1), exp(clamp(-4,5)) outputs (B,1,H/4,W/4);
    末层 bias 初始化为 0.182（早期 exp 输出 ~1.2 m） / last-layer bias initialized to 0.182 (early exp output ~1.2 m).
训练损失使用官方 DepthLoss26（SILog + 多尺度梯度匹配，dlog/dgrad 两项） / Training loss uses official DepthLoss26 (SILog + multi-scale gradient matching, dlog/dgrad two terms);
评估无 NMS / bbox，用 DepthMetrics(max_depth=100) 计算 delta1/abs_rel/rmse/silog，fitness=delta1 / Evaluation has no NMS / bbox; DepthMetrics(max_depth=100) computes delta1/abs_rel/rmse/silog, fitness=delta1.
"""

import argparse
import copy
import importlib
import importlib.util
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import quantization as quant_pkg

# 复用检测网络的 backbone/neck 组件、尺度缩放助手与训练基础设施 / Reuse detection network's backbone/neck components, scale scaling helpers and training infrastructure
# （文件名 networks_yolo26-detect.py 含 '-'，不能直接 import，用文件路径加载） / (Filename networks_yolo26-detect.py contains '-', cannot import directly; load via file path)
det_spec = importlib.util.spec_from_file_location(
    "networks_yolo26_detect",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "networks_yolo26-detect.py"),
)
det = importlib.util.module_from_spec(det_spec)
det_spec.loader.exec_module(det)

# ultralytics 提供数据管道 / 损失 / 深度指标 / 可视化 / ultralytics provides data pipeline / loss / depth metrics / visualization
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import DepthLoss26
from ultralytics.utils.metrics import DepthMetrics
from ultralytics.utils.plotting import plot_images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR_BASE = os.path.join(BASE_DIR, "model", "yolo26-depth")
MODEL_DIR = _MODEL_DIR_BASE


def _model_dir_for(scale=None, quant_method=None):
    """按网络名 + 尺度 + 量化后端返回产物目录。 / Return output directory by network name + scale + quantization backend."""
    path = _MODEL_DIR_BASE
    if scale is not None:
        path = os.path.join(path, scale)
    if quant_method is not None:
        path = os.path.join(path, quant_method)
    os.makedirs(path, exist_ok=True)
    return path
ULTRA_DIR = os.path.join(BASE_DIR, "ultralytics", "ultralytics")
# 默认官方数据集（首次运行自动下载到 dataset/；--data 可手动指定其他目录） /
# Default official dataset (auto-downloaded to dataset/ on first run; --data for a manual dir)
DATA_YAML = det.DEFAULT_DATA_YAML["depth"]

IMGSZ = 640
NUM_CLASSES = 1  # depth 为稠密回归任务，nc=1 仅为兼容 yaml 解析 / depth is dense regression; nc=1 only for yaml parser compatibility
STRIDES = (8, 16, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 量化后端可切换：dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact / Quantization backend switchable: dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact
# backbone 组件从检测网络复用，因此切换时同步切换检测网络模块内绑定的后端算子 / Backbone components are reused from detection network, so bound backend ops in detection module must switch in sync.
# ---------------------------------------------------------------------------
DEFAULT_QUANT_METHOD = "lsqplus_v1"
QUANT_METHOD = DEFAULT_QUANT_METHOD
Q = quant_pkg.load_quant_backend(DEFAULT_QUANT_METHOD)

QuantAdd = Q.QuantAdd
QuantCat = Q.QuantCat
QuantConcat = Q.QuantConcat
QuantConv2d = Q.QuantConv2d
QuantConvTranspose2d = Q.QuantConvTranspose2d
QuantMaxPool = Q.QuantMaxPool
QuantSiLU = getattr(Q, 'QuantSiLU', None)
QuantSigmoid = getattr(Q, 'QuantSigmoid', None)
QuantMatMul = getattr(Q, 'QuantMatMul', None)


def set_quant_method(method):
    """切换量化后端（必须在构建 QuantYOLO26Depth 之前调用）。 / Switch quantization backend (must be called before constructing QuantYOLO26Depth)."""
    global Q, QUANT_METHOD
    global QuantAdd, QuantCat, QuantConcat, QuantConv2d, QuantConvTranspose2d, QuantMaxPool, QuantSiLU, QuantSigmoid, QuantMatMul

    # 复用的 backbone/neck block 类内部引用的是 det 模块的全局算子，必须先同步切换 / Reused backbone/neck block classes reference global ops in det module; must switch in sync first
    det.set_quant_method(method)
    Q = quant_pkg.load_quant_backend(method)
    QUANT_METHOD = method
    QuantAdd = det.QuantAdd
    QuantCat = det.QuantCat
    QuantConcat = det.QuantConcat
    QuantConv2d = det.QuantConv2d
    QuantConvTranspose2d = Q.QuantConvTranspose2d
    QuantMaxPool = det.QuantMaxPool
    QuantSiLU = getattr(Q, 'QuantSiLU', None)
    QuantSigmoid = getattr(Q, 'QuantSigmoid', None)
    QuantMatMul = getattr(Q, 'QuantMatMul', None)
    return Q


# ============================== 深度头组件 / Depth Head Components ==============================


class DepthHead(nn.Module):
    """YOLO26-depth 深度头（复刻 ultralytics.nn.modules.head.Depth，yolo26-depth.yaml 层 23） / YOLO26-depth depth head (reproduces ultralytics.nn.modules.head.Depth, layer 23 of yolo26-depth.yaml).

    输入 P3/P4/P5 三尺度特征，先把每层 1x1 Conv 投影到 c_mid，再自顶向下 / Takes P3/P4/P5 multi-scale features, projects each to c_mid via 1x1 Conv, then top-down
    双线性 2x 上采样（align_corners=True，与官方发布权重一致）逐级融合并过 refine（Conv3x3×2）， / bilinear 2x upsampling (align_corners=True, consistent with official released weights) fuses level by level through refine (Conv3x3×2),
    最后 head（Conv3x3 -> ConvTranspose2d 2x -> Conv3x3 -> Conv2d 1x1）输出 log-depth， / finally head (Conv3x3 -> ConvTranspose2d 2x -> Conv3x3 -> Conv2d 1x1) outputs log-depth,
    exp(clamp(-4, 5)) 得到 (B, 1, H/4, W/4) 深度图（P2 分辨率） / exp(clamp(-4, 5)) yields (B, 1, H/4, W/4) depth map (P2 resolution).

    训练时返回 {"depth": tensor} 供 DepthLoss26 使用；eval 时应用 log-affine 标定 / During training returns {"depth": tensor} for DepthLoss26; during eval applies log-affine calibration
    depth = depth.pow(cal_a) * cal_b.exp()（默认恒等）；export 时再 4x 双线性上采样到输入分辨率 / depth = depth.pow(cal_a) * cal_b.exp() (identity by default); in export mode additionally 4x bilinear upsamples to input resolution.

    属性命名与官方完全一致（proj/refine/head/cal_a/cal_b）， / Attribute naming matches official exactly (proj/refine/head/cal_a/cal_b),
    可直接加载 yolo26{scale}-depth.pt 中 model.23.* 的权重 / can directly load weights from model.23.* in yolo26{scale}-depth.pt.
    """

    export = False  # export 模式标志（导出 ONNX 时置 True，输出上采样到输入分辨率） / export mode flag (set True when exporting ONNX, output upsampled to input resolution)

    def __init__(self, ch=(256, 512, 1024), c_mid=256, quant=False):
        super().__init__()
        self.nl = len(ch)  # 金字塔层数 / number of pyramid levels

        # 每层 1x1 Conv 投影到 c_mid / per-level 1x1 Conv projection to c_mid
        self.proj = nn.ModuleList(det.Conv(c, c_mid, k=1, quant=quant) for c in ch)

        # 自顶向下 nl-1 次融合后的细化块（最粗层不细化） / refinement blocks after each of the nl-1 fusion steps (coarsest level not refined)
        self.refine = nn.ModuleList(
            nn.Sequential(det.Conv(c_mid, c_mid, k=3, quant=quant), det.Conv(c_mid, c_mid, k=3, quant=quant))
            for _ in ch[:-1]
        )
        # 量化模型中融合残差加法用 QuantAdd；浮点模型用普通加法（None 占位） / fusion residual add uses QuantAdd in quant model; plain add in float model (None placeholder)
        self.fuse_adds = (
            nn.ModuleList(det.QuantAdd(a_bits=8, quant_inference=True) for _ in ch[:-1])
            if quant else None
        )

        if quant:
            upsample = QuantConvTranspose2d(
                c_mid // 2, c_mid // 2, kernel_size=2, stride=2, padding=0, bias=True,
                a_bits=8, w_bits=8, per_channel=True,
            )
            last_conv = QuantConv2d(c_mid // 4, 1, kernel_size=1, a_bits=8, w_bits=8, per_channel=True)
        else:
            upsample = nn.ConvTranspose2d(c_mid // 2, c_mid // 2, kernel_size=2, stride=2, bias=True)
            last_conv = nn.Conv2d(c_mid // 4, 1, kernel_size=1)
        self.head = nn.Sequential(
            det.Conv(c_mid, c_mid // 2, k=3, quant=quant),
            upsample,
            det.Conv(c_mid // 2, c_mid // 4, k=3, quant=quant),
            last_conv,
        )

        # log-affine 标定 d' = exp(a·log d + b)，默认恒等（a=1, b=0） / scale-only log-affine calibration d' = exp(a·log d + b); identity by default (a=1, b=0)
        self.register_buffer("cal_a", torch.ones(1))
        self.register_buffer("cal_b", torch.zeros(1))
        self.bias_init()

    def bias_init(self):
        """末层 1x1 Conv 偏置填 0.182，使早期 exp() 输出 ~1.2 m、训练初期数值稳定（官方做法） /
        Fill last 1x1 Conv bias with 0.182 so early exp() outputs ~1.2 m and stay well-conditioned (official approach)."""
        self.head[-1].bias.data.fill_(0.182)

    def forward(self, x):
        # 全部投影到同一通道维 / project all levels to the same channel dim
        feats = [self.proj[i](x[i]) for i in range(self.nl)]

        out = feats[-1]
        for i in range(self.nl - 2, -1, -1):
            # align_corners=True 与官方发布权重一致；相邻金字塔层尺度恒为 2x，静态导出友好 /
            # align_corners=True baked into official released weights; consecutive pyramid levels keep upsample a static 2x, export-friendly
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
            out = self.fuse_adds[i](out, feats[i]) if self.fuse_adds is not None else out + feats[i]
            out = self.refine[i](out)

        out = self.head(out)  # (B, 1, H/4, W/4)
        depth = torch.exp(out.clamp(-4.0, 5.0))

        if self.training:
            return {"depth": depth}

        depth = depth.pow(self.cal_a) * self.cal_b.exp()
        if self.export:
            depth = F.interpolate(depth, scale_factor=4.0, mode="bilinear", align_corners=False)
        return depth


# ============================== 网络主体 / Network Body ==============================


class YOLO26Depth(det.YOLO26):
    """yolo26n-depth（scale=n）深度估计网络。quant=False 浮点模型，True 为伪量化模型。 / yolo26n-depth (scale=n) depth estimation network. quant=False float model, True fake-quantized model.

    层 0-22 与检测网络一致（直接构建后替换层 23），topology 编号/save 集合不变， / Layers 0-22 identical to detection network (built directly then replace layer 23); topology indices/save set unchanged,
    因此 yolo26n-depth.pt 的 state_dict（model.0.* ~ model.23.*）可直接加载。 / so state_dict of yolo26n-depth.pt (model.0.* ~ model.23.*) can be loaded directly.
    """

    def __init__(self, nc=NUM_CLASSES, quant=False, scale=det.DEFAULT_SCALE):
        # 先构建检测网络得到完整的 0-22 backbone/neck，再把层 23 的 Detect 换成 DepthHead / First build detection network to get complete 0-22 backbone/neck, then replace layer 23's Detect with DepthHead
        super().__init__(nc=nc, quant=quant, scale=scale)
        head = DepthHead(
            # c_mid 固定 256，不随尺度缩放（官方 yolo26-depth.yaml: Depth, [256]；parse_model 原样透传） /
            # c_mid fixed at 256, not scaled by model scale (official yolo26-depth.yaml: Depth, [256]; parse_model passes through)
            ch=det.detect_head_channels(scale), c_mid=256, quant=quant,
        )
        det._tag(head, 23, [16, 19, 22])
        self.model[-1] = head
        self._register_quantizer_buffers()
        self._initialize_head()


class FloatYOLO26Depth(YOLO26Depth):
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
        super().__init__(nc=nc, quant=False, scale=scale)


class QuantYOLO26Depth(YOLO26Depth):
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
        super().__init__(nc=nc, quant=True, scale=scale)


# ============================== 数据 / 损失 / 评估 / Data / Loss / Evaluation ==============================


def _make_cfg(num_workers):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = IMGSZ
    cfg.task = "depth"  # build_yolo_dataset 据此选用 DepthDataset（RGB + 配对 PNG/NPY 深度图） / build_yolo_dataset selects DepthDataset accordingly (RGB + paired PNG/NPY depth maps)
    cfg.workers = num_workers
    return cfg


def _build_loaders(batch_size, num_workers, data, calibration=False):
    """构建 dataloader；train 模式额外返回 cfg / train_set（close_mosaic 需要引用） /
    Build dataloaders; train mode additionally returns cfg / train_set (required by close_mosaic).

    train：640x640 拉伸 + 翻转等增强（depth 任务 mosaic/mixup/copy_paste 由 DepthDataset 强制关闭） /
    train: 640x640 stretch + flip etc. augmentations (mosaic/mixup/copy_paste forcibly disabled by DepthDataset);
    val/calibration：无增强的 640x640 拉伸（depth val letterbox 为 scale_fill 拉伸，不做 padding） /
    val/calibration: no-aug 640x640 stretch (depth val letterbox is scale_fill stretch, no padding).
    """
    cfg = _make_cfg(num_workers)
    if calibration:
        dataset = build_yolo_dataset(cfg, data["val"], batch_size, data, mode="val", rect=False)
        return build_dataloader(
            dataset, batch_size, num_workers, shuffle=False, rank=-1, device=device
        )

    train_set = build_yolo_dataset(cfg, data["train"], batch_size, data, mode="train", rect=False)
    train_loader = build_dataloader(
        train_set, batch_size, num_workers, shuffle=True, rank=-1, device=device
    )
    val_set = build_yolo_dataset(cfg, data["val"], batch_size, data, mode="val", rect=True)
    val_loader = build_dataloader(
        val_set, batch_size, num_workers, shuffle=False, rank=-1, device=device
    )
    return train_loader, val_loader, cfg, train_set


def get_dataloaders(batch_size=8, num_workers=2, calibration=False):
    """复用 ultralytics 官方 depth8 数据管道（batch 含 "img" 与 "depth"） / Reuse ultralytics official depth8 data pipeline (batch contains "img" and "depth")."""
    data = det.get_data_dict(DATA_YAML, "depth")
    if calibration:
        return _build_loaders(batch_size, num_workers, data, calibration=True), data
    train_loader, val_loader, _, _ = _build_loaders(batch_size, num_workers, data)
    return train_loader, val_loader, data


class _LossShim:
    """DepthLoss26 只需要 model.args（dlog/dgrad/dlam）与 model.parameters() / DepthLoss26 only needs model.args (dlog/dgrad/dlam) and model.parameters()."""

    def __init__(self, depth_head, epochs):
        self.args = SimpleNamespace(dlog=1.0, dgrad=0.5, dlam=1.0, epochs=epochs)
        self.model = [None] * 23 + [depth_head]

    def parameters(self):
        return self.model[-1].parameters()


def build_criterion(model, epochs):
    return DepthLoss26(_LossShim(model.model[-1], epochs))


def _maybe_visualize(batch, preds, names, viz_dir, viz_prefix, viz_state):
    """用 ultralytics 官方 plot_images 保存本批的 GT 深度拼图与预测深度拼图（val_batch 风格） /
    Use official ultralytics plot_images to save GT depth mosaic and predicted depth mosaic of this batch (val_batch style).

    labels 传 {"depth": ...} 即可触发深度热力图叠加（无 bbox/cls） / passing labels={"depth": ...} triggers depth heatmap overlay (no bbox/cls).
    viz_state 跨 batch 记录已保存图片数 / viz_state tracks saved image count across batches.
    """
    os.makedirs(viz_dir, exist_ok=True)
    bs = batch["img"].shape[0]
    bi = viz_state["batch"]
    # GT 深度拼图 / GT depth mosaic
    plot_images(
        labels={"depth": batch["depth"]},
        images=batch["img"],
        paths=batch.get("im_file"),
        fname=os.path.join(viz_dir, f"{viz_prefix}_batch{bi}_labels.jpg"),
        names=names,
        threaded=False,  # 训练循环内同步执行，避免线程堆积 / run synchronously inside training loop to avoid thread pile-up
    )
    # 预测深度拼图（与官方 DepthValidator.plot_predictions 一致） / predicted depth mosaic (consistent with official DepthValidator.plot_predictions)
    plot_images(
        labels={"depth": preds},
        images=batch["img"],
        paths=batch.get("im_file"),
        fname=os.path.join(viz_dir, f"{viz_prefix}_batch{bi}_pred.jpg"),
        names=names,
        threaded=False,
    )
    viz_state["saved"] += bs
    viz_state["batch"] += 1


@torch.no_grad()
def evaluate(model, val_loader, data, batch_size=8, max_batches=None,
             viz_dir=None, viz_max=30, viz_prefix="eval"):
    """在 depth8 验证集上计算深度指标 delta1/delta2/delta3/abs_rel/rmse/silog，fitness=delta1。若 viz_dir 不为空，
    额外把前 viz_max 张验证图的 GT/预测深度拼图保存到 {viz_dir}/fvisualize/。 / Compute depth metrics delta1/delta2/delta3/abs_rel/rmse/silog on depth8 validation set, fitness=delta1.
    If viz_dir is set, additionally save GT/prediction depth mosaics of the first viz_max val images into {viz_dir}/fvisualize/.

    与官方 DepthValidator 一致：无 NMS、无 bbox；预测深度与 GT 分辨率不同时 / Consistent with official DepthValidator: no NMS, no bbox; when predicted depth and GT resolutions differ,
    先 bilinear（align_corners=True）插值到 GT 尺寸再 update_stats。 / bilinear-interpolate (align_corners=True) pred to GT size before update_stats.
    """
    model.eval()
    metrics = DepthMetrics(max_depth=float(data.get("max_depth") or 100.0))
    names = data.names if hasattr(data, "names") else data["names"]
    num_images = len(val_loader.dataset)
    steps = max_batches or math.ceil(num_images / batch_size)
    # 可视化输出目录与计数器 / visualization output dir and counters
    viz_out = os.path.join(viz_dir, "fvisualize") if viz_dir else None
    viz_state = {"saved": 0, "batch": 0}
    if viz_out:
        os.makedirs(viz_out, exist_ok=True)

    for batch_index, batch in enumerate(val_loader):
        if batch_index >= steps:
            break
        images = batch["img"].float().to(device) / 255.0
        preds = model(images)  # (B, 1, H/4, W/4)，已应用 cal_a/cal_b 标定 / (B, 1, H/4, W/4), calibration applied
        gt_depth = batch["depth"].float().to(device)
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        if preds.shape[-2:] != gt_depth.shape[-2:]:
            preds = F.interpolate(preds.float(), size=gt_depth.shape[-2:], mode="bilinear", align_corners=True)
        metrics.update_stats(preds, gt_depth)

        # 每次评估都可视化前几批（失败仅告警，绝不影响评估） / visualize first batches on every evaluate (failure only warns, never breaks eval)
        if viz_out and viz_state["saved"] < viz_max:
            try:
                _maybe_visualize(batch, preds, names, viz_out, viz_prefix, viz_state)
            except Exception as exc:
                print(f"      [viz] 可视化保存失败（仅告警）: {exc} / visualization save failed (warn only): {exc}")

    metrics.process()
    # results_dict 含 metrics/delta1、metrics/abs_rel、metrics/rmse、metrics/silog 与 fitness(=delta1) /
    # results_dict contains metrics/delta1, metrics/abs_rel, metrics/rmse, metrics/silog and fitness (=delta1)
    return metrics.results_dict


# ============================== checkpoint / 权重复制 / 导出 / Checkpoint / Weight Copy / Export ==============================


def load_checkpoint(model, path, return_meta=False):
    return det.load_checkpoint(model, path, return_meta=return_meta)


def save_checkpoint(model, path, **metadata):
    det.save_checkpoint(model, path, **metadata)


def load_pretrained(model, path=None, scale=det.DEFAULT_SCALE):
    """加载官方 yolo26{scale}-depth.pt；自动跳过 shape mismatch 参数 / Load official yolo26{scale}-depth.pt; auto-skip shape mismatch parameters.

    本地缺失的官方 depth 权重（如 yolo26n-depth.pt）会按 ultralytics 方式自动下载；仅离线或自定义路径缺失时才从头训练 /
    Missing official depth weights (e.g. yolo26n-depth.pt) are auto-downloaded like ultralytics; random init only when offline or custom path missing.
    """
    if path is None:
        path = os.path.join(ULTRA_DIR, f"{det.base_name(scale, 'depth')}.pt")
    # 官方权重名缺失时自动下载（同 ultralytics）；自定义路径缺失则从头训练 /
    # Auto-download official asset names (like ultralytics); missing custom path -> train from scratch
    path = det.resolve_pretrained_path(path)
    if path is None:
        print(f"[Pretrain] [warn] 未找到预训练权重，{det.model_name(scale)}-depth 将从头训练")
        return model
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].float().state_dict()

    # 过滤 shape mismatch 的 key（与 det.load_pretrained 同一约定） / Filter shape mismatch keys (same convention as det.load_pretrained)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    skipped_shape = [k for k, v in state_dict.items()
                     if k in model_sd and model_sd[k].shape != v.shape]
    missing_unexpected = set(model_sd.keys()) - set(filtered.keys())

    model.load_state_dict(filtered, strict=False)
    print(f"[Pretrain] 已加载 {path}")
    print(f"  ✓ 成功加载: {len(filtered)} 参数")
    if skipped_shape:
        print(f"  ⚠️  跳过 shape mismatch: {len(skipped_shape)} 参数")
    if missing_unexpected:
        print(f"  ⚠️  missing/unexpected: {len(missing_unexpected)} 参数")
    return model


def copy_float_to_quant(float_model, quant_model):
    return det.copy_float_to_quant(float_model, quant_model)


def build_float_model(quant_model, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
    """把量化模型的反量化权重灌进同结构的干净浮点模型（用于导出纯浮点 ONNX）。 / Inject dequantized weights from quantized model into a clean float model of same structure (for exporting pure-float ONNX)."""
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_state = quant_model.state_dict()
    for name, module in quant_model.named_modules():
        if quant_pkg.is_weight_quant_module(module):
            quant_state[f"{name}.weight"] = quant_pkg.dequantized_weight(module)

    float_model = FloatYOLO26Depth(nc=nc, scale=scale)
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


def export_onnx(float_model, onnx_path, opset=16, imgsz=None):
    """导出 ONNX，输入形状完全固定为 [1, 3, H, W]（无 dynamic_axes，图尺寸清晰可见），输出 depth [1, 1, H, W] /
    Export ONNX with fully static input shape [1, 3, H, W] (no dynamic_axes, graph dimensions clearly visible), output depth [1, 1, H, W].

    导出前置 model.model[-1].export = True：深度头在 eval 标定基础上再 4x 双线性上采样，输出与输入同分辨率 /
    Set model.model[-1].export = True before export: depth head additionally 4x bilinear upsamples on top of eval calibration, output matches input resolution.
    """
    H = W = imgsz if imgsz is not None else IMGSZ
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    float_model.eval()
    head = float_model.model[-1]
    head.export = True
    model_device = next(float_model.parameters()).device
    dummy = torch.randn(1, 3, H, W, device=model_device)
    try:
        torch.onnx.export(
            float_model,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            input_names=["images"],
            output_names=["depth"],
            # 无 dynamic_axes → 输入输出形状完全固定 / No dynamic_axes → all shapes fully fixed
        )
    finally:
        head.export = False  # 恢复 eval 输出分辨率（H/4, W/4），避免影响后续评估 / restore eval output resolution (H/4, W/4) to avoid affecting subsequent evaluation
    # onnxsim 简化（若已安装） / onnxsim simplify (if installed)
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
    """训练保存 best checkpoint 时同步导出 ONNX；失败仅告警，绝不影响训练。 / Export ONNX synchronously when saving best checkpoint during training; failure only warns, never affects training.

    导出后模型被置为 eval，由下一轮 train_one_epoch 的 model.train() 恢复。 / After export model is set to eval, restored by model.train() in next round of train_one_epoch.
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
        print(f"      ONNX checker 通过，算子数: {len(op_types)}（含 Conv/ConvTranspose/Concat 等）")
    except ImportError:
        print("      [skip] 未安装 onnx，跳过结构检查")

    try:
        ort = importlib.import_module("onnxruntime")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        torch.manual_seed(0)
        float_model.cpu().eval()
        head = float_model.model[-1]
        head.export = True  # 与导出的 ONNX 输出分辨率一致（H, W） / match exported ONNX output resolution (H, W)
        x = torch.randn(1, 3, IMGSZ, IMGSZ)
        with torch.no_grad():
            y_torch = float_model(x).numpy()
        head.export = False
        y_onnx = session.run(["depth"], {"images": x.numpy()})[0]
        max_diff = float(np.abs(y_torch - y_onnx).max())
        ref_mag = float(np.abs(y_torch).max())
        print(f"      onnxruntime vs PyTorch 最大绝对误差: {max_diff:.3e}（参考幅度 {ref_mag:.3e}）")
        # 深度输出为 exp 后的正数（米），量级随场景变化，纯绝对阈值过严：max(1e-3 绝对, 1e-5 相对) /
        # Depth output is positive (meters) after exp, magnitude varies with scene; pure abs threshold too strict: max(1e-3 abs, 1e-5 rel)
        assert max_diff < max(1e-3, 1e-5 * ref_mag), (
            "ONNX 数值误差过大"
        )
    except ImportError:
        print("      [skip] 未安装 onnxruntime，跳过数值比对")

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
    quant_checkpoint = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'depth')}.pth")
    save_checkpoint(quant_model, quant_checkpoint, **(meta or {}))

    quant_params = collect_quant_params(quant_model)
    json_path = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'depth')}_quant_params.json")
    pth_path = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'depth')}_quant_params.pth")
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
    float_checkpoint = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'depth')}_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'depth')}_float.onnx")
    export_onnx(float_model, onnx_path, opset=16)
    print(f"[4/5] 干净浮点 ONNX 已写出: {onnx_path}")

    verify(float_model, quant_model, onnx_path, quant_params)
    print("[5/5] 完成。")
    return quant_checkpoint, json_path, pth_path, float_checkpoint, onnx_path


# ============================== 训练各阶段 / Training Stages ==============================


_ModelEMA = det._ModelEMA
_auto_lr = det._auto_lr
_build_optimizer = det._build_optimizer


def train_one_epoch(model, loader, criterion, optimizer, epoch, epochs, nb, ema=None,
                    nbs=64, warmup_epochs=3.0, lrf=0.01, max_batches=None):
    """与检测版一致的官方训练循环；损失项为 dlog/dgrad 两项。 / Official training loop consistent with detection version; loss items are dlog/dgrad (two terms).

    criterion 为 DepthLoss26，直接 __call__ 调用：返回已乘 batch_size 的损失向量 / criterion is DepthLoss26, called via __call__ directly: returns loss vector already multiplied by batch_size
    （官方直接 backward，不除以 batch）与未缩放的 items dict（dlog_loss / dgrad_loss）。 / (official backward directly, no batch divide) and unscaled items dict (dlog_loss / dgrad_loss).
    返回 (总损失, {loss_name: 平均值}) 以兼容任意项数的损失。 / Returns (total loss, {loss_name: average}) to be compatible with any number of loss terms.
    """
    model.train()
    batch_size = loader.batch_size
    accumulate = max(round(nbs / batch_size), 1)
    warmup_steps = round(min(warmup_epochs, max(epochs - 1, 0)) * nb) if warmup_epochs > 0 else 0
    lf = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lf(epoch)

    total_steps = nb if max_batches is None else min(nb, max_batches)
    loss_names = criterion.loss_names
    loss_items_sum = torch.zeros(len(loss_names))
    last_opt_step = 0
    for i, batch in enumerate(loader):
        if i >= total_steps:
            break
        ni = i + nb * epoch
        if ni < warmup_steps:
            xi = [0, warmup_steps]
            accumulate = max(1, int(np.interp(ni, xi, [1, nbs / batch_size]).round()))
            for group in optimizer.param_groups:
                group["lr"] = float(
                    np.interp(ni, xi, [0.0, group["initial_lr"] * lf(epoch)])
                )

        images = batch["img"].float().to(device) / 255.0
        batch["depth"] = batch["depth"].float()  # 与官方 preprocess_batch 一致：深度保持 float32 / consistent with official preprocess_batch: keep depth as float32
        preds = model(images)
        loss_vec, loss_items = criterion(preds, batch)  # DepthLoss26.__call__（非 .loss） / DepthLoss26.__call__ (not .loss)
        loss_vec.sum().backward()

        if ni - last_opt_step >= accumulate:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            last_opt_step = ni

        loss_items_sum += torch.stack([loss_items[name].detach().cpu() for name in loss_names])

    avg_items = (loss_items_sum / total_steps).tolist()
    return float(sum(avg_items)), dict(zip(loss_names, avg_items))


def float_train(batch_size=16, lr=None, epochs=100, num_classes=NUM_CLASSES, num_workers=2,
                max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                resume=False):
    print(f"========== Float training ({det.model_name(scale)}-depth / depth8) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr(num_classes)
    print(
        f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4（官方 optimizer=auto 配方）"
        " | warmup 3ep | 梯度累积 nbs=64 | EMA | close_mosaic=10"
    )

    data = det.get_data_dict(DATA_YAML, "depth")
    train_loader, val_loader, cfg, train_set = _build_loaders(batch_size, num_workers, data)
    nb = len(train_loader)

    float_model = FloatYOLO26Depth(num_classes, scale=scale).to(device)
    det.model_info(float_model, imgsz=IMGSZ)
    load_pretrained(float_model, scale=scale)

    criterion = build_criterion(float_model, epochs)
    optimizer = _build_optimizer(float_model, lr=lr)
    ema = _ModelEMA(float_model)

    close_mosaic = 10
    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    model_dir = _model_dir_for(scale=scale)
    checkpoint_path = os.path.join(model_dir, f"{det.base_name(scale, 'depth')}_best.pth")
    last_checkpoint = os.path.join(model_dir, f"{det.base_name(scale, 'depth')}_last.pth")

    # resume：从 _last.pth 恢复，接续训练 / Resume: restore from _last.pth, continue training
    start_epoch = 0
    if resume and os.path.exists(last_checkpoint):
        print(f"[Float] Resume from {last_checkpoint}")
        _, meta = det.load_checkpoint(float_model, last_checkpoint, return_meta=True)
        ema.ema.load_state_dict(float_model.state_dict())
        start_epoch = meta.get("epoch", 0)
        best_fitness = meta.get("fitness", -1.0)
        best_meta = meta
        best_epoch = start_epoch
        print(f"[Float] 从 epoch {start_epoch} 接续训练，当前 best delta1={meta.get('metrics/delta1', 0):.4f}")
    elif resume:
        print(f"[Float] Resume 开启但找不到 {last_checkpoint}，从头训练")

    for epoch in range(start_epoch, epochs):
        if epoch == epochs - close_mosaic:
            print("[Float] 关闭 dataloader mosaic（最后 10 个 epoch）")
            train_set.close_mosaic(copy.copy(cfg))
            train_loader.reset()

        train_loss, items = train_one_epoch(
            float_model, train_loader, criterion, optimizer, epoch, epochs, nb, ema=ema,
            max_batches=max_train_batches,
        )
        dlog_loss, dgrad_loss = items["dlog_loss"], items["dgrad_loss"]
        metrics = evaluate(
            ema.ema, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
            viz_dir=model_dir, viz_prefix=f"float_ep{epoch + 1:03d}"
        )
        fitness = metrics["fitness"]  # fitness = delta1（越高越好） / fitness = delta1 (higher is better)

        epoch_meta = {
            "stage": "float",
            "scale": det.get_scale(scale),
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "dlog_loss": float(dlog_loss),
            "dgrad_loss": float(dgrad_loss),
            "fitness": float(fitness),
            **metrics,
        }
        save_checkpoint(ema.ema, last_checkpoint, **epoch_meta)

        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch + 1
            best_meta = epoch_meta
            save_checkpoint(ema.ema, checkpoint_path, **best_meta)
            _try_export_onnx(ema.ema, os.path.splitext(checkpoint_path)[0] + ".onnx")

        print(
            f"[Float] Epoch [{epoch + 1}/{epochs}] | loss:{train_loss:.4f} "
            f"(dlog:{dlog_loss:.3f} dgrad:{dgrad_loss:.3f}) | "
            f"delta1:{metrics['metrics/delta1']:.4f} | abs_rel:{metrics['metrics/abs_rel']:.4f} "
            f"rmse:{metrics['metrics/rmse']:.4f} silog:{metrics['metrics/silog']:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)
    best_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=model_dir, viz_prefix="float_best"
    )
    print(f"[Float] Best checkpoint: {checkpoint_path}（epoch {best_epoch}/{epochs}）")
    print(
        f"[Float] Best delta1:{best_metrics['metrics/delta1']:.4f} | "
        f"abs_rel:{best_metrics['metrics/abs_rel']:.4f} rmse:{best_metrics['metrics/rmse']:.4f} "
        f"silog:{best_metrics['metrics/silog']:.4f}"
    )
    return checkpoint_path, best_fitness, best_meta


@torch.no_grad()
def calibrate_quantizer(quant_model, calibration_loader, calibration_batches=20,
                         float_model=None):
    """校准量化器（委托给 detect 模块的升级版本）。 / Calibrate quantizer (delegates to upgraded version in detect module)."""
    return det.calibrate_quantizer(quant_model, calibration_loader,
                                    calibration_batches, float_model=float_model)


def PTQ_calibration(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                    calibration_batches=20, num_workers=2, max_eval_batches=None,
                    scale=det.DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== PTQ calibration ({det.model_name(scale)}-depth / {tag}) ==========")
    print(f"Device: {device}")

    calibration_loader, data = get_dataloaders(batch_size, num_workers, calibration=True)
    _, val_loader, _ = get_dataloaders(batch_size, num_workers)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth（与 QAT 命名一致），fallback 到旧版无后缀 .pth / Prefer _best.pth (consistent with QAT naming), fallback to old no-suffix .pth
    float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'depth')}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'depth')}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatYOLO26Depth(num_classes, scale=scale).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = QuantYOLO26Depth(num_classes, scale=scale).to(device)
    det.model_info(ptq_model, imgsz=IMGSZ)
    copy_float_to_quant(float_model, ptq_model)

    calibrate_quantizer(ptq_model, calibration_loader, calibration_batches,
                         float_model=float_model)

    metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix=f"ptq_{tag}"
    )
    print(
        f"[PTQ-{tag}] delta1:{metrics['metrics/delta1']:.4f} | "
        f"abs_rel:{metrics['metrics/abs_rel']:.4f} rmse:{metrics['metrics/rmse']:.4f} "
        f"silog:{metrics['metrics/silog']:.4f}"
    )

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
    final_metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix=f"ptq_{tag}_reload"
    )
    print(
        f"[PTQ] 重载 checkpoint 后 delta1:{final_metrics['metrics/delta1']:.4f} "
        f"abs_rel:{final_metrics['metrics/abs_rel']:.4f}"
    )
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=16, lr=None, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                 resume=False):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training ({det.model_name(scale)}-depth / {tag}) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr(num_classes) * 0.1
    print(f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4 | warmup 3ep | 梯度累积 nbs=64")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)
    nb = len(train_loader)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    ptq_checkpoint = os.path.join(quant_dir, f"ptq_{tag}_{det.base_name(scale, 'depth')}.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = QuantYOLO26Depth(num_classes, scale=scale).to(device)
    det.model_info(qat_model, imgsz=IMGSZ)
    _, ptq_meta = load_checkpoint(qat_model, ptq_checkpoint, return_meta=True)
    saved_method = ptq_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] PTQ checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    saved_scale = ptq_meta.get("scale")
    if saved_scale is not None and saved_scale != det.get_scale(scale):
        print(f"      [warn] PTQ checkpoint 记录的尺度为 yolo26{saved_scale}，当前为 yolo26{det.get_scale(scale)}")
    quant_pkg.freeze_batch_init(qat_model)

    criterion = build_criterion(qat_model, epochs)
    optimizer = _build_optimizer(qat_model, lr=lr)

    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    best_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'depth')}_best.pth")
    last_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'depth')}_last.pth")

    # resume：从 _last.pth 恢复，接续训练 / Resume: restore from _last.pth, continue training
    start_epoch = 0
    if resume and os.path.exists(last_checkpoint):
        print(f"[QAT-{tag}] Resume from {last_checkpoint}")
        _, meta = load_checkpoint(qat_model, last_checkpoint, return_meta=True)
        start_epoch = meta.get("epoch", 0)
        best_fitness = meta.get("fitness", -1.0)
        best_meta = meta
        best_epoch = start_epoch
        print(f"[QAT-{tag}] 从 epoch {start_epoch} 接续训练，当前 best delta1={meta.get('metrics/delta1', 0):.4f}")
    elif resume:
        print(f"[QAT-{tag}] Resume 开启但找不到 {last_checkpoint}，从头训练")

    for epoch in range(start_epoch, epochs):
        train_loss, items = train_one_epoch(
            qat_model, train_loader, criterion, optimizer, epoch, epochs, nb,
            max_batches=max_train_batches,
        )
        dlog_loss, dgrad_loss = items["dlog_loss"], items["dgrad_loss"]
        metrics = evaluate(
            qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
            viz_dir=quant_dir, viz_prefix=f"qat_{tag}_ep{epoch + 1:03d}"
        )
        fitness = metrics["fitness"]  # fitness = delta1（越高越好） / fitness = delta1 (higher is better)

        epoch_meta = {
            "stage": "qat",
            "scale": det.get_scale(scale),
            "quant_method": tag,
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "dlog_loss": float(dlog_loss),
            "dgrad_loss": float(dgrad_loss),
            "fitness": float(fitness),
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
            f"[QAT-{tag}] Epoch [{epoch + 1}/{epochs}] | loss:{train_loss:.4f} "
            f"(dlog:{dlog_loss:.3f} dgrad:{dgrad_loss:.3f}) | "
            f"delta1:{metrics['metrics/delta1']:.4f} | abs_rel:{metrics['metrics/abs_rel']:.4f} "
            f"rmse:{metrics['metrics/rmse']:.4f} silog:{metrics['metrics/silog']:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    best_metrics = evaluate(
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix=f"qat_{tag}_best"
    )
    print(
        f"[QAT-{tag}] Best epoch:{best_epoch}/{epochs} | "
        f"delta1:{best_metrics['metrics/delta1']:.4f} abs_rel:{best_metrics['metrics/abs_rel']:.4f} "
        f"rmse:{best_metrics['metrics/rmse']:.4f} silog:{best_metrics['metrics/silog']:.4f} | "
        f"Best: {best_checkpoint}"
    )
    qat_checkpoint = save_quant_outputs(
        qat_model, f"qat_{tag}", nc=num_classes, meta=best_meta, scale=scale,
        model_dir=quant_dir,
    )[0]
    return qat_checkpoint, best_fitness, best_meta


def compare_precision(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                      num_workers=2, max_eval_batches=None, scale=det.DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== Float vs QAT precision ({det.model_name(scale)}-depth / depth8 / {tag}) ==========")
    print(f"Device: {device}")

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth，fallback 到无后缀 / Prefer _best.pth, fallback to no-suffix
    float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'depth')}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'depth')}.pth")
    qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'depth')}_best.pth")
    if not os.path.exists(qat_checkpoint):
        qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'depth')}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26Depth(num_classes, scale=scale).to(device)
    float_model, float_meta = load_checkpoint(float_model, float_checkpoint, return_meta=True)
    float_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix="compare_float"
    )

    qat_model = QuantYOLO26Depth(num_classes, scale=scale).to(device)
    qat_model, qat_meta = load_checkpoint(qat_model, qat_checkpoint, return_meta=True)
    saved_method = qat_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] QAT checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    saved_scale = qat_meta.get("scale")
    if saved_scale is not None and saved_scale != det.get_scale(scale):
        print(f"      [warn] QAT checkpoint 记录的尺度为 yolo26{saved_scale}，当前为 yolo26{det.get_scale(scale)}")
    quant_pkg.freeze_batch_init(qat_model)
    qat_metrics = evaluate(
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix="compare_qat"
    )

    print(
        f"[Compare] Float | best epoch:{float_meta.get('epoch', '-')}/"
        f"{float_meta.get('total_epochs', '-')} | "
        f"delta1:{float_metrics['metrics/delta1']:.4f} | "
        f"abs_rel:{float_metrics['metrics/abs_rel']:.4f} "
        f"rmse:{float_metrics['metrics/rmse']:.4f} silog:{float_metrics['metrics/silog']:.4f}"
    )
    print(
        f"[Compare] QAT   | best epoch:{qat_meta.get('epoch', '-')}/"
        f"{qat_meta.get('total_epochs', '-')} | "
        f"delta1:{qat_metrics['metrics/delta1']:.4f} | "
        f"abs_rel:{qat_metrics['metrics/abs_rel']:.4f} "
        f"rmse:{qat_metrics['metrics/rmse']:.4f} silog:{qat_metrics['metrics/silog']:.4f}"
    )
    print(
        f"[Compare] delta | delta1:{qat_metrics['metrics/delta1'] - float_metrics['metrics/delta1']:+.4f} | "
        f"abs_rel:{qat_metrics['metrics/abs_rel'] - float_metrics['metrics/abs_rel']:+.4f} "
        f"rmse:{qat_metrics['metrics/rmse'] - float_metrics['metrics/rmse']:+.4f} "
        f"silog:{qat_metrics['metrics/silog'] - float_metrics['metrics/silog']:+.4f}"
    )
    return {"float": float_metrics, "qat": qat_metrics, "float_meta": float_meta, "qat_meta": qat_meta}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="YOLO26-depth: 浮点训练(加载 yolo26{scale}-depth.pt，缺失则从头训练) -> PTQ -> QAT -> delta1/abs_rel/rmse/silog 对比，模型尺度与量化方法可选"
    )
    parser.add_argument(
        "--model",
        choices=det.MODEL_CHOICES,
        default=f"yolo26{det.DEFAULT_SCALE}",
        help="模型尺度（默认 yolo26n；仓库不随附 depth 预训练权重时从头训练）",
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
    parser.add_argument("--float-batch-size", type=int, default=16)
    parser.add_argument("--qat-batch-size", type=int, default=16)
    parser.add_argument("--ptq-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--float-epochs", type=int, default=100)
    parser.add_argument("--qat-epochs", type=int, default=None,
                        help="默认 max(1, float-epochs*0.2)")
    parser.add_argument("--float-lr", type=float, default=None,
                        help="默认 optimizer=auto: AdamW lr=round(0.002*5/(4+nc), 6)")
    parser.add_argument("--qat-lr", type=float, default=None,
                        help="默认 float auto lr x 0.1")
    parser.add_argument("--calibration-batches", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--data",
        default=DATA_YAML,
        help="数据集 yaml 路径或数据集目录（默认自动下载 depth8.yaml 到 dataset/；可手动指定其他目录）",
    )
    parser.add_argument("--resume", action="store_true",
                        help="从 _last.pth checkpoint 接续训练")
    return parser


if __name__ == "__main__":
    quant_pkg.install_print_timestamp()  # print 加分钟级时间戳 / minute-precision timestamp for print
    args = build_arg_parser().parse_args()

    # 根据 --data 动态覆盖全局 NUM_CLASSES 和 DATA_YAML（默认走自动下载） /
    # Override global NUM_CLASSES / DATA_YAML per --data (default auto-downloads)
    DATA_YAML = args.data  # 复用变量名；函数内部引用它 / Reuse variable name; functions reference it internally
    try:
        _tmp = det.get_data_dict(args.data, "depth")
        NUM_CLASSES = len(_tmp["names"])
    except Exception:
        pass  # yaml 解析失败则保留默认 1 / Keep default 1 if yaml parsing fails
    print(f"数据集: {args.data} | 类别数: {NUM_CLASSES}")

    scale = det.get_scale(args.model)
    print(f"Python: {sys.executable}")
    print(f"torch: {torch.__version__} | Device: {device}")
    print(f"模型: {det.model_name(scale)}-depth | 量化方法: {args.quant} | 阶段: {args.stage}")

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
