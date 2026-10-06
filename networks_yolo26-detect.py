"""YOLO26 检测网络的 aLSQ+ 量化感知训练流程（coco128，尺度可选 n/s/m/l/x） /
aLSQ+ Quantization-Aware Training pipeline for YOLO26 detection network
(coco128, scales n/s/m/l/x optional).

流程 / Pipeline:
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练 / float training（默认加载 yolo26{scale}.pt 预训练权重 / loads yolo26{scale}.pt pretrained weights by default）
    PTQ_calibration()                               # 训练后量化校准 / post-training quantization calibration
    QAT_training(lr=qat_lr, epochs=qat_epochs)      # 量化感知训练 / quantization-aware training
    compare_precision()                             # 浮点 vs QAT 的 mAP 对比 / mAP comparison between float vs QAT

模型尺度用 --model yolo26n|yolo26s|yolo26m|yolo26l|yolo26x 选择（默认 yolo26n） /
Model scale selected via --model yolo26n|yolo26s|yolo26m|yolo26l|yolo26x (default yolo26n):
通道按 make_divisible(min(c, max_ch)*width, 8)、重复次数按 max(round(n*depth), 1)
缩放，与 ultralytics parse_model 完全一致 / channels scaled via make_divisible(min(c, max_ch)*width, 8),
repeat counts via max(round(n*depth), 1), identical to ultralytics parse_model.
本地缺失的官方预训练权重（任意尺度）会像 ultralytics 一样从 GitHub Releases 自动下载；仅离线或自定义路径缺失时才从头训练 /
Missing official pretrained weights (any scale) are auto-downloaded from GitHub Releases like ultralytics;
random init only happens when offline or a custom path is missing.

网络结构与 ultralytics/cfg/models/26/yolo26.yaml（scale=n）逐层对齐 /
Network architecture aligned layer-by-layer with ultralytics/cfg/models/26/yolo26.yaml (scale=n):
Conv / C3k2(C3k) / SPPF / C2PSA(Attention) / PAN-FPN / Detect(reg_max=1).
说明 / Note: yolo26n 原始 end2end 双头（one2many+one2one）这里只保留 one2many 单头 /
original yolo26n end2end dual-head (one2many+one2one), only one2many head kept here;
损失仍使用官方 TaskAlignedAssigner(topk=10) + CIoU 的 v8DetectionLoss /
loss still uses official v8DetectionLoss with TaskAlignedAssigner(topk=10) + CIoU;
backbone / neck / 检测头拓扑与官方完全一致，可直接加载 yolo26n.pt 的权重 /
backbone / neck / detect head topology fully consistent with official, can directly load yolo26n.pt weights
（one2one_* 双头权重会被跳过 / one2one_* dual-head weights are skipped）.
"""

import argparse
import copy
import importlib
import json
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import quantization as quant_pkg

# ultralytics 提供数据管道 / 损失 / 解码 / NMS / mAP / ultralytics provides data pipeline / loss / decode / NMS / mAP
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, YAML
from ultralytics.utils.checks import check_file
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.torch_utils import model_info
from ultralytics.utils.plotting import plot_images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR_BASE = os.path.join(BASE_DIR, "model", "yolo26-detect")
MODEL_DIR = _MODEL_DIR_BASE


def _model_dir_for(scale=None, quant_method=None):
    """按网络名 + 尺度 + 量化后端(+量化配置标签+运行变体后缀)返回产物目录 /
    Return artifact directory by net name + scale + quant backend (+ quant config tag + run variant suffix)."""
    path = _MODEL_DIR_BASE
    if scale is not None:
        path = os.path.join(path, scale)
    if quant_method is not None:
        # seed/run_tag 变体后缀：多种子与校准敏感度实验避免覆盖主实验产物 /
        # seed/run_tag variant suffix: multi-seed & calibration-sensitivity runs never overwrite main artifacts
        suffix = (f"_seed{RUN_SEED}" if RUN_SEED else "") + (f"_{RUN_TAG}" if RUN_TAG else "")
        path = os.path.join(path, quant_method + _quant_cfg_tag() + suffix)
    os.makedirs(path, exist_ok=True)
    return path
ULTRA_DIR = os.path.join(BASE_DIR, "ultralytics", "ultralytics")

# ---------------------------------------------------------------------------
# 数据集解析：优先自动下载，--data 可手动指定任意 yaml / 目录 /
# Dataset resolution: auto-download by default; --data can point to any yaml / dir
# ---------------------------------------------------------------------------
# 自动下载根目录（ultralytics 按 yaml 内 download URL 把数据下载解压到这里） /
# Auto-download root (ultralytics downloads + unzips data here per yaml's download URL)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# 各任务默认的 ultralytics 官方数据集 yaml（裸文件名，位于 ultralytics 包 cfg/datasets/；
# 首次运行时自动下载：detect/seg/pose 用 coco8 系列，obb 用 dota8 多光谱，depth 用 depth8） /
# Default official dataset yamls per task (bare names inside ultralytics cfg/datasets/;
# auto-downloaded on first run: coco8 family for detect/seg/pose, dota8 multispectral for obb, depth8 for depth)
DEFAULT_DATA_YAML = {
    "detect": "coco8.yaml",
    "seg": "coco8-seg.yaml",
    "pose": "coco8-pose.yaml",
    "obb": "dota8-multispectral.yaml",
    "depth": "depth8.yaml",
}

# detect 任务当前使用的数据集（main 中可被 --data 覆盖） / dataset yaml used by detect (overridable via --data in main)
DATA_YAML = DEFAULT_DATA_YAML["detect"]

_DATASETS_DIR_PATCHED = False


def _patch_datasets_dir():
    """把 ultralytics 的 DATASETS_DIR 在本进程内重定向到项目 dataset/。

    / Redirect ultralytics' DATASETS_DIR to the project dataset/ dir for this process only.
    ultralytics 的 check_det_dataset 下载 zip 时使用模块级 DATASETS_DIR 常量（导入时绑定，
    不受 settings.json 影响），这里直接替换两个模块中的符号；不写用户全局配置，不影响其他项目。
    / check_det_dataset downloads zips to the module-level DATASETS_DIR (bound at import);
    patch the symbol in both modules in-process. No global settings.json side effects.
    """
    global _DATASETS_DIR_PATCHED
    if _DATASETS_DIR_PATCHED:
        return
    os.makedirs(DATASET_DIR, exist_ok=True)
    import ultralytics.utils as _ultra_utils
    import ultralytics.data.utils as _ultra_data_utils
    target = Path(DATASET_DIR)
    _ultra_utils.DATASETS_DIR = target
    _ultra_data_utils.DATASETS_DIR = target
    _DATASETS_DIR_PATCHED = True


def resolve_dataset_yaml(spec=None, task="detect"):
    """返回一份可直接传给 check_det_dataset 的 yaml 路径。

    / Return a yaml path ready for check_det_dataset.

    1) spec 指向已存在的 yaml 文件或数据集目录 → 原样返回（手动加载其他目录，
       例如本机已有的 ultralytics/ultralytics/data/datasets/...）；
    2) spec 显式给定但不存在 → 原样交给 ultralytics 处理（支持 URL / 官方报错提示）；
    3) 默认裸文件名（coco8.yaml 等）→ 用 ultralytics 的 check_file 定位包内官方 yaml，
       在 dataset/ 下生成一份 path 改写为绝对路径的便携副本并返回；随后 check_det_dataset
       在数据缺失时会走 ultralytics 原生 safe_download 流程自动下载解压到 dataset/。

    / 1) existing yaml/dir → returned as-is (manual datasets);
      2) explicit but missing spec → passed through to ultralytics (URL / official error);
      3) bare official name → locate packaged yaml via check_file, write a portable copy
         under dataset/ with an absolute path; check_det_dataset then auto-downloads via
         its native safe_download pipeline into dataset/.
    """
    name = str(spec or DEFAULT_DATA_YAML[task])
    p = Path(name)
    if p.exists():
        return name  # 手动目录 / 已有 yaml 优先 / manual path takes priority
    # 显式指定的非默认路径但不存在 → 交给 ultralytics（URL / 官方报错）；
    # 注意 argparse 默认值本身就是官方裸名，需按默认名处理，不能当成手动路径 /
    # explicit non-default missing spec → let ultralytics handle; the argparse default
    # itself is the bare official name, so it must follow the auto-download branch
    if spec is not None and name != DEFAULT_DATA_YAML.get(task):
        return name

    # 默认官方数据集：生成便携副本 / default official dataset: emit portable copy
    _patch_datasets_dir()
    stock = check_file(name)  # ultralytics 在其包内 cfg/datasets/ 查找 / search packaged cfg/datasets/
    cfg = YAML.load(stock, append_filename=True)
    cfg.pop("yaml_file", None)
    # 解压目录名取下载 zip 的文件名（coco8.zip→coco8, depth8-png.zip→depth8-png） /
    # extract dir derived from zip URL basename
    url = str(cfg.get("download", ""))
    stem = Path(url.split("?")[0]).stem if url else p.stem
    cfg["path"] = os.path.join(DATASET_DIR, stem)  # 绝对路径，任何机器均可移植 / absolute, portable
    os.makedirs(DATASET_DIR, exist_ok=True)
    out_yaml = os.path.join(DATASET_DIR, p.name)
    YAML.save(out_yaml, cfg)
    return out_yaml


def get_data_dict(spec=None, task="detect"):
    """解析数据集并返回 data 字典；数据缺失时由 ultralytics 自动下载。

    / Resolve dataset and return its data dict; auto-download via ultralytics if missing.
    """
    return check_det_dataset(resolve_dataset_yaml(spec, task))

IMGSZ = 640
NUM_CLASSES = 80
STRIDES = (8, 16, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 模型尺度选择：与 ultralytics/cfg/models/26/yolo26.yaml 的 scales 完全一致 /
# Model scale selection: fully consistent with ultralytics/cfg/models/26/yolo26.yaml scales
#   # [depth, width, max_channels]
#   n: [0.50, 0.25, 1024]  # 260 layers, 2,572,280 parameters, 6.1 GFLOPs
#   s: [0.50, 0.50, 1024]  # 260 layers, 10,009,784 parameters, 22.8 GFLOPs
#   m: [0.50, 1.00, 512]   # 280 layers, 21,896,248 parameters, 75.4 GFLOPs
#   l: [1.00, 1.00, 512]   # 392 layers, 26,299,704 parameters, 93.8 GFLOPs
#   x: [1.00, 1.50, 512]   # 392 layers, 58,993,368 parameters, 209.5 GFLOPs
# ---------------------------------------------------------------------------
YOLO26_SCALES = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}
DEFAULT_SCALE = "n"
MODEL_CHOICES = tuple(f"yolo26{s}" for s in YOLO26_SCALES)

# ---------------------------------------------------------------------------
# 量化超参：禁止在量化层构建处硬编码，统一从此处读取（由 CLI 在构建量化模型前写入） /
# Quantization hyperparameters: must NOT be hardcoded at quant-layer construction;
# all layers read them from here (written from CLI before the quantized model is built).
#   a_bits       激活量化位宽 / activation quantization bit-width
#   w_bits       权重量级化位宽 / weight quantization bit-width
#   per_channel  权重 per-channel(True) 或 per-tensor(False) / weight per-channel vs per-tensor
#   all_positive 激活无符号量化（默认 True：SiLU 后激活非负）；权重是否无符号由 w_all_positive 单独控制 /
#                unsigned activation quantization (default True: post-SiLU activations are non-negative);
#                weight unsigned-ness is controlled separately by w_all_positive
#   w_all_positive 权重无符号量化（默认 False：权重有符号。对有符号权重开 True 会破坏符号平衡，仅用于对照实验）/
#                unsigned weight quantization (default False = signed weights; enabling on signed weights breaks sign balance, for ablation only)
#   mixed_quant  混合量化(True)：首层 stem 与任务头保持 FP32，其余层按上面的位宽量化；False=整网统一量化 /
#                mixed quantization: stem (layer 0) and task head stay FP32, other layers use above bit-width; False = uniform quantization
#   pact_w_quant PACT 后端的权重量级化器选择（仅 quant=pact 时生效）：lsqplus_v1(默认)/dorefa(原始 PACT 实现)/minmax/lsqplus_v2 /
#                weight quantizer for the pact backend only: lsqplus_v1 (default) / dorefa (original PACT impl) / minmax / lsqplus_v2
# ---------------------------------------------------------------------------
QUANT_CFG = SimpleNamespace(a_bits=8, w_bits=8, per_channel=True,
                            all_positive=True, w_all_positive=False, mixed_quant=False,
                            pact_w_quant='lsqplus_v1')
# 历史/当前默认参数仅用于 CLI 默认值；所有实验目录都强制带显式配置标签，避免新旧默认互相覆盖 /
# Defaults below only seed the CLI; every experiment dir gets an explicit config tag so old/new defaults never overwrite each other.
DEFAULT_QUANT_CFG = SimpleNamespace(a_bits=8, w_bits=8, per_channel=True,
                                    all_positive=True, w_all_positive=False, mixed_quant=False,
                                    pact_w_quant='lsqplus_v1')

# 运行级变体（与量化配置无关，由 --seed/--run-tag 设置）：非默认时产物目录自动加后缀，避免覆盖主实验 /
# Run-level variants (set via --seed/--run-tag, independent of quant config): non-default values add a directory suffix
RUN_SEED = 0
RUN_TAG = ""


def _quant_cfg_tag(cfg=None):
    """生成量化配置目录/日志标签；始终显式携带完整配置名称 / Build quant-config tag with full names.

    标签组成 / Tag parts:
      _a{ab}w{wb}_{per_channel|per_tensor}{_act_unsigned|_act_signed}[_weight_unsigned][_mixed][_wquant_{method}]
      act_unsigned = 激活无符号（unsigned activations）；act_signed = 激活有符号（signed activations）
      weight_unsigned = 权重无符号（unsigned weights；缺省为有符号 signed）
      _wquant_{method} = pact 后端权重量级化器非默认(lsqplus_v1)时追加 / appended when pact weight quantizer differs from default (lsqplus_v1)
    """
    cfg = cfg if cfg is not None else QUANT_CFG
    return (
        f"_a{cfg.a_bits}w{cfg.w_bits}_"
        f"{'per_channel' if cfg.per_channel else 'per_tensor'}"
        f"{'_act_unsigned' if cfg.all_positive else '_act_signed'}"
        f"{'_weight_unsigned' if cfg.w_all_positive else ''}"
        f"{'_mixed' if cfg.mixed_quant else ''}"
        f"{'' if getattr(cfg, 'pact_w_quant', DEFAULT_QUANT_CFG.pact_w_quant) == DEFAULT_QUANT_CFG.pact_w_quant else '_wquant_' + cfg.pact_w_quant}"
    )


def add_quant_cfg_args(parser):
    """给 argparse 注册量化超参（a_bits/w_bits/per_channel/all_positive），所有任务共用 /
    Register quant hyperparameter CLI args shared by all tasks."""
    parser.add_argument("--a-bits", type=int, default=DEFAULT_QUANT_CFG.a_bits,
                        help="激活量化位宽（默认 8）/ activation quantization bits (default 8)")
    parser.add_argument("--w-bits", type=int, default=DEFAULT_QUANT_CFG.w_bits,
                        help="权重量级化位宽（默认 8）/ weight quantization bits (default 8)")
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_QUANT_CFG.per_channel,
                        help="权重 per-channel 量化；--no-per-channel 表示 per-tensor（默认 per-channel）/ "
                             "weight per-channel quant; --no-per-channel = per-tensor (default per-channel)")
    parser.add_argument("--all-positive", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_QUANT_CFG.all_positive,
                        help="激活无符号量化（默认开启：SiLU 后非负特征）；--no-all-positive 改回有符号 / unsigned activation quant (default on for post-SiLU features); --no-all-positive forces signed")
    parser.add_argument("--w-all-positive", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_QUANT_CFG.w_all_positive,
                        help="权重无符号量化；默认关闭（权重有符号，开启易导致精度崩溃，仅对照用）/ unsigned weight quant; default off (signed weights; enabling may wreck accuracy, ablation only)")
    parser.add_argument("--mixed-quant", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_QUANT_CFG.mixed_quant,
                        help="混合量化：首层 stem + 任务头保持 FP32，其余按 --a-bits/--w-bits 量化；默认关闭（整网 int8）/ "
                             "mixed quant: stem + task head stay FP32, rest quantized; default off (uniform int8)")
    parser.add_argument("--pact-w-quant", type=str, default=DEFAULT_QUANT_CFG.pact_w_quant,
                        choices=['dorefa', 'minmax', 'lsqplus_v1', 'lsqplus_v2'],
                        help="PACT 后端的权重量级化器（仅 --quant pact 生效，默认 lsqplus_v1；dorefa 与原始 PACT 实现一致）/ "
                             "weight quantizer for pact backend (only with --quant pact; default lsqplus_v1; dorefa matches original PACT)")


def apply_quant_cfg_args(args):
    """把 CLI 量化超参写入 QUANT_CFG（必须在量化模型构建前调用）/ Write CLI quant args into QUANT_CFG (call before building quant model)."""
    QUANT_CFG.a_bits = args.a_bits
    QUANT_CFG.w_bits = args.w_bits
    QUANT_CFG.per_channel = args.per_channel
    QUANT_CFG.all_positive = args.all_positive
    QUANT_CFG.w_all_positive = args.w_all_positive
    QUANT_CFG.mixed_quant = args.mixed_quant
    QUANT_CFG.pact_w_quant = args.pact_w_quant
    tag = _quant_cfg_tag()
    print(f"量化配置 / quant cfg: a_bits={QUANT_CFG.a_bits} w_bits={QUANT_CFG.w_bits} "
          f"per_channel={QUANT_CFG.per_channel} act_all_positive={QUANT_CFG.all_positive} "
          f"w_all_positive={QUANT_CFG.w_all_positive} mixed_quant={QUANT_CFG.mixed_quant} "
          f"pact_w_quant={QUANT_CFG.pact_w_quant}"
          f"{f' 标签/tag={tag}' if tag else ''}")



def get_scale(scale=DEFAULT_SCALE):
    """归一化模型尺度名：允许传 'n' / 'yolo26n' / 'YOLO26n' /
    Normalize model scale name: accepts 'n' / 'yolo26n' / 'YOLO26n'."""
    s = str(scale)[-1].lower()
    if s not in YOLO26_SCALES:
        raise ValueError(f"未知模型尺度 {scale!r}（可选 {', '.join(MODEL_CHOICES)}）")
    return s


def make_divisible(x, divisor=8):
    """ultralytics 的通道取整规则 / ultralytics channel round rule."""
    return math.ceil(x / divisor) * divisor


def scaled_channels(channels, scale=DEFAULT_SCALE):
    """ultralytics parse_model 通道缩放：make_divisible(min(c, max_ch) * width, 8) /
    ultralytics parse_model channel scaling."""
    _, width, max_channels = YOLO26_SCALES[get_scale(scale)]
    return make_divisible(min(channels, max_channels) * width)


def scaled_repeats(repeats, scale=DEFAULT_SCALE):
    """ultralytics parse_model 层重复次数缩放：max(round(n * depth), 1) /
    ultralytics parse_model layer repeat scaling."""
    depth, _, _ = YOLO26_SCALES[get_scale(scale)]
    return max(round(repeats * depth), 1)


def model_name(scale=DEFAULT_SCALE):
    """官方风格模型名，如 YOLO26n / YOLO26x / Official-style model name, e.g. YOLO26n / YOLO26x."""
    return f"YOLO26{get_scale(scale).upper()}"


def base_name(scale=DEFAULT_SCALE, task=""):
    """权重 / checkpoint 基础名，如 yolo26n、yolo26s-seg / Base name for weights / checkpoint, e.g. yolo26n, yolo26s-seg."""
    return f"yolo26{get_scale(scale)}" + (f"-{task}" if task else "")


def detect_head_channels(scale=DEFAULT_SCALE):
    """Detect 头三尺度输入通道（层 16/19/22 输出，yaml 基准 256/512/1024） /
    Three-scale input channels for Detect head (layer 16/19/22 outputs, yaml base 256/512/1024)."""
    return tuple(scaled_channels(c, scale) for c in (256, 512, 1024))

# ---------------------------------------------------------------------------
# 量化后端可切换：dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact /
# Quant backend is switchable: dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact
# 每个后端文件都实现了同一套算子；set_quant_method() 切换后再实例化量化模型即可 /
# Each backend file implements the same operator set; call set_quant_method() before instantiating quant model.
# ---------------------------------------------------------------------------
DEFAULT_QUANT_METHOD = "lsqplus_v1"
QUANT_METHOD = DEFAULT_QUANT_METHOD
Q = quant_pkg.load_quant_backend(DEFAULT_QUANT_METHOD)

QuantAdd = Q.QuantAdd
QuantCat = Q.QuantCat
QuantConcat = Q.QuantConcat
QuantConv2d = Q.QuantConv2d
QuantMaxPool = Q.QuantMaxPool
QuantSiLU = getattr(Q, 'QuantSiLU', None)
QuantSigmoid = getattr(Q, 'QuantSigmoid', None)
QuantReLU = getattr(Q, 'QuantReLU', None)


def set_quant_method(method):
    """切换量化后端（必须在构建 QuantYOLO26 之前调用） /
    Switch quant backend (must call before constructing QuantYOLO26)."""
    global Q, QUANT_METHOD
    global QuantAdd, QuantCat, QuantConcat, QuantConv2d, QuantMaxPool, QuantSiLU, QuantSigmoid, QuantReLU

    Q = quant_pkg.load_quant_backend(method)
    QUANT_METHOD = method
    QuantAdd = Q.QuantAdd
    QuantCat = Q.QuantCat
    QuantConcat = Q.QuantConcat
    QuantConv2d = Q.QuantConv2d
    QuantMaxPool = Q.QuantMaxPool
    QuantSiLU = getattr(Q, 'QuantSiLU', None)
    QuantSigmoid = getattr(Q, 'QuantSigmoid', None)
    QuantReLU = getattr(Q, 'QuantReLU', None)
    return Q


# ============================== 基础组件 / Basic Components ==============================


class FloatAdd(nn.Module):
    """浮点残差加法（无参数） / Float residual addition (parameter-free)."""

    def forward(self, a, b):
        return a + b

def autopad(kernel_size, padding=None, dilation=1):
    if padding is not None:
        return padding
    return (kernel_size if isinstance(kernel_size, int) else kernel_size[0]) * dilation // 2


def make_bn(channels):
    # 与 ultralytics 保持一致：eps=1e-3, momentum=0.03 / Keep consistent with ultralytics: eps=1e-3, momentum=0.03
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)


def _quant_layer_kwargs():
    """量化卷积层公共构造参数；pact 后端额外注入 w_quant（其余后端类不接受该参数，按鸭子类型跳过）/
    Common kwargs for quantized conv layers; pact backend additionally gets w_quant (skipped for backends whose signature lacks it)."""
    kw = dict(a_bits=QUANT_CFG.a_bits, w_bits=QUANT_CFG.w_bits,
              per_channel=QUANT_CFG.per_channel, all_positive=QUANT_CFG.all_positive,
              w_all_positive=QUANT_CFG.w_all_positive)
    import inspect
    if 'w_quant' in inspect.signature(QuantConv2d.__init__).parameters:
        kw['w_quant'] = QUANT_CFG.pact_w_quant
    return kw


class Conv(nn.Module):
    """Conv + BN + SiLU；quant=True 时卷积换成 QuantConv2d / Conv + BN + SiLU; quant=True replaces conv with QuantConv2d.

    与 ultralytics.nn.modules.Conv 同名同结构（self.conv / self.bn / self.act） /
    Same name and structure as ultralytics.nn.modules.Conv (self.conv / self.bn / self.act).
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False, quant=False):
        super().__init__()
        if quant:
            self.conv = QuantConv2d(
                c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d),
                dilation=d, groups=g, bias=bias,
                **_quant_layer_kwargs(),
            )
        else:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = make_bn(c2)
        if quant and QuantSiLU is not None:
            self.act = QuantSiLU(a_bits=QUANT_CFG.a_bits, quant_inference=False) if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        else:
            self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """标准瓶颈块：cv1(3x3) -> cv2(3x3)，同通道时带残差加法 /
    Standard bottleneck: cv1(3x3) -> cv2(3x3), with residual add when channels match."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, quant=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1, quant=quant)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, quant=quant)
        self.has_add = shortcut and c1 == c2
        self.add_op = QuantAdd(a_bits=QUANT_CFG.a_bits, quant_inference=True) if quant else FloatAdd()

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return self.add_op(x, out) if self.has_add else out


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions（C3k 的基类） / CSP Bottleneck with 3 convolutions (base class of C3k)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, quant=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, quant=quant)
        self.cv2 = Conv(c1, c_, 1, 1, quant=quant)
        self.cv3 = Conv(2 * c_, c2, 1, 1, quant=quant)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0, quant=quant) for _ in range(n))
        )
        self.cat_op = QuantCat(2, a_bits=QUANT_CFG.a_bits) if quant else torch.cat

    def forward(self, x):
        return self.cv3(self._cat([self.m(self.cv1(x)), self.cv2(x)], 1))

    def _cat(self, tensors, dim):
        return self.cat_op(tensors, dim) if isinstance(self.cat_op, QuantCat) else torch.cat(tensors, dim)


class C3k(C3):
    """C3k：内部堆叠 n 个 3x3 Bottleneck（yolo26n 中 n=2） /
    C3k: stacks n 3x3 Bottlenecks internally (n=2 in yolo26n)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, quant=False):
        super().__init__(c1, c2, n, shortcut, g, e, quant=quant)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0, quant=quant) for _ in range(n))
        )


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions（C3k2 的基类） / CSP Bottleneck with 2 convolutions (base class of C3k2)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, quant=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, quant=quant)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1, quant=quant)
        self.cat_op = QuantCat(n + 2, a_bits=QUANT_CFG.a_bits) if quant else torch.cat

    def _cat(self, tensors, dim=1):
        return self.cat_op(tensors, dim) if isinstance(self.cat_op, QuantCat) else torch.cat(tensors, dim)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self._cat(y, 1))


class C3k2(C2f):
    """yolo26n 主干核心块。c3k=True 时使用 C3k，attn=True 时使用 Bottleneck+PSABlock /
    yolo26n backbone core block. Uses C3k when c3k=True, Bottleneck+PSABlock when attn=True."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True, quant=False):
        super().__init__(c1, c2, n, shortcut, g, e, quant=quant)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g, quant=quant),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1), quant=quant),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g, quant=quant)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g, quant=quant)
            for _ in range(n)
        )


class Attention(nn.Module):
    """PSA 多头自注意力（qkv / proj / pe 为卷积；matmul+softmax 保持浮点） /
    PSA multi-head self-attention (qkv / proj / pe are convolutions; matmul+softmax stays float)."""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5, quant=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False, quant=quant)
        self.proj = Conv(dim, dim, 1, act=False, quant=quant)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False, quant=quant)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(out)


class PSABlock(nn.Module):
    """注意力 + FFN，两条残差加法 / Attention + FFN, two residual additions."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, quant=False):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads, quant=quant)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1, quant=quant), Conv(c * 2, c, 1, act=False, quant=quant))
        self.add = shortcut
        self.add1 = QuantAdd(a_bits=QUANT_CFG.a_bits, quant_inference=True) if quant else FloatAdd()
        self.add2 = QuantAdd(a_bits=QUANT_CFG.a_bits, quant_inference=True) if quant else FloatAdd()

    def forward(self, x):
        x = self.add1(x, self.attn(x)) if self.add else self.attn(x)
        x = self.add2(x, self.ffn(x)) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """C2PSA：cv1 降维 -> PSABlock 序列 -> cv2 融合 / C2PSA: cv1 dim-reduce -> PSABlock sequence -> cv2 fuse."""

    def __init__(self, c1, c2, n=1, e=0.5, quant=False):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, quant=quant)
        self.cv2 = Conv(2 * self.c, c1, 1, 1, quant=quant)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1), quant=quant) for _ in range(n))
        )
        self.cat_op = QuantCat(2, a_bits=QUANT_CFG.a_bits) if quant else torch.cat

    def _cat(self, tensors, dim=1):
        return self.cat_op(tensors, dim) if isinstance(self.cat_op, QuantCat) else torch.cat(tensors, dim)

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(self._cat([a, b], 1))


class SPPF(nn.Module):
    """YOLO26 版 SPPF：cv1 不带激活，串联 3 次 MaxPool，shortcut 残差加法 /
    YOLO26-style SPPF: cv1 no activation, 3 chained MaxPools, shortcut residual add."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False, quant=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False, quant=quant)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1, quant=quant)
        if quant:
            self.m = QuantMaxPool(kernel_size=k, stride=1, padding=k // 2, a_bits=QUANT_CFG.a_bits, quant_inference=True)
        else:
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2
        self.cat_op = QuantCat(n + 1, a_bits=QUANT_CFG.a_bits) if quant else torch.cat
        self.add_op = QuantAdd(a_bits=QUANT_CFG.a_bits, quant_inference=True) if quant else FloatAdd()

    def _cat(self, tensors, dim=1):
        return self.cat_op(tensors, dim) if isinstance(self.cat_op, QuantCat) else torch.cat(tensors, dim)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        out = self.cv2(self._cat(y, 1))
        return self.add_op(out, x) if self.add else out


class Concat(nn.Module):
    """PAN-FPN 中的拼接节点（对应 yaml 里的 Concat 层） /
    Concat node in PAN-FPN (corresponds to Concat layer in yaml)."""

    def __init__(self, dimension=1, quant=False):
        super().__init__()
        self.d = dimension
        self.op = QuantConcat(a_bits=QUANT_CFG.a_bits, quant_inference=True) if quant else None

    def forward(self, xs):
        if self.op is not None:
            return self.op(xs[0], xs[1], self.d)
        return torch.cat(xs, self.d)


class Detect(nn.Module):
    """YOLO26 Detect 检测头（单 one2many 头，reg_max=1，分类支路为 DWConv） /
    YOLO26 Detect head (single one2many head, reg_max=1, class branch uses DWConv).

    训练时返回 dict(boxes/scores/feats) 供 v8DetectionLoss 使用 /
    During training returns dict(boxes/scores/feats) for v8DetectionLoss;
    评估时返回解码后的 (B, 4+nc, num_anchors) 检测张量（xywh + sigmoid 分数） /
    During eval returns decoded (B, 4+nc, num_anchors) detection tensor (xywh + sigmoid scores).
    """

    def __init__(self, nc=NUM_CLASSES, reg_max=1, ch=(64, 128, 256), quant=False):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)
        # 混合量化时整个检测头保持 FP32（quant=True 但 hq=False）；seg/pose 子类复用此标志 /
        # Under mixed quantization the whole head stays FP32 (quant=True but hq=False); reused by seg/pose subclasses.
        hq = quant and not QUANT_CFG.mixed_quant
        self.head_quant = hq
        c2 = max((16, ch[0] // 4, reg_max * 4))
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3, quant=hq),
                Conv(c2, c2, 3, quant=hq),
                QuantConv2d(c2, 4 * reg_max, 1, **_quant_layer_kwargs())
                if hq else nn.Conv2d(c2, 4 * reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x, quant=hq), Conv(x, c3, 1, quant=hq)),
                nn.Sequential(Conv(c3, c3, 3, g=c3, quant=hq), Conv(c3, c3, 1, quant=hq)),
                QuantConv2d(c3, nc, 1, **_quant_layer_kwargs())
                if hq else nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.dfl = nn.Identity()
        self._anchors = None
        self._strides_tensor = None
        self._feat_shape = None

    def forward_head(self, x):
        bs = x[0].shape[0]
        boxes = torch.cat([self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x}

    def forward(self, x):
        preds = self.forward_head(x)
        if self.training:
            return preds
        # 推理：ltrb 距离 -> xywh 框（×stride），分类 sigmoid / Inference: ltrb distances -> xywh bbox (×stride), class sigmoid
        shape = x[0].shape
        if self._feat_shape != shape:
            # 注册为非持久 buffer：随 .cpu()/.cuda()/deepcopy 正确迁移，且不写入 state_dict /
            # Register as non-persistent buffers so .cpu()/.cuda()/deepcopy move them, without entering state_dict
            _anchors, _strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            # 可能已以普通属性存在（__init__ 置 None / 旧版缓存）或已注册（特征尺寸变化），先清理 /
            # May pre-exist as a plain attr (None from __init__ / legacy cache) or be registered (feature-shape change): clear first
            self.__dict__.pop('_anchors', None); self.__dict__.pop('_strides_tensor', None)
            self._buffers.pop('_anchors', None); self._buffers.pop('_strides_tensor', None)
            self.register_buffer('_anchors', _anchors, persistent=False)
            self.register_buffer('_strides_tensor', _strides, persistent=False)
            self._feat_shape = shape
        dbox = dist2bbox(self.dfl(preds["boxes"]), self._anchors.unsqueeze(0), xywh=True, dim=1)
        dbox = dbox * self._strides_tensor
        return torch.cat((dbox, preds["scores"].sigmoid()), 1)

    def bias_init(self):
        for i, (box_head, cls_head) in enumerate(zip(self.cv2, self.cv3)):
            box_head[-1].bias.data[:] = 2.0
            cls_head[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


# ============================== 网络主体 / Network Body ==============================


def _tag(module, index, source):
    module.i = index
    module.f = source
    return module


class YOLO26(nn.Module):
    """YOLO26 检测网络（scale 可选 n/s/m/l/x） / YOLO26 detection network (scale n/s/m/l/x optional).
    quant=False 浮点模型，True 为 aLSQ+ 伪量化模型 / quant=False for float model, True for aLSQ+ pseudo-quantized model.

    层编号 / 拓扑 / 命名与 ultralytics 解析 yolo26.yaml 得到的 DetectionModel 一致 /
    Layer index / topology / naming consistent with DetectionModel parsed from yolo26.yaml by ultralytics,
    通道按 make_divisible(min(c, max_ch)*width, 8)、重复次数按 max(round(n*depth), 1)
    缩放（与 parse_model 相同），因此浮点模型可直接加载对应尺度 yolo26{scale}.pt 的
    state_dict / channels scaled via make_divisible(min(c, max_ch)*width, 8), repeats via
    max(round(n*depth), 1) (same as parse_model), so float model can directly load yolo26{scale}.pt state_dict.
    """

    def __init__(self, nc=NUM_CLASSES, quant=False, scale=DEFAULT_SCALE):
        super().__init__()
        self.quant = quant
        self.nc = nc
        self.scale = get_scale(scale)
        self.yaml_file = f"{base_name(self.scale)}.yaml"  # 供官方 model_info 打印模型名 / for official model_info to print model name
        s = self.scale
        C = lambda c: scaled_channels(c, s)  # 通道缩放 / channel scaling
        N = lambda r: scaled_repeats(r, s)   # 层重复次数缩放 / layer repeat scaling
        # ultralytics parse_model 特殊规则：M/L/X 尺度所有 C3k2 强制 c3k=True /
        # ultralytics parse_model special rule: all C3k2 forced c3k=True for M/L/X scales
        # （tasks.py: `if scale in {"m", "l", "x"}: args[3:4] = [True]`，覆盖 yaml 的 False） /
        # (tasks.py: overrides yaml's False)
        c3k_all = s in ("m", "l", "x")
        layers = []

        # ---------------- backbone ----------------
        # 混合量化：首层 stem（3→64）保持 FP32，其余层正常量化 / Mixed quant: stem (3→64) stays FP32, other layers quantized
        stem_quant = quant and not QUANT_CFG.mixed_quant
        layers += [_tag(Conv(3, C(64), 3, 2, quant=stem_quant), 0, -1)]               # P1/2
        layers += [_tag(Conv(C(64), C(128), 3, 2, quant=quant), 1, -1)]          # P2/4
        layers += [_tag(C3k2(C(128), C(256), n=N(2), c3k=c3k_all, e=0.25, quant=quant), 2, -1)]
        layers += [_tag(Conv(C(256), C(256), 3, 2, quant=quant), 3, -1)]         # P3/8
        layers += [_tag(C3k2(C(256), C(512), n=N(2), c3k=c3k_all, e=0.25, quant=quant), 4, -1)]
        layers += [_tag(Conv(C(512), C(512), 3, 2, quant=quant), 5, -1)]         # P4/16
        layers += [_tag(C3k2(C(512), C(512), n=N(2), c3k=True, quant=quant), 6, -1)]
        layers += [_tag(Conv(C(512), C(1024), 3, 2, quant=quant), 7, -1)]        # P5/32
        layers += [_tag(C3k2(C(1024), C(1024), n=N(2), c3k=True, quant=quant), 8, -1)]
        layers += [_tag(SPPF(C(1024), C(1024), k=5, n=3, shortcut=True, quant=quant), 9, -1)]
        layers += [_tag(C2PSA(C(1024), C(1024), n=N(2), quant=quant), 10, -1)]

        # ---------------- head (PAN-FPN) ----------------
        layers += [_tag(nn.Upsample(scale_factor=2, mode="nearest"), 11, -1)]
        layers += [_tag(Concat(1, quant=quant), 12, [-1, 6])]
        layers += [_tag(C3k2(C(1024) + C(512), C(512), n=N(2), c3k=True, quant=quant), 13, -1)]

        layers += [_tag(nn.Upsample(scale_factor=2, mode="nearest"), 14, -1)]
        layers += [_tag(Concat(1, quant=quant), 15, [-1, 4])]
        layers += [_tag(C3k2(C(512) + C(512), C(256), n=N(2), c3k=True, quant=quant), 16, -1)]  # P3

        layers += [_tag(Conv(C(256), C(256), 3, 2, quant=quant), 17, -1)]
        layers += [_tag(Concat(1, quant=quant), 18, [-1, 13])]
        layers += [_tag(C3k2(C(256) + C(512), C(512), n=N(2), c3k=True, quant=quant), 19, -1)]  # P4

        layers += [_tag(Conv(C(512), C(512), 3, 2, quant=quant), 20, -1)]
        layers += [_tag(Concat(1, quant=quant), 21, [-1, 10])]
        layers += [_tag(C3k2(C(512) + C(1024), C(1024), n=N(1), c3k=True, attn=True, quant=quant), 22, -1)]  # P5

        layers += [_tag(Detect(nc=nc, reg_max=1, ch=detect_head_channels(s), quant=quant), 23, [16, 19, 22])]

        self.model = nn.ModuleList(layers)
        self._register_quantizer_buffers()
        self.save = {23}
        self.save.update(
            s for m in layers if isinstance(m.f, list) for s in m.f if s != -1
        )

        self._initialize_head()

    def _register_quantizer_buffers(self):
        """把量化器的 init_state 注册为持久化 buffer / Register quantizer init_state as persistent buffer.

        LSQPlus*Quantizer 原生的 init_state 是普通 int，不会进入 state_dict，
        checkpoint 重载后会回到 0，导致前向时用本批统计量覆盖已校准/训练好的 s /
        Native LSQPlus*Quantizer init_state is a plain int, not in state_dict;
        after checkpoint reload it reverts to 0, causing current-batch stats to
        overwrite calibrated/trained s during forward.
        注册成 buffer 后可随 checkpoint 保存与恢复 / Registering as buffer saves/restores with checkpoint.
        """
        for module in self.modules():
            if "init_state" in dict(module.named_buffers()):
                continue
            if hasattr(module, "init_state"):
                value = int(module.init_state)
                del module.init_state  # 普通 int 属性需先删除才能注册同名 buffer / plain int attribute must be deleted first to register same-name buffer
                module.register_buffer(
                    "init_state", torch.tensor(value), persistent=True
                )

    def _initialize_head(self):
        """用一次 dummy 前向推算 stride 并初始化 Detect 偏置（官方做法） /
        Use one dummy forward to infer strides and init Detect bias (official approach).

        注意：dummy 输入不能用全零！LSQ v1 的 activation_quantizer 用全零
        初始化 s=0，之后 torch.div(x, 0) → NaN / Note: dummy input must NOT be all zeros!
        LSQ v1 activation_quantizer initializes s=0 with all-zeros, causing torch.div(x,0) → NaN.
        用 randn 让每个 quantizer 得到合理的初始 s / Use randn so each quantizer gets a reasonable initial s.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMGSZ, IMGSZ) * 0.1  # 小随机噪声，避免全零 / small random noise, avoid all zeros
            feats = self._forward_features(dummy)
            head = self.model[-1]
            head.stride = torch.tensor([IMGSZ / f.shape[-2] for f in feats])
            head.bias_init()
            head._feat_shape = None
        if was_training:
            self.train()

    def _forward_features(self, x):
        y = []
        feats = None
        for m in self.model:
            if m.i == 23:
                feats = [x if j == -1 else y[j] for j in m.f]
                x = m(feats)
            else:
                if isinstance(m.f, list):
                    x = [x if j == -1 else y[j] for j in m.f]
                elif m.f != -1:
                    x = y[m.f]
                x = m(x)
            y.append(x if m.i in self.save else None)
        return feats if feats is not None else x

    def forward(self, x):
        y = []
        for m in self.model:
            if isinstance(m.f, list):
                x = [x if j == -1 else y[j] for j in m.f]
            elif m.f != -1:
                x = y[m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


class FloatYOLO26(YOLO26):
    def __init__(self, nc=NUM_CLASSES, scale=DEFAULT_SCALE):
        super().__init__(nc=nc, quant=False, scale=scale)


class QuantYOLO26(YOLO26):
    def __init__(self, nc=NUM_CLASSES, scale=DEFAULT_SCALE):
        super().__init__(nc=nc, quant=True, scale=scale)


# ============================== 数据 / 损失 / 评估 / Data / Loss / Evaluation ==============================


def _make_cfg(num_workers):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = IMGSZ
    cfg.task = "detect"
    cfg.workers = num_workers
    return cfg


def _build_loaders(batch_size, num_workers, data, calibration=False):
    """构建 dataloader；train 模式额外返回 cfg / train_set（close_mosaic 需要引用） /
    Build dataloader; train mode additionally returns cfg / train_set (required by close_mosaic).

    train：640x640 + mosaic/翻转等增强；val：rect letterbox；
    calibration：无增强的 640x640 letterbox /
    train: 640x640 + mosaic/flip etc. augmentations; val: rect letterbox;
    calibration: no-aug 640x640 letterbox.
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
    """复用 ultralytics 官方 coco128 数据管道 / Reuse ultralytics official coco128 data pipeline."""
    data = get_data_dict(DATA_YAML, "detect")
    if calibration:
        return _build_loaders(batch_size, num_workers, data, calibration=True), data
    train_loader, val_loader, _, _ = _build_loaders(batch_size, num_workers, data)
    return train_loader, val_loader, data


class _LossShim:
    """v8DetectionLoss 只需要 model.args / model.model[-1] / model.parameters() /
    v8DetectionLoss only needs model.args / model.model[-1] / model.parameters()."""

    def __init__(self, detect_head, epochs):
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=epochs)
        self.model = [None] * 23 + [detect_head]
        self.class_weights = None

    def parameters(self):
        return self.model[-1].parameters()


def build_criterion(model, epochs):
    return v8DetectionLoss(_LossShim(model.model[-1], epochs), tal_topk=10)


IOU_VECTOR = torch.linspace(0.5, 0.95, 10)


def _match_predictions(pred_labels, pred_bboxes, gt_labels, gt_bboxes, iou_vector):
    """复刻 ultralytics BaseValidator._process_batch：在 10 个 IoU 阈值上匹配预测与 GT /
    Replicate ultralytics BaseValidator._process_batch: match predictions and GT at 10 IoU thresholds."""
    iou = box_iou(gt_bboxes, pred_bboxes)
    correct = torch.zeros(pred_bboxes.shape[0], iou_vector.numel(), dtype=torch.bool)
    correct_class = gt_labels[:, None] == pred_labels[None, :]
    for i, threshold in enumerate(iou_vector):
        matches_idx = torch.where((iou >= threshold) & correct_class)
        if matches_idx[0].shape[0] == 0:
            continue
        matches = torch.cat(
            (torch.stack(matches_idx, 1), iou[matches_idx[0], matches_idx[1]][:, None]), 1
        ).cpu().numpy()
        if matches_idx[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), i] = True
    return correct


def _maybe_visualize(batch, predictions, names, viz_dir, viz_prefix, viz_state):
    """用 ultralytics 官方 plot_images 保存本批的 GT 拼图与预测拼图（val_batch 风格，与官方验证器一致） /
    Use official ultralytics plot_images to save GT and prediction mosaics of this batch (val_batch style, same as official validators).

    viz_state 跨 batch 记录已保存图片数 / viz_state tracks saved image count across batches.
    """
    os.makedirs(viz_dir, exist_ok=True)
    bs = batch["img"].shape[0]
    bi = viz_state["batch"]
    # GT 拼图（labels）：cls + 归一化 xywh 框 + batch_idx，与官方 plot_val_samples 一致 /
    # GT mosaic (labels): cls + normalized xywh boxes + batch_idx, same as official plot_val_samples
    plot_images(
        labels={
            "cls": batch["cls"].squeeze(-1),
            "bboxes": batch["bboxes"],
            "batch_idx": batch["batch_idx"],
        },
        images=batch["img"],
        paths=batch.get("im_file"),
        fname=os.path.join(viz_dir, f"{viz_prefix}_batch{bi}_labels.jpg"),
        names=names,
        threaded=False,  # 训练循环内同步执行，避免线程堆积 / run synchronously inside training loop to avoid thread pile-up
    )
    # 预测拼图（preds）：与官方 plot_predictions 一致，bboxes 需 xyxy→xywh /
    # Prediction mosaic (preds): same as official plot_predictions, bboxes need xyxy→xywh
    if any(p.shape[0] for p in predictions):
        plot_images(
            labels={
                "cls": torch.cat([p[:, 5] for p in predictions]),
                "conf": torch.cat([p[:, 4] for p in predictions]),
                "bboxes": xyxy2xywh(torch.cat([p[:, :4] for p in predictions])),
                "batch_idx": torch.cat(
                    [torch.full((p.shape[0],), i) for i, p in enumerate(predictions)]
                ),
            },
            images=batch["img"],
            paths=batch.get("im_file"),
            fname=os.path.join(viz_dir, f"{viz_prefix}_batch{bi}_pred.jpg"),
            names=names,
            threaded=False,
        )
    viz_state["saved"] += bs
    viz_state["batch"] += 1


@torch.no_grad()
def evaluate(model, val_loader, data, batch_size=8, max_batches=None, conf_thres=0.001,
             iou_thres=0.7, viz_dir=None, viz_max=30, viz_prefix="eval"):
    """在验证集上计算 mAP50 / mAP50-95（NMS + ap_per_class，与官方一致）。
    若 viz_dir 不为空，额外把前 viz_max 张验证图的 GT/预测拼图保存到 {viz_dir}/fvisualize/ /
    Compute mAP50 / mAP50-95 on val set (NMS + ap_per_class, consistent with official).
    If viz_dir is set, additionally save GT/prediction mosaics of the first viz_max val images into {viz_dir}/fvisualize/."""
    model.eval()
    stats_conf, stats_pcls, stats_tcls, stats_tp = [], [], [], []
    names = data.names if hasattr(data, 'names') else data["names"]
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
        predictions = non_max_suppression(
            model(images),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            multi_label=True,
            agnostic=False,
            max_det=300,
        )
        image_size = batch["img"].shape[2:]

        # 每次评估都可视化前几批（失败仅告警，绝不影响评估） / visualize first batches on every evaluate (failure only warns, never breaks eval)
        if viz_out and viz_state["saved"] < viz_max:
            try:
                _maybe_visualize(batch, predictions, names, viz_out, viz_prefix, viz_state)
            except Exception as exc:
                print(f"      [viz] 可视化保存失败（仅告警）: {exc} / visualization save failed (warn only): {exc}")

        for sample_index, pred in enumerate(predictions):
            index = batch["batch_idx"] == sample_index
            gt_cls = batch["cls"][index].squeeze(-1)
            gt_boxes = batch["bboxes"][index]
            if gt_cls.shape[0]:
                gt_boxes = xywh2xyxy(gt_boxes) * torch.tensor(image_size)[[1, 0, 1, 0]]

            true_positive = torch.zeros(pred.shape[0], IOU_VECTOR.numel(), dtype=torch.bool)
            if pred.shape[0] and gt_cls.shape[0]:
                true_positive = _match_predictions(
                    pred[:, 5].cpu(), pred[:, :4].cpu(), gt_cls, gt_boxes, IOU_VECTOR
                )

            stats_conf.append(pred[:, 4].cpu())
            stats_pcls.append(pred[:, 5].cpu())
            stats_tp.append(true_positive)
            stats_tcls.append(gt_cls)

    conf_all = torch.cat(stats_conf).numpy()
    pred_cls_all = torch.cat(stats_pcls).numpy()
    tp_all = torch.cat(stats_tp).numpy()
    target_cls_all = torch.cat(stats_tcls).numpy()

    if conf_all.shape[0] == 0 or target_cls_all.shape[0] == 0:
        return {"map50": 0.0, "map": 0.0, "precision": 0.0, "recall": 0.0}

    _, _, precision, recall, _, ap, _, _, _, _, _, _ = ap_per_class(
        tp_all, conf_all, pred_cls_all, target_cls_all, plot=False, names=names
    )
    return {
        "map50": float(ap[:, 0].mean()),
        "map": float(ap.mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
    }


# ============================== checkpoint / 权重复制 / 导出 / checkpoint / Weight Copy / Export ==============================


def load_checkpoint(model, path, return_meta=False):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        meta = checkpoint.get("meta", {})
    else:
        state_dict = checkpoint
        meta = {}

    # 过滤 shape mismatch（不同数据集 nc 变化时 Detect head） / Filter shape mismatch (Detect head when nc changes across datasets)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    if len(filtered) < len(state_dict):
        skipped = [k for k, v in state_dict.items()
                   if k in model_sd and model_sd[k].shape != v.shape]
        print(f"[load_checkpoint] 跳过 {len(skipped)} 个 shape mismatch 参数")
    if len(filtered) < len(model_sd):
        missing = set(model_sd.keys()) - set(filtered.keys())
        # 只打印非 QuantCat 的 missing（QuantCat init_state=0 是正常的） / Only print non-QuantCat missing (QuantCat init_state=0 is normal)
        meaningful = [k for k in missing if 'QuantCat' not in k or 'init_state' not in k]
        if meaningful:
            print(f"[load_checkpoint] missing {len(meaningful)} 参数 (可能是 QuantCat init)")

    model.load_state_dict(filtered, strict=False)
    return (model, meta) if return_meta else model


def save_checkpoint(model, path, **metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if metadata:
        torch.save({"state_dict": model.state_dict(), "meta": metadata}, path)
    else:
        torch.save(model.state_dict(), path)


def resolve_pretrained_path(path):
    """解析预训练权重路径：官方权重名缺失时像 ultralytics 一样自动下载 /
    Resolve the pretrained-weight path: when an official asset name is missing locally, auto-download it like ultralytics.

    - 路径已存在：原样返回（用户自定义文件或本地官方权重均可） / Existing path (user file or local official weight): returned as-is.
    - 文件名属于 ultralytics 官方发布资产（如 yolo26n.pt / yolo26s-seg.pt）：调用
      ``ultralytics.utils.downloads.attempt_download_asset`` 从 GitHub Releases 下载到该路径 /
      Basename is an official ultralytics release asset (e.g. yolo26n.pt / yolo26s-seg.pt): downloaded from
      GitHub Releases into that path via ``attempt_download_asset``.
    - 其余自定义路径缺失或下载失败：返回 None，由调用方按“从头训练”处理 /
      Any other missing custom path, or download failure: returns None; caller falls back to random init.

    参考 / Reference: https://github.com/ultralytics/ultralytics/blob/main/README.md
    """
    if path and os.path.exists(path):
        return path
    fname = os.path.basename(path) if path else ""
    try:
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES, attempt_download_asset
        if fname in GITHUB_ASSETS_NAMES:
            print(f"[Pretrain] 本地未找到官方权重 {fname}，按 ultralytics 方式从 GitHub Releases 自动下载 ...")
            downloaded = attempt_download_asset(path)  # 传入绝对路径时会直接下载到该路径 / absolute path downloads in place
            if downloaded and os.path.exists(downloaded):
                print(f"[Pretrain] 自动下载完成: {downloaded}")
                return downloaded
            print(f"[Pretrain] [warn] 自动下载未产出文件: {fname}")
    except Exception as e:  # 离线 / 网络受限环境下优雅退回从头训练 / Graceful fallback to random init when offline
        print(f"[Pretrain] [warn] 自动下载 {fname} 失败（{e}）")
    return None


def load_pretrained(model, path=None, scale=DEFAULT_SCALE):
    """加载官方 yolo26{scale}.pt 权重；自动跳过 one2one_* 双头权重和 shape mismatch（nc 变化） /
    Load official yolo26{scale}.pt weights; auto-skip one2one_* dual-head weights and shape mismatch (nc changes).

    path 为官方权重名（如 yolo26n.pt / yolo26s.pt）且本地缺失时，会像 ultralytics 一样自动从 GitHub Releases 下载；
    path 为用户自定义路径时直接加载该文件，缺失则从头训练 /
    When path is an official asset name (e.g. yolo26n.pt / yolo26s.pt) missing locally, it is auto-downloaded from
    GitHub Releases like ultralytics; a user-defined path is loaded directly, and training starts from scratch if it is missing.
    """
    if path is None:
        path = os.path.join(ULTRA_DIR, f"{base_name(scale)}.pt")
    path = resolve_pretrained_path(path)
    if path is None:
        print(f"[Pretrain] [warn] 未找到预训练权重，{model_name(scale)} 将从头训练")
        return model
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].float().state_dict()

    # 过滤 shape mismatch 的 key（COCO 80类 → VOC 20类 时 Detect head 分类层） /
    # Filter shape mismatch keys (Detect head class layer when COCO 80 classes → VOC 20 classes)
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
        print(f"  ⚠️  跳过 shape mismatch: {len(skipped_shape)} 参数 (Detect head nc 变化)")
    if missing_unexpected:
        print(f"  ⚠️  missing/unexpected: {len(missing_unexpected)} 参数")
    return model


def copy_float_to_quant(float_model, quant_model):
    float_state = float_model.state_dict()
    quant_state = quant_model.state_dict()
    for key, value in float_state.items():
        if key in quant_state and quant_state[key].shape == value.shape:
            quant_state[key] = value.detach().clone().to(device=quant_state[key].device)
    quant_model.load_state_dict(quant_state)
    # 灌入真实权重后，让所有量化器从第 0 批重新统计 scale/beta（构建模型时的 /
    # After injecting real weights, let all quantizers re-collect scale/beta from batch 0 (dummy zero-input
    # dummy 零输入前向可能已经把 init_state 推到 1）：各后端的量化器在 /
    # forward during model construction may have already pushed init_state to 1): each backend's quantizer at
    # init_state==0 的首次前向都会用自己的原生公式按真实权重/激活重新初始化 /
    # first forward with init_state==0 re-initializes using its native formula on real weights/activations.
    quant_pkg.reset_quantizer_states(quant_model)
    return quant_model


def build_float_model(quant_model, nc=NUM_CLASSES, scale=DEFAULT_SCALE, use_clip=False):
    """构建纯浮点参考模型（用于导出纯浮点 ONNX）/
    Build the pure-float reference model (for pure-float ONNX export).

    直接委托 quantization.build_float_model：在量化模型深拷贝上烘焙反量化权重，
    激活量化器按 use_clip 替换——False（默认）换恒等（彻底去伪量化，交付给板端 PTQ
    工具自校准）；True 换仅截断算子（保留 clip、去掉 round，参考图与真实整型推理
    数值行为一致）。nc/scale 参数仅为兼容旧调用签名，不再使用。
    / Delegates to quantization.build_float_model: bake dequantized weights on a deep
    copy; activation quantizers are replaced per use_clip — False (default) identity
    (fully de-fake-quantized, for board-side PTQ calibration); True clip-only ops
    (keep clip, drop rounding; reference graph matches real integer inference).
    nc/scale are kept only for signature compatibility and are unused.
    """
    return quant_pkg.build_float_model(quant_model, use_clip=use_clip)


def collect_quant_params(quant_model):
    # 各后端 scale/zero_point 提取差异由 quantization 包统一处理 / Backend-specific scale/zero_point extraction unified by quantization package
    return quant_pkg.collect_quant_params(quant_model)


def export_onnx(float_model, onnx_path, opset=16, imgsz=None):
    """导出 ONNX，输入形状完全固定为 [1, 3, H, W]（无 dynamic_axes，图尺寸清晰可见） /
    Export ONNX with fully static input shape [1, 3, H, W] (no dynamic_axes, graph dimensions clearly visible).

    Args / 参数:
        float_model: 浮点模型（或已反量化的"干净"模型） / Float model (or dequantized "clean" model)
        onnx_path: 输出 .onnx 路径 / Output .onnx path
        opset: ONNX opset，默认 16 / ONNX opset, default 16
        imgsz: 输入图像尺寸 H=W；默认模块级 IMGSZ(640) / Input image size H=W; default module-level IMGSZ (640)
    """
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
        # 无 dynamic_axes → 输入形状完全固定 [1, 3, H, W] / No dynamic_axes → input shape fully fixed [1, 3, H, W]
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
    """训练保存 best checkpoint 时同步导出 ONNX；失败仅告警，绝不影响训练 /
    Export ONNX synchronously when saving best checkpoint during training; failure only warns, never affects training.

    导出后模型被置为 eval，由下一轮 train_one_epoch 的 model.train() 恢复 /
    Model is set to eval after export; next train_one_epoch model.train() restores it.
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
        print(f"      ONNX checker 通过，算子数: {len(op_types)}（含 Conv/Concat/Reshape/Sigmoid 等）")
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


def save_quant_outputs(quant_model, prefix, nc=NUM_CLASSES, meta=None, scale=DEFAULT_SCALE,
                      model_dir=None):
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    if model_dir is None:
        model_dir = _MODEL_DIR_BASE
    quant_checkpoint = os.path.join(model_dir, f"{prefix}_{base_name(scale)}.pth")
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
    print(f"[2/5] 量化参数已写出: {json_path} / {pth_path}（{len(quant_params)} 个张量）")

    float_model = build_float_model(quant_model, nc=nc, scale=scale)
    float_checkpoint = os.path.join(model_dir, f"{prefix}_{base_name(scale)}_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(model_dir, f"{prefix}_{base_name(scale)}_float.onnx")
    export_onnx(float_model, onnx_path, opset=16)
    print(f"[4/5] 干净浮点 ONNX 已写出: {onnx_path}")

    verify(float_model, quant_model, onnx_path, quant_params)
    print("[5/5] 完成。")
    return quant_checkpoint, json_path, pth_path, float_checkpoint, onnx_path


# ============================== 训练各阶段 / Training Stages ==============================


class _ModelEMA:
    """官方 ModelEMA 精简版：对训练权重做指数滑动平均，验证与保存均使用 EMA 权重 /
    Lightweight official ModelEMA: exponential moving average on training weights,
    EMA weights used for validation and saving.

    decay = 0.9999 * (1 - exp(-updates / 2000))，更新次数少时近似直接跟随模型 /
    decay = 0.9999 * (1 - exp(-updates / 2000)); few updates approximate direct model copy.
    """

    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.updates = 0
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        model_sd = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value *= d
                value += (1 - d) * model_sd[key].detach()
            else:
                value.copy_(model_sd[key])


def _auto_lr(nc=NUM_CLASSES):
    """官方 optimizer=auto 的 AdamW 学习率，随类别数自适应（nc=80 → 0.000119） /
    Official optimizer=auto AdamW learning rate, adaptive to class count (nc=80 → 0.000119)."""
    return round(0.002 * 5 / (4 + nc), 6)


def _build_optimizer(model, lr=None, decay=5e-4):
    """官方 optimizer=auto 配方：AdamW(betas=(0.9, 0.999))，三参数组分组 /
    Official optimizer=auto recipe: AdamW(betas=(0.9, 0.999)), three param groups.

    weight 组做 weight decay；BN 权重与 bias 不做 decay（量化器 scale s / 偏移 /
    beta 也归入无衰减组）；lr=None 时按类别数自适应 / weight group has weight decay;
    BN weights and biases have no decay (quantizer scale s / offset beta also in no-decay group);
    lr=None → adaptive by class count.
    """
    if lr is None:
        lr = _auto_lr()
    norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k and isinstance(v, type))
    g_wd, g_bn, g_bias = [], [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if "bias" in name:
                g_bias.append(param)
            elif isinstance(module, norm_types) or name in {"s", "beta"}:
                g_bn.append(param)
            else:
                g_wd.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": g_wd, "weight_decay": decay, "param_group": "weight"},
            {"params": g_bn, "weight_decay": 0.0, "param_group": "bn"},
            {"params": g_bias, "weight_decay": 0.0, "param_group": "bias"},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def train_one_epoch(model, loader, criterion, optimizer, epoch, epochs, nb, ema=None,
                    nbs=64, warmup_epochs=3.0, lrf=0.01, max_batches=None):
    """官方 BaseTrainer 训练循环复刻：线性 lr 衰减 + warmup（lr 与梯度累积同步插值） /
    Official BaseTrainer training loop replica: linear lr decay + warmup (lr interpolated
    synchronously with gradient accumulation)
    + 梯度累积到 nbs + 梯度裁剪(10.0) + EMA 更新 /
    + gradient accumulation to nbs + gradient clipping (10.0) + EMA update.

    criterion 返回已乘 batch_size 的损失向量（官方直接 backward，不除以 batch） /
    criterion returns loss vector already multiplied by batch_size (official backward directly, no batch divide)
    与未缩放的 items dict（box_loss / cls_loss / l1_loss） / and unscaled items dict (box_loss / cls_loss / l1_loss).
    """
    model.train()
    batch_size = loader.batch_size
    accumulate = max(round(nbs / batch_size), 1)
    warmup_steps = round(min(warmup_epochs, max(epochs - 1, 0)) * nb) if warmup_epochs > 0 else 0
    # 线性衰减（官方默认 cos_lr=False）：lf(0)=1 → lf(epochs)=lrf / Linear decay (official default cos_lr=False): lf(0)=1 → lf(epochs)=lrf
    lf = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf

    # 官方每个 epoch 开始时 scheduler.step()：lr = initial_lr * lf(epoch) / Official scheduler.step() at epoch start: lr = initial_lr * lf(epoch)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lf(epoch)

    total_steps = nb if max_batches is None else min(nb, max_batches)
    loss_items_sum = torch.zeros(3)
    last_opt_step = 0
    for i, batch in enumerate(loader):
        if i >= total_steps:
            break
        ni = i + nb * epoch  # 自训练开始的累计 batch 数 / cumulative batch count since training start
        if ni < warmup_steps:
            xi = [0, warmup_steps]
            accumulate = max(1, int(np.interp(ni, xi, [1, nbs / batch_size]).round()))
            for group in optimizer.param_groups:
                # optimizer=auto 时 warmup_bias_lr=0.0：所有组 lr 从 0 爬升 / optimizer=auto warmup_bias_lr=0.0: all group lr ramps from 0
                group["lr"] = float(
                    np.interp(ni, xi, [0.0, group["initial_lr"] * lf(epoch)])
                )

        images = batch["img"].float().to(device) / 255.0
        preds = model(images)
        loss_vec, loss_items = criterion.loss(preds, batch)
        loss_vec.sum().backward()

        if ni - last_opt_step >= accumulate:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            last_opt_step = ni

        loss_items_sum += torch.stack(
            [
                loss_items["box_loss"].detach().cpu(),
                loss_items["cls_loss"].detach().cpu(),
                loss_items["l1_loss"].detach().cpu(),
            ]
        )

    avg_items = (loss_items_sum / total_steps).tolist()
    return float(sum(avg_items)), avg_items


def float_train(batch_size=16, lr=None, epochs=100, num_classes=NUM_CLASSES, num_workers=2,
                max_train_batches=None, max_eval_batches=None, scale=DEFAULT_SCALE,
                resume=False):
    print(f"========== Float training ({model_name(scale)} / coco128) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr()
    print(
        f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4（官方 optimizer=auto 配方）"
        " | warmup 3ep | 梯度累积 nbs=64 | EMA | close_mosaic=10"
    )

    data = get_data_dict(DATA_YAML, "detect")
    train_loader, val_loader, cfg, train_set = _build_loaders(batch_size, num_workers, data)
    nb = len(train_loader)

    float_model = FloatYOLO26(num_classes, scale=scale).to(device)
    model_info(float_model, imgsz=IMGSZ)
    load_pretrained(float_model, scale=scale)

    criterion = build_criterion(float_model, epochs)
    optimizer = _build_optimizer(float_model, lr=lr)
    ema = _ModelEMA(float_model)

    close_mosaic = 10
    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    # best checkpoint 加 _best 后缀，与 PTQ/QAT 命名一致 / best checkpoint uses _best suffix, consistent with PTQ/QAT naming
    model_dir = _model_dir_for(scale=scale)
    checkpoint_path = os.path.join(model_dir, f"{base_name(scale)}_best.pth")
    last_checkpoint = os.path.join(model_dir, f"{base_name(scale)}_last.pth")

    # resume：从 _last.pth 恢复，接续训练 / resume: restore from _last.pth, continue training
    start_epoch = 0
    if resume and os.path.exists(last_checkpoint):
        print(f"[Float] Resume from {last_checkpoint}")
        _, meta = load_checkpoint(float_model, last_checkpoint, return_meta=True)
        ema.ema.load_state_dict(float_model.state_dict())
        start_epoch = meta.get("epoch", 0)
        best_fitness = meta.get("fitness", -1.0)
        best_meta = meta
        best_epoch = start_epoch
        print(f"[Float] 从 epoch {start_epoch} 接续训练，当前 best mAP50={meta.get('map50', 0):.4f}")
    elif resume:
        print(f"[Float] Resume 开启但找不到 {last_checkpoint}，从头训练")

    for epoch in range(start_epoch, epochs):
        if epoch == epochs - close_mosaic:
            print("[Float] 关闭 dataloader mosaic（最后 10 个 epoch）")
            train_set.close_mosaic(copy.copy(cfg))
            train_loader.reset()

        train_loss, (box_loss, cls_loss, l1_loss) = train_one_epoch(
            float_model, train_loader, criterion, optimizer, epoch, epochs, nb, ema=ema,
            max_batches=max_train_batches,
        )
        metrics = evaluate(
            ema.ema, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
            viz_dir=model_dir, viz_prefix=f"float_ep{epoch + 1:03d}"
        )
        fitness = 0.9 * metrics["map"] + 0.1 * metrics["map50"]

        epoch_meta = {
            "stage": "float",
            "scale": get_scale(scale),
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "box_loss": float(box_loss),
            "cls_loss": float(cls_loss),
            "l1_loss": float(l1_loss),
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
            f"(box:{box_loss:.3f} cls:{cls_loss:.3f} l1:{l1_loss:.3f}) | "
            f"mAP50:{metrics['map50']:.4f} mAP50-95:{metrics['map']:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)  # 载入最优 EMA 权重，供后续 PTQ 使用 / Load best EMA weights for subsequent PTQ
    best_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=model_dir, viz_prefix="float_best"
    )
    print(f"[Float] Best checkpoint: {checkpoint_path}（epoch {best_epoch}/{epochs}）")
    print(f"[Float] Best mAP50:{best_metrics['map50']:.4f} mAP50-95:{best_metrics['map']:.4f}")
    return checkpoint_path, best_fitness, best_meta


def _collect_float_ranges_full(float_model, calibration_loader, calibration_batches,
                               normalize_img=True):
    """收集 float 模型中每个 Conv/Linear + eltwise 算子的运行 min/max 范围 /
    Collect runtime min/max ranges for each Conv/Linear + eltwise op in the float model.

    Hook 策略（全部是 named module 的 forward_pre_hook，不 monkeypatch torch.cat/add 等） /
    Hook strategy (all named module forward_hooks, no monkeypatching torch.cat/add etc.):
      - Conv2d/ConvTranspose2d/Linear: 输入 (单 tensor) / input (single tensor)
      - FloatAdd: 两个输入 (A, C) — 同名 QuantAdd / two inputs (A, C) — same-name QuantAdd
      - MaxPool2d: 输入 (单 tensor) — 同名 QuantMaxPool / input (single tensor) — same-name QuantMaxPool
      - Concat: 输入列表 xs = [tensor, ...] — QuantConcat 在 {name}.op / input list xs = [tensor, ...] — QuantConcat at {name}.op
      - QuantCat: float 里是 torch.cat 函数调用 (C2f/C3k/SPPF/...)，无同名模块， /
        QuantCat: torch.cat function call in float (C2f/C3k/SPPF/...), no same-name module,
        靠安全网处理 / handled by safety net

    Returns:
        module_input_ranges: dict[name] = list of [min, max]
          - Conv/Linear: 1 个输入 → [[min, max]] / 1 input
          - FloatAdd: 2 个输入 → [[A_min, A_max], [C_min, C_max]] / 2 inputs
          - Concat: N 个输入 → [[xs[0]_min, xs[0]_max], ...] / N inputs
          - MaxPool2d: 1 个输入 → [[min, max]] / 1 input
    """
    module_input_ranges = {}

    def make_single_input_hook(name):
        def hook(module, inputs, output):
            if isinstance(inputs, tuple) and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                x = inputs[0].detach()
                cur_min, cur_max = x.min(), x.max()
                if name not in module_input_ranges:
                    module_input_ranges[name] = [[cur_min, cur_max]]
                else:
                    slot = module_input_ranges[name][0]
                    slot[0] = torch.minimum(slot[0], cur_min)
                    slot[1] = torch.maximum(slot[1], cur_max)
        return hook

    def make_multi_input_hook(name):
        """展开 tuple/list 中的所有 tensor 输入 / Flatten all tensor inputs in tuple/list."""
        def hook(module, inputs, output):
            tensors = []
            for a in inputs:
                if isinstance(a, torch.Tensor):
                    tensors.append(a)
                elif isinstance(a, (list, tuple)):
                    tensors.extend(t for t in a if isinstance(t, torch.Tensor))
            if not tensors:
                return
            if name not in module_input_ranges:
                module_input_ranges[name] = [[t.detach().min(), t.detach().max()] for t in tensors]
            else:
                slot = module_input_ranges[name]
                for i, t in enumerate(tensors):
                    if i < len(slot):
                        slot[i][0] = torch.minimum(slot[i][0], t.detach().min())
                        slot[i][1] = torch.maximum(slot[i][1], t.detach().max())
        return hook

    hooks = []
    float_model.eval()
    for name, module in float_model.named_modules():
        cls = type(module).__name__
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_single_input_hook(name)))
        elif cls == 'FloatAdd':
            hooks.append(module.register_forward_hook(make_multi_input_hook(name)))
        elif cls == 'MaxPool2d':
            hooks.append(module.register_forward_hook(make_single_input_hook(name)))
        elif cls == 'Concat':
            hooks.append(module.register_forward_hook(make_multi_input_hook(name)))
        # 激活函数：单输入 / Activation functions: single input
        elif isinstance(module, (nn.SiLU, nn.Sigmoid, nn.ReLU)):
            hooks.append(module.register_forward_hook(make_single_input_hook(name)))

    with torch.no_grad():
        for batch_index, batch in enumerate(calibration_loader):
            img = batch["img"].float().to(device)
            if normalize_img:
                img = img / 255.0
            float_model(img)
            if batch_index + 1 >= calibration_batches:
                break

    for h in hooks:
        h.remove()

    conv_count = sum(1 for k in module_input_ranges
                     if any(k.endswith(s) for s in ('.conv', '.cv')))
    eltwise_count = len(module_input_ranges) - conv_count
    print(f"[Calib] Collected float ranges: {len(module_input_ranges)} modules "
          f"({conv_count} Conv/Linear, {eltwise_count} eltwise)")
    return module_input_ranges


def _set_quantizer_frozen(q):
    """把量化器标记为已初始化/冻结（兼容 int 与 Tensor 两种 init_state） /
    Mark quantizer as initialized/frozen (supports both int and Tensor init_state).

    与 quantization.freeze_batch_init 的单量化器版本语义一致：只改标志位， /
    Same semantics as quantization.freeze_batch_init single-quantizer version:
    不动已写入的 scale/beta/alpha / only changes flag bits, does not touch written scale/beta/alpha.
    """
    init_state = getattr(q, "init_state", None)
    if isinstance(init_state, torch.Tensor):
        init_state.fill_(quant_pkg.INIT_STATE_FROZEN)
    elif isinstance(init_state, int):
        q.init_state = quant_pkg.INIT_STATE_FROZEN
    init_flag = getattr(q, "init", None)
    if type(init_flag) is int:
        q.init = 1


def _apply_minmax_to_quantizer(q, cur_min, cur_max, eps=1e-8):
    """通用：用 min-max 范围初始化一个量化器，支持全部 7 个后端 /
    Generic: initialize a quantizer with min-max range, supports all 7 backends.

    cur_min / cur_max 允许是 Tensor 或 python float，内部统一转成 float32 Tensor /
    cur_min / cur_max can be Tensor or python float, internally converted to float32 Tensor.
    各后端的 scale 语义不同，按各自原生公式赋值 / Each backend has different scale semantics, assigned by its native formula:
      - minmax:  写 r_min/r_max 并按其 forward 公式重算 scale/zero_point / write r_min/r_max and recompute scale/zero_point per its forward formula;
      - pact:    alpha = 绝对值最大（与其原生自初始化一致） / alpha = max abs value (consistent with its native self-init);
      - dorefa:  新版与 lsqplus 同公式（s + beta 非对称），走 beta 分支 / new dorefa shares lsqplus formula (asymmetric s + beta), handled by the beta branch;
      - lsqplus: s = (max-min)/(Qp-Qn)，beta = min - s*Qn（非对称精确覆盖） / s = (max-min)/(Qp-Qn), beta = min - s*Qn (asymmetric exact coverage);
      - lsq:     对称网格，取能覆盖 [min, max] 的最小 scale / symmetric grid, take smallest scale covering [min, max].
    """
    cur_min = torch.as_tensor(cur_min, dtype=torch.float32)
    cur_max = torch.as_tensor(cur_max, dtype=torch.float32)
    if float(cur_min) > float(cur_max):
        cur_min, cur_max = cur_max, cur_min

    # minmax 后端：r_min/r_max buffer + 原生 scale/zero_point 公式 / minmax backend: r_min/r_max buffer + native scale/zero_point formula
    if hasattr(q, "r_min") and hasattr(q, "scale"):
        if getattr(q, "all_positive", False):
            qmin, qmax = 0, q.q_range
            r_min = cur_min.clamp(min=0)
        else:
            qmin, qmax = -(2 ** (q.a_bits - 1)), 2 ** (q.a_bits - 1) - 1
            r_min = cur_min
        r_max = cur_max
        feps = torch.finfo(torch.float32).eps
        scale = (torch.clamp(r_max - r_min, min=feps) / (qmax - qmin)).reshape(1)
        zero_point = (qmin - r_min / scale).round().clamp(qmin, qmax).reshape(1)
        q.r_min.copy_(r_min.reshape(1).to(q.r_min.device))
        q.r_max.copy_(r_max.reshape(1).to(q.r_max.device))
        q.scale.copy_(scale.to(q.scale.device))
        q.zero_point.copy_(zero_point.to(q.zero_point.device))
        q.init = 1
        return True

    # PACT：对称截断阈值 alpha（原生自初始化用的就是绝对值最大） / PACT: symmetric clipping threshold alpha (native self-init uses max abs value)
    if hasattr(q, "alpha"):
        cur_max_abs = torch.maximum(cur_min.abs(), cur_max.abs()).clamp(min=eps)
        q.alpha.data.copy_(cur_max_abs.reshape(q.alpha.shape).to(q.alpha.device))
        _set_quantizer_frozen(q)
        return True

    Qn = getattr(q, "Qn", None)
    Qp = getattr(q, "Qp", None)
    if hasattr(q, "s") and Qn is not None:
        if hasattr(q, "_set_init_state") and not hasattr(q, "beta"):
            # 旧版 dorefa（tanh 域归一化）已废弃；保留此分支仅为兼容旧 checkpoint /
            # Legacy dorefa (tanh-domain normalization) is deprecated; kept only for old checkpoints
            if getattr(q, "all_positive", False):
                cur_s = cur_max.clamp(min=1e-6)
            else:
                cur_s = torch.maximum(cur_min.abs(), cur_max.abs()).clamp(min=1e-6)
            q.s.data.copy_(cur_s.reshape(q.s.shape).to(q.s.device))
        elif hasattr(q, "beta"):
            # lsqplus / 新版 dorefa：非对称 scale + beta 精确覆盖 [min, max] /
            # lsqplus / new dorefa: asymmetric scale + beta exact coverage of [min, max]
            cur_s = torch.clamp(cur_max - cur_min, min=eps) / (Qp - Qn)
            q.s.data.copy_(cur_s.reshape(q.s.shape).to(q.s.device))
            cur_beta = cur_min - cur_s * Qn
            q.beta.data.copy_(cur_beta.reshape(q.beta.shape).to(q.beta.device))
        else:
            # lsq：对称网格 [-Qn*s, Qp*s]，取能覆盖 [min, max] 的最小 scale / lsq: symmetric grid [-Qn*s, Qp*s], take smallest scale covering [min, max]
            if Qn < 0:
                cur_s = torch.maximum(cur_max / Qp, cur_min / Qn).clamp(min=eps)
            else:
                cur_s = (cur_max / Qp).clamp(min=eps)
            q.s.data.copy_(cur_s.reshape(q.s.shape).to(q.s.device))
        _set_quantizer_frozen(q)
        return True

    return False


def _init_quantizers_from_float(float_model, quant_model, calibration_loader,
                                calibration_batches=20, normalize_img=True):
    """用 float 模型的激活范围 + 权重范围初始化 quant 模型的量化器 /
    Initialize quant model quantizers using float model's activation + weight ranges.

    覆盖三类量化点（全部直接从 float 模型收集，无级联误差） / Covers three types of quantization points
    (all collected directly from float model, no cascading error):
      1. Conv/Linear 的 activation_quantizer —— float forward 输入范围 / Conv/Linear activation_quantizer — float forward input range
      2. Conv/Linear 的 weight_quantizer —— 当前权重的 min-max（避免 LSQ+ 3σ 饱和） / Conv/Linear weight_quantizer — current weight min-max (avoid LSQ+ 3σ saturation)
      3. FloatAdd/MaxPool2d/Concat 的 eltwise 量化器：
           - FloatAdd(name) → QuantAdd(name).activation_quantizer0/1
           - MaxPool2d(name) → QuantMaxPool(name).activation_quantizer
           - Concat(name) → QuantConcat(name.op).activation_quantizer0/1

    QuantCat（torch.cat 函数调用）无同名 float 模块，交给安全网处理 /
    QuantCat (torch.cat function call) has no same-name float module, handled by safety net.
    """
    module_input_ranges = _collect_float_ranges_full(
        float_model, calibration_loader, calibration_batches, normalize_img=normalize_img
    )

    eps = 1e-8
    aq_count = 0
    wq_count = 0
    eltwise_count = 0
    quant_model_modules = dict(quant_model.named_modules())

    for name, float_range_list in module_input_ranges.items():
        # float_range_list: [[min0, max0], [min1, max1], ...]
        # 统一处理成 list of tensors / Normalize to list of tensors
        ranges = [[torch.as_tensor(m, dtype=torch.float32),
                   torch.as_tensor(x, dtype=torch.float32)]
                  for m, x in float_range_list]

        if name not in quant_model_modules:
            # 可能是 Concat → QuantConcat 在 {name}.op / Might be Concat → QuantConcat at {name}.op
            qname = name + '.op'
            if qname not in quant_model_modules:
                continue
            qmodule = quant_model_modules[qname]
        else:
            qmodule = quant_model_modules[name]

        cls_name = type(qmodule).__name__

        # ── Conv/Linear: activation_quantizer + weight_quantizer ──
        if cls_name in ('QuantConv2d', 'QuantConvTranspose2d', 'QuantLinear'):
            aq = getattr(qmodule, "activation_quantizer", None)
            if aq is not None and _apply_minmax_to_quantizer(aq, ranges[0][0], ranges[0][1], eps):
                aq_count += 1

            wq = getattr(qmodule, "weight_quantizer", None)
            if wq is not None and hasattr(wq, "s"):
                weight = qmodule.weight.detach()
                Qn, Qp = wq.Qn, wq.Qp
                if getattr(wq, "per_channel", False) and weight.dim() >= 2:
                    w_flat = weight.contiguous().view(weight.size(0), -1)
                    w_min = w_flat.min(dim=1)[0]
                    w_max = w_flat.max(dim=1)[0]
                else:
                    w_min, w_max = weight.min(), weight.max()
                if hasattr(wq, "beta"):
                    cur_s = torch.clamp(w_max - w_min, min=eps) / (Qp - Qn)
                else:
                    if Qn < 0:
                        cur_s = torch.maximum(w_max / Qp, w_min / Qn).clamp(min=eps)
                    else:
                        cur_s = (w_max / Qp).clamp(min=eps)
                wq.s.data.copy_(cur_s.reshape(wq.s.shape).to(wq.s.device))
                if hasattr(wq, "beta"):
                    cur_beta = w_min - cur_s * Qn
                    wq.beta.data.copy_(cur_beta.reshape(wq.beta.shape).to(wq.beta.device))
                _set_quantizer_frozen(wq)
                wq_count += 1
            continue

        # ── QuantAdd: activation_quantizer0 (A), activation_quantizer1 (C) ──
        if cls_name == 'QuantAdd':
            for i, attr in enumerate(('activation_quantizer0', 'activation_quantizer1')):
                if i < len(ranges):
                    q = getattr(qmodule, attr, None)
                    if q is not None and _apply_minmax_to_quantizer(q, ranges[i][0], ranges[i][1], eps):
                        eltwise_count += 1
            continue

        # ── QuantMaxPool: activation_quantizer (单输入 / single input) ──
        if cls_name == 'QuantMaxPool':
            aq = getattr(qmodule, "activation_quantizer", None)
            if aq is not None and len(ranges) >= 1:
                if _apply_minmax_to_quantizer(aq, ranges[0][0], ranges[0][1], eps):
                    eltwise_count += 1
            continue

        # ── QuantConcat: activation_quantizer0/1 ──
        if cls_name == 'QuantConcat':
            for i, attr in enumerate(('activation_quantizer0', 'activation_quantizer1')):
                if i < len(ranges):
                    q = getattr(qmodule, attr, None)
                    if q is not None and _apply_minmax_to_quantizer(q, ranges[i][0], ranges[i][1], eps):
                        eltwise_count += 1
            continue

        # ── 其他量化模块：通用 activation_quantizer / Other quant modules: generic activation_quantizer ──
        aq = getattr(qmodule, "activation_quantizer", None)
        if aq is not None and len(ranges) >= 1:
            if _apply_minmax_to_quantizer(aq, ranges[0][0], ranges[0][1], eps):
                aq_count += 1

    print(f"[Calib] Initialized {aq_count} activation + {wq_count} weight "
          f"+ {eltwise_count} eltwise quantizers from float ranges")
    return aq_count, wq_count


def _safety_net_calibrate(quant_model, calibration_loader, calibration_batches=20,
                          normalize_img=True):
    """安全网校准：在真实前向中完成剩余量化点的初始化 /
    Safety net calibration: initialize remaining quantization points during real forward passes.

    Conv/Linear 已由 _init_quantizers_from_float 按浮点范围初始化并冻结 /
    Conv/Linear already initialized and frozen from float ranges by _init_quantizers_from_float;
    本函数处理其余量化算子（QuantAdd/QuantCat/QuantConcat/QuantMaxPool 等）： /
    this function handles remaining quant ops (QuantAdd/QuantCat/QuantConcat/QuantMaxPool etc.):

      1. 给它们挂 forward_pre_hook，记录各输入张量跨 batch 的运行 min/max /
         Attach forward_pre_hook to record per-input-tensor runtime min/max across batches;
      2. 跑若干批真实前向 —— 各后端的原生自初始化机制正常触发 /
         Run several batches of real forward — each backend's native self-init triggers normally
         （LSQ+/LSQ 的 EMA、minmax 的运行统计、PACT 的首批 absmax、 /
         (LSQ+/LSQ EMA, minmax runtime stats, PACT first-batch absmax,
         dorefa 的首批+EMA），不存在"垃圾 scale 被冻结"的路径； /
         dorefa first-batch + EMA); no path where "garbage scale gets frozen";
      3. 结束后按记录的运行 min/max 统一覆盖赋值 —— 修正 PACT 只看首批、 /
         After completion, uniformly overwrite per recorded runtime min/max — fixes PACT first-batch-only,
         LSQ 家族 EMA 滞后的问题（minmax 保留其原生 percentile 统计，不覆盖）。 /
         LSQ family EMA lag issues (minmax keeps its native percentile stats, not overwritten).
    """
    records = {}
    hooks = []

    def make_pre_hook(name):
        def pre_hook(module, args):
            # 展开输入张量：QuantCat 是 (tensor_list, dim)，Add/Concat 是 /
            # Flatten input tensors: QuantCat is (tensor_list, dim), Add/Concat is
            # (A, C[, dim])，MaxPool 是 (x,)，统一抽成 tensor 列表 /
            # (A, C[, dim]), MaxPool is (x,), uniformly extract into tensor list
            tensors = []
            for a in args:
                if isinstance(a, torch.Tensor):
                    tensors.append(a)
                elif isinstance(a, (list, tuple)):
                    tensors.extend(t for t in a if isinstance(t, torch.Tensor))
            if not tensors:
                return
            slot = records.get(name)
            if slot is None:
                records[name] = [[t.detach().min(), t.detach().max()] for t in tensors]
            else:
                for i, t in enumerate(tensors):
                    if i < len(slot):
                        slot[i][0] = torch.minimum(slot[i][0], t.detach().min())
                        slot[i][1] = torch.maximum(slot[i][1], t.detach().max())
        return pre_hook

    for name, module in quant_model.named_modules():
        if quant_pkg.is_weight_quant_module(module):
            continue  # Conv/Linear：已按 float 范围初始化 / Conv/Linear: already initialized from float ranges
        if hasattr(module, "quantizers") and isinstance(module.quantizers, nn.ModuleList):
            hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))
        elif hasattr(module, "activation_quantizer0") and hasattr(module, "activation_quantizer1"):
            hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))
        elif hasattr(module, "activation_quantizer"):
            hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))

    with torch.no_grad():
        for batch_index, batch in enumerate(calibration_loader):
            img = batch["img"].float().to(device)
            if normalize_img:
                img = img / 255.0
            quant_model(img)
            if batch_index + 1 >= calibration_batches:
                break

    for h in hooks:
        h.remove()

    # 按记录的运行 min/max 统一赋值 —— 只覆盖未初始化的 / Uniformly assign per recorded runtime min/max —— only overwrite uninitialized ones
    quant_modules = dict(quant_model.named_modules())
    assigned = 0
    skipped = 0

    def _is_already_initialized(q):
        """LSQ 家族 init_state=FROZEN; minmax/pact init=1 (已初始化标记) /
        LSQ family init_state=FROZEN; minmax/pact init=1 (initialized flag)."""
        ist = getattr(q, 'init_state', None)
        if isinstance(ist, torch.Tensor) and int(ist.flatten()[0]) >= quant_pkg.INIT_STATE_FROZEN:
            return True
        if isinstance(ist, int) and ist >= quant_pkg.INIT_STATE_FROZEN:
            return True
        if hasattr(q, 'init') and type(q.init) is int and q.init == 1:
            return True
        return False

    def _assign_recorded(q, mn, mx):
        # 已由 float 范围初始化 → 跳过（不要覆盖更准确的 float 范围） / Already initialized from float ranges → skip (don't overwrite more accurate float ranges)
        if _is_already_initialized(q):
            return 'skip'
        # minmax 后端已在安全网前向中按原生 percentile 机制自收集，保留其结果 / minmax backend already self-collected via native percentile in safety net forward, keep its result
        if hasattr(q, "r_min"):
            return 'skip'
        return _apply_minmax_to_quantizer(q, mn, mx)

    for name, slot in records.items():
        module = quant_modules.get(name)
        if module is None:
            continue
        qlist = getattr(module, "quantizers", None)
        if qlist is not None and isinstance(qlist, nn.ModuleList):
            for i, q in enumerate(qlist):
                if i >= len(slot): continue
                r = _assign_recorded(q, slot[i][0], slot[i][1])
                if r == 'skip': skipped += 1
                elif r: assigned += 1
            continue
        q0 = getattr(module, "activation_quantizer0", None)
        if q0 is not None:
            if len(slot) >= 1:
                r = _assign_recorded(q0, slot[0][0], slot[0][1])
                if r == 'skip': skipped += 1
                elif r: assigned += 1
            q1 = getattr(module, "activation_quantizer1", None)
            if q1 is not None and len(slot) >= 2:
                r = _assign_recorded(q1, slot[1][0], slot[1][1])
                if r == 'skip': skipped += 1
                elif r: assigned += 1
            continue
        aq = getattr(module, "activation_quantizer", None)
        if aq is not None and len(slot) >= 1:
            r = _assign_recorded(aq, slot[0][0], slot[0][1])
            if r == 'skip': skipped += 1
            elif r: assigned += 1

    print(f"[Calib] Safety net: recorded {len(records)} eltwise/cat/pool ops, "
          f"assigned {assigned} / skipped {skipped} quantizers")


@torch.no_grad()
def calibrate_quantizer(quant_model, calibration_loader, calibration_batches=20,
                        float_model=None, normalize_img=True):
    """校准量化器 / Calibrate quantizers. 如果提供 float_model, 用 float 模型的激活+权重范围独立校准 /
    If float_model provided, use float model's activation + weight ranges for independent calibration,
    避免级联误差和 3σ 饱和问题 / avoiding cascading error and 3σ saturation issues.

    流程（float_model 路径）/ Pipeline (float_model path):
      1. reset: 所有量化器回到未初始化状态 / reset: all quantizers return to uninitialized state ——
         构建模型时 _initialize_head 的 dummy 零输入前向可能已把垃圾 scale 写进量化器，必须先清掉标志位； /
         dummy zero-input forward during _initialize_head may have written garbage scale, must clear flag bits first;
      2. Conv/Linear 的激活/权重量化器用 float 范围初始化并冻结（每层独立，无级联误差；权重用 min-max 精确覆盖，无 3σ 饱和）/
         Conv/Linear activation/weight quantizers init and freeze from float ranges (per-layer independent, no cascading error; weights covered by min-max exactly, no 3σ saturation);
      3. 安全网: 跑若干批真实前向，其余量化算子（QuantAdd/QuantCat/QuantConcat/QuantMaxPool）按各后端原生机制自初始化，再统一按记录的运行 min/max 赋值 —— 不存在"垃圾 scale 被冻结"的路径； /
         Safety net: run several batches real forward, remaining quant ops self-init per backend native mechanism, then uniformly overwrite per recorded runtime min/max — no path where "garbage scale gets frozen";
      4. freeze 全部量化器 / freeze all quantizers.

    无 float_model 时退化为纯 quant 前向自校准（各后端原生机制） / Without float_model, degenerates to pure quant forward self-calibration (backend native mechanism).

    Args:
        normalize_img: 数据 loader 输出是否需要 /255 归一化 / Whether dataloader output needs /255 normalization.
            detect/seg/pose 的数据 loader 输出 [0,255]，需要 True / detect/seg/pose data loader outputs [0,255], needs True.
            cls 的数据 loader 已经归一化到 [0,1]，用 False / cls data loader already normalized to [0,1], use False.
    """
    quant_model.eval()
    quant_pkg.reset_quantizer_states(quant_model)

    if float_model is not None:
        _init_quantizers_from_float(
            float_model, quant_model, calibration_loader, calibration_batches,
            normalize_img=normalize_img,
        )

    # 安全网：覆盖剩余量化点（float_model=None 时这是唯一的校准手段） / Safety net: cover remaining quant points (this is the only calibration when float_model=None)
    _safety_net_calibrate(
        quant_model, calibration_loader, calibration_batches,
        normalize_img=normalize_img,
    )

    quant_pkg.freeze_batch_init(quant_model)


def PTQ_calibration(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                    calibration_batches=20, num_workers=2, max_eval_batches=None,
                    scale=DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== PTQ calibration ({model_name(scale)} / {tag}) ==========")
    print(f"Device: {device}")

    calibration_loader, data = get_dataloaders(batch_size, num_workers, calibration=True)
    _, val_loader, _ = get_dataloaders(batch_size, num_workers)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth（与 QAT 命名一致），fallback 到旧版无后缀 .pth / Prefer _best.pth (consistent with QAT naming), fallback to legacy no-suffix .pth
    float_checkpoint = os.path.join(float_dir, f"{base_name(scale)}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{base_name(scale)}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatYOLO26(num_classes, scale=scale).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = QuantYOLO26(num_classes, scale=scale).to(device)
    model_info(ptq_model, imgsz=IMGSZ)
    copy_float_to_quant(float_model, ptq_model)

    calibrate_quantizer(ptq_model, calibration_loader, calibration_batches,
                         float_model=float_model)

    metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix=f"ptq_{tag}"
    )
    print(f"[PTQ-{tag}] mAP50:{metrics['map50']:.4f} mAP50-95:{metrics['map']:.4f}")

    ptq_meta = {
        "stage": "ptq",
        "scale": get_scale(scale),
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
    print(f"[PTQ] 重载 checkpoint 后 mAP50:{final_metrics['map50']:.4f} mAP50-95:{final_metrics['map']:.4f}")
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=16, lr=None, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None, scale=DEFAULT_SCALE,
                 resume=False):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training ({model_name(scale)} / {tag}) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr() * 0.1  # QAT 微调用 float auto lr 的 1/10 / QAT fine-tuning uses 1/10 of float auto lr
    print(f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4 | warmup 3ep | 梯度累积 nbs=64")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)
    nb = len(train_loader)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    ptq_checkpoint = os.path.join(quant_dir, f"ptq_{tag}_{base_name(scale)}.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = QuantYOLO26(num_classes, scale=scale).to(device)
    model_info(qat_model, imgsz=IMGSZ)
    _, ptq_meta = load_checkpoint(qat_model, ptq_checkpoint, return_meta=True)
    saved_method = ptq_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] PTQ checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    saved_scale = ptq_meta.get("scale")
    if saved_scale is not None and saved_scale != get_scale(scale):
        print(f"      [warn] PTQ checkpoint 记录的尺度为 yolo26{saved_scale}，当前为 yolo26{get_scale(scale)}")
    quant_pkg.freeze_batch_init(qat_model)

    criterion = build_criterion(qat_model, epochs)
    optimizer = _build_optimizer(qat_model, lr=lr)

    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    best_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{base_name(scale)}_best.pth")
    last_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{base_name(scale)}_last.pth")

    # resume：从 _last.pth 恢复，接续训练 / resume: restore from _last.pth, continue training
    start_epoch = 0
    if resume and os.path.exists(last_checkpoint):
        print(f"[QAT-{tag}] Resume from {last_checkpoint}")
        _, meta = load_checkpoint(qat_model, last_checkpoint, return_meta=True)
        start_epoch = meta.get("epoch", 0)
        best_fitness = meta.get("fitness", -1.0)
        best_meta = meta
        best_epoch = start_epoch
        print(f"[QAT-{tag}] 从 epoch {start_epoch} 接续训练，当前 best mAP50={meta.get('map50', 0):.4f}")
    elif resume:
        print(f"[QAT-{tag}] Resume 开启但找不到 {last_checkpoint}，从头训练")

    for epoch in range(start_epoch, epochs):
        train_loss, (box_loss, cls_loss, l1_loss) = train_one_epoch(
            qat_model, train_loader, criterion, optimizer, epoch, epochs, nb,
            max_batches=max_train_batches,
        )
        metrics = evaluate(
            qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
            viz_dir=quant_dir, viz_prefix=f"qat_{tag}_ep{epoch + 1:03d}"
        )
        fitness = 0.9 * metrics["map"] + 0.1 * metrics["map50"]

        epoch_meta = {
            "stage": "qat",
            "scale": get_scale(scale),
            "quant_method": tag,
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "box_loss": float(box_loss),
            "cls_loss": float(cls_loss),
            "l1_loss": float(l1_loss),
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
            f"(box:{box_loss:.3f} cls:{cls_loss:.3f} l1:{l1_loss:.3f}) | "
            f"mAP50:{metrics['map50']:.4f} mAP50-95:{metrics['map']:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    best_metrics = evaluate(
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix=f"qat_{tag}_best"
    )
    print(
        f"[QAT-{tag}] Best epoch:{best_epoch}/{epochs} | "
        f"mAP50:{best_metrics['map50']:.4f} mAP50-95:{best_metrics['map']:.4f} | "
        f"Best: {best_checkpoint}"
    )
    qat_checkpoint = save_quant_outputs(
        qat_model, f"qat_{tag}", nc=num_classes, meta=best_meta, scale=scale,
        model_dir=quant_dir,
    )[0]
    return qat_checkpoint, best_fitness, best_meta


def compare_precision(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                      num_workers=2, max_eval_batches=None, scale=DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== Float vs QAT precision ({model_name(scale)} / coco128 / {tag}) ==========")
    print(f"Device: {device}")

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth，fallback 到无后缀 / Prefer _best.pth, fallback to no-suffix
    float_checkpoint = os.path.join(float_dir, f"{base_name(scale)}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{base_name(scale)}.pth")
    qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{base_name(scale)}_best.pth")
    if not os.path.exists(qat_checkpoint):
        qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{base_name(scale)}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26(num_classes, scale=scale).to(device)
    float_model, float_meta = load_checkpoint(float_model, float_checkpoint, return_meta=True)
    float_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix="compare_float"
    )

    qat_model = QuantYOLO26(num_classes, scale=scale).to(device)
    qat_model, qat_meta = load_checkpoint(qat_model, qat_checkpoint, return_meta=True)
    saved_method = qat_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] QAT checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    saved_scale = qat_meta.get("scale")
    if saved_scale is not None and saved_scale != get_scale(scale):
        print(f"      [warn] QAT checkpoint 记录的尺度为 yolo26{saved_scale}，当前为 yolo26{get_scale(scale)}")
    quant_pkg.freeze_batch_init(qat_model)
    qat_metrics = evaluate(
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix="compare_qat"
    )

    print(
        f"[Compare] Float | best epoch:{float_meta.get('epoch', '-')}/"
        f"{float_meta.get('total_epochs', '-')} | "
        f"P:{float_metrics['precision']:.4f} R:{float_metrics['recall']:.4f} "
        f"mAP50:{float_metrics['map50']:.4f} mAP50-95:{float_metrics['map']:.4f}"
    )
    print(
        f"[Compare] QAT   | best epoch:{qat_meta.get('epoch', '-')}/"
        f"{qat_meta.get('total_epochs', '-')} | "
        f"P:{qat_metrics['precision']:.4f} R:{qat_metrics['recall']:.4f} "
        f"mAP50:{qat_metrics['map50']:.4f} mAP50-95:{qat_metrics['map']:.4f}"
    )
    print(
        f"[Compare] delta | mAP50:{qat_metrics['map50'] - float_metrics['map50']:+.4f} "
        f"mAP50-95:{qat_metrics['map'] - float_metrics['map']:+.4f}"
    )
    return {"float": float_metrics, "qat": qat_metrics, "float_meta": float_meta, "qat_meta": qat_meta}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="YOLO26: 浮点训练(加载 yolo26{scale}.pt) -> PTQ -> QAT -> mAP 对比，模型尺度与量化方法可选"
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=f"yolo26{DEFAULT_SCALE}",
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
    # 量化超参（不再硬编码）：--a-bits/--w-bits/--per-channel/--all-positive / Quant hyperparameters (no longer hardcoded)
    add_quant_cfg_args(parser)
    # 冒烟/快速验证用：每个 epoch / 评估最多跑多少个 batch，默认不限制（完整训练） / Smoke test / quick validation: max batches per epoch / eval, default unlimited (full training)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--data",
        default=DATA_YAML,
        help="数据集 yaml 路径或数据集目录（默认自动下载 coco8.yaml 到 dataset/；可手动指定其他目录）",
    )
    parser.add_argument("--resume", action="store_true",
                        help="从 _last.pth checkpoint 接续训练")
    # 运行级变体：多种子 / 校准敏感度等补充实验，目录自动加后缀不覆盖主实验 /
    # Run-level variants for multi-seed / calibration-sensitivity runs; auto dir suffix, never overwrite main runs
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（默认 0；非 0 时产物目录加 _seed{N} 后缀）/ random seed (default 0; non-zero adds _seed{N} dir suffix)")
    parser.add_argument("--run-tag", type=str, default="",
                        help="额外产物目录后缀（如 calib5；默认空）/ extra artifact dir suffix (e.g. calib5; default empty)")
    return parser


if __name__ == "__main__":
    quant_pkg.install_print_timestamp()  # print 加分钟级时间戳 / minute-precision timestamp for print
    args = build_arg_parser().parse_args()

    # 写入量化超参（须在任何量化模型构建之前）/ Apply quant hyperparameters before any quantized model is built
    apply_quant_cfg_args(args)

    # 运行级变体：设置随机种子与目录后缀标签 / Run-level variants: set random seed and directory suffix tag
    RUN_SEED = args.seed
    RUN_TAG = args.run_tag.strip().lstrip("_")
    if RUN_SEED:
        random.seed(RUN_SEED)
        np.random.seed(RUN_SEED)
        torch.manual_seed(RUN_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RUN_SEED)
    if RUN_SEED or RUN_TAG:
        print(f"运行变体 / Run variant: seed={RUN_SEED} run_tag={RUN_TAG or '-'}")

    # 根据 --data 动态覆盖全局 NUM_CLASSES 和 DATA_YAML（默认走自动下载） /
    # Override global NUM_CLASSES / DATA_YAML per --data (default auto-downloads)
    DATA_YAML = args.data  # 复用变量名；函数内部引用它 / Reuse variable name; referenced inside functions
    try:
        _tmp = get_data_dict(args.data, "detect")
        NUM_CLASSES = len(_tmp["names"])
    except Exception:
        pass  # yaml 解析失败则保留默认 80 / Keep default 80 if yaml parsing fails
    print(f"数据集: {args.data} | 类别数: {NUM_CLASSES}")

    scale = get_scale(args.model)
    print(f"Python: {sys.executable}")
    print(f"torch: {torch.__version__} | Device: {device}")
    print(f"模型: {model_name(scale)} | 量化方法: {args.quant} | 阶段: {args.stage}")

    tag = args.quant
    qat_epochs = (
        args.qat_epochs
        if args.qat_epochs is not None
        else max(1, int(round(args.float_epochs * 0.2)))
    )

    if args.stage in ("all", "float"):
        # lr=None → 官方 optimizer=auto：AdamW lr=round(0.002*5/(4+nc), 6)（nc=80 → 0.000119）/ lr=None → official optimizer=auto: AdamW lr=round(0.002*5/(4+nc), 6) (nc=80 → 0.000119)
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
