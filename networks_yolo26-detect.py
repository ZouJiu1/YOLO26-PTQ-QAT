"""YOLO26 检测网络的 aLSQ+ 量化感知训练流程（coco128，尺度可选 n/s/m/l/x）。

流程：
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练（默认加载 yolo26{scale}.pt 预训练权重）
    PTQ_calibration()                               # 训练后量化校准
    QAT_training(lr=qat_lr, epochs=qat_epochs)      # 量化感知训练
    compare_precision()                             # 浮点 vs QAT 的 mAP 对比

模型尺度用 --model yolo26n|yolo26s|yolo26m|yolo26l|yolo26x 选择（默认 yolo26n）：
通道按 make_divisible(min(c, max_ch)*width, 8)、重复次数按 max(round(n*depth), 1)
缩放，与 ultralytics parse_model 完全一致。仓库仅随附 yolo26n.pt 预训练权重，
其余尺度会自动跳过加载、从头训练。

网络结构与 ultralytics/cfg/models/26/yolo26.yaml（scale=n）逐层对齐：
Conv / C3k2(C3k) / SPPF / C2PSA(Attention) / PAN-FPN / Detect(reg_max=1)。
说明：yolo26n 原始 end2end 双头（one2many+one2one）这里只保留 one2many 单头，
损失仍使用官方 TaskAlignedAssigner(topk=10) + CIoU 的 v8DetectionLoss；
backbone / neck / 检测头拓扑与官方完全一致，可直接加载 yolo26n.pt 的权重
（one2one_* 双头权重会被跳过）。
"""

import argparse
import copy
import importlib
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import quantization as quant_pkg

# ultralytics 提供数据管道 / 损失 / 解码 / NMS / mAP
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.torch_utils import model_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ULTRA_DIR = os.path.join(BASE_DIR, "ultralytics", "ultralytics")
COCO128_YAML = os.path.join(ULTRA_DIR, "cfg", "datasets", "coco128.yaml")
# COCO128_YAML = os.path.join(ULTRA_DIR, "cfg", "datasets", "VOC.yaml")

IMGSZ = 640
NUM_CLASSES = 80
STRIDES = (8, 16, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 模型尺度选择：与 ultralytics/cfg/models/26/yolo26.yaml 的 scales 完全一致
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


def get_scale(scale=DEFAULT_SCALE):
    """归一化模型尺度名：允许传 'n' / 'yolo26n' / 'YOLO26n'。"""
    s = str(scale)[-1].lower()
    if s not in YOLO26_SCALES:
        raise ValueError(f"未知模型尺度 {scale!r}（可选 {', '.join(MODEL_CHOICES)}）")
    return s


def make_divisible(x, divisor=8):
    """ultralytics 的通道取整规则。"""
    return math.ceil(x / divisor) * divisor


def scaled_channels(channels, scale=DEFAULT_SCALE):
    """ultralytics parse_model 通道缩放：make_divisible(min(c, max_ch) * width, 8)。"""
    _, width, max_channels = YOLO26_SCALES[get_scale(scale)]
    return make_divisible(min(channels, max_channels) * width)


def scaled_repeats(repeats, scale=DEFAULT_SCALE):
    """ultralytics parse_model 层重复次数缩放：max(round(n * depth), 1)。"""
    depth, _, _ = YOLO26_SCALES[get_scale(scale)]
    return max(round(repeats * depth), 1)


def model_name(scale=DEFAULT_SCALE):
    """官方风格模型名，如 YOLO26n / YOLO26x。"""
    return f"YOLO26{get_scale(scale).upper()}"


def base_name(scale=DEFAULT_SCALE, task=""):
    """权重 / checkpoint 基础名，如 yolo26n、yolo26s-seg。"""
    return f"yolo26{get_scale(scale)}" + (f"-{task}" if task else "")


def detect_head_channels(scale=DEFAULT_SCALE):
    """Detect 头三尺度输入通道（层 16/19/22 输出，yaml 基准 256/512/1024）。"""
    return tuple(scaled_channels(c, scale) for c in (256, 512, 1024))

# ---------------------------------------------------------------------------
# 量化后端可切换：dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact
# 每个后端文件都实现了同一套算子；set_quant_method() 切换后再实例化量化模型即可。
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
    """切换量化后端（必须在构建 QuantYOLO26 之前调用）。"""
    global Q, QUANT_METHOD, QuantSiLU, QuantSigmoid, QuantMatMul
    global QuantAdd, QuantCat, QuantConcat, QuantConv2d, QuantMaxPool, QuantSiLU, QuantSigmoid, QuantMatMul
    global QuantSiLU, QuantSigmoid, QuantReLU

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


# ============================== 基础组件 ==============================


class FloatAdd(nn.Module):
    """浮点残差加法（无参数）。"""

    def forward(self, a, b):
        return a + b

def autopad(kernel_size, padding=None, dilation=1):
    if padding is not None:
        return padding
    return (kernel_size if isinstance(kernel_size, int) else kernel_size[0]) * dilation // 2


def make_bn(channels):
    # 与 ultralytics 保持一致：eps=1e-3, momentum=0.03
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)


class Conv(nn.Module):
    """Conv + BN + SiLU；quant=True 时卷积换成 QuantConv2d。

    与 ultralytics.nn.modules.Conv 同名同结构（self.conv / self.bn / self.act）。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bias=False, quant=False):
        super().__init__()
        if quant:
            self.conv = QuantConv2d(
                c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d),
                dilation=d, groups=g, bias=bias, a_bits=8, w_bits=8, per_channel=True,
            )
        else:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.bn = make_bn(c2)
        if quant and QuantSiLU is not None:
            self.act = QuantSiLU(a_bits=8, quant_inference=False) if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        else:
            self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """标准瓶颈块：cv1(3x3) -> cv2(3x3)，同通道时带残差加法。"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, quant=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1, quant=quant)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, quant=quant)
        self.has_add = shortcut and c1 == c2
        self.add_op = QuantAdd(a_bits=8, quant_inference=True) if quant else FloatAdd()

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return self.add_op(x, out) if self.has_add else out


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions（C3k 的基类）。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, quant=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1, quant=quant)
        self.cv2 = Conv(c1, c_, 1, 1, quant=quant)
        self.cv3 = Conv(2 * c_, c2, 1, 1, quant=quant)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0, quant=quant) for _ in range(n))
        )
        self.cat_op = QuantCat(2, a_bits=8) if quant else torch.cat

    def forward(self, x):
        return self.cv3(self._cat([self.m(self.cv1(x)), self.cv2(x)], 1))

    def _cat(self, tensors, dim):
        return self.cat_op(tensors, dim) if isinstance(self.cat_op, QuantCat) else torch.cat(tensors, dim)


class C3k(C3):
    """C3k：内部堆叠 n 个 3x3 Bottleneck（yolo26n 中 n=2）。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, quant=False):
        super().__init__(c1, c2, n, shortcut, g, e, quant=quant)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0, quant=quant) for _ in range(n))
        )


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions（C3k2 的基类）。"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, quant=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, quant=quant)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1, quant=quant)
        self.cat_op = QuantCat(n + 2, a_bits=8) if quant else torch.cat

    def _cat(self, tensors, dim=1):
        return self.cat_op(tensors, dim) if isinstance(self.cat_op, QuantCat) else torch.cat(tensors, dim)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(self._cat(y, 1))


class C3k2(C2f):
    """yolo26n 主干核心块。c3k=True 时使用 C3k，attn=True 时使用 Bottleneck+PSABlock。"""

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
    """PSA 多头自注意力（qkv / proj / pe 为卷积；matmul+softmax 保持浮点）。"""

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
    """注意力 + FFN，两条残差加法。"""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, quant=False):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads, quant=quant)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1, quant=quant), Conv(c * 2, c, 1, act=False, quant=quant))
        self.add = shortcut
        self.add1 = QuantAdd(a_bits=8, quant_inference=True) if quant else FloatAdd()
        self.add2 = QuantAdd(a_bits=8, quant_inference=True) if quant else FloatAdd()

    def forward(self, x):
        x = self.add1(x, self.attn(x)) if self.add else self.attn(x)
        x = self.add2(x, self.ffn(x)) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """C2PSA：cv1 降维 -> PSABlock 序列 -> cv2 融合。"""

    def __init__(self, c1, c2, n=1, e=0.5, quant=False):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, quant=quant)
        self.cv2 = Conv(2 * self.c, c1, 1, 1, quant=quant)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1), quant=quant) for _ in range(n))
        )
        self.cat_op = QuantCat(2, a_bits=8) if quant else torch.cat

    def _cat(self, tensors, dim=1):
        return self.cat_op(tensors, dim) if isinstance(self.cat_op, QuantCat) else torch.cat(tensors, dim)

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(self._cat([a, b], 1))


class SPPF(nn.Module):
    """YOLO26 版 SPPF：cv1 不带激活，串联 3 次 MaxPool，shortcut 残差加法。"""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False, quant=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False, quant=quant)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1, quant=quant)
        if quant:
            self.m = QuantMaxPool(kernel_size=k, stride=1, padding=k // 2, a_bits=8, quant_inference=True)
        else:
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2
        self.cat_op = QuantCat(n + 1, a_bits=8) if quant else torch.cat
        self.add_op = QuantAdd(a_bits=8, quant_inference=True) if quant else FloatAdd()

    def _cat(self, tensors, dim=1):
        return self.cat_op(tensors, dim) if isinstance(self.cat_op, QuantCat) else torch.cat(tensors, dim)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        out = self.cv2(self._cat(y, 1))
        return self.add_op(out, x) if self.add else out


class Concat(nn.Module):
    """PAN-FPN 中的拼接节点（对应 yaml 里的 Concat 层）。"""

    def __init__(self, dimension=1, quant=False):
        super().__init__()
        self.d = dimension
        self.op = QuantConcat(a_bits=8, quant_inference=True) if quant else None

    def forward(self, xs):
        if self.op is not None:
            return self.op(xs[0], xs[1], self.d)
        return torch.cat(xs, self.d)


class Detect(nn.Module):
    """YOLO26 Detect 检测头（单 one2many 头，reg_max=1，分类支路为 DWConv）。

    训练时返回 dict(boxes/scores/feats) 供 v8DetectionLoss 使用；
    评估时返回解码后的 (B, 4+nc, num_anchors) 检测张量（xywh + sigmoid 分数）。
    """

    def __init__(self, nc=NUM_CLASSES, reg_max=1, ch=(64, 128, 256), quant=False):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, reg_max * 4))
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3, quant=quant),
                Conv(c2, c2, 3, quant=quant),
                QuantConv2d(c2, 4 * reg_max, 1, a_bits=8, w_bits=8, per_channel=True)
                if quant else nn.Conv2d(c2, 4 * reg_max, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x, quant=quant), Conv(x, c3, 1, quant=quant)),
                nn.Sequential(Conv(c3, c3, 3, g=c3, quant=quant), Conv(c3, c3, 1, quant=quant)),
                QuantConv2d(c3, nc, 1, a_bits=8, w_bits=8, per_channel=True)
                if quant else nn.Conv2d(c3, nc, 1),
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
        # 推理：ltrb 距离 -> xywh 框（×stride），分类 sigmoid
        shape = x[0].shape
        if self._feat_shape != shape:
            self._anchors, self._strides_tensor = (
                a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5)
            )
            self._feat_shape = shape
        dbox = dist2bbox(self.dfl(preds["boxes"]), self._anchors.unsqueeze(0), xywh=True, dim=1)
        dbox = dbox * self._strides_tensor
        return torch.cat((dbox, preds["scores"].sigmoid()), 1)

    def bias_init(self):
        for i, (box_head, cls_head) in enumerate(zip(self.cv2, self.cv3)):
            box_head[-1].bias.data[:] = 2.0
            cls_head[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


# ============================== 网络主体 ==============================


def _tag(module, index, source):
    module.i = index
    module.f = source
    return module


class YOLO26(nn.Module):
    """YOLO26 检测网络（scale 可选 n/s/m/l/x）。quant=False 浮点模型，True 为 aLSQ+ 伪量化模型。

    层编号 / 拓扑 / 命名与 ultralytics 解析 yolo26.yaml 得到的 DetectionModel 一致，
    通道按 make_divisible(min(c, max_ch)*width, 8)、重复次数按 max(round(n*depth), 1)
    缩放（与 parse_model 相同），因此浮点模型可直接加载对应尺度 yolo26{scale}.pt 的
    state_dict。
    """

    def __init__(self, nc=NUM_CLASSES, quant=False, scale=DEFAULT_SCALE):
        super().__init__()
        self.quant = quant
        self.nc = nc
        self.scale = get_scale(scale)
        self.yaml_file = f"{base_name(self.scale)}.yaml"  # 供官方 model_info 打印模型名
        s = self.scale
        C = lambda c: scaled_channels(c, s)  # 通道缩放
        N = lambda r: scaled_repeats(r, s)   # 层重复次数缩放
        # ultralytics parse_model 特殊规则：M/L/X 尺度所有 C3k2 强制 c3k=True
        # （tasks.py: `if scale in {"m", "l", "x"}: args[3:4] = [True]`，覆盖 yaml 的 False）
        c3k_all = s in ("m", "l", "x")
        layers = []

        # ---------------- backbone ----------------
        layers += [_tag(Conv(3, C(64), 3, 2, quant=quant), 0, -1)]               # P1/2
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
        """把量化器的 init_state 注册为持久化 buffer。

        LSQPlus*Quantizer 原生的 init_state 是普通 int，不会进入 state_dict，
        checkpoint 重载后会回到 0，导致前向时用本批统计量覆盖已校准/训练好的 s。
        注册成 buffer 后可随 checkpoint 保存与恢复。
        """
        for module in self.modules():
            if "init_state" in dict(module.named_buffers()):
                continue
            if hasattr(module, "init_state"):
                value = int(module.init_state)
                del module.init_state  # 普通 int 属性需先删除才能注册同名 buffer
                module.register_buffer(
                    "init_state", torch.tensor(value), persistent=True
                )

    def _initialize_head(self):
        """用一次 dummy 前向推算 stride 并初始化 Detect 偏置（官方做法）。

        注意：dummy 输入不能用全零！LSQ v1 的 activation_quantizer 用全零
        初始化 s=0，之后 torch.div(x, 0) → NaN。
        用 randn 让每个 quantizer 得到合理的初始 s。
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMGSZ, IMGSZ) * 0.1  # 小随机噪声，避免全零
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


# ============================== 数据 / 损失 / 评估 ==============================


def _make_cfg(num_workers):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = IMGSZ
    cfg.task = "detect"
    cfg.workers = num_workers
    return cfg


def _build_loaders(batch_size, num_workers, data, calibration=False):
    """构建 dataloader；train 模式额外返回 cfg / train_set（close_mosaic 需要引用）。

    train：640x640 + mosaic/翻转等增强；val：rect letterbox；
    calibration：无增强的 640x640 letterbox。
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
    """复用 ultralytics 官方 coco128 数据管道。"""
    data = check_det_dataset(COCO128_YAML)
    if calibration:
        return _build_loaders(batch_size, num_workers, data, calibration=True), data
    train_loader, val_loader, _, _ = _build_loaders(batch_size, num_workers, data)
    return train_loader, val_loader, data


class _LossShim:
    """v8DetectionLoss 只需要 model.args / model.model[-1] / model.parameters()。"""

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
    """复刻 ultralytics BaseValidator._process_batch：在 10 个 IoU 阈值上匹配预测与 GT。"""
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


@torch.no_grad()
def evaluate(model, val_loader, data, batch_size=8, max_batches=None, conf_thres=0.001, iou_thres=0.7):
    """在验证集上计算 mAP50 / mAP50-95（NMS + ap_per_class，与官方一致）。"""
    model.eval()
    stats_conf, stats_pcls, stats_tcls, stats_tp = [], [], [], []
    names = data.names if hasattr(data, 'names') else data["names"]
    num_images = len(val_loader.dataset)
    steps = max_batches or math.ceil(num_images / batch_size)

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


# ============================== checkpoint / 权重复制 / 导出 ==============================


def load_checkpoint(model, path, return_meta=False):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        meta = checkpoint.get("meta", {})
    else:
        state_dict = checkpoint
        meta = {}

    # 过滤 shape mismatch（不同数据集 nc 变化时 Detect head）
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    if len(filtered) < len(state_dict):
        skipped = [k for k, v in state_dict.items()
                   if k in model_sd and model_sd[k].shape != v.shape]
        print(f"[load_checkpoint] 跳过 {len(skipped)} 个 shape mismatch 参数")
    if len(filtered) < len(model_sd):
        missing = set(model_sd.keys()) - set(filtered.keys())
        # 只打印非 QuantCat 的 missing（QuantCat init_state=0 是正常的）
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


def load_pretrained(model, path=None, scale=DEFAULT_SCALE):
    """加载官方 yolo26{scale}.pt 权重；自动跳过 one2one_* 双头权重和 shape mismatch（nc 变化）。

    仓库仅随附 yolo26n.pt；s/m/l/x 缺失时跳过加载，模型随机初始化从头训练。
    """
    if path is None:
        path = os.path.join(ULTRA_DIR, f"{base_name(scale)}.pt")
    if not os.path.exists(path):
        print(f"[Pretrain] [warn] 未找到预训练权重 {path}，{model_name(scale)} 将从头训练")
        return model
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].float().state_dict()

    # 过滤 shape mismatch 的 key（COCO 80类 → VOC 20类 时 Detect head 分类层）
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
    # 灌入真实权重后，让所有量化器从第 0 批重新统计 scale/beta（构建模型时的
    # dummy 零输入前向可能已经把 init_state 推到 1）：各后端的量化器在
    # init_state==0 的首次前向都会用自己的原生公式按真实权重/激活重新初始化。
    quant_pkg.reset_quantizer_states(quant_model)
    return quant_model


def build_float_model(quant_model, nc=NUM_CLASSES, scale=DEFAULT_SCALE):
    """把量化模型的反量化权重灌进同结构的干净浮点模型（用于导出纯浮点 ONNX）。"""
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_state = quant_model.state_dict()
    for name, module in quant_model.named_modules():
        if quant_pkg.is_weight_quant_module(module):
            quant_state[f"{name}.weight"] = quant_pkg.dequantized_weight(module)

    float_model = FloatYOLO26(nc=nc, scale=scale)
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
    # 各后端 scale/zero_point 提取差异由 quantization 包统一处理
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
        dynamic_axes={"images": {0: "batch_size"}, "preds": {0: "batch_size"}},
    )


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

    try:
        ort = importlib.import_module("onnxruntime")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        torch.manual_seed(0)
        float_model.cpu().eval()
        x = torch.randn(1, 3, IMGSZ, IMGSZ)
        with torch.no_grad():
            y_torch = float_model(x).numpy()
        y_onnx = session.run(["preds"], {"images": x.numpy()})[0]
        max_diff = float(np.abs(y_torch - y_onnx).max())
        ref_mag = float(np.abs(y_torch).max())
        print(f"      onnxruntime vs PyTorch 最大绝对误差: {max_diff:.3e}（参考幅度 {ref_mag:.3e}）")
        # 输出含大数量级解码坐标（如 0~imgsz 的 box 值），纯绝对阈值过严：
        # 改为 max(1e-3 绝对, 1e-5 相对)，仍足以抓住导出结构错误
        assert max_diff < max(1e-3, 1e-5 * ref_mag), (
            f"ONNX 数值误差过大: {max_diff}（参考幅度 {ref_mag:.3e}）"
        )
    except ImportError:
        print("      [skip] 未安装 onnxruntime，跳过数值比对")

    required_keys = quant_pkg.required_quant_keys(quant_model)
    missing = [key for key in required_keys if key not in quant_params]
    assert not missing, f"以下量化参数不完整: {missing[:10]}"
    print(f"      量化参数完整性检查通过（{len(quant_params)} 个张量）")


def save_quant_outputs(quant_model, prefix, nc=NUM_CLASSES, meta=None, scale=DEFAULT_SCALE):
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_{base_name(scale)}.pth")
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
    print(f"[2/5] 量化参数已写出: {json_path} / {pth_path}（{len(quant_params)} 个张量）")

    float_model = build_float_model(quant_model, nc=nc, scale=scale)
    float_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_{base_name(scale)}_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(MODEL_DIR, f"{prefix}_{base_name(scale)}_float.onnx")
    export_onnx(float_model, onnx_path, opset=16)
    print(f"[4/5] 干净浮点 ONNX 已写出: {onnx_path}")

    verify(float_model, quant_model, onnx_path, quant_params)
    print("[5/5] 完成。")
    return quant_checkpoint, json_path, pth_path, float_checkpoint, onnx_path


# ============================== 训练各阶段 ==============================


class _ModelEMA:
    """官方 ModelEMA 精简版：对训练权重做指数滑动平均，验证与保存均使用 EMA 权重。

    decay = 0.9999 * (1 - exp(-updates / 2000))，更新次数少时近似直接跟随模型。
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
    """官方 optimizer=auto 的 AdamW 学习率，随类别数自适应（nc=80 → 0.000119）。"""
    return round(0.002 * 5 / (4 + nc), 6)


def _build_optimizer(model, lr=None, decay=5e-4):
    """官方 optimizer=auto 配方：AdamW(betas=(0.9, 0.999))，三参数组分组。

    weight 组做 weight decay；BN 权重与 bias 不做 decay（量化器 scale s / 偏移
    beta 也归入无衰减组）；lr=None 时按类别数自适应。
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
    """官方 BaseTrainer 训练循环复刻：线性 lr 衰减 + warmup（lr 与梯度累积同步插值）
    + 梯度累积到 nbs + 梯度裁剪(10.0) + EMA 更新。

    criterion 返回已乘 batch_size 的损失向量（官方直接 backward，不除以 batch）
    与未缩放的 items dict（box_loss / cls_loss / l1_loss）。
    """
    model.train()
    batch_size = loader.batch_size
    accumulate = max(round(nbs / batch_size), 1)
    warmup_steps = round(min(warmup_epochs, max(epochs - 1, 0)) * nb) if warmup_epochs > 0 else 0
    # 线性衰减（官方默认 cos_lr=False）：lf(0)=1 → lf(epochs)=lrf
    lf = lambda x: max(1 - x / epochs, 0) * (1.0 - lrf) + lrf

    # 官方每个 epoch 开始时 scheduler.step()：lr = initial_lr * lf(epoch)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lf(epoch)

    total_steps = nb if max_batches is None else min(nb, max_batches)
    loss_items_sum = torch.zeros(3)
    last_opt_step = 0
    for i, batch in enumerate(loader):
        if i >= total_steps:
            break
        ni = i + nb * epoch  # 自训练开始的累计 batch 数
        if ni < warmup_steps:
            xi = [0, warmup_steps]
            accumulate = max(1, int(np.interp(ni, xi, [1, nbs / batch_size]).round()))
            for group in optimizer.param_groups:
                # optimizer=auto 时 warmup_bias_lr=0.0：所有组 lr 从 0 爬升
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
                max_train_batches=None, max_eval_batches=None, scale=DEFAULT_SCALE):
    print(f"========== Float training ({model_name(scale)} / coco128) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr()
    print(
        f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4（官方 optimizer=auto 配方）"
        " | warmup 3ep | 梯度累积 nbs=64 | EMA | close_mosaic=10"
    )

    data = check_det_dataset(COCO128_YAML)
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
    checkpoint_path = os.path.join(MODEL_DIR, f"{base_name(scale)}.pth")
    last_checkpoint = os.path.join(MODEL_DIR, f"{base_name(scale)}_last.pth")

    for epoch in range(epochs):
        if epoch == epochs - close_mosaic:
            print("[Float] 关闭 dataloader mosaic（最后 10 个 epoch）")
            train_set.close_mosaic(copy(cfg))
            train_loader.reset()

        train_loss, (box_loss, cls_loss, l1_loss) = train_one_epoch(
            float_model, train_loader, criterion, optimizer, epoch, epochs, nb, ema=ema,
            max_batches=max_train_batches,
        )
        metrics = evaluate(
            ema.ema, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
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
            **metrics,
        }
        save_checkpoint(ema.ema, last_checkpoint, **epoch_meta)

        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch + 1
            best_meta = epoch_meta
            save_checkpoint(ema.ema, checkpoint_path, **best_meta)

        print(
            f"[Float] Epoch [{epoch + 1}/{epochs}] | loss:{train_loss:.4f} "
            f"(box:{box_loss:.3f} cls:{cls_loss:.3f} l1:{l1_loss:.3f}) | "
            f"mAP50:{metrics['map50']:.4f} mAP50-95:{metrics['map']:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)  # 载入最优 EMA 权重，供后续 PTQ 使用
    best_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(f"[Float] Best checkpoint: {checkpoint_path}（epoch {best_epoch}/{epochs}）")
    print(f"[Float] Best mAP50:{best_metrics['map50']:.4f} mAP50-95:{best_metrics['map']:.4f}")
    return checkpoint_path, best_fitness, best_meta


def _collect_float_ranges_full(float_model, calibration_loader, calibration_batches,
                               normalize_img=True):
    """收集 float 模型中每个 Conv/Linear + eltwise 算子的运行 min/max 范围。

    Hook 策略（全部是 named module 的 forward_pre_hook，不 monkeypatch torch.cat/add 等）：
      - Conv2d/ConvTranspose2d/Linear: 输入 (单 tensor)
      - FloatAdd: 两个输入 (A, C) — 同名 QuantAdd
      - MaxPool2d: 输入 (单 tensor) — 同名 QuantMaxPool
      - Concat: 输入列表 xs = [tensor, ...] — QuantConcat 在 {name}.op
      - QuantCat: float 里是 torch.cat 函数调用 (C2f/C3k/SPPF/...)，无同名模块，
        靠安全网处理

    Returns:
        module_input_ranges: dict[name] = list of [min, max]
          - Conv/Linear: 1 个输入 → [[min, max]]
          - FloatAdd: 2 个输入 → [[A_min, A_max], [C_min, C_max]]
          - Concat: N 个输入 → [[xs[0]_min, xs[0]_max], ...]
          - MaxPool2d: 1 个输入 → [[min, max]]
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
        """展开 tuple/list 中的所有 tensor 输入。"""
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
        # 激活函数：单输入
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
    """把量化器标记为已初始化/冻结（兼容 int 与 Tensor 两种 init_state）。

    与 quantization.freeze_batch_init 的单量化器版本语义一致：只改标志位，
    不动已写入的 scale/beta/alpha。
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
    """通用：用 min-max 范围初始化一个量化器，支持全部 7 个后端。

    cur_min / cur_max 允许是 Tensor 或 python float，内部统一转成 float32 Tensor。
    各后端的 scale 语义不同，按各自原生公式赋值：
      - minmax:  写 r_min/r_max 并按其 forward 公式重算 scale/zero_point；
      - pact:    alpha = 绝对值最大（与其原生自初始化一致）；
      - dorefa:  s = 激活幅度上界（归一化尺度，不是网格 scale）；
      - lsqplus: s = (max-min)/(Qp-Qn)，beta = min - s*Qn（非对称精确覆盖）；
      - lsq:     对称网格，取能覆盖 [min, max] 的最小 scale。
    """
    cur_min = torch.as_tensor(cur_min, dtype=torch.float32)
    cur_max = torch.as_tensor(cur_max, dtype=torch.float32)
    if float(cur_min) > float(cur_max):
        cur_min, cur_max = cur_max, cur_min

    # minmax 后端：r_min/r_max buffer + 原生 scale/zero_point 公式
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

    # PACT：对称截断阈值 alpha（原生自初始化用的就是绝对值最大）
    if hasattr(q, "alpha"):
        cur_max_abs = torch.maximum(cur_min.abs(), cur_max.abs()).clamp(min=eps)
        q.alpha.data.copy_(cur_max_abs.reshape(q.alpha.shape).to(q.alpha.device))
        _set_quantizer_frozen(q)
        return True

    Qn = getattr(q, "Qn", None)
    Qp = getattr(q, "Qp", None)
    if hasattr(q, "s") and Qn is not None:
        if hasattr(q, "_set_init_state"):
            # dorefa：s 是激活幅度上界（x/s 归一化到 [-1,1] 再量化）
            if getattr(q, "all_positive", False):
                cur_s = cur_max.clamp(min=1e-6)
            else:
                cur_s = torch.maximum(cur_min.abs(), cur_max.abs()).clamp(min=1e-6)
            q.s.data.copy_(cur_s.reshape(q.s.shape).to(q.s.device))
        elif hasattr(q, "beta"):
            # lsqplus：非对称 scale + beta 精确覆盖 [min, max]
            cur_s = torch.clamp(cur_max - cur_min, min=eps) / (Qp - Qn)
            q.s.data.copy_(cur_s.reshape(q.s.shape).to(q.s.device))
            cur_beta = cur_min - cur_s * Qn
            q.beta.data.copy_(cur_beta.reshape(q.beta.shape).to(q.beta.device))
        else:
            # lsq：对称网格 [-Qn*s, Qp*s]，取能覆盖 [min, max] 的最小 scale
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
    """用 float 模型的激活范围 + 权重范围初始化 quant 模型的量化器。

    覆盖三类量化点（全部直接从 float 模型收集，无级联误差）：
      1. Conv/Linear 的 activation_quantizer —— float forward 输入范围
      2. Conv/Linear 的 weight_quantizer —— 当前权重的 min-max（避免 LSQ+ 3σ 饱和）
      3. FloatAdd/MaxPool2d/Concat 的 eltwise 量化器：
           - FloatAdd(name) → QuantAdd(name).activation_quantizer0/1
           - MaxPool2d(name) → QuantMaxPool(name).activation_quantizer
           - Concat(name) → QuantConcat(name.op).activation_quantizer0/1

    QuantCat（torch.cat 函数调用）无同名 float 模块，交给安全网处理。
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
        # 统一处理成 list of tensors
        ranges = [[torch.as_tensor(m, dtype=torch.float32),
                   torch.as_tensor(x, dtype=torch.float32)]
                  for m, x in float_range_list]

        if name not in quant_model_modules:
            # 可能是 Concat → QuantConcat 在 {name}.op
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

        # ── QuantMaxPool: activation_quantizer (单输入) ──
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

        # ── 其他量化模块：通用 activation_quantizer ──
        aq = getattr(qmodule, "activation_quantizer", None)
        if aq is not None and len(ranges) >= 1:
            if _apply_minmax_to_quantizer(aq, ranges[0][0], ranges[0][1], eps):
                aq_count += 1

    print(f"[Calib] Initialized {aq_count} activation + {wq_count} weight "
          f"+ {eltwise_count} eltwise quantizers from float ranges")
    return aq_count, wq_count


def _safety_net_calibrate(quant_model, calibration_loader, calibration_batches=20,
                          normalize_img=True):
    """安全网校准：在真实前向中完成剩余量化点的初始化。

    Conv/Linear 已由 _init_quantizers_from_float 按浮点范围初始化并冻结；
    本函数处理其余量化算子（QuantAdd/QuantCat/QuantConcat/QuantMaxPool 等）：

      1. 给它们挂 forward_pre_hook，记录各输入张量跨 batch 的运行 min/max；
      2. 跑若干批真实前向 —— 各后端的原生自初始化机制正常触发
         （LSQ+/LSQ 的 EMA、minmax 的运行统计、PACT 的首批 absmax、
         dorefa 的首批+EMA），不存在"垃圾 scale 被冻结"的路径；
      3. 结束后按记录的运行 min/max 统一覆盖赋值 —— 修正 PACT 只看首批、
         LSQ 家族 EMA 滞后的问题（minmax 保留其原生 percentile 统计，不覆盖）。
    """
    records = {}
    hooks = []

    def make_pre_hook(name):
        def pre_hook(module, args):
            # 展开输入张量：QuantCat 是 (tensor_list, dim)，Add/Concat 是
            # (A, C[, dim])，MaxPool 是 (x,)，统一抽成 tensor 列表
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
            continue  # Conv/Linear：已按 float 范围初始化
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

    # 按记录的运行 min/max 统一赋值 —— 只覆盖未初始化的
    quant_modules = dict(quant_model.named_modules())
    assigned = 0
    skipped = 0

    def _is_already_initialized(q):
        """LSQ 家族 init_state=FROZEN; minmax/pact init=1 (已初始化标记)。"""
        ist = getattr(q, 'init_state', None)
        if isinstance(ist, torch.Tensor) and int(ist.flatten()[0]) >= quant_pkg.INIT_STATE_FROZEN:
            return True
        if isinstance(ist, int) and ist >= quant_pkg.INIT_STATE_FROZEN:
            return True
        if hasattr(q, 'init') and type(q.init) is int and q.init == 1:
            return True
        return False

    def _assign_recorded(q, mn, mx):
        # 已由 float 范围初始化 → 跳过（不要覆盖更准确的 float 范围）
        if _is_already_initialized(q):
            return 'skip'
        # minmax 后端已在安全网前向中按原生 percentile 机制自收集，保留其结果
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
    """校准量化器。如果提供 float_model, 用 float 模型的激活+权重范围独立校准,
    避免级联误差和 3σ 饱和问题。

    流程（float_model 路径）:
      1. reset: 所有量化器回到未初始化状态 —— 构建模型时 _initialize_head 的
         dummy 零输入前向可能已把垃圾 scale 写进量化器，必须先清掉标志位；
      2. Conv/Linear 的激活/权重量化器用 float 范围初始化并冻结（每层独立，
         无级联误差；权重用 min-max 精确覆盖，无 3σ 饱和）；
      3. 安全网: 跑若干批真实前向，其余量化算子（QuantAdd/QuantCat/
         QuantConcat/QuantMaxPool）按各后端原生机制自初始化，再统一按记录的
         运行 min/max 赋值 —— 不存在"垃圾 scale 被冻结"的路径；
      4. freeze 全部量化器。

    无 float_model 时退化为纯 quant 前向自校准（各后端原生机制）。

    Args:
        normalize_img: 数据 loader 输出是否需要 /255 归一化。
            detect/seg/pose 的数据 loader 输出 [0,255]，需要 True。
            cls 的数据 loader 已经归一化到 [0,1]，用 False。
    """
    quant_model.eval()
    quant_pkg.reset_quantizer_states(quant_model)

    if float_model is not None:
        _init_quantizers_from_float(
            float_model, quant_model, calibration_loader, calibration_batches,
            normalize_img=normalize_img,
        )

    # 安全网：覆盖剩余量化点（float_model=None 时这是唯一的校准手段）
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

    float_checkpoint = os.path.join(MODEL_DIR, f"{base_name(scale)}.pth")
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
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
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
        ptq_model, f"ptq_{tag}", nc=num_classes, meta=ptq_meta, scale=scale
    )[0]

    load_checkpoint(ptq_model, ptq_checkpoint)
    quant_pkg.freeze_batch_init(ptq_model)
    final_metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(f"[PTQ] 重载 checkpoint 后 mAP50:{final_metrics['map50']:.4f} mAP50-95:{final_metrics['map']:.4f}")
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=16, lr=None, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None, scale=DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training ({model_name(scale)} / {tag}) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr() * 0.1  # QAT 微调用 float auto lr 的 1/10
    print(f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4 | warmup 3ep | 梯度累积 nbs=64")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)
    nb = len(train_loader)

    ptq_checkpoint = os.path.join(MODEL_DIR, f"ptq_{tag}_{base_name(scale)}.pth")
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
    best_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_{base_name(scale)}_best.pth")
    last_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_{base_name(scale)}_last.pth")

    for epoch in range(epochs):
        train_loss, (box_loss, cls_loss, l1_loss) = train_one_epoch(
            qat_model, train_loader, criterion, optimizer, epoch, epochs, nb,
            max_batches=max_train_batches,
        )
        metrics = evaluate(
            qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
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
            **metrics,
        }
        save_checkpoint(qat_model, last_checkpoint, **epoch_meta)

        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch + 1
            best_meta = epoch_meta
            save_checkpoint(qat_model, best_checkpoint, **best_meta)

        print(
            f"[QAT-{tag}] Epoch [{epoch + 1}/{epochs}] | loss:{train_loss:.4f} "
            f"(box:{box_loss:.3f} cls:{cls_loss:.3f} l1:{l1_loss:.3f}) | "
            f"mAP50:{metrics['map50']:.4f} mAP50-95:{metrics['map']:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    best_metrics = evaluate(
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(
        f"[QAT-{tag}] Best epoch:{best_epoch}/{epochs} | "
        f"mAP50:{best_metrics['map50']:.4f} mAP50-95:{best_metrics['map']:.4f} | "
        f"Best: {best_checkpoint}"
    )
    qat_checkpoint = save_quant_outputs(
        qat_model, f"qat_{tag}", nc=num_classes, meta=best_meta, scale=scale
    )[0]
    return qat_checkpoint, best_fitness, best_meta


def compare_precision(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                      num_workers=2, max_eval_batches=None, scale=DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== Float vs QAT precision ({model_name(scale)} / coco128 / {tag}) ==========")
    print(f"Device: {device}")

    float_checkpoint = os.path.join(MODEL_DIR, f"{base_name(scale)}.pth")
    # 优先 _best.pth（最新训练保存的 best），fallback 到 .pth
    qat_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_{base_name(scale)}_best.pth")
    if not os.path.exists(qat_checkpoint):
        qat_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_{base_name(scale)}.pth")
    ptq_checkpoint = os.path.join(MODEL_DIR, f"ptq_{tag}_{base_name(scale)}.pth")
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
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
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
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
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
    # 冒烟/快速验证用：每个 epoch / 评估最多跑多少个 batch，默认不限制（完整训练）
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--data",
        default=COCO128_YAML,
        help="数据集 yaml 路径（默认 coco128.yaml；VOC 可用 ultralytics/ultralytics/cfg/datasets/VOC.yaml）",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    # 根据 --data yaml 动态覆盖全局 NUM_CLASSES 和 COCO128_YAML
    COCO128_YAML = args.data  # 复用变量名；函数内部引用它
    try:
        _tmp = check_det_dataset(args.data)
        NUM_CLASSES = len(_tmp["names"])
    except Exception:
        pass  # yaml 解析失败则保留默认 80
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
        # lr=None → 官方 optimizer=auto：AdamW lr=round(0.002*5/(4+nc), 6)（nc=80 → 0.000119）
        float_train(
            batch_size=args.float_batch_size,
            lr=args.float_lr,
            epochs=args.float_epochs,
            num_classes=NUM_CLASSES,
            num_workers=args.num_workers,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
            scale=scale,
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
