"""YOLO26n 检测网络的 aLSQ+ 量化感知训练流程（coco128）。

流程：
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练（默认加载 yolo26n.pt 预训练权重）
    PTQ_calibration()                               # 训练后量化校准
    QAT_training(lr=qat_lr, epochs=qat_epochs)      # 量化感知训练
    compare_precision()                             # 浮点 vs QAT 的 mAP 对比

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ULTRA_DIR = os.path.join(BASE_DIR, "ultralytics", "ultralytics")
COCO128_YAML = os.path.join(ULTRA_DIR, "cfg", "datasets", "coco128.yaml")
# COCO128_YAML = os.path.join(ULTRA_DIR, "cfg", "datasets", "VOC.yaml")
PRETRAINED_WEIGHTS = os.path.join(ULTRA_DIR, "yolo26n.pt")

IMGSZ = 640
NUM_CLASSES = 80
# yolo26n scale=n: depth=0.50, width=0.25
STRIDES = (8, 16, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def set_quant_method(method):
    """切换量化后端（必须在构建 QuantYOLO26 之前调用）。"""
    global Q, QUANT_METHOD
    global QuantAdd, QuantCat, QuantConcat, QuantConv2d, QuantMaxPool

    Q = quant_pkg.load_quant_backend(method)
    QUANT_METHOD = method
    QuantAdd = Q.QuantAdd
    QuantCat = Q.QuantCat
    QuantConcat = Q.QuantConcat
    QuantConv2d = Q.QuantConv2d
    QuantMaxPool = Q.QuantMaxPool
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
    """yolo26n（scale=n）检测网络。quant=False 为浮点模型，True 为 aLSQ+ 伪量化模型。

    层编号 / 拓扑 / 命名与 ultralytics 解析 yolo26.yaml 得到的 DetectionModel 一致，
    因此浮点模型可直接加载 yolo26n.pt 的 state_dict。
    """

    def __init__(self, nc=NUM_CLASSES, quant=False):
        super().__init__()
        self.quant = quant
        self.nc = nc
        layers = []

        # ---------------- backbone ----------------
        layers += [_tag(Conv(3, 16, 3, 2, quant=quant), 0, -1)]          # P1/2
        layers += [_tag(Conv(16, 32, 3, 2, quant=quant), 1, -1)]         # P2/4
        layers += [_tag(C3k2(32, 64, n=1, c3k=False, e=0.25, quant=quant), 2, -1)]
        layers += [_tag(Conv(64, 64, 3, 2, quant=quant), 3, -1)]         # P3/8
        layers += [_tag(C3k2(64, 128, n=1, c3k=False, e=0.25, quant=quant), 4, -1)]
        layers += [_tag(Conv(128, 128, 3, 2, quant=quant), 5, -1)]       # P4/16
        layers += [_tag(C3k2(128, 128, n=1, c3k=True, quant=quant), 6, -1)]
        layers += [_tag(Conv(128, 256, 3, 2, quant=quant), 7, -1)]       # P5/32
        layers += [_tag(C3k2(256, 256, n=1, c3k=True, quant=quant), 8, -1)]
        layers += [_tag(SPPF(256, 256, k=5, n=3, shortcut=True, quant=quant), 9, -1)]
        layers += [_tag(C2PSA(256, 256, n=1, quant=quant), 10, -1)]

        # ---------------- head (PAN-FPN) ----------------
        layers += [_tag(nn.Upsample(scale_factor=2, mode="nearest"), 11, -1)]
        layers += [_tag(Concat(1, quant=quant), 12, [-1, 6])]
        layers += [_tag(C3k2(256 + 128, 128, n=1, c3k=True, quant=quant), 13, -1)]

        layers += [_tag(nn.Upsample(scale_factor=2, mode="nearest"), 14, -1)]
        layers += [_tag(Concat(1, quant=quant), 15, [-1, 4])]
        layers += [_tag(C3k2(128 + 128, 64, n=1, c3k=True, quant=quant), 16, -1)]  # P3

        layers += [_tag(Conv(64, 64, 3, 2, quant=quant), 17, -1)]
        layers += [_tag(Concat(1, quant=quant), 18, [-1, 13])]
        layers += [_tag(C3k2(64 + 128, 128, n=1, c3k=True, quant=quant), 19, -1)]  # P4

        layers += [_tag(Conv(128, 128, 3, 2, quant=quant), 20, -1)]
        layers += [_tag(Concat(1, quant=quant), 21, [-1, 10])]
        layers += [_tag(C3k2(128 + 256, 256, n=1, c3k=True, attn=True, quant=quant), 22, -1)]  # P5

        layers += [_tag(Detect(nc=nc, reg_max=1, ch=(64, 128, 256), quant=quant), 23, [16, 19, 22])]

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
        """用一次 dummy 前向推算 stride 并初始化 Detect 偏置（官方做法）。"""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            feats = self._forward_features(torch.zeros(1, 3, IMGSZ, IMGSZ))
            head = self.model[-1]
            head.stride = torch.tensor([IMGSZ / f.shape[-2] for f in feats])
            head.bias_init()
            # dummy 前向时 stride 还是 0，Detect 已缓存了零 stride_tensor，
            # 必须令缓存失效，让首次真实前向用正确 stride 重建 anchors。
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
    def __init__(self, nc=NUM_CLASSES):
        super().__init__(nc=nc, quant=False)


class QuantYOLO26(YOLO26):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__(nc=nc, quant=True)


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
    """在 coco128 验证集上计算 mAP50 / mAP50-95（NMS + ap_per_class，与官方一致）。"""
    model.eval()
    stats_conf, stats_pcls, stats_tcls, stats_tp = [], [], [], []
    names = data["names"]
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
    model.load_state_dict(state_dict)
    return (model, meta) if return_meta else model


def save_checkpoint(model, path, **metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if metadata:
        torch.save({"state_dict": model.state_dict(), "meta": metadata}, path)
    else:
        torch.save(model.state_dict(), path)


def load_pretrained(model, path=PRETRAINED_WEIGHTS):
    """加载官方 yolo26n.pt 权重；单头模型会自动跳过 one2one_* 双头权重。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].float().state_dict()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    skipped = [k for k in unexpected if "one2one_" in k]
    other_unexpected = [k for k in unexpected if "one2one_" not in k]
    if other_unexpected:
        print(f"[Pretrain] 警告：{len(other_unexpected)} 个未识别权重未加载: {other_unexpected[:3]}")
    print(f"[Pretrain] 已加载 {path}（跳过 {len(skipped)} 个 one2one 双头参数，缺失 {len(missing)} 个）")
    if missing:
        print(f"[Pretrain] 缺失参数: {missing[:5]}")
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


def build_float_model(quant_model, nc=NUM_CLASSES):
    """把量化模型的反量化权重灌进同结构的干净浮点模型（用于导出纯浮点 ONNX）。"""
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_state = quant_model.state_dict()
    for name, module in quant_model.named_modules():
        if quant_pkg.is_weight_quant_module(module):
            quant_state[f"{name}.weight"] = quant_pkg.dequantized_weight(module)

    float_model = FloatYOLO26(nc=nc)
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
        print(f"      onnxruntime vs PyTorch 最大绝对误差: {max_diff:.3e}")
        assert max_diff < 1e-3, f"ONNX 数值误差过大: {max_diff}"
    except ImportError:
        print("      [skip] 未安装 onnxruntime，跳过数值比对")

    required_keys = quant_pkg.required_quant_keys(quant_model)
    missing = [key for key in required_keys if key not in quant_params]
    assert not missing, f"以下量化参数不完整: {missing[:10]}"
    print(f"      量化参数完整性检查通过（{len(quant_params)} 个张量）")


def save_quant_outputs(quant_model, prefix, nc=NUM_CLASSES, meta=None):
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_yolo26n.pth")
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

    float_model = build_float_model(quant_model, nc=nc)
    float_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_yolo26n_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(MODEL_DIR, f"{prefix}_yolo26n_float.onnx")
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
                max_train_batches=None, max_eval_batches=None):
    print("========== Float training (YOLO26n / coco128) ==========")
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

    float_model = FloatYOLO26(num_classes).to(device)
    load_pretrained(float_model, PRETRAINED_WEIGHTS)

    criterion = build_criterion(float_model, epochs)
    optimizer = _build_optimizer(float_model, lr=lr)
    ema = _ModelEMA(float_model)

    close_mosaic = 10
    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    checkpoint_path = os.path.join(MODEL_DIR, "yolo26n.pth")
    last_checkpoint = os.path.join(MODEL_DIR, "yolo26n_last.pth")

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


@torch.no_grad()
def calibrate_quantizer(quant_model, calibration_loader, calibration_batches=20):
    quant_model.eval()
    for batch_index, batch in enumerate(calibration_loader):
        quant_model(batch["img"].float().to(device) / 255.0)
        if batch_index + 1 >= calibration_batches:
            break
    quant_pkg.freeze_batch_init(quant_model)


def PTQ_calibration(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                    calibration_batches=20, num_workers=2, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== PTQ calibration (YOLO26n / {tag}) ==========")
    print(f"Device: {device}")

    calibration_loader, data = get_dataloaders(batch_size, num_workers, calibration=True)
    _, val_loader, _ = get_dataloaders(batch_size, num_workers)

    float_checkpoint = os.path.join(MODEL_DIR, "yolo26n.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatYOLO26(num_classes).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = QuantYOLO26(num_classes).to(device)
    copy_float_to_quant(float_model, ptq_model)

    calibrate_quantizer(ptq_model, calibration_loader, calibration_batches)

    metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(f"[PTQ-{tag}] mAP50:{metrics['map50']:.4f} mAP50-95:{metrics['map']:.4f}")

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
    final_metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(f"[PTQ] 重载 checkpoint 后 mAP50:{final_metrics['map50']:.4f} mAP50-95:{final_metrics['map']:.4f}")
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=16, lr=None, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training (YOLO26n / {tag}) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr() * 0.1  # QAT 微调用 float auto lr 的 1/10
    print(f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4 | warmup 3ep | 梯度累积 nbs=64")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)
    nb = len(train_loader)

    ptq_checkpoint = os.path.join(MODEL_DIR, f"ptq_{tag}_yolo26n.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = QuantYOLO26(num_classes).to(device)
    _, ptq_meta = load_checkpoint(qat_model, ptq_checkpoint, return_meta=True)
    saved_method = ptq_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] PTQ checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    quant_pkg.freeze_batch_init(qat_model)

    criterion = build_criterion(qat_model, epochs)
    optimizer = _build_optimizer(qat_model, lr=lr)

    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    best_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_yolo26n_best.pth")
    last_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_yolo26n_last.pth")

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
        qat_model, f"qat_{tag}", nc=num_classes, meta=best_meta
    )[0]
    return qat_checkpoint, best_fitness, best_meta


def compare_precision(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                      num_workers=2, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== Float vs QAT precision (YOLO26n / coco128 / {tag}) ==========")
    print(f"Device: {device}")

    float_checkpoint = os.path.join(MODEL_DIR, "yolo26n.pth")
    qat_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_yolo26n.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26(num_classes).to(device)
    float_model, float_meta = load_checkpoint(float_model, float_checkpoint, return_meta=True)
    float_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )

    qat_model = QuantYOLO26(num_classes).to(device)
    qat_model, qat_meta = load_checkpoint(qat_model, qat_checkpoint, return_meta=True)
    saved_method = qat_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] QAT checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
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
        description="YOLO26n: 浮点训练(加载 yolo26n.pt) -> PTQ -> QAT -> mAP 对比，量化方法可选"
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
        # lr=None → 官方 optimizer=auto：AdamW lr=round(0.002*5/(4+nc), 6)（nc=80 → 0.000119）
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
