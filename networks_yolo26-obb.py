"""YOLO26 OBB 旋转框检测网络的 aLSQ+ 量化感知训练流程（dota8-multispectral，尺度可选 n/s/m/l/x） /
aLSQ+ Quantization-Aware Training pipeline for YOLO26 OBB (oriented bounding box) network
(dota8-multispectral, scales n/s/m/l/x optional).

流程 / Pipeline:
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练 / float training（默认加载 yolo26{scale}-obb.pt 预训练权重 / loads yolo26{scale}-obb.pt pretrained weights by default）
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

网络结构与 ultralytics/cfg/models/26/yolo26-obb.yaml（scale=n）逐层对齐 /
Network architecture aligned layer-by-layer with ultralytics/cfg/models/26/yolo26-obb.yaml (scale=n):
backbone / neck（层 0-22）与 networks_yolo26-detect.py 完全一致，直接复用 /
backbone/neck (layers 0-22) identical to networks_yolo26-detect.py, directly reused;
检测头为 OBB26（层 23，继承 det.Detect，仅新增 cv4 角度分支） /
head is OBB26 (layer 23, inherits det.Detect, only adds cv4 angle branch).
Conv / C3k2(C3k) / SPPF / C2PSA(Attention) / PAN-FPN / OBB26(reg_max=1, ne=1 角度分支 / angle branch).
说明 / Note: yolo26n-obb 原始 end2end 双头（one2many+one2one）这里只保留 one2many 单头 /
original yolo26n-obb end2end dual-head (one2many+one2one), only one2many head kept here;
损失仍使用官方 RotatedTaskAlignedAssigner(topk=10) 的 v8OBBLoss（box/cls/l1/angle 四项） /
loss still uses official v8OBBLoss with RotatedTaskAlignedAssigner(topk=10) (box/cls/l1/angle);
数据集为 dota8-multispectral（10 通道 TIFF，15 类，OBB 标注 xywhr），首层 Conv 输入通道随 /
dataset is dota8-multispectral (10-channel TIFF, 15 classes, OBB xywhr labels); first Conv input
data['channels'] 自适应（预训练首层为 3 通道，shape mismatch 自动跳过、重新初始化） /
channels adapt to data['channels'] (pretrained first conv is 3-channel, skipped on shape mismatch);
backbone / neck / OBB 检测头拓扑与官方完全一致，可直接加载 yolo26n-obb.pt 的权重 /
backbone / neck / obb head topology fully consistent with official, can directly load yolo26n-obb.pt weights
（one2one_* 双头权重会被跳过 / one2one_* dual-head weights are skipped）.
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

import quantization as quant_pkg

# 复用检测网络的 backbone/neck 组件、尺度缩放助手与训练基础设施 / Reuse detection network's backbone/neck components, scale scaling helpers and training infrastructure
# （文件名 networks_yolo26-detect.py 含 '-'，不能直接 import，用文件路径加载） / (Filename networks_yolo26-detect.py contains '-', cannot import directly; load via file path)
det_spec = importlib.util.spec_from_file_location(
    "networks_yolo26_detect",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "networks_yolo26-detect.py"),
)
det = importlib.util.module_from_spec(det_spec)
det_spec.loader.exec_module(det)

# ultralytics 提供数据管道 / 损失 / 解码 / NMS / mAP / ultralytics provides data pipeline / loss / decode / NMS / mAP
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.tal import make_anchors, dist2rbox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, batch_probiou
from ultralytics.utils.plotting import plot_images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR_BASE = os.path.join(BASE_DIR, "model", "yolo26-obb")
MODEL_DIR = _MODEL_DIR_BASE


def _model_dir_for(scale=None, quant_method=None):
    """按网络名 + 尺度 + 量化后端返回产物目录 / Return artifact directory by net name + scale + quant backend."""
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
DATA_YAML = det.DEFAULT_DATA_YAML["obb"]
# 备选：RGB OBB 可改成 "dota8.yaml" / Alternative RGB OBB: use "dota8.yaml"

IMGSZ = 640
NUM_CLASSES = 15
IN_CHANNELS = 10  # 多光谱输入通道数，main 中按 data['channels'] 覆盖 / multispectral input channels, overridden by data['channels'] in main
STRIDES = (8, 16, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 量化后端可切换：dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact / Quantization backend switchable: dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact
# backbone 组件从检测网络复用，因此切换时同步切换检测网络模块内绑定的后端算子 / Backbone components are reused from detection network, so bound backend ops in detection module must switch in sync.
# ---------------------------------------------------------------------------
DEFAULT_QUANT_METHOD = det.DEFAULT_QUANT_METHOD
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
    """切换量化后端（必须在构建 QuantYOLO26OBB 之前调用）。 / Switch quantization backend (must be called before constructing QuantYOLO26OBB)."""
    global Q, QUANT_METHOD
    global QuantAdd, QuantCat, QuantConcat, QuantConv2d, QuantMaxPool, QuantSiLU, QuantSigmoid, QuantReLU

    # 复用的 backbone/neck block 类内部引用的是 det 模块的全局算子，必须先同步切换 / Reused backbone/neck block classes reference global ops in det module; must switch in sync first
    det.set_quant_method(method)
    Q = quant_pkg.load_quant_backend(method)
    QUANT_METHOD = method
    QuantAdd = det.QuantAdd
    QuantCat = det.QuantCat
    QuantConcat = det.QuantConcat
    QuantConv2d = det.QuantConv2d
    QuantMaxPool = det.QuantMaxPool
    QuantSiLU = getattr(Q, 'QuantSiLU', None)
    QuantSigmoid = getattr(Q, 'QuantSigmoid', None)
    QuantReLU = getattr(Q, 'QuantReLU', None)
    return Q


# ============================== OBB 检测头组件 / OBB Head Components ==============================

class OBBDetect(det.Detect):
    """YOLO26 OBB 旋转框检测头（单 one2many 头，reg_max=1，分类支路为 DWConv，含 cv4 角度分支） /
    YOLO26 OBB head (single one2many head, reg_max=1, class branch uses DWConv, with cv4 angle branch).

    继承 det.Detect（cv2 框 / cv3 分类支路完全复用），仅新增 cv4 角度分支并覆盖 self.no /
    Inherits det.Detect (cv2 box / cv3 class branches fully reused), only adds cv4 angle branch and overrides self.no.
    与官方 OBB26 一致：角度分支输出 raw logits（不做 sigmoid*pi 缩放） /
    Same as official OBB26: angle branch outputs raw logits (no sigmoid*pi scaling).

    训练时返回 dict(boxes/scores/angle/feats) 供 v8OBBLoss 使用 /
    During training returns dict(boxes/scores/angle/feats) for v8OBBLoss;
    评估时返回解码后的 (B, 4+nc+1, num_anchors) 张量（xywh 像素 + sigmoid 分数 + 角度弧度） /
    During eval returns decoded (B, 4+nc+1, num_anchors) tensor (xywh pixels + sigmoid scores + angle rad).
    """

    def __init__(self, nc=NUM_CLASSES, ne=1, reg_max=1, ch=(64, 128, 256), quant=False):
        super().__init__(nc=nc, reg_max=reg_max, ch=ch, quant=quant)
        self.ne = ne  # 角度等额外参数个数 / number of extra params (angle)
        self.no = nc + reg_max * 4 + ne  # 覆盖父类：加上角度通道 / override parent: add angle channels
        c4 = max(ch[0] // 4, ne)  # 角度分支隐藏通道，与官方 OBB 一致 / angle branch hidden channels, same as official OBB
        # cv4 角度分支：官方 OBB 为 Conv(x, c4, 3) + Conv(c4, c4, 3) + Conv2d(c4, ne, 1) /
        # cv4 angle branch: official OBB uses Conv(x, c4, 3) + Conv(c4, c4, 3) + Conv2d(c4, ne, 1)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                det.Conv(x, c4, 3, quant=quant),
                det.Conv(c4, c4, 3, quant=quant),
                QuantConv2d(c4, ne, 1, a_bits=8, w_bits=8, per_channel=True)
                if quant else nn.Conv2d(c4, ne, 1),
            )
            for x in ch
        )

    def forward_head(self, x):
        bs = x[0].shape[0]
        boxes = torch.cat([self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        # OBB26：角度为 raw logits，训练时由 v8OBBLoss 直接回归 /
        # OBB26: raw angle logits, directly regressed by v8OBBLoss during training
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "angle": angle, "feats": x}

    def forward(self, x):
        preds = self.forward_head(x)
        if self.training:
            return preds
        # 推理：ltrb 距离 + 角度 -> 旋转框 xywh（×stride），分类 sigmoid /
        # Inference: ltrb distances + angle -> rotated bbox xywh (×stride), class sigmoid
        shape = x[0].shape
        if self._feat_shape != shape:
            self._anchors, self._strides_tensor = (
                a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5)
            )
            self._feat_shape = shape
        dbox = dist2rbox(self.dfl(preds["boxes"]), preds["angle"], self._anchors.unsqueeze(0), dim=1)
        dbox = dbox * self._strides_tensor
        # 输出 (B, 4+nc+1, N)：xywh 像素 + sigmoid 分数 + 角度（弧度，raw） /
        # Output (B, 4+nc+1, N): xywh pixels + sigmoid scores + angle (radians, raw)
        return torch.cat((dbox, preds["scores"].sigmoid(), preds["angle"]), 1)


# ============================== 网络主体 / Network Body ==============================


class YOLO26OBB(det.YOLO26):
    """YOLO26 OBB 旋转框检测网络（scale 可选 n/s/m/l/x） / YOLO26 OBB detection network (scale n/s/m/l/x optional).
    quant=False 浮点模型，True 为 aLSQ+ 伪量化模型 / quant=False for float model, True for aLSQ+ pseudo-quantized model.

    层 0-22 与检测网络一致（直接构建后替换层 23 为 OBBDetect），topology 编号/save 集合不变， /
    Layers 0-22 identical to detection network (built directly then replace layer 23 with OBBDetect); topology indices/save set unchanged,
    因此 yolo26{scale}-obb.pt 的 state_dict（model.0.* ~ model.23.*）可直接加载， /
    so state_dict of yolo26{scale}-obb.pt (model.0.* ~ model.23.*) can be loaded directly,
    end2end 双头 one2one_* 参数被跳过。 / end2end dual-head one2one_* parameters are skipped.
    """

    def __init__(self, nc=NUM_CLASSES, quant=False, scale=det.DEFAULT_SCALE, ch_in=None):
        # ch_in=None 时读全局 IN_CHANNELS（main 会按 data['channels'] 覆盖）；多光谱为 10 /
        # ch_in=None reads global IN_CHANNELS (main overrides per data['channels']); 10 for multispectral
        self.ch_in = ch_in if ch_in is not None else IN_CHANNELS
        # 先构建检测网络得到完整的 0-22 backbone/neck，再把层 23 的 Detect 换成 OBBDetect /
        # First build detection network to get complete 0-22 backbone/neck, then replace layer 23's Detect with OBBDetect
        super().__init__(nc=nc, quant=quant, scale=scale)
        head = OBBDetect(nc=nc, ne=1, reg_max=1, ch=det.detect_head_channels(scale), quant=quant)
        det._tag(head, 23, [16, 19, 22])
        self.model[-1] = head
        # 多光谱输入通道数 != 3 时替换首层 Conv（detect 默认按 RGB 3 通道构建） /
        # Replace first Conv when input channels != 3 (detect builds RGB 3-channel by default)
        if self.ch_in != 3:
            c0 = self.model[0].conv.out_channels
            self.model[0] = det.Conv(self.ch_in, c0, 3, 2, quant=quant)
            det._tag(self.model[0], 0, -1)
        self.yaml_file = f"{det.base_name(self.scale, 'obb')}.yaml"  # 供官方 model_info 打印模型名 / for official model_info to print model name
        self._register_quantizer_buffers()
        self._initialize_head()  # 换了头（可能还有首层），需重新推算 stride / 初始化偏置 / head (and maybe first conv) replaced: re-infer strides / re-init bias

    def _initialize_head(self):
        """用一次 dummy 前向推算 stride 并初始化检测头偏置（官方做法） /
        Use one dummy forward to infer strides and init head bias (official approach).

        dummy 输入通道数取当前首层 Conv 的 in_channels（父类构建期 3 通道、换首层后为多光谱通道）， /
        dummy input channel count follows current first Conv's in_channels (3 during parent build, multispectral after swap);
        不能用全零！LSQ v1 的 activation_quantizer 用全零初始化 s=0，之后 torch.div(x, 0) → NaN /
        must NOT be all zeros! LSQ v1 activation_quantizer initializes s=0 with all-zeros, causing torch.div(x,0) → NaN.
        用 randn 让每个 quantizer 得到合理的初始 s / Use randn so each quantizer gets a reasonable initial s.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(1, self.model[0].conv.in_channels, IMGSZ, IMGSZ) * 0.1  # 小随机噪声，避免全零 / small random noise, avoid all zeros
            feats = self._forward_features(dummy)
            head = self.model[-1]
            head.stride = torch.tensor([IMGSZ / f.shape[-2] for f in feats])
            head.bias_init()
            head._feat_shape = None
        if was_training:
            self.train()


class FloatYOLO26OBB(YOLO26OBB):
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE, ch_in=None):
        super().__init__(nc=nc, quant=False, scale=scale, ch_in=ch_in)


class QuantYOLO26OBB(YOLO26OBB):
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE, ch_in=None):
        super().__init__(nc=nc, quant=True, scale=scale, ch_in=ch_in)


# ============================== 数据 / 损失 / 评估 / Data / Loss / Evaluation ==============================


def _make_cfg(num_workers):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = IMGSZ
    cfg.task = "obb"  # OBB 旋转框任务 / OBB oriented bounding box task
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
    """复用 ultralytics 官方 dota8-multispectral 数据管道 / Reuse ultralytics official dota8-multispectral data pipeline."""
    data = det.get_data_dict(DATA_YAML, "obb")
    if calibration:
        return _build_loaders(batch_size, num_workers, data, calibration=True), data
    train_loader, val_loader, _, _ = _build_loaders(batch_size, num_workers, data)
    return train_loader, val_loader, data


class _LossShim:
    """v8OBBLoss 只需要 model.args / model.model[-1] / model.parameters() /
    v8OBBLoss only needs model.args / model.model[-1] / model.parameters()."""

    def __init__(self, detect_head, epochs):
        # angle=1.0：v8OBBLoss 的角度损失增益 / angle=1.0: angle loss gain required by v8OBBLoss
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, angle=1.0, epochs=epochs)
        self.model = [None] * 23 + [detect_head]
        self.class_weights = None

    def parameters(self):
        return self.model[-1].parameters()


def build_criterion(model, epochs):
    return v8OBBLoss(_LossShim(model.model[-1], epochs), tal_topk=10)


IOU_VECTOR = torch.linspace(0.5, 0.95, 10)


def _match_predictions(pred_labels, pred_bboxes, gt_labels, gt_bboxes, iou_vector):
    """复刻 ultralytics OBBValidator._process_batch：用 batch_probiou 在 10 个 IoU 阈值上匹配旋转框 /
    Replicate ultralytics OBBValidator._process_batch: match rotated boxes at 10 IoU thresholds with batch_probiou.

    pred_bboxes / gt_bboxes 均为 xywhr（5 列，像素单位，角度弧度） /
    both pred_bboxes and gt_bboxes are xywhr (5 cols, pixel units, angle in radians).
    """
    iou = batch_probiou(gt_bboxes, pred_bboxes)
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
    OBB：bboxes 为 xywhr（5 列），plot_images 自动检测并画旋转框 /
    OBB: bboxes are xywhr (5 cols), plot_images auto-detects and draws rotated boxes;
    多光谱图像自动截取前 3 通道显示 / multispectral images auto-cropped to first 3 channels.
    """
    os.makedirs(viz_dir, exist_ok=True)
    bs = batch["img"].shape[0]
    bi = viz_state["batch"]
    # GT 拼图（labels）：cls + 归一化 xywhr 框 + batch_idx，与官方 plot_val_samples 一致 /
    # GT mosaic (labels): cls + normalized xywhr boxes + batch_idx, same as official plot_val_samples
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
    # 预测拼图（preds）：rotated NMS 输出为 [x,y,w,h,conf,cls,angle]，取 xywh+angle 拼成 xywhr /
    # Prediction mosaic (preds): rotated NMS outputs [x,y,w,h,conf,cls,angle]; take xywh+angle as xywhr
    if any(p.shape[0] for p in predictions):
        plot_images(
            labels={
                "cls": torch.cat([p[:, 5] for p in predictions]),
                "conf": torch.cat([p[:, 4] for p in predictions]),
                "bboxes": torch.cat(
                    [torch.cat([p[:, :4], p[:, 6:7]], dim=-1) for p in predictions]
                ),
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
    """在验证集上计算 OBB mAP50 / mAP50-95（rotated NMS + batch_probiou + ap_per_class，与官方 OBBValidator 一致）。
    若 viz_dir 不为空，额外把前 viz_max 张验证图的 GT/预测拼图保存到 {viz_dir}/fvisualize/ /
    Compute OBB mAP50 / mAP50-95 on val set (rotated NMS + batch_probiou + ap_per_class, consistent with official OBBValidator).
    If viz_dir is set, additionally save GT/prediction mosaics of the first viz_max val images into {viz_dir}/fvisualize/."""
    model.eval()
    stats_conf, stats_pcls, stats_tcls, stats_tp = [], [], [], []
    names = data.names if hasattr(data, 'names') else data["names"]
    nc = len(names)  # OBB 输出含角度列，NMS 必须显式传 nc / OBB output has angle column, NMS needs explicit nc
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
            nc=nc,  # 显式类别数，否则角度列会被误当类别 / explicit nc, otherwise the angle column is mistaken for a class
            multi_label=True,
            agnostic=False,
            max_det=300,
            rotated=True,  # OBB 旋转框 NMS（fast_nms + batch_probiou） / rotated box NMS (fast_nms + batch_probiou)
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
            gt_boxes = batch["bboxes"][index]  # 归一化 xywhr（5 列） / normalized xywhr (5 cols)
            if gt_cls.shape[0]:
                # 仅 xywh 乘图像尺寸，角度保持弧度不变 / scale only xywh to pixels, angle stays in radians
                scale = torch.tensor(image_size)[[1, 0, 1, 0]]
                gt_boxes = torch.cat([gt_boxes[:, :4] * scale, gt_boxes[:, 4:5]], dim=1)

            # rotated NMS 输出 [x,y,w,h,conf,cls,angle]，匹配时取 xywhr /
            # rotated NMS outputs [x,y,w,h,conf,cls,angle]; take xywhr for matching
            pred_boxes = torch.cat([pred[:, :4], pred[:, 6:7]], dim=1)
            true_positive = torch.zeros(pred.shape[0], IOU_VECTOR.numel(), dtype=torch.bool)
            if pred.shape[0] and gt_cls.shape[0]:
                true_positive = _match_predictions(
                    pred[:, 5].cpu(), pred_boxes.cpu(), gt_cls, gt_boxes, IOU_VECTOR
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


load_checkpoint = det.load_checkpoint
save_checkpoint = det.save_checkpoint


def load_pretrained(model, path=None, scale=det.DEFAULT_SCALE):
    """加载官方 yolo26{scale}-obb.pt 权重；自动跳过 one2one_* 双头权重和 shape mismatch /
    Load official yolo26{scale}-obb.pt weights; auto-skip one2one_* dual-head weights and shape mismatch.

    本地缺失的官方权重（如 yolo26s-obb.pt）会按 ultralytics 方式自动下载；仅离线或自定义路径缺失时才从头训练 /
    Missing official weights (e.g. yolo26s-obb.pt) are auto-downloaded like ultralytics; random init only when offline or custom path missing.
    """
    if path is None:
        path = os.path.join(ULTRA_DIR, f"{det.base_name(scale, 'obb')}.pt")
    # 官方权重名缺失时自动下载（同 ultralytics）；自定义路径缺失则从头训练 /
    # Auto-download official asset names (like ultralytics); missing custom path -> train from scratch
    path = det.resolve_pretrained_path(path)
    if path is None:
        print(f"[Pretrain] [warn] 未找到预训练权重，{det.model_name(scale)}-obb 将从头训练")
        return model
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].float().state_dict()

    # 过滤 shape mismatch 的 key（首层 Conv 3→10 通道 / Detect head nc 变化等） /
    # Filter shape mismatch keys (first Conv 3->10 channels / Detect head nc changes etc.)
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
        print(f"  ⚠️  跳过 shape mismatch: {len(skipped_shape)} 参数 (首层 3→10 通道 / nc 变化)")
    if missing_unexpected:
        print(f"  ⚠️  missing/unexpected: {len(missing_unexpected)} 参数")
    return model


copy_float_to_quant = det.copy_float_to_quant


def build_float_model(quant_model, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
    """把量化模型的反量化权重灌进同结构的干净浮点模型（用于导出纯浮点 ONNX） /
    Inject dequantized weights from quant model into a clean float model of same architecture (for pure-float ONNX export)."""
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_state = quant_model.state_dict()
    for name, module in quant_model.named_modules():
        if quant_pkg.is_weight_quant_module(module):
            quant_state[f"{name}.weight"] = quant_pkg.dequantized_weight(module)

    float_model = FloatYOLO26OBB(nc=nc, scale=scale)
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


collect_quant_params = quant_pkg.collect_quant_params


def export_onnx(float_model, onnx_path, opset=16, imgsz=None):
    """导出 ONNX，输入形状完全固定为 [1, C, H, W]（无 dynamic_axes，图尺寸清晰可见） /
    Export ONNX with fully static input shape [1, C, H, W] (no dynamic_axes, graph dimensions clearly visible).

    C 取模型首层输入通道（多光谱为 10） / C equals model first-layer input channels (10 for multispectral).

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
    ch_in = getattr(float_model, "ch_in", IN_CHANNELS)
    dummy = torch.randn(1, ch_in, H, W, device=model_device)
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

    try:
        ort = importlib.import_module("onnxruntime")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        torch.manual_seed(0)
        float_model.cpu().eval()
        x = torch.randn(1, getattr(float_model, "ch_in", IN_CHANNELS), IMGSZ, IMGSZ)
        with torch.no_grad():
            y_torch = float_model(x).numpy()
        y_onnx = session.run(["preds"], {"images": x.numpy()})[0]
        max_diff = float(np.abs(y_torch - y_onnx).max())
        ref_mag = float(np.abs(y_torch).max())
        print(f"      onnxruntime vs PyTorch 最大绝对误差: {max_diff:.3e}（参考幅度 {ref_mag:.3e}）")
        # 输出含大数量级解码坐标（如 0~imgsz 的 box 值），纯绝对阈值过严： /
        # Output contains large-magnitude decoded coordinates (e.g. 0~imgsz box values); pure abs threshold too strict:
        # 改为 max(1e-3 绝对, 1e-5 相对)，仍足以抓住导出结构错误 / change to max(1e-3 abs, 1e-5 rel), still sufficient to catch export structural errors
        assert max_diff < max(1e-3, 1e-5 * ref_mag), (
            f"ONNX 数值误差过大: {max_diff}（参考幅度 {ref_mag:.3e}）"
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
    quant_checkpoint = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'obb')}.pth")
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
    float_checkpoint = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'obb')}_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'obb')}_float.onnx")
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
    """官方 BaseTrainer 训练循环复刻：线性 lr 衰减 + warmup（lr 与梯度累积同步插值） /
    Official BaseTrainer training loop replica: linear lr decay + warmup (lr interpolated
    synchronously with gradient accumulation)
    + 梯度累积到 nbs + 梯度裁剪(10.0) + EMA 更新 /
    + gradient accumulation to nbs + gradient clipping (10.0) + EMA update.

    criterion 返回已乘 batch_size 的损失向量（官方直接 backward，不除以 batch） /
    criterion returns loss vector already multiplied by batch_size (official backward directly, no batch divide)
    与未缩放的 items dict；loss 项数按 criterion.loss_names 泛化（OBB 为 box/cls/l1/angle 四项） /
    and unscaled items dict; loss item count generalized via criterion.loss_names (4 for OBB: box/cls/l1/angle).
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

    loss_names = list(criterion.loss_names)
    total_steps = nb if max_batches is None else min(nb, max_batches)
    loss_items_sum = torch.zeros(len(loss_names))
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
            [loss_items[name].detach().cpu() for name in loss_names]
        )

    avg_items = (loss_items_sum / total_steps).tolist()
    return float(sum(avg_items)), dict(zip(loss_names, avg_items))


def float_train(batch_size=16, lr=None, epochs=100, num_classes=NUM_CLASSES, num_workers=2,
                max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                resume=False):
    print(f"========== Float training ({det.model_name(scale)}-obb / dota8-multispectral OBB) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr(num_classes)
    print(
        f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4（官方 optimizer=auto 配方）"
        " | warmup 3ep | 梯度累积 nbs=64 | EMA | close_mosaic=10"
    )

    data = det.get_data_dict(DATA_YAML, "obb")
    train_loader, val_loader, cfg, train_set = _build_loaders(batch_size, num_workers, data)
    nb = len(train_loader)

    float_model = FloatYOLO26OBB(num_classes, scale=scale).to(device)
    det.model_info(float_model, imgsz=IMGSZ)
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
    checkpoint_path = os.path.join(model_dir, f"{det.base_name(scale, 'obb')}_best.pth")
    last_checkpoint = os.path.join(model_dir, f"{det.base_name(scale, 'obb')}_last.pth")

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

        train_loss, loss_items = train_one_epoch(
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
            "scale": det.get_scale(scale),
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            **{k: float(v) for k, v in loss_items.items()},
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

        loss_str = " ".join(f"{k.replace('_loss', '')}:{v:.3f}" for k, v in loss_items.items())
        print(
            f"[Float] Epoch [{epoch + 1}/{epochs}] | loss:{train_loss:.4f} "
            f"({loss_str}) | "
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


# float 范围初始化 / 安全网校准 / minmax 赋值等校准实现全部复用 det 模块（obb 无特殊逻辑） /
# float-range init / safety-net calibration / minmax assign implementations all reused from det module (no obb-specific logic)
calibrate_quantizer = det.calibrate_quantizer


def PTQ_calibration(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                    calibration_batches=20, num_workers=2, max_eval_batches=None,
                    scale=det.DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== PTQ calibration ({det.model_name(scale)}-obb / {tag}) ==========")
    print(f"Device: {device}")

    calibration_loader, data = get_dataloaders(batch_size, num_workers, calibration=True)
    _, val_loader, _ = get_dataloaders(batch_size, num_workers)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth（与 QAT 命名一致），fallback 到旧版无后缀 .pth / Prefer _best.pth (consistent with QAT naming), fallback to legacy no-suffix .pth
    float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'obb')}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'obb')}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatYOLO26OBB(num_classes, scale=scale).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = QuantYOLO26OBB(num_classes, scale=scale).to(device)
    det.model_info(ptq_model, imgsz=IMGSZ)
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
    print(f"[PTQ] 重载 checkpoint 后 mAP50:{final_metrics['map50']:.4f} mAP50-95:{final_metrics['map']:.4f}")
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=16, lr=None, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                 resume=False):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training ({det.model_name(scale)}-obb / {tag}) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr(num_classes) * 0.1  # QAT 微调用 float auto lr 的 1/10 / QAT fine-tuning uses 1/10 of float auto lr
    print(f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4 | warmup 3ep | 梯度累积 nbs=64")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)
    nb = len(train_loader)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    ptq_checkpoint = os.path.join(quant_dir, f"ptq_{tag}_{det.base_name(scale, 'obb')}.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = QuantYOLO26OBB(num_classes, scale=scale).to(device)
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
    best_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'obb')}_best.pth")
    last_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'obb')}_last.pth")

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
        train_loss, loss_items = train_one_epoch(
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
            "scale": det.get_scale(scale),
            "quant_method": tag,
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            **{k: float(v) for k, v in loss_items.items()},
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

        loss_str = " ".join(f"{k.replace('_loss', '')}:{v:.3f}" for k, v in loss_items.items())
        print(
            f"[QAT-{tag}] Epoch [{epoch + 1}/{epochs}] | loss:{train_loss:.4f} "
            f"({loss_str}) | "
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
                      num_workers=2, max_eval_batches=None, scale=det.DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== Float vs QAT precision ({det.model_name(scale)}-obb / dota8-multispectral OBB / {tag}) ==========")
    print(f"Device: {device}")

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth，fallback 到无后缀 / Prefer _best.pth, fallback to no-suffix
    float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'obb')}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'obb')}.pth")
    qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'obb')}_best.pth")
    if not os.path.exists(qat_checkpoint):
        qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'obb')}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26OBB(num_classes, scale=scale).to(device)
    float_model, float_meta = load_checkpoint(float_model, float_checkpoint, return_meta=True)
    float_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix="compare_float"
    )

    qat_model = QuantYOLO26OBB(num_classes, scale=scale).to(device)
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
        description="YOLO26-obb: 浮点训练(加载 yolo26{scale}-obb.pt，缺失则从头训练) -> PTQ -> QAT -> mAP 对比，模型尺度与量化方法可选"
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
    # 冒烟/快速验证用：每个 epoch / 评估最多跑多少个 batch，默认不限制（完整训练） / Smoke test / quick validation: max batches per epoch / eval, default unlimited (full training)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--data",
        default=DATA_YAML,
        help="数据集 yaml 路径或数据集目录（默认自动下载 dota8-multispectral.yaml 到 dataset/；RGB OBB 可用 dota8.yaml 或手动指定其他目录）",
    )
    parser.add_argument("--resume", action="store_true",
                        help="从 _last.pth checkpoint 接续训练")
    return parser


if __name__ == "__main__":
    quant_pkg.install_print_timestamp()  # print 加分钟级时间戳 / minute-precision timestamp for print
    args = build_arg_parser().parse_args()

    # 根据 --data yaml 动态覆盖全局 NUM_CLASSES / DATA_YAML / IN_CHANNELS /
    # Dynamically override global NUM_CLASSES / DATA_YAML / IN_CHANNELS per --data yaml
    DATA_YAML = args.data  # 复用变量名；函数内部引用它 / Reuse variable name; referenced inside functions
    try:
        _tmp = det.get_data_dict(args.data, "obb")
        NUM_CLASSES = len(_tmp["names"])
        IN_CHANNELS = int(_tmp.get("channels", 3))  # 多光谱通道数自适应 / adapt multispectral channels
    except Exception:
        pass  # yaml 解析失败则保留默认 15 类 / 10 通道 / Keep default 15 classes / 10 channels if yaml parsing fails
    print(f"数据集: {args.data} | 类别数: {NUM_CLASSES} | 输入通道: {IN_CHANNELS}")

    scale = det.get_scale(args.model)
    print(f"Python: {sys.executable}")
    print(f"torch: {torch.__version__} | Device: {device}")
    print(f"模型: {det.model_name(scale)}-obb | 量化方法: {args.quant} | 阶段: {args.stage}")

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
