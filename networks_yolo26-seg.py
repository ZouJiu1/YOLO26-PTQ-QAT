"""YOLO26-seg 实例分割网络的量化感知训练流程（coco128-seg，nc=80，尺度可选 n/s/m/l/x） / Quantization-aware training pipeline for YOLO26-seg instance segmentation network (coco128-seg, nc=80, scales selectable n/s/m/l/x).

流程 / Pipeline:
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练 / float training（默认加载 / default load yolo26{scale}-seg.pt 预训练权重 / pretrained weights）
    PTQ_calibration()                               # 训练后量化校准 / post-training quantization calibration
    QAT_training(lr=qat_lr, epochs=qat_epochs)      # 量化感知训练 / quantization-aware training
    compare_precision()                             # 浮点 vs QAT 的 box/mask mAP 对比 / box/mask mAP comparison between float and QAT

模型尺度用 --model yolo26n|yolo26s|yolo26m|yolo26l|yolo26x 选择（默认 yolo26n） / Model scale selected via --model yolo26n|yolo26s|yolo26m|yolo26l|yolo26x (default yolo26n).
backbone / neck（层 0-22）与 networks_yolo26-detect.py 完全一致，直接复用 / backbone/neck (layers 0-22) identical to networks_yolo26-detect.py, directly reused;
分割头为 ultralytics yolo26-seg.yaml 的 Segment26（层 23）/ segmentation head is Segment26 from ultralytics yolo26-seg.yaml (layer 23):
    cv2/cv3 检测框/分类支路与检测模型相同 / cv2/cv3 bounding box/classification branches same as detection model; 额外的 cv4 输出 32 个 mask 系数 / additional cv4 outputs 32 mask coefficients;
    Proto26 融合 P3/P4/P5 多尺度特征 / Proto26 fuses P3/P4/P5 multi-scale features, 生成 (B,32,160,160) mask prototypes / generates (B,32,160,160) mask prototypes,
    并带一条 semantic 辅助分割支路（仅训练时返回，用于 sem_loss） / with an auxiliary semantic segmentation branch (returned only during training, used for sem_loss).
训练损失使用官方 v8SegmentationLoss（TaskAlignedAssigner + BCEDice）/ Training loss uses official v8SegmentationLoss (TaskAlignedAssigner + BCEDice);
推理输出 (检测张量[xywh+scores+mask系数], protos) / Inference outputs (detection tensor [xywh+scores+mask coefficients], protos), 经 NMS + process_mask 得到实例掩码 / instance masks obtained via NMS + process_mask,
按 mask_iou 在 10 个 IoU 阈值上计算 mask mAP / mask mAP computed via mask_iou at 10 IoU thresholds.
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

# ultralytics 提供数据管道 / 损失 / 解码 / NMS / mask 处理 / mAP / ultralytics provides data pipeline / loss / decode / NMS / mask processing / mAP
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh, process_mask
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou, mask_iou
from ultralytics.utils.torch_utils import model_info
from ultralytics.utils.plotting import plot_images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR_BASE = os.path.join(BASE_DIR, "model", "yolo26-seg")
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
# 默认官方数据集（首次运行自动下载到 dataset/；--data 可手动指定其他目录） /
# Default official dataset (auto-downloaded to dataset/ on first run; --data for a manual dir)
DATA_YAML = det.DEFAULT_DATA_YAML["seg"]

IMGSZ = 640
NUM_CLASSES = 80
NM = 32          # mask 系数个数 / number of mask coefficients（官方 yolo26-seg.yaml 各尺度固定 32，不随 width 缩放 / official yolo26-seg.yaml fixed 32 for all scales, not scaled by width）
# npr（mask 系数头隐层通道，yaml 基准 256）按尺度缩放 / npr (mask coefficient head hidden channel, yaml baseline 256) scales by scale:
#   make_divisible(min(256, max_ch) * width, 8)（n→64 / s→128 / m→256 / l→256 / x→384），
#   由 det.scaled_channels(256, scale) 计算（见 ultralytics/nn/tasks.py parse_model） / computed by det.scaled_channels(256, scale) (see ultralytics/nn/tasks.py parse_model).
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
    """切换量化后端（必须在构建 QuantYOLO26Seg 之前调用）。 / Switch quantization backend (must be called before constructing QuantYOLO26Seg)."""
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


# ============================== 分割头组件 / Segmentation Head Components ==============================


class Proto26(nn.Module):
    """复刻 ultralytics.nn.modules.block.Proto26（yolo26-seg, nm=32，npr 随尺度缩放）。 / Reproduces ultralytics.nn.modules.block.Proto26 (yolo26-seg, nm=32, npr scales with model scale).

    多尺度融合：P3 特征 + 上采样的 P4/P5 细化特征（两次残差加法）， / Multi-scale fusion: P3 features + upsampled P4/P5 refined features (two residual additions),
    经 feat_fuse 后送入 Proto（Conv3x3 -> ConvTranspose2d 上采样 2x -> Conv3x3 -> Conv1x1）， / After feat_fuse, fed into Proto (Conv3x3 -> ConvTranspose2d upsample 2x -> Conv3x3 -> Conv1x1),
    输出 (B, nm=32, 160, 160) prototypes；训练时额外返回 semseg 的 (B, nc, 160, 160) 语义图。 / Outputs (B, nm=32, 160, 160) prototypes; during training additionally returns semseg's (B, nc, 160, 160) semantic map.

    属性命名与官方完全一致（cv1/upsample/cv2/cv3/feat_refine/feat_fuse/semseg）， / Attribute naming matches official exactly (cv1/upsample/cv2/cv3/feat_refine/feat_fuse/semseg),
    可直接加载 yolo26{scale}-seg.pt 中 model.23.proto.* 的权重。 / Can directly load weights from model.23.proto.* in yolo26{scale}-seg.pt.
    """

    def __init__(self, ch=(64, 128, 256), npr=64, nm=NM, nc=NUM_CLASSES, quant=False):
        super().__init__()
        # Proto(npr, npr, nm): cv1/cv2 为 npr 通道，upsample 为 2x 转置卷积，cv3 输出 nm 通道 / Proto(npr, npr, nm): cv1/cv2 have npr channels, upsample is 2x transposed conv, cv3 outputs nm channels
        self.cv1 = det.Conv(npr, npr, k=3, quant=quant)
        if quant:
            self.upsample = QuantConvTranspose2d(
                npr, npr, kernel_size=2, stride=2, padding=0, bias=True,
                **det._quant_layer_kwargs(),
            )
        else:
            self.upsample = nn.ConvTranspose2d(npr, npr, 2, 2, 0, bias=True)
        self.cv2 = det.Conv(npr, npr, k=3, quant=quant)
        self.cv3 = det.Conv(npr, nm, quant=quant)

        self.feat_refine = nn.ModuleList(det.Conv(x, ch[0], k=1, quant=quant) for x in ch[1:])
        self.refine_adds = (
            nn.ModuleList(det.QuantAdd(a_bits=det.QUANT_CFG.a_bits, quant_inference=True) for _ in ch[1:])
            if quant else None
        )
        self.feat_fuse = det.Conv(ch[0], npr, k=3, quant=quant)
        self.semseg = nn.Sequential(
            det.Conv(ch[0], npr, k=3, quant=quant),
            det.Conv(npr, npr, k=3, quant=quant),
            QuantConv2d(npr, nc, 1, **det._quant_layer_kwargs())
            if quant else nn.Conv2d(npr, nc, 1),
        )

    def forward(self, x):
        feat = x[0]
        for i, refine in enumerate(self.feat_refine):
            up_feat = F.interpolate(refine(x[i + 1]), scale_factor=2 ** (i + 1), mode="nearest")
            if self.refine_adds is not None:
                feat = self.refine_adds[i](feat, up_feat)
            else:
                feat = feat + up_feat
        p = self.cv3(self.cv2(self.upsample(self.cv1(self.feat_fuse(feat)))))
        if self.training:
            semantic = self.semseg(feat)
            return p, semantic
        return p


class Segment(det.Detect):
    """YOLO26-seg 分割头（单 one2many 头，reg_max=1）。 / YOLO26-seg segmentation head (single one2many head, reg_max=1).

    训练时返回 dict(boxes/scores/feats/mask_coefficient/proto) 供 v8SegmentationLoss 使用， / During training returns dict(boxes/scores/feats/mask_coefficient/proto) for v8SegmentationLoss,
    proto 为 (protos, semantic) 二元组；评估时返回 (检测张量, protos)： / proto is a (protos, semantic) tuple; during evaluation returns (detection tensor, protos):
    检测张量通道顺序 xywh(4) + sigmoid 分类(nc) + 原始 mask 系数(nm)，与 NMS 的 / Detection tensor channel order xywh(4) + sigmoid classification(nc) + raw mask coefficients(nm), consistent with NMS's
    split((4, nc, extra)) / process_mask 约定一致。 / split((4, nc, extra)) / process_mask convention.
    """

    def __init__(self, nc=NUM_CLASSES, nm=NM, npr=64, reg_max=1, ch=(64, 128, 256), quant=False):
        super().__init__(nc=nc, reg_max=reg_max, ch=ch, quant=quant)
        hq = self.head_quant  # 混合量化时整个分割头（含 Proto/cv4/semseg）保持 FP32 / Under mixed quant the whole seg head (Proto/cv4/semseg) stays FP32
        self.nm = nm
        self.npr = npr
        # 属性名必须为 proto，与官方 Segment26 / yolo26n-seg.pt 的 model.23.proto.* 对齐 / Attribute name must be proto, aligned with official Segment26 / model.23.proto.* in yolo26n-seg.pt
        self.proto = Proto26(ch, npr, nm, nc, quant=hq)
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                det.Conv(x, c4, 3, quant=hq),
                det.Conv(c4, c4, 3, quant=hq),
                QuantConv2d(c4, self.nm, 1, **det._quant_layer_kwargs())
                if hq else nn.Conv2d(c4, self.nm, 1),
            )
            for x in ch
        )
        self._anchors = None
        self._strides_tensor = None
        self._feat_shape = None

    def forward(self, x):
        bs = x[0].shape[0]
        boxes = torch.cat(
            [self.cv2[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1
        )
        scores = torch.cat(
            [self.cv3[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        mask_coefficient = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], dim=-1
        )

        proto = self.proto(x)
        if self.training:
            return {
                "boxes": boxes,
                "scores": scores,
                "feats": x,
                "mask_coefficient": mask_coefficient,
                "proto": proto,
            }

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
        dbox = dist2bbox(boxes, self._anchors.unsqueeze(0), xywh=True, dim=1)
        dbox = dbox * self._strides_tensor
        det = torch.cat((dbox, scores.sigmoid(), mask_coefficient), 1)
        # eval 时 Proto26 只返回 protos 张量 / In eval mode Proto26 only returns protos tensor
        proto_p = proto[0] if isinstance(proto, tuple) else proto
        return det, proto_p


# ============================== 网络主体 / Network Body ==============================


class YOLO26Seg(det.YOLO26):
    """yolo26n-seg（scale=n）分割网络。quant=False 浮点模型，True 为伪量化模型。 / yolo26n-seg (scale=n) segmentation network. quant=False float model, True fake-quantized model.

    层 0-22 与检测网络一致（直接构建后替换层 23），topology 编号/save 集合不变， / Layers 0-22 identical to detection network (built directly then replace layer 23); topology indices/save set unchanged,
    因此 yolo26n-seg.pt 的 state_dict（model.0.* ~ model.23.*）可直接加载， / so state_dict of yolo26n-seg.pt (model.0.* ~ model.23.*) can be loaded directly,
    end2end 双头 one2one_* 参数被跳过。 / end2end dual-head one2one_* parameters are skipped.
    """

    def __init__(self, nc=NUM_CLASSES, quant=False, scale=det.DEFAULT_SCALE):
        # 先构建检测网络得到完整的 0-22 backbone/neck，再把层 23 的 Detect 换成 Segment / First build detection network to get complete 0-22 backbone/neck, then replace layer 23's Detect with Segment
        super().__init__(nc=nc, quant=quant, scale=scale)
        head = Segment(
            nc=nc, nm=NM, npr=det.scaled_channels(256, scale),
            reg_max=1, ch=det.detect_head_channels(scale), quant=quant,
        )
        det._tag(head, 23, [16, 19, 22])
        self.model[-1] = head
        self._register_quantizer_buffers()
        self._initialize_head()


class FloatYOLO26Seg(YOLO26Seg):
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
        super().__init__(nc=nc, quant=False, scale=scale)


class QuantYOLO26Seg(YOLO26Seg):
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
        super().__init__(nc=nc, quant=True, scale=scale)


# ============================== 数据 / 损失 / 评估 / Data / Loss / Evaluation ==============================


def _make_cfg(num_workers):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = IMGSZ
    cfg.task = "segment"  # YOLODataset 据此生成 polygon -> bitmap masks / sem_masks / YOLODataset generates polygon -> bitmap masks / sem_masks accordingly
    cfg.workers = num_workers
    return cfg


def _build_loaders(batch_size, num_workers, data, calibration=False):
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
    """复用 ultralytics 官方 coco128-seg 数据管道（含 masks / sem_masks）。 / Reuse ultralytics official coco128-seg data pipeline (includes masks / sem_masks)."""
    data = det.get_data_dict(DATA_YAML, "seg")
    if calibration:
        return _build_loaders(batch_size, num_workers, data, calibration=True), data
    train_loader, val_loader, _, _ = _build_loaders(batch_size, num_workers, data)
    return train_loader, val_loader, data


class _LossShim:
    """v8SegmentationLoss 只需要 model.args / model.model[-1] / model.parameters()。 / v8SegmentationLoss only needs model.args / model.model[-1] / model.parameters()."""

    def __init__(self, segment_head, epochs):
        self.args = SimpleNamespace(
            box=7.5, cls=0.5, dfl=1.5, epochs=epochs, overlap_mask=True
        )
        self.model = [None] * 23 + [segment_head]
        self.class_weights = None

    def parameters(self):
        return self.model[-1].parameters()


def build_criterion(model, epochs):
    return v8SegmentationLoss(_LossShim(model.model[-1], epochs), tal_topk=10)


IOU_VECTOR = torch.linspace(0.5, 0.95, 10)


def _match_predictions(pred_labels, pred_bboxes, gt_labels, gt_bboxes, iou_vector):
    """复刻 ultralytics BaseValidator._process_batch：在 10 个 IoU 阈值上匹配预测与 GT。 / Reproduces ultralytics BaseValidator._process_batch: match predictions with GT at 10 IoU thresholds."""
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


def _match_masks(pred_labels, pred_masks, gt_labels, gt_masks, iou_vector):
    """与框匹配同构，但 IoU 用 mask_iou（pred_masks/gt_masks 均已二值化）。 / Structurally same as box matching, but IoU uses mask_iou (pred_masks/gt_masks are both binarized)."""
    # mask_iou 只接受二维 (N, H*W)，先拉平 / mask_iou only accepts 2D (N, H*W), flatten first
    n_gt, n_pred = gt_masks.shape[0], pred_masks.shape[0]
    gt_flat = gt_masks.reshape(n_gt, -1).float()
    pred_flat = pred_masks.reshape(n_pred, -1).float()
    iou = mask_iou(gt_flat, pred_flat)
    correct = torch.zeros(pred_masks.shape[0], iou_vector.numel(), dtype=torch.bool)
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


def _maybe_visualize(batch, predictions, protos, names, viz_dir, viz_prefix, viz_state):
    """用 ultralytics 官方 plot_images 保存本批的 GT 拼图与预测拼图（含分割掩码，val_batch 风格） /
    Use official ultralytics plot_images to save GT and prediction mosaics of this batch (with seg masks, val_batch style).

    viz_state 跨 batch 记录已保存图片数 / viz_state tracks saved image count across batches.
    """
    os.makedirs(viz_dir, exist_ok=True)
    bs = batch["img"].shape[0]
    bi = viz_state["batch"]
    image_size = tuple(batch["img"].shape[2:])
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
    # 预测拼图（preds）：与官方 SegmentationValidator.plot_predictions 一致，含 masks（process_mask upsample 到图像分辨率） /
    # Prediction mosaic (preds): same as official SegmentationValidator.plot_predictions, incl. masks (process_mask upsampled to image size)
    if any(p.shape[0] for p in predictions):
        mask_list = [
            process_mask(protos[i], p[:, 6:6 + NM], p[:, :4], shape=image_size, upsample=True)
            for i, p in enumerate(predictions) if p.shape[0]
        ]
        plot_images(
            labels={
                "cls": torch.cat([p[:, 5] for p in predictions]),
                "conf": torch.cat([p[:, 4] for p in predictions]),
                "bboxes": xyxy2xywh(torch.cat([p[:, :4] for p in predictions])),
                "batch_idx": torch.cat(
                    [torch.full((p.shape[0],), i) for i, p in enumerate(predictions)]
                ),
                **({"masks": torch.cat(mask_list)} if mask_list else {}),
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
    """在 coco128-seg 验证集上同时计算 box mAP 与 mask mAP。若 viz_dir 不为空，
    额外把前 viz_max 张验证图的 GT/预测拼图（含掩码）保存到 {viz_dir}/fvisualize/。 / Compute both box mAP and mask mAP on coco128-seg validation set.
    If viz_dir is set, additionally save GT/prediction mosaics (with masks) of the first viz_max val images into {viz_dir}/fvisualize/.

    与官方 SegmentationValidator 一致：NMS 后的 mask 系数与 protos 相乘， / Consistent with official SegmentationValidator: post-NMS mask coefficients multiplied with protos,
    在 proto 分辨率（160x160，图像 /4）上二值化并按框裁剪；GT overlap 索引掩码 / Binarize at proto resolution (160x160, image /4) and crop by boxes; GT overlap index masks
    拆分后插值到同一分辨率，用 mask_iou 在 10 个阈值上匹配。 / split then interpolated to same resolution, matched via mask_iou at 10 thresholds.
    """
    model.eval()
    stats_conf, stats_pcls, stats_tcls = [], [], []
    stats_tp_box, stats_tp_mask = [], []
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
        det, protos = model(images)
        predictions = non_max_suppression(
            det,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            multi_label=True,
            agnostic=False,
            max_det=300,
            nc=NUM_CLASSES,  # 显式指定，否则 nm=32 个 mask 系数会被当成类别 / Specify explicitly, otherwise nm=32 mask coefficients would be treated as classes
        )
        image_size = batch["img"].shape[2:]
        mh, mw = protos.shape[2:]

        # 每次评估都可视化前几批（失败仅告警，绝不影响评估） / visualize first batches on every evaluate (failure only warns, never breaks eval)
        if viz_out and viz_state["saved"] < viz_max:
            try:
                _maybe_visualize(batch, predictions, protos, names, viz_out, viz_prefix, viz_state)
            except Exception as exc:
                print(f"      [viz] 可视化保存失败（仅告警）: {exc} / visualization save failed (warn only): {exc}")

        for sample_index, pred in enumerate(predictions):
            index = batch["batch_idx"] == sample_index
            gt_cls = batch["cls"][index].squeeze(-1)
            gt_boxes = batch["bboxes"][index]
            nl = gt_cls.shape[0]
            if nl:
                gt_boxes = xywh2xyxy(gt_boxes) * torch.tensor(image_size)[[1, 0, 1, 0]]

            # GT 实例掩码：overlap_mask=True 时每张图是一张 1..nl 的索引图 / GT instance masks: when overlap_mask=True each image is an index map 1..nl
            gt_mask_i = batch["masks"][sample_index].float().to(device)  # (1, gh, gw)
            if nl:
                gt_masks = gt_mask_i == torch.arange(1, nl + 1, device=device).view(nl, 1, 1)
                gt_masks = gt_masks.float()
                if tuple(gt_masks.shape[-2:]) != (mh, mw):
                    gt_masks = F.interpolate(
                        gt_masks[None], (mh, mw), mode="bilinear", align_corners=False
                    )[0].gt_(0.5)
            else:
                gt_masks = torch.zeros((0, mh, mw), device=device)

            tp_box = torch.zeros(pred.shape[0], IOU_VECTOR.numel(), dtype=torch.bool)
            tp_mask = torch.zeros(pred.shape[0], IOU_VECTOR.numel(), dtype=torch.bool)
            if pred.shape[0] and nl:
                tp_box = _match_predictions(
                    pred[:, 5].cpu(), pred[:, :4].cpu(), gt_cls, gt_boxes, IOU_VECTOR
                )
                coeff = pred[:, 6:6 + NM]
                pred_masks = process_mask(
                    protos[sample_index], coeff, pred[:, :4], shape=tuple(image_size)
                )  # (N, mh, mw) uint8
                tp_mask = _match_masks(
                    pred[:, 5].cpu(),
                    pred_masks.float().cpu(),
                    gt_cls,
                    gt_masks.cpu(),
                    IOU_VECTOR,
                )

            stats_conf.append(pred[:, 4].cpu())
            stats_pcls.append(pred[:, 5].cpu())
            stats_tp_box.append(tp_box)
            stats_tp_mask.append(tp_mask)
            stats_tcls.append(gt_cls)

    conf_all = torch.cat(stats_conf).numpy()
    pred_cls_all = torch.cat(stats_pcls).numpy()
    tcls_all = torch.cat(stats_tcls).numpy()

    def _ap(tp_all):
        if conf_all.shape[0] == 0 or tcls_all.shape[0] == 0:
            return 0.0, 0.0, 0.0, 0.0
        _, _, precision, recall, _, ap, *_ = ap_per_class(
            tp_all, conf_all, pred_cls_all, tcls_all, plot=False, names=names
        )
        return float(ap[:, 0].mean()), float(ap.mean()), float(precision.mean()), float(recall.mean())

    box_map50, box_map, box_p, box_r = _ap(torch.cat(stats_tp_box).numpy())
    mask_map50, mask_map, mask_p, mask_r = _ap(torch.cat(stats_tp_mask).numpy())
    return {
        "map50": box_map50, "map": box_map, "precision": box_p, "recall": box_r,
        "mask_map50": mask_map50, "mask_map": mask_map,
        "mask_precision": mask_p, "mask_recall": mask_r,
    }


# ============================== checkpoint / 权重复制 / 导出 / Checkpoint / Weight Copy / Export ==============================


def load_checkpoint(model, path, return_meta=False):
    return det.load_checkpoint(model, path, return_meta=return_meta)


def save_checkpoint(model, path, **metadata):
    det.save_checkpoint(model, path, **metadata)


def load_pretrained(model, path=None, scale=det.DEFAULT_SCALE):
    """加载官方 yolo26{scale}-seg.pt；单头模型跳过 one2one_* 双头权重。 / Load official yolo26{scale}-seg.pt; single-head model skips one2one_* dual-head weights.

    本地缺失的官方权重（如 yolo26s-seg.pt）会按 ultralytics 方式自动下载；仅离线或自定义路径缺失时才从头训练。 /
    Missing official weights (e.g. yolo26s-seg.pt) are auto-downloaded like ultralytics; random init only when offline or custom path missing.
    """
    if path is None:
        path = os.path.join(ULTRA_DIR, f"{det.base_name(scale, 'seg')}.pt")
    # 官方权重名缺失时自动下载（同 ultralytics）；自定义路径缺失则从头训练 /
    # Auto-download official asset names (like ultralytics); missing custom path -> train from scratch
    path = det.resolve_pretrained_path(path)
    if path is None:
        print(f"[Pretrain] [warn] 未找到预训练权重，{det.model_name(scale)}-seg 将从头训练")
        return model
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].float().state_dict()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    skipped = [k for k in unexpected if "one2one_" in k]
    other_unexpected = [k for k in unexpected if "one2one_" not in k]
    if other_unexpected:
        print(f"[Pretrain] 警告：{len(other_unexpected)} 个未识别权重未加载: {other_unexpected[:3]}")
    print(f"[Pretrain] 已加载 {path}（跳过 {len(skipped)} 个 one2one 双头参数，缺失 {len(missing)} 个）")
    seg_missing = [k for k in missing if "proto" in k or "cv4" in k]
    if seg_missing:
        print(f"[Pretrain] 分割头缺失参数: {seg_missing[:5]}")
    return model


def copy_float_to_quant(float_model, quant_model):
    return det.copy_float_to_quant(float_model, quant_model)


def build_float_model(quant_model, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE, use_clip=False):
    """把量化模型的反量化权重灌进同结构的干净浮点模型（用于导出纯浮点 ONNX）。 / Inject dequantized weights from quantized model into a clean float model of same structure (for exporting pure-float ONNX)."""
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
        output_names=["preds", "proto"],
        # 无 dynamic_axes → 输入输出形状完全固定 / No dynamic_axes → all shapes fully fixed
    )
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
    quant_checkpoint = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'seg')}.pth")
    save_checkpoint(quant_model, quant_checkpoint, **(meta or {}))

    quant_params = collect_quant_params(quant_model)
    json_path = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'seg')}_quant_params.json")
    pth_path = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'seg')}_quant_params.pth")
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
    float_checkpoint = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'seg')}_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(model_dir, f"{prefix}_{det.base_name(scale, 'seg')}_float.onnx")
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
    """与检测版一致的官方训练循环；损失项为 box/seg/cls/l1/sem 五项。 / Official training loop consistent with detection version; loss items are box/seg/cls/l1/sem (five terms)."""
    model.train()
    batch_size = loader.batch_size
    accumulate = max(round(nbs / batch_size), 1)
    warmup_steps = round(min(warmup_epochs, max(epochs - 1, 0)) * nb) if warmup_epochs > 0 else 0
    lf = lambda x_step: max(1 - x_step / epochs, 0) * (1.0 - lrf) + lrf

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lf(epoch)

    total_steps = nb if max_batches is None else min(nb, max_batches)
    loss_items_sum = torch.zeros(5)
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
                loss_items["seg_loss"].detach().cpu(),
                loss_items["cls_loss"].detach().cpu(),
                loss_items["l1_loss"].detach().cpu(),
                loss_items["sem_loss"].detach().cpu(),
            ]
        )

    avg_items = (loss_items_sum / total_steps).tolist()
    return float(sum(avg_items)), avg_items


def float_train(batch_size=16, lr=None, epochs=100, num_classes=NUM_CLASSES, num_workers=2,
                max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                resume=False):
    print(f"========== Float training ({det.model_name(scale)}-seg / coco128-seg) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr()
    print(
        f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4（官方 optimizer=auto 配方）"
        " | warmup 3ep | 梯度累积 nbs=64 | EMA | close_mosaic=10"
    )

    data = det.get_data_dict(DATA_YAML, "seg")
    train_loader, val_loader, cfg, train_set = _build_loaders(batch_size, num_workers, data)
    nb = len(train_loader)

    float_model = FloatYOLO26Seg(num_classes, scale=scale).to(device)
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
    checkpoint_path = os.path.join(model_dir, f"{det.base_name(scale, 'seg')}_best.pth")
    last_checkpoint = os.path.join(model_dir, f"{det.base_name(scale, 'seg')}_last.pth")

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
        print(f"[Float] 从 epoch {start_epoch} 接续训练，当前 best mask mAP50={meta.get('mask_map50', 0):.4f}")
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
        box_loss, seg_loss, cls_loss, l1_loss, sem_loss = items
        metrics = evaluate(
            ema.ema, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
            viz_dir=model_dir, viz_prefix=f"float_ep{epoch + 1:03d}"
        )
        fitness = 0.9 * metrics["mask_map"] + 0.1 * metrics["mask_map50"]

        epoch_meta = {
            "stage": "float",
            "scale": det.get_scale(scale),
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "box_loss": float(box_loss),
            "seg_loss": float(seg_loss),
            "cls_loss": float(cls_loss),
            "l1_loss": float(l1_loss),
            "sem_loss": float(sem_loss),
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
            f"(box:{box_loss:.3f} seg:{seg_loss:.3f} cls:{cls_loss:.3f} sem:{sem_loss:.3f}) | "
            f"box mAP50:{metrics['map50']:.4f} | mask mAP50:{metrics['mask_map50']:.4f} "
            f"mAP50-95:{metrics['mask_map']:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)
    best_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=model_dir, viz_prefix="float_best"
    )
    print(f"[Float] Best checkpoint: {checkpoint_path}（epoch {best_epoch}/{epochs}）")
    print(
        f"[Float] Best box mAP50:{best_metrics['map50']:.4f} | "
        f"mask mAP50:{best_metrics['mask_map50']:.4f} mAP50-95:{best_metrics['mask_map']:.4f}"
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
    print(f"========== PTQ calibration ({det.model_name(scale)}-seg / {tag}) ==========")
    print(f"Device: {device}")

    calibration_loader, data = get_dataloaders(batch_size, num_workers, calibration=True)
    _, val_loader, _ = get_dataloaders(batch_size, num_workers)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth（与 QAT 命名一致），fallback 到旧版无后缀 .pth / Prefer _best.pth (consistent with QAT naming), fallback to old no-suffix .pth
    float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'seg')}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'seg')}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatYOLO26Seg(num_classes, scale=scale).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = QuantYOLO26Seg(num_classes, scale=scale).to(device)
    det.model_info(ptq_model, imgsz=IMGSZ)
    copy_float_to_quant(float_model, ptq_model)

    calibrate_quantizer(ptq_model, calibration_loader, calibration_batches,
                         float_model=float_model)

    metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix=f"ptq_{tag}"
    )
    print(
        f"[PTQ-{tag}] box mAP50:{metrics['map50']:.4f} | "
        f"mask mAP50:{metrics['mask_map50']:.4f} mAP50-95:{metrics['mask_map']:.4f}"
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
        f"[PTQ] 重载 checkpoint 后 mask mAP50:{final_metrics['mask_map50']:.4f} "
        f"mAP50-95:{final_metrics['mask_map']:.4f}"
    )
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=16, lr=None, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                 resume=False):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training ({det.model_name(scale)}-seg / {tag}) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr() * 0.1
    print(f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4 | warmup 3ep | 梯度累积 nbs=64")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)
    nb = len(train_loader)

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    ptq_checkpoint = os.path.join(quant_dir, f"ptq_{tag}_{det.base_name(scale, 'seg')}.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = QuantYOLO26Seg(num_classes, scale=scale).to(device)
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
    best_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'seg')}_best.pth")
    last_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'seg')}_last.pth")

    # resume：从 _last.pth 恢复，接续训练 / Resume: restore from _last.pth, continue training
    start_epoch = 0
    if resume and os.path.exists(last_checkpoint):
        print(f"[QAT-{tag}] Resume from {last_checkpoint}")
        _, meta = load_checkpoint(qat_model, last_checkpoint, return_meta=True)
        start_epoch = meta.get("epoch", 0)
        best_fitness = meta.get("fitness", -1.0)
        best_meta = meta
        best_epoch = start_epoch
        print(f"[QAT-{tag}] 从 epoch {start_epoch} 接续训练，当前 best mAP50={meta.get('mask_map50', 0):.4f}")
    elif resume:
        print(f"[QAT-{tag}] Resume 开启但找不到 {last_checkpoint}，从头训练")

    for epoch in range(start_epoch, epochs):
        train_loss, items = train_one_epoch(
            qat_model, train_loader, criterion, optimizer, epoch, epochs, nb,
            max_batches=max_train_batches,
        )
        box_loss, seg_loss, cls_loss, l1_loss, sem_loss = items
        metrics = evaluate(
            qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
            viz_dir=quant_dir, viz_prefix=f"qat_{tag}_ep{epoch + 1:03d}"
        )
        fitness = 0.9 * metrics["mask_map"] + 0.1 * metrics["mask_map50"]

        epoch_meta = {
            "stage": "qat",
            "scale": det.get_scale(scale),
            "quant_method": tag,
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "box_loss": float(box_loss),
            "seg_loss": float(seg_loss),
            "cls_loss": float(cls_loss),
            "l1_loss": float(l1_loss),
            "sem_loss": float(sem_loss),
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
            f"[QAT-{tag}] Epoch [{epoch + 1}/{epochs}] | loss:{train_loss:.4f} "
            f"(box:{box_loss:.3f} seg:{seg_loss:.3f} cls:{cls_loss:.3f} sem:{sem_loss:.3f}) | "
            f"box mAP50:{metrics['map50']:.4f} | mask mAP50:{metrics['mask_map50']:.4f} "
            f"mAP50-95:{metrics['mask_map']:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    best_metrics = evaluate(
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix=f"qat_{tag}_best"
    )
    print(
        f"[QAT-{tag}] Best epoch:{best_epoch}/{epochs} | "
        f"mask mAP50:{best_metrics['mask_map50']:.4f} mAP50-95:{best_metrics['mask_map']:.4f} | "
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
    print(f"========== Float vs QAT precision ({det.model_name(scale)}-seg / coco128-seg / {tag}) ==========")
    print(f"Device: {device}")

    float_dir = _model_dir_for(scale=scale)
    quant_dir = _model_dir_for(scale=scale, quant_method=quant_method)
    # 优先 _best.pth，fallback 到无后缀 / Prefer _best.pth, fallback to no-suffix
    float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'seg')}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(float_dir, f"{det.base_name(scale, 'seg')}.pth")
    qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'seg')}_best.pth")
    if not os.path.exists(qat_checkpoint):
        qat_checkpoint = os.path.join(quant_dir, f"qat_{tag}_{det.base_name(scale, 'seg')}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26Seg(num_classes, scale=scale).to(device)
    float_model, float_meta = load_checkpoint(float_model, float_checkpoint, return_meta=True)
    float_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches,
        viz_dir=quant_dir, viz_prefix="compare_float"
    )

    qat_model = QuantYOLO26Seg(num_classes, scale=scale).to(device)
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
        f"box mAP50:{float_metrics['map50']:.4f} | "
        f"mask P:{float_metrics['mask_precision']:.4f} R:{float_metrics['mask_recall']:.4f} "
        f"mAP50:{float_metrics['mask_map50']:.4f} mAP50-95:{float_metrics['mask_map']:.4f}"
    )
    print(
        f"[Compare] QAT   | best epoch:{qat_meta.get('epoch', '-')}/"
        f"{qat_meta.get('total_epochs', '-')} | "
        f"box mAP50:{qat_metrics['map50']:.4f} | "
        f"mask P:{qat_metrics['mask_precision']:.4f} R:{qat_metrics['mask_recall']:.4f} "
        f"mAP50:{qat_metrics['mask_map50']:.4f} mAP50-95:{qat_metrics['mask_map']:.4f}"
    )
    print(
        f"[Compare] delta | box mAP50:{qat_metrics['map50'] - float_metrics['map50']:+.4f} | "
        f"mask mAP50:{qat_metrics['mask_map50'] - float_metrics['mask_map50']:+.4f} "
        f"mAP50-95:{qat_metrics['mask_map'] - float_metrics['mask_map']:+.4f}"
    )
    return {"float": float_metrics, "qat": qat_metrics, "float_meta": float_meta, "qat_meta": qat_meta}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="YOLO26-seg: 浮点训练(加载 yolo26{scale}-seg.pt) -> PTQ -> QAT -> box/mask mAP 对比，模型尺度与量化方法可选"
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
    # 量化超参（不再硬编码）：--a-bits/--w-bits/--per-channel/--all-positive / Quant hyperparameters (no longer hardcoded)
    det.add_quant_cfg_args(parser)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--data",
        default=DATA_YAML,
        help="数据集 yaml 路径或数据集目录（默认自动下载 coco8-seg.yaml 到 dataset/；可手动指定其他目录）",
    )
    parser.add_argument("--resume", action="store_true",
                        help="从 _last.pth checkpoint 接续训练")
    return parser


if __name__ == "__main__":
    quant_pkg.install_print_timestamp()  # print 加分钟级时间戳 / minute-precision timestamp for print
    args = build_arg_parser().parse_args()

    # 写入量化超参（det.QUANT_CFG 是所有量化层读取的唯一来源）/ Apply quant hyperparameters (single source read by all quant layers)
    det.apply_quant_cfg_args(args)

    # 根据 --data 动态覆盖全局 NUM_CLASSES 和 DATA_YAML（默认走自动下载） /
    # Override global NUM_CLASSES / DATA_YAML per --data (default auto-downloads)
    DATA_YAML = args.data  # 复用变量名；函数内部引用它 / Reuse variable name; functions reference it internally
    try:
        _tmp = det.get_data_dict(args.data, "seg")
        NUM_CLASSES = len(_tmp["names"])
    except Exception:
        pass  # yaml 解析失败则保留默认 80 / Keep default 80 if yaml parsing fails
    print(f"数据集: {args.data} | 类别数: {NUM_CLASSES}")

    scale = det.get_scale(args.model)
    print(f"Python: {sys.executable}")
    print(f"torch: {torch.__version__} | Device: {device}")
    print(f"模型: {det.model_name(scale)}-seg | 量化方法: {args.quant} | 阶段: {args.stage}")

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
