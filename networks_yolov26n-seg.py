"""YOLO26n-seg 实例分割网络的量化感知训练流程（coco128-seg，nc=80）。

流程：
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练（默认加载 yolo26n-seg.pt 预训练权重）
    PTQ_calibration()                               # 训练后量化校准
    QAT_training(lr=qat_lr, epochs=qat_epochs)      # 量化感知训练
    compare_precision()                             # 浮点 vs QAT 的 box/mask mAP 对比

backbone / neck（层 0-22）与 networks_yolov26n.py 完全一致，直接复用；
分割头为 ultralytics yolo26-seg.yaml 的 Segment26（层 23）：
    cv2/cv3 检测框/分类支路与检测模型相同；额外的 cv4 输出 32 个 mask 系数；
    Proto26 融合 P3/P4/P5 多尺度特征，生成 (B,32,160,160) mask prototypes，
    并带一条 semantic 辅助分割支路（仅训练时返回，用于 sem_loss）。
训练损失使用官方 v8SegmentationLoss（TaskAlignedAssigner + BCEDice）；
推理输出 (检测张量[xywh+scores+mask系数], protos)，经 NMS + process_mask 得到实例掩码，
按 mask_iou 在 10 个 IoU 阈值上计算 mask mAP。
"""

import argparse
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

# 复用检测网络的 backbone/neck 组件与训练基础设施
# （文件名 networks_yolov26n-detect.py 含 '-'，不能直接 import，用文件路径加载）
det_spec = importlib.util.spec_from_file_location(
    "networks_yolov26n_detect",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "networks_yolov26n-detect.py"),
)
det = importlib.util.module_from_spec(det_spec)
det_spec.loader.exec_module(det)

# ultralytics 提供数据管道 / 损失 / 解码 / NMS / mask 处理 / mAP
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.utils.ops import xywh2xyxy, process_mask
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou, mask_iou

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ULTRA_DIR = os.path.join(BASE_DIR, "ultralytics", "ultralytics")
COCO128_SEG_YAML = os.path.join(ULTRA_DIR, "cfg", "datasets", "coco128-seg.yaml")
PRETRAINED_WEIGHTS = os.path.join(ULTRA_DIR, "yolo26n-seg.pt")

IMGSZ = 640
NUM_CLASSES = 80
NM = 32          # mask 系数个数
# yolo26-seg.yaml 里 npr=256 会按 width=0.25 缩放：make_divisible(256*0.25, 8)=64
# （见 ultralytics/nn/tasks.py parse_model: args[2] = make_divisible(min(args[2], max_channels)*width, 8)）
NPR = 64
STRIDES = (8, 16, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 量化后端可切换：dorefa / lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / pact
# backbone 组件从检测网络复用，因此切换时同步切换检测网络模块内绑定的后端算子。
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


def set_quant_method(method):
    """切换量化后端（必须在构建 QuantYOLO26Seg 之前调用）。"""
    global Q, QUANT_METHOD
    global QuantAdd, QuantCat, QuantConcat, QuantConv2d, QuantConvTranspose2d, QuantMaxPool

    # 复用的 backbone/neck block 类内部引用的是 det 模块的全局算子，必须先同步切换
    det.set_quant_method(method)
    Q = quant_pkg.load_quant_backend(method)
    QUANT_METHOD = method
    QuantAdd = det.QuantAdd
    QuantCat = det.QuantCat
    QuantConcat = det.QuantConcat
    QuantConv2d = det.QuantConv2d
    QuantConvTranspose2d = Q.QuantConvTranspose2d
    QuantMaxPool = det.QuantMaxPool
    return Q


# ============================== 分割头组件 ==============================


class Proto26(nn.Module):
    """复刻 ultralytics.nn.modules.block.Proto26（yolo26n-seg, npr=64, nm=32）。

    多尺度融合：P3 特征 + 上采样的 P4/P5 细化特征（两次残差加法），
    经 feat_fuse 后送入 Proto（Conv3x3 -> ConvTranspose2d 上采样 2x -> Conv3x3 -> Conv1x1），
    输出 (B, nm=32, 160, 160) prototypes；训练时额外返回 semseg 的 (B, nc, 160, 160) 语义图。

    属性命名与官方完全一致（cv1/upsample/cv2/cv3/feat_refine/feat_fuse/semseg），
    可直接加载 yolo26n-seg.pt 中 model.23.proto.* 的权重。
    """

    def __init__(self, ch=(64, 128, 256), npr=NPR, nm=NM, nc=NUM_CLASSES, quant=False):
        super().__init__()
        # Proto(npr, npr, nm): cv1/cv2 为 npr 通道，upsample 为 2x 转置卷积，cv3 输出 nm 通道
        self.cv1 = det.Conv(npr, npr, k=3, quant=quant)
        if quant:
            self.upsample = QuantConvTranspose2d(
                npr, npr, kernel_size=2, stride=2, padding=0, bias=True,
                a_bits=8, w_bits=8, per_channel=True,
            )
        else:
            self.upsample = nn.ConvTranspose2d(npr, npr, 2, 2, 0, bias=True)
        self.cv2 = det.Conv(npr, npr, k=3, quant=quant)
        self.cv3 = det.Conv(npr, nm, quant=quant)

        self.feat_refine = nn.ModuleList(det.Conv(x, ch[0], k=1, quant=quant) for x in ch[1:])
        self.refine_adds = (
            nn.ModuleList(det.QuantAdd(a_bits=8, quant_inference=True) for _ in ch[1:])
            if quant else None
        )
        self.feat_fuse = det.Conv(ch[0], npr, k=3, quant=quant)
        self.semseg = nn.Sequential(
            det.Conv(ch[0], npr, k=3, quant=quant),
            det.Conv(npr, npr, k=3, quant=quant),
            QuantConv2d(npr, nc, 1, a_bits=8, w_bits=8, per_channel=True)
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
    """YOLO26-seg 分割头（单 one2many 头，reg_max=1）。

    训练时返回 dict(boxes/scores/feats/mask_coefficient/proto) 供 v8SegmentationLoss 使用，
    proto 为 (protos, semantic) 二元组；评估时返回 (检测张量, protos)：
    检测张量通道顺序 xywh(4) + sigmoid 分类(nc) + 原始 mask 系数(nm)，与 NMS 的
    split((4, nc, extra)) / process_mask 约定一致。
    """

    def __init__(self, nc=NUM_CLASSES, nm=NM, npr=NPR, reg_max=1, ch=(64, 128, 256), quant=False):
        super().__init__(nc=nc, reg_max=reg_max, ch=ch, quant=quant)
        self.nm = nm
        self.npr = npr
        # 属性名必须为 proto，与官方 Segment26 / yolo26n-seg.pt 的 model.23.proto.* 对齐
        self.proto = Proto26(ch, npr, nm, nc, quant=quant)
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                det.Conv(x, c4, 3, quant=quant),
                det.Conv(c4, c4, 3, quant=quant),
                QuantConv2d(c4, self.nm, 1, a_bits=8, w_bits=8, per_channel=True)
                if quant else nn.Conv2d(c4, self.nm, 1),
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
            self._anchors, self._strides_tensor = (
                a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5)
            )
            self._feat_shape = shape
        dbox = dist2bbox(boxes, self._anchors.unsqueeze(0), xywh=True, dim=1)
        dbox = dbox * self._strides_tensor
        det = torch.cat((dbox, scores.sigmoid(), mask_coefficient), 1)
        # eval 时 Proto26 只返回 protos 张量
        proto_p = proto[0] if isinstance(proto, tuple) else proto
        return det, proto_p


# ============================== 网络主体 ==============================


class YOLO26Seg(det.YOLO26):
    """yolo26n-seg（scale=n）分割网络。quant=False 浮点模型，True 为伪量化模型。

    层 0-22 与检测网络一致（直接构建后替换层 23），topology 编号/save 集合不变，
    因此 yolo26n-seg.pt 的 state_dict（model.0.* ~ model.23.*）可直接加载，
    end2end 双头 one2one_* 参数被跳过。
    """

    def __init__(self, nc=NUM_CLASSES, quant=False):
        # 先构建检测网络得到完整的 0-22 backbone/neck，再把层 23 的 Detect 换成 Segment
        super().__init__(nc=nc, quant=quant)
        head = Segment(nc=nc, nm=NM, npr=NPR, reg_max=1, ch=(64, 128, 256), quant=quant)
        det._tag(head, 23, [16, 19, 22])
        self.model[-1] = head
        self._register_quantizer_buffers()
        self._initialize_head()


class FloatYOLO26Seg(YOLO26Seg):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__(nc=nc, quant=False)


class QuantYOLO26Seg(YOLO26Seg):
    def __init__(self, nc=NUM_CLASSES):
        super().__init__(nc=nc, quant=True)


# ============================== 数据 / 损失 / 评估 ==============================


def _make_cfg(num_workers):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = IMGSZ
    cfg.task = "segment"  # YOLODataset 据此生成 polygon -> bitmap masks / sem_masks
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
    """复用 ultralytics 官方 coco128-seg 数据管道（含 masks / sem_masks）。"""
    data = check_det_dataset(COCO128_SEG_YAML)
    if calibration:
        return _build_loaders(batch_size, num_workers, data, calibration=True), data
    train_loader, val_loader, _, _ = _build_loaders(batch_size, num_workers, data)
    return train_loader, val_loader, data


class _LossShim:
    """v8SegmentationLoss 只需要 model.args / model.model[-1] / model.parameters()。"""

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


def _match_masks(pred_labels, pred_masks, gt_labels, gt_masks, iou_vector):
    """与框匹配同构，但 IoU 用 mask_iou（pred_masks/gt_masks 均已二值化）。"""
    # mask_iou 只接受二维 (N, H*W)，先拉平
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


@torch.no_grad()
def evaluate(model, val_loader, data, batch_size=8, max_batches=None, conf_thres=0.001, iou_thres=0.7):
    """在 coco128-seg 验证集上同时计算 box mAP 与 mask mAP。

    与官方 SegmentationValidator 一致：NMS 后的 mask 系数与 protos 相乘，
    在 proto 分辨率（160x160，图像 /4）上二值化并按框裁剪；GT overlap 索引掩码
    拆分后插值到同一分辨率，用 mask_iou 在 10 个阈值上匹配。
    """
    model.eval()
    stats_conf, stats_pcls, stats_tcls = [], [], []
    stats_tp_box, stats_tp_mask = [], []
    names = data["names"]
    num_images = len(val_loader.dataset)
    steps = max_batches or math.ceil(num_images / batch_size)

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
            nc=NUM_CLASSES,  # 显式指定，否则 nm=32 个 mask 系数会被当成类别
        )
        image_size = batch["img"].shape[2:]
        mh, mw = protos.shape[2:]

        for sample_index, pred in enumerate(predictions):
            index = batch["batch_idx"] == sample_index
            gt_cls = batch["cls"][index].squeeze(-1)
            gt_boxes = batch["bboxes"][index]
            nl = gt_cls.shape[0]
            if nl:
                gt_boxes = xywh2xyxy(gt_boxes) * torch.tensor(image_size)[[1, 0, 1, 0]]

            # GT 实例掩码：overlap_mask=True 时每张图是一张 1..nl 的索引图
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


# ============================== checkpoint / 权重复制 / 导出 ==============================


def load_checkpoint(model, path, return_meta=False):
    return det.load_checkpoint(model, path, return_meta=return_meta)


def save_checkpoint(model, path, **metadata):
    det.save_checkpoint(model, path, **metadata)


def load_pretrained(model, path=PRETRAINED_WEIGHTS):
    """加载官方 yolo26n-seg.pt；单头模型跳过 one2one_* 双头权重。"""
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


def build_float_model(quant_model, nc=NUM_CLASSES):
    """把量化模型的反量化权重灌进同结构的干净浮点模型（用于导出纯浮点 ONNX）。"""
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_state = quant_model.state_dict()
    for name, module in quant_model.named_modules():
        if quant_pkg.is_weight_quant_module(module):
            quant_state[f"{name}.weight"] = quant_pkg.dequantized_weight(module)

    float_model = FloatYOLO26Seg(nc=nc)
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
        output_names=["preds", "proto"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "preds": {0: "batch_size"},
            "proto": {0: "batch_size"},
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
        print(f"      ONNX checker 通过，算子数: {len(op_types)}（含 Conv/ConvTranspose/Concat 等）")
    except ImportError:
        print("      [skip] 未安装 onnx，跳过结构检查")

    try:
        ort = importlib.import_module("onnxruntime")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        torch.manual_seed(0)
        float_model.cpu().eval()
        x = torch.randn(1, 3, IMGSZ, IMGSZ)
        with torch.no_grad():
            y_torch_det, y_torch_proto = float_model(x)
        y_onnx = session.run(["preds", "proto"], {"images": x.numpy()})
        max_diff_det = float(np.abs(y_torch_det.numpy() - y_onnx[0]).max())
        max_diff_proto = float(np.abs(y_torch_proto.numpy() - y_onnx[1]).max())
        print(
            f"      onnxruntime vs PyTorch 最大绝对误差: det {max_diff_det:.3e} / "
            f"proto {max_diff_proto:.3e}"
        )
        assert max(max_diff_det, max_diff_proto) < 1e-3, "ONNX 数值误差过大"
    except ImportError:
        print("      [skip] 未安装 onnxruntime，跳过数值比对")

    required_keys = quant_pkg.required_quant_keys(quant_model)
    missing = [key for key in required_keys if key not in quant_params]
    assert not missing, f"以下量化参数不完整: {missing[:10]}"
    print(f"      量化参数完整性检查通过（{len(quant_params)} 个张量）")


def save_quant_outputs(quant_model, prefix, nc=NUM_CLASSES, meta=None):
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_yolo26nseg.pth")
    save_checkpoint(quant_model, quant_checkpoint, **(meta or {}))

    quant_params = collect_quant_params(quant_model)
    json_path = os.path.join(MODEL_DIR, f"{prefix}_yolo26nseg_quant_params.json")
    pth_path = os.path.join(MODEL_DIR, f"{prefix}_yolo26nseg_quant_params.pth")
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
    float_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_yolo26nseg_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(MODEL_DIR, f"{prefix}_yolo26nseg_float.onnx")
    export_onnx(float_model, onnx_path, opset=16)
    print(f"[4/5] 干净浮点 ONNX 已写出: {onnx_path}")

    verify(float_model, quant_model, onnx_path, quant_params)
    print("[5/5] 完成。")
    return quant_checkpoint, json_path, pth_path, float_checkpoint, onnx_path


# ============================== 训练各阶段 ==============================


_ModelEMA = det._ModelEMA
_auto_lr = det._auto_lr
_build_optimizer = det._build_optimizer


def train_one_epoch(model, loader, criterion, optimizer, epoch, epochs, nb, ema=None,
                    nbs=64, warmup_epochs=3.0, lrf=0.01, max_batches=None):
    """与检测版一致的官方训练循环；损失项为 box/seg/cls/l1/sem 五项。"""
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
                max_train_batches=None, max_eval_batches=None):
    print("========== Float training (YOLO26n-seg / coco128-seg) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr()
    print(
        f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4（官方 optimizer=auto 配方）"
        " | warmup 3ep | 梯度累积 nbs=64 | EMA | close_mosaic=10"
    )

    data = check_det_dataset(COCO128_SEG_YAML)
    train_loader, val_loader, cfg, train_set = _build_loaders(batch_size, num_workers, data)
    nb = len(train_loader)

    float_model = FloatYOLO26Seg(num_classes).to(device)
    load_pretrained(float_model, PRETRAINED_WEIGHTS)

    criterion = build_criterion(float_model, epochs)
    optimizer = _build_optimizer(float_model, lr=lr)
    ema = _ModelEMA(float_model)

    close_mosaic = 10
    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    checkpoint_path = os.path.join(MODEL_DIR, "yolo26n-seg.pth")
    last_checkpoint = os.path.join(MODEL_DIR, "yolo26n-seg_last.pth")

    for epoch in range(epochs):
        if epoch == epochs - close_mosaic:
            print("[Float] 关闭 dataloader mosaic（最后 10 个 epoch）")
            train_set.close_mosaic(cfg)
            train_loader.reset()

        train_loss, items = train_one_epoch(
            float_model, train_loader, criterion, optimizer, epoch, epochs, nb, ema=ema,
            max_batches=max_train_batches,
        )
        box_loss, seg_loss, cls_loss, l1_loss, sem_loss = items
        metrics = evaluate(
            ema.ema, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
        )
        fitness = 0.9 * metrics["mask_map"] + 0.1 * metrics["mask_map50"]

        epoch_meta = {
            "stage": "float",
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "box_loss": float(box_loss),
            "seg_loss": float(seg_loss),
            "cls_loss": float(cls_loss),
            "l1_loss": float(l1_loss),
            "sem_loss": float(sem_loss),
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
            f"(box:{box_loss:.3f} seg:{seg_loss:.3f} cls:{cls_loss:.3f} sem:{sem_loss:.3f}) | "
            f"box mAP50:{metrics['map50']:.4f} | mask mAP50:{metrics['mask_map50']:.4f} "
            f"mAP50-95:{metrics['mask_map']:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)
    best_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(f"[Float] Best checkpoint: {checkpoint_path}（epoch {best_epoch}/{epochs}）")
    print(
        f"[Float] Best box mAP50:{best_metrics['map50']:.4f} | "
        f"mask mAP50:{best_metrics['mask_map50']:.4f} mAP50-95:{best_metrics['mask_map']:.4f}"
    )
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
    print(f"========== PTQ calibration (YOLO26n-seg / {tag}) ==========")
    print(f"Device: {device}")

    calibration_loader, data = get_dataloaders(batch_size, num_workers, calibration=True)
    _, val_loader, _ = get_dataloaders(batch_size, num_workers)

    float_checkpoint = os.path.join(MODEL_DIR, "yolo26n-seg.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatYOLO26Seg(num_classes).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = QuantYOLO26Seg(num_classes).to(device)
    copy_float_to_quant(float_model, ptq_model)

    calibrate_quantizer(ptq_model, calibration_loader, calibration_batches)

    metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(
        f"[PTQ-{tag}] box mAP50:{metrics['map50']:.4f} | "
        f"mask mAP50:{metrics['mask_map50']:.4f} mAP50-95:{metrics['mask_map']:.4f}"
    )

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
    print(
        f"[PTQ] 重载 checkpoint 后 mask mAP50:{final_metrics['mask_map50']:.4f} "
        f"mAP50-95:{final_metrics['mask_map']:.4f}"
    )
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=16, lr=None, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training (YOLO26n-seg / {tag}) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr() * 0.1
    print(f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4 | warmup 3ep | 梯度累积 nbs=64")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)
    nb = len(train_loader)

    ptq_checkpoint = os.path.join(MODEL_DIR, f"ptq_{tag}_yolo26nseg.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = QuantYOLO26Seg(num_classes).to(device)
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
    best_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_yolo26nseg_best.pth")
    last_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_yolo26nseg_last.pth")

    for epoch in range(epochs):
        train_loss, items = train_one_epoch(
            qat_model, train_loader, criterion, optimizer, epoch, epochs, nb,
            max_batches=max_train_batches,
        )
        box_loss, seg_loss, cls_loss, l1_loss, sem_loss = items
        metrics = evaluate(
            qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
        )
        fitness = 0.9 * metrics["mask_map"] + 0.1 * metrics["mask_map50"]

        epoch_meta = {
            "stage": "qat",
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
            f"(box:{box_loss:.3f} seg:{seg_loss:.3f} cls:{cls_loss:.3f} sem:{sem_loss:.3f}) | "
            f"box mAP50:{metrics['map50']:.4f} | mask mAP50:{metrics['mask_map50']:.4f} "
            f"mAP50-95:{metrics['mask_map']:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    best_metrics = evaluate(
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(
        f"[QAT-{tag}] Best epoch:{best_epoch}/{epochs} | "
        f"mask mAP50:{best_metrics['mask_map50']:.4f} mAP50-95:{best_metrics['mask_map']:.4f} | "
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
    print(f"========== Float vs QAT precision (YOLO26n-seg / coco128-seg / {tag}) ==========")
    print(f"Device: {device}")

    float_checkpoint = os.path.join(MODEL_DIR, "yolo26n-seg.pth")
    qat_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_yolo26nseg.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26Seg(num_classes).to(device)
    float_model, float_meta = load_checkpoint(float_model, float_checkpoint, return_meta=True)
    float_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )

    qat_model = QuantYOLO26Seg(num_classes).to(device)
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
        description="YOLO26n-seg: 浮点训练(加载 yolo26n-seg.pt) -> PTQ -> QAT -> box/mask mAP 对比，量化方法可选"
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
