"""YOLO26n-pose 关键点网络的 aLSQ+ 量化感知训练流程（coco8-pose）。

流程：
    float_train(lr=float_lr, epochs=float_epochs)   # 浮点训练（默认加载 yolo26n-pose.pt 预训练权重）
    PTQ_calibration()                               # 训练后量化校准
    QAT_training(lr=qat_lr, epochs=qat_epochs)      # 量化感知训练
    compare_precision()                             # 浮点 vs QAT 的 box / pose mAP 对比

网络结构与 ultralytics/cfg/models/26/yolo26-pose.yaml（scale=n）逐层对齐：
Conv / C3k2(C3k) / SPPF / C2PSA(Attention) / PAN-FPN / Pose26(reg_max=1)。
说明：yolo26n-pose 原始 end2end 双头（one2many+one2one）这里只保留 one2many 单头，
损失使用官方 PoseLoss26（TaskAlignedAssigner + OKS keypoint_loss + BCE 可见性
+ RealNVP 归一化流 RLE 损失，由 cv4_sigma 输出的 sigma 驱动）；
backbone / neck / pose 头拓扑（含 flow_model）与官方完全一致，可直接加载
yolo26n-pose.pt 的权重（one2one_* 双头权重会被跳过）。

关键点：
  * coco-pose 只有 1 个类别（person），所以 nc=1，kpt_shape=[17, 3]；
  * 推理输出通道 = xywh(4) + 类别(1) + 关键点(17*3=51) = 56，NMS 后 reshape 成 (N,17,3)；
  * 训练头 cv4(->85ch) 后接 cv4_kpts(->51) 与 cv4_sigma(->34，训练专用)；
  * flow_model 仅在训练损失里使用（推理不经过），因此始终保持浮点 nn.Linear，不替换成
    QuantLinear；cv4/cv4_kpts/cv4_sigma 中的 Conv2d 则参与量化。
  * Pose26 的关键点解码为 (raw + anchor) * stride（无 v8 Pose 的 *2-0.5 偏移），
    可见性通道走 sigmoid。
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

# 复用检测网络的 backbone/neck 组件与训练基础设施
# （文件名 networks_yolo26-detect.py 含 '-'，不能直接 import，用文件路径加载）
det_spec = importlib.util.spec_from_file_location(
    "networks_yolo26_detect",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "networks_yolo26-detect.py"),
)
det = importlib.util.module_from_spec(det_spec)
det_spec.loader.exec_module(det)

# ultralytics 提供数据管道 / 损失 / 解码 / NMS / OKS / mAP
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset, build_dataloader
from ultralytics.utils.loss import PoseLoss26
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class, box_iou, kpt_iou, OKS_SIGMA
from ultralytics.utils.torch_utils import model_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ULTRA_DIR = os.path.join(BASE_DIR, "ultralytics", "ultralytics")
COCO8_POSE_YAML = os.path.join(ULTRA_DIR, "cfg", "datasets", "coco8-pose.yaml")
PRETRAINED_WEIGHTS = os.path.join(ULTRA_DIR, "yolo26n-pose.pt")

IMGSZ = 640
NUM_CLASSES = 1          # coco-pose 只有 person
KPT_SHAPE = [17, 3]      # 17 个关键点，x/y/visibility
NKPT = KPT_SHAPE[0]
NDIM = KPT_SHAPE[1]
NK = NKPT * NDIM         # 51
NK_SIGMA = NKPT * 2      # 34（每个关键点 sigma_x / sigma_y）
# Pose26: c4 = max(ch[0]//4, nkpt*(ndim+2)) = max(16, 85) = 85
C4 = max(64 // 4, NKPT * (NDIM + 2))
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
QuantMaxPool = Q.QuantMaxPool
QuantSiLU = getattr(Q, 'QuantSiLU', None)
QuantSigmoid = getattr(Q, 'QuantSigmoid', None)
QuantMatMul = getattr(Q, 'QuantMatMul', None)


def set_quant_method(method):
    """切换量化后端（必须在构建 QuantYOLO26Pose 之前调用）。"""
    global Q, QUANT_METHOD
    global QuantAdd, QuantCat, QuantConcat, QuantConv2d, QuantMaxPool, QuantSiLU, QuantSigmoid, QuantMatMul

    # 复用的 backbone/neck block 类内部引用的是 det 模块的全局算子，必须先同步切换
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
    QuantMatMul = getattr(Q, 'QuantMatMul', None)
    return Q


# ============================== RealNVP 归一化流 ==============================


class RealNVP(nn.Module):
    """复刻 ultralytics.nn.modules.block.RealNVP（PoseLoss26 的 RLE 损失需要）。

    仅在训练损失里对归一化残差做 log_prob；推理不经过此模块，因此保持纯浮点
    nn.Linear，不参与量化。属性 / buffer 命名（loc/cov/mask/s/t）与官方一致，
    可直接加载 yolo26n-pose.pt 中 model.23.flow_model.* 的权重。
    """

    @staticmethod
    def nets():
        return nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh()
        )

    @staticmethod
    def nett():
        return nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2)
        )

    def __init__(self):
        super().__init__()
        # loc/cov 现在已不参与计算，但仍注册以兼容旧 checkpoint
        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)

    def backward_p(self, x):
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        # 二维标准正态闭式 log N(z; 0, I)
        return -0.5 * (z.float() ** 2).sum(-1) - math.log(2 * math.pi) + log_det


# ============================== Pose26 检测/关键点头 ==============================


class Pose(det.Detect):
    """YOLO26-pose 关键点头（单 one2many 头，reg_max=1，Pose26 解码方式）。

    训练时返回 dict(boxes/scores/feats/kpts/kpts_sigma) 供官方 PoseLoss26 使用；
    评估时返回解码后的检测张量，通道顺序：
        xywh(4) + sigmoid 类别(nc) + 关键点(nk=51)
    与 NMS 的 split((4, nc, extra)) 约定一致，NMS 后把后 51 列 reshape 为 (N,17,3)。

    结构（与官方 Pose26 对齐）：
        cv4[i]  = Conv3x3 -> Conv3x3（输出 c4=85，无末端 1x1）
        cv4_kpts[i] = Conv1x1(c4 -> nk=51)
        cv4_sigma[i] = Conv1x1(c4 -> nk_sigma=34)（仅训练）
        flow_model = RealNVP()（仅训练损失使用，纯浮点）
    """

    def __init__(self, nc=NUM_CLASSES, kpt_shape=None, reg_max=1,
                 ch=None, quant=False, scale=det.DEFAULT_SCALE):
        kpt_shape = KPT_SHAPE if kpt_shape is None else list(kpt_shape)
        if ch is None:
            ch = det.detect_head_channels(scale)
        super().__init__(nc=nc, reg_max=reg_max, ch=ch, quant=quant)
        self.kpt_shape = list(kpt_shape)
        self.nkpt, self.ndim = self.kpt_shape[0], self.kpt_shape[1]
        self.nk = self.nkpt * self.ndim
        self.nk_sigma = self.nkpt * 2
        # RLE 归一化流（训练损失专用，永不量化）
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, self.nkpt * (self.ndim + 2))
        self.c4 = c4
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                det.Conv(x, c4, 3, quant=quant),
                det.Conv(c4, c4, 3, quant=quant),
            )
            for x in ch
        )
        self.cv4_kpts = nn.ModuleList(
            QuantConv2d(c4, self.nk, 1, a_bits=8, w_bits=8, per_channel=True)
            if quant else nn.Conv2d(c4, self.nk, 1)
            for _ in ch
        )
        self.cv4_sigma = nn.ModuleList(
            QuantConv2d(c4, self.nk_sigma, 1, a_bits=8, w_bits=8, per_channel=True)
            if quant else nn.Conv2d(c4, self.nk_sigma, 1)
            for _ in ch
        )

    def forward(self, x):
        # 检测支路（cv2/cv3）直接复用父类
        preds = self.forward_head(x)
        bs = x[0].shape[0]

        pose_feats = [self.cv4[i](x[i]) for i in range(self.nl)]
        kpts = torch.cat(
            [self.cv4_kpts[i](pose_feats[i]).view(bs, self.nk, -1) for i in range(self.nl)],
            dim=-1,
        )

        if self.training:
            kpts_sigma = torch.cat(
                [self.cv4_sigma[i](pose_feats[i]).view(bs, self.nk_sigma, -1)
                 for i in range(self.nl)],
                dim=-1,
            )
            preds["kpts"] = kpts
            preds["kpts_sigma"] = kpts_sigma
            return preds

        # ---------------- 推理：Pose26 解码 (raw + anchor) * stride ----------------
        shape = x[0].shape
        if self._feat_shape != shape:
            self._anchors, self._strides_tensor = (
                a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5)
            )
            self._feat_shape = shape

        num_anchors = kpts.shape[-1]
        y = kpts.view(bs, self.nkpt, self.ndim, num_anchors)
        # 缓存的 _anchors 为转置后的 (2, N)（dist2bbox 约定），取两行网格坐标
        ax = self._anchors[0].view(1, 1, num_anchors)
        ay = self._anchors[1].view(1, 1, num_anchors)
        st = self._strides_tensor.view(1, 1, num_anchors)
        kx = (y[:, :, 0:1, :] + ax) * st
        ky = (y[:, :, 1:2, :] + ay) * st
        kv = y[:, :, 2:3, :].sigmoid()
        kpts_decoded = torch.cat([kx, ky, kv], dim=2).view(bs, self.nk, num_anchors)

        dbox = dist2bbox(preds["boxes"], self._anchors.unsqueeze(0), xywh=True, dim=1)
        dbox = dbox * self._strides_tensor
        return torch.cat((dbox, preds["scores"].sigmoid(), kpts_decoded), 1)


# ============================== 网络主体 ==============================


class YOLO26Pose(det.YOLO26):
    """yolo26n-pose（scale=n）关键点网络。quant=False 浮点模型，True 为伪量化模型。

    层 0-22 与检测网络一致（直接构建后替换层 23），topology 编号/save 集合不变，
    因此 yolo26n-pose.pt 的 state_dict（model.0.* ~ model.23.*，含 flow_model）
    可直接加载，end2end 双头 one2one_* 参数被跳过。
    """

    def __init__(self, nc=NUM_CLASSES, quant=False, scale=det.DEFAULT_SCALE):
        # 先构建检测网络得到完整的 0-22 backbone/neck，再把层 23 的 Detect 换成 Pose
        super().__init__(nc=nc, quant=quant, scale=scale)
        head = Pose(nc=nc, kpt_shape=KPT_SHAPE, reg_max=1, quant=quant, scale=scale)
        det._tag(head, 23, [16, 19, 22])
        self.model[-1] = head
        self._register_quantizer_buffers()
        self._initialize_head()


class FloatYOLO26Pose(YOLO26Pose):
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
        super().__init__(nc=nc, quant=False, scale=scale)


class QuantYOLO26Pose(YOLO26Pose):
    def __init__(self, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
        super().__init__(nc=nc, quant=True, scale=scale)


# ============================== 数据 / 损失 / 评估 ==============================


def _make_cfg(num_workers):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = IMGSZ
    cfg.task = "pose"  # YOLODataset 据此生成 keypoints 标注
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
    """复用 ultralytics 官方 coco8-pose 数据管道（含 keypoints）。"""
    data = check_det_dataset(COCO8_POSE_YAML)
    if calibration:
        return _build_loaders(batch_size, num_workers, data, calibration=True), data
    train_loader, val_loader, _, _ = _build_loaders(batch_size, num_workers, data)
    return train_loader, val_loader, data


class _LossShim:
    """PoseLoss26 只需要 model.args / model.model[-1] / model.parameters()。"""

    def __init__(self, pose_head, epochs):
        self.args = SimpleNamespace(
            box=7.5, cls=0.5, dfl=1.5,
            pose=12.0, kobj=1.0, rle=1.0,   # ultralytics/cfg/default.yaml 官方增益
            epochs=epochs,
        )
        self.model = [None] * 23 + [pose_head]
        self.class_weights = None

    def parameters(self):
        return self.model[-1].parameters()


def build_criterion(model, epochs):
    # 官方 PoseLoss26：v8PoseLoss(OKS keypoint loss + BCE 可见性) + RealNVP RLE
    return PoseLoss26(_LossShim(model.model[-1], epochs), tal_topk=10)


IOU_VECTOR = torch.linspace(0.5, 0.95, 10)


def _match_iou(pred_labels, gt_labels, iou, iou_vector):
    """复刻 ultralytics BaseValidator.match_predictions：在 10 个 IoU 阈值上贪心匹配。

    iou 形状 (num_gt, num_pred)，box IoU 或 pose OKS 均可。
    """
    correct = torch.zeros(iou.shape[1], iou_vector.numel(), dtype=torch.bool)
    correct_class = gt_labels[:, None] == pred_labels[None, :]
    for level, threshold in enumerate(iou_vector):
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
        correct[matches[:, 1].astype(int), level] = True
    return correct


@torch.no_grad()
def evaluate(model, val_loader, data, batch_size=8, max_batches=None, conf_thres=0.001, iou_thres=0.7):
    """在 coco8-pose 验证集上同时计算 box mAP 与 pose(OKS) mAP。

    与官方 PoseValidator._process_batch 一致：NMS 后把额外 51 列 reshape 成
    (N,17,3) 关键点；GT 关键点由归一化坐标缩放到 letterbox 输入尺寸；
    OKS 的面积项取 GT 框 xywh 的 w*h*0.53（cocoeval 约定），
    用 kpt_iou(OKS) 在 10 个阈值上匹配。
    """
    model.eval()
    stats_conf, stats_pcls, stats_tcls = [], [], []
    stats_tp_box, stats_tp_pose = [], []
    names = data.names if hasattr(data, "names") else data["names"]
    num_images = len(val_loader.dataset)
    steps = max_batches or math.ceil(num_images / batch_size)
    sigma = OKS_SIGMA if KPT_SHAPE == [17, 3] else np.ones(NKPT) / NKPT

    for batch_index, batch in enumerate(val_loader):
        if batch_index >= steps:
            break
        images = batch["img"].float().to(device) / 255.0
        det = model(images)
        predictions = non_max_suppression(
            det,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            multi_label=True,
            agnostic=False,
            max_det=300,
            nc=NUM_CLASSES,  # 显式指定，否则 nk=51 个关键点通道会被当成类别
        )
        image_size = batch["img"].shape[2:]  # (h, w)

        for sample_index, pred in enumerate(predictions):
            index = batch["batch_idx"] == sample_index
            gt_cls = batch["cls"][index].squeeze(-1)
            gt_boxes = batch["bboxes"][index]
            nl = gt_cls.shape[0]
            if nl:
                gt_boxes = xywh2xyxy(gt_boxes) * torch.tensor(image_size)[[1, 0, 1, 0]]

            # GT 关键点：归一化坐标 -> letterbox 输入像素坐标
            gt_kpts = batch["keypoints"][index].float().to(device)
            if nl:
                gt_kpts = gt_kpts.clone()
                gt_kpts[..., 0] *= image_size[1]
                gt_kpts[..., 1] *= image_size[0]

            # 统一搬 CPU（detect evaluate 同理，ultralytics box_iou 等 metric 函数不跨设备）
            pred_boxes = pred[:, :4].cpu()
            gt_boxes = gt_boxes.cpu()
            gt_cls = gt_cls.cpu()
            gt_kpts = gt_kpts.cpu()
            pred_kpts = pred[:, 6:6 + NK].cpu().view(-1, NKPT, NDIM)

            tp_box = torch.zeros(pred.shape[0], IOU_VECTOR.numel(), dtype=torch.bool)
            tp_pose = torch.zeros(pred.shape[0], IOU_VECTOR.numel(), dtype=torch.bool)
            if pred.shape[0] and nl:
                tp_box = _match_iou(
                    pred[:, 5].cpu(), gt_cls,
                    box_iou(gt_boxes, pred_boxes), IOU_VECTOR
                )
                # 官方 cocoeval 约定：OKS 面积 = box 面积 * 0.53
                area = xyxy2xywh(gt_boxes)[:, 2:].prod(1) * 0.53
                oks = kpt_iou(gt_kpts, pred_kpts, area=area, sigma=sigma)
                tp_pose = _match_iou(pred[:, 5].cpu(), gt_cls, oks.cpu(), IOU_VECTOR)

            stats_conf.append(pred[:, 4].cpu())
            stats_pcls.append(pred[:, 5].cpu())
            stats_tp_box.append(tp_box)
            stats_tp_pose.append(tp_pose)
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
    pose_map50, pose_map, pose_p, pose_r = _ap(torch.cat(stats_tp_pose).numpy())
    return {
        "map50": box_map50, "map": box_map, "precision": box_p, "recall": box_r,
        "pose_map50": pose_map50, "pose_map": pose_map,
        "pose_precision": pose_p, "pose_recall": pose_r,
    }


# ============================== checkpoint / 权重复制 / 导出 ==============================


def load_checkpoint(model, path, return_meta=False):
    return det.load_checkpoint(model, path, return_meta=return_meta)


def save_checkpoint(model, path, **metadata):
    det.save_checkpoint(model, path, **metadata)


def load_pretrained(model, path=None, scale=det.DEFAULT_SCALE):
    """加载官方 yolo26{scale}-pose.pt 权重；单头模型跳过 one2one_* 双头权重。

    仓库仅随附 yolo26n-pose.pt；s/m/l/x 缺失时跳过加载，模型随机初始化从头训练。
    """
    if path is None:
        path = os.path.join(ULTRA_DIR, f"{det.base_name(scale, 'pose')}.pt")
    if not os.path.exists(path):
        print(f"[Pretrain] [warn] 未找到预训练权重 {path}，{det.model_name(scale)}-pose 将从头训练")
        return model
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"].float().state_dict()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    skipped = [k for k in unexpected if "one2one_" in k]
    other_unexpected = [k for k in unexpected if "one2one_" not in k]
    if other_unexpected:
        print(f"[Pretrain] 警告：{len(other_unexpected)} 个未识别权重未加载: {other_unexpected[:3]}")
    print(f"[Pretrain] 已加载 {path}（跳过 {len(skipped)} 个 one2one 双头参数，缺失 {len(missing)} 个）")
    pose_missing = [k for k in missing if "cv4" in k or "flow_model" in k]
    if pose_missing:
        print(f"[Pretrain] pose 头缺失参数: {pose_missing[:5]}")
    return model


def copy_float_to_quant(float_model, quant_model):
    return det.copy_float_to_quant(float_model, quant_model)


def build_float_model(quant_model, nc=NUM_CLASSES, scale=det.DEFAULT_SCALE):
    """把量化模型的反量化权重灌进同结构的干净浮点模型（用于导出纯浮点 ONNX）。"""
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    quant_state = quant_model.state_dict()
    for name, module in quant_model.named_modules():
        if quant_pkg.is_weight_quant_module(module):
            quant_state[f"{name}.weight"] = quant_pkg.dequantized_weight(module)

    float_model = FloatYOLO26Pose(nc=nc, scale=scale)
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
    # onnxsim 简化（若已安装）
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
    """训练保存 best checkpoint 时同步导出 ONNX；失败仅告警，绝不影响训练。

    导出后模型被置为 eval，由下一轮 train_one_epoch 的 model.train() 恢复。
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
        print(f"      ONNX checker 通过，算子数: {len(op_types)}（含 Conv/Concat 等）")
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
        ref_mag = float(np.abs(y_torch.numpy()).max())
        print(f"      onnxruntime vs PyTorch 最大绝对误差: preds {max_diff:.3e}（参考幅度 {ref_mag:.3e}）")
        # 输出含大数量级解码坐标，纯绝对阈值过严：max(1e-3 绝对, 1e-5 相对)
        assert max_diff < max(1e-3, 1e-5 * ref_mag), "ONNX 数值误差过大"
    except ImportError:
        print("      [skip] 未安装 onnxruntime，跳过数值比对")

    required_keys = quant_pkg.required_quant_keys(quant_model)
    missing = [key for key in required_keys if key not in quant_params]
    assert not missing, f"以下量化参数不完整: {missing[:10]}"
    print(f"      量化参数完整性检查通过（{len(quant_params)} 个张量）")


def save_quant_outputs(quant_model, prefix, nc=NUM_CLASSES, meta=None, scale=det.DEFAULT_SCALE):
    quant_pkg.freeze_batch_init(quant_model)
    quant_model.eval()

    base = det.base_name(scale, 'pose')
    quant_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_{base}.pth")
    save_checkpoint(quant_model, quant_checkpoint, **(meta or {}))

    quant_params = collect_quant_params(quant_model)
    json_path = os.path.join(MODEL_DIR, f"{prefix}_{base}_quant_params.json")
    pth_path = os.path.join(MODEL_DIR, f"{prefix}_{base}_quant_params.pth")
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
    float_checkpoint = os.path.join(MODEL_DIR, f"{prefix}_{base}_float.pth")
    save_checkpoint(float_model, float_checkpoint)
    print(f"[3/5] 干净浮点权重已写出: {float_checkpoint}")

    onnx_path = os.path.join(MODEL_DIR, f"{prefix}_{base}_float.onnx")
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
    """与检测/分割版一致的官方训练循环；损失项为 box/pose/kobj/cls/l1/rle 六项。"""
    model.train()
    batch_size = loader.batch_size
    accumulate = max(round(nbs / batch_size), 1)
    warmup_steps = round(min(warmup_epochs, max(epochs - 1, 0)) * nb) if warmup_epochs > 0 else 0
    lf = lambda x_step: max(1 - x_step / epochs, 0) * (1.0 - lrf) + lrf

    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lf(epoch)

    total_steps = nb if max_batches is None else min(nb, max_batches)
    loss_items_sum = torch.zeros(6)
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
        # ultralytics PoseLoss26 内部会把 batch_idx、offsets 等 tensor 混用在不同设备上
        # （新版本 bug），需先把 batch 里所有 tensor 统一搬到 device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
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
                loss_items["pose_loss"].detach().cpu(),
                loss_items["kobj_loss"].detach().cpu(),
                loss_items["cls_loss"].detach().cpu(),
                loss_items["l1_loss"].detach().cpu(),
                loss_items["rle_loss"].detach().cpu(),
            ]
        )

    if last_opt_step == 0:  # 所有 batch 都还没触发 optimizer step（smoke 极限场景兜底）
        optimizer.step()
        optimizer.zero_grad()

    num_steps = total_steps
    avg_items = (loss_items_sum / max(num_steps, 1)).tolist()
    return sum(avg_items), avg_items


def float_train(batch_size=16, lr=None, epochs=100, num_classes=NUM_CLASSES, num_workers=2,
                max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                resume=False):
    print(f"========== Float training ({det.model_name(scale)}-pose / coco8-pose) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr()
    print(
        f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4（官方 optimizer=auto 配方）"
        " | warmup 3ep | 梯度累积 nbs=64 | EMA | close_mosaic=10"
    )

    data = check_det_dataset(COCO8_POSE_YAML)
    train_loader, val_loader, cfg, train_set = _build_loaders(batch_size, num_workers, data)
    nb = len(train_loader)

    float_model = FloatYOLO26Pose(num_classes, scale=scale).to(device)
    model_info(float_model, imgsz=IMGSZ)
    load_pretrained(float_model, scale=scale)

    criterion = build_criterion(float_model, epochs)
    optimizer = _build_optimizer(float_model, lr=lr)
    ema = _ModelEMA(float_model)

    close_mosaic = 10
    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    base = det.base_name(scale, 'pose')
    checkpoint_path = os.path.join(MODEL_DIR, f"{base}_best.pth")
    last_checkpoint = os.path.join(MODEL_DIR, f"{base}_last.pth")

    # resume：从 _last.pth 恢复，接续训练
    start_epoch = 0
    if resume and os.path.exists(last_checkpoint):
        print(f"[Float] Resume from {last_checkpoint}")
        _, meta = det.load_checkpoint(float_model, last_checkpoint, return_meta=True)
        ema.ema.load_state_dict(float_model.state_dict())
        start_epoch = meta.get("epoch", 0)
        best_fitness = meta.get("fitness", -1.0)
        best_meta = meta
        best_epoch = start_epoch
        print(f"[Float] 从 epoch {start_epoch} 接续训练，当前 best pose mAP50={meta.get('pose_map50', 0):.4f}")
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
        box_loss, pose_loss, kobj_loss, cls_loss, l1_loss, rle_loss = items
        metrics = evaluate(
            ema.ema, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
        )
        fitness = 0.9 * metrics["pose_map"] + 0.1 * metrics["pose_map50"]

        epoch_meta = {
            "stage": "float",
            "scale": det.get_scale(scale),
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "box_loss": float(box_loss),
            "pose_loss": float(pose_loss),
            "kobj_loss": float(kobj_loss),
            "cls_loss": float(cls_loss),
            "l1_loss": float(l1_loss),
            "rle_loss": float(rle_loss),
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
            f"(box:{box_loss:.3f} pose:{pose_loss:.3f} kobj:{kobj_loss:.3f} "
            f"cls:{cls_loss:.3f} rle:{rle_loss:.3f}) | "
            f"box mAP50:{metrics['map50']:.4f} | pose mAP50:{metrics['pose_map50']:.4f} "
            f"mAP50-95:{metrics['pose_map']:.4f}"
        )

    load_checkpoint(float_model, checkpoint_path)
    best_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(f"[Float] Best checkpoint: {checkpoint_path}（epoch {best_epoch}/{epochs}）")
    print(
        f"[Float] Best box mAP50:{best_metrics['map50']:.4f} | "
        f"pose mAP50:{best_metrics['pose_map50']:.4f} mAP50-95:{best_metrics['pose_map']:.4f}"
    )
    return checkpoint_path, best_fitness, best_meta


@torch.no_grad()
def calibrate_quantizer(quant_model, calibration_loader, calibration_batches=20,
                         float_model=None):
    """校准量化器（委托给 detect 模块的升级版本）。"""
    return det.calibrate_quantizer(quant_model, calibration_loader,
                                    calibration_batches, float_model=float_model)


def PTQ_calibration(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                    calibration_batches=20, num_workers=2, max_eval_batches=None,
                    scale=det.DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== PTQ calibration ({det.model_name(scale)}-pose / {tag}) ==========")
    print(f"Device: {device}")

    calibration_loader, data = get_dataloaders(batch_size, num_workers, calibration=True)
    _, val_loader, _ = get_dataloaders(batch_size, num_workers)

    base = det.base_name(scale, 'pose')
    # 优先 _best.pth（与 QAT 命名一致），fallback 到旧版无后缀 .pth
    float_checkpoint = os.path.join(MODEL_DIR, f"{base}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(MODEL_DIR, f"{base}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")

    float_model = FloatYOLO26Pose(num_classes, scale=scale).to(device)
    load_checkpoint(float_model, float_checkpoint)

    ptq_model = QuantYOLO26Pose(num_classes, scale=scale).to(device)
    model_info(ptq_model, imgsz=IMGSZ)
    copy_float_to_quant(float_model, ptq_model)

    calibrate_quantizer(ptq_model, calibration_loader, calibration_batches,
                         float_model=float_model)

    metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(
        f"[PTQ-{tag}] box mAP50:{metrics['map50']:.4f} | "
        f"pose mAP50:{metrics['pose_map50']:.4f} mAP50-95:{metrics['pose_map']:.4f}"
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
        ptq_model, f"ptq_{tag}", nc=num_classes, meta=ptq_meta, scale=scale
    )[0]

    load_checkpoint(ptq_model, ptq_checkpoint)
    quant_pkg.freeze_batch_init(ptq_model)
    final_metrics = evaluate(
        ptq_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(
        f"[PTQ] 重载 checkpoint 后 pose mAP50:{final_metrics['pose_map50']:.4f} "
        f"mAP50-95:{final_metrics['pose_map']:.4f}"
    )
    return ptq_checkpoint


def QAT_training(quant_method=DEFAULT_QUANT_METHOD, batch_size=16, lr=None, epochs=20,
                 num_classes=NUM_CLASSES, num_workers=2,
                 max_train_batches=None, max_eval_batches=None, scale=det.DEFAULT_SCALE,
                 resume=False):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== QAT training ({det.model_name(scale)}-pose / {tag}) ==========")
    print(f"Device: {device}")
    if lr is None:
        lr = _auto_lr() * 0.1
    print(f"Optimizer: AdamW(lr={lr:g}, betas=(0.9, 0.999)) wd=5e-4 | warmup 3ep | 梯度累积 nbs=64")

    train_loader, val_loader, data = get_dataloaders(batch_size, num_workers)
    nb = len(train_loader)

    base = det.base_name(scale, 'pose')
    ptq_checkpoint = os.path.join(MODEL_DIR, f"ptq_{tag}_{base}.pth")
    if not os.path.exists(ptq_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 PTQ 权重，请先运行 PTQ_calibration('{tag}'): {ptq_checkpoint}"
        )

    qat_model = QuantYOLO26Pose(num_classes, scale=scale).to(device)
    model_info(qat_model, imgsz=IMGSZ)
    _, ptq_meta = load_checkpoint(qat_model, ptq_checkpoint, return_meta=True)
    saved_method = ptq_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] PTQ checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    saved_scale = ptq_meta.get("scale")
    if saved_scale is not None and saved_scale != det.get_scale(scale):
        print(f"      [warn] PTQ checkpoint 记录的尺度为 yolo26{saved_scale}-pose，当前为 yolo26{det.get_scale(scale)}-pose")
    quant_pkg.freeze_batch_init(qat_model)

    criterion = build_criterion(qat_model, epochs)
    optimizer = _build_optimizer(qat_model, lr=lr)

    best_fitness = -1.0
    best_epoch = -1
    best_meta = {}
    best_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_{base}_best.pth")
    last_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_{base}_last.pth")

    # resume：从 _last.pth 恢复，接续训练
    start_epoch = 0
    if resume and os.path.exists(last_checkpoint):
        print(f"[QAT-{tag}] Resume from {last_checkpoint}")
        _, meta = load_checkpoint(qat_model, last_checkpoint, return_meta=True)
        start_epoch = meta.get("epoch", 0)
        best_fitness = meta.get("fitness", -1.0)
        best_meta = meta
        best_epoch = start_epoch
        print(f"[QAT-{tag}] 从 epoch {start_epoch} 接续训练，当前 best mAP50={meta.get('pose_map50', 0):.4f}")
    elif resume:
        print(f"[QAT-{tag}] Resume 开启但找不到 {last_checkpoint}，从头训练")

    for epoch in range(start_epoch, epochs):
        train_loss, items = train_one_epoch(
            qat_model, train_loader, criterion, optimizer, epoch, epochs, nb,
            max_batches=max_train_batches,
        )
        box_loss, pose_loss, kobj_loss, cls_loss, l1_loss, rle_loss = items
        metrics = evaluate(
            qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
        )
        fitness = 0.9 * metrics["pose_map"] + 0.1 * metrics["pose_map50"]

        epoch_meta = {
            "stage": "qat",
            "scale": det.get_scale(scale),
            "quant_method": tag,
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "lr": lr,
            "train_loss": float(train_loss),
            "box_loss": float(box_loss),
            "pose_loss": float(pose_loss),
            "kobj_loss": float(kobj_loss),
            "cls_loss": float(cls_loss),
            "l1_loss": float(l1_loss),
            "rle_loss": float(rle_loss),
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
            f"(box:{box_loss:.3f} pose:{pose_loss:.3f} kobj:{kobj_loss:.3f} "
            f"cls:{cls_loss:.3f} rle:{rle_loss:.3f}) | "
            f"box mAP50:{metrics['map50']:.4f} | pose mAP50:{metrics['pose_map50']:.4f} "
            f"mAP50-95:{metrics['pose_map']:.4f}"
        )

    load_checkpoint(qat_model, best_checkpoint)
    best_metrics = evaluate(
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )
    print(
        f"[QAT-{tag}] Best epoch:{best_epoch}/{epochs} | "
        f"pose mAP50:{best_metrics['pose_map50']:.4f} mAP50-95:{best_metrics['pose_map']:.4f} | "
        f"Best: {best_checkpoint}"
    )
    qat_checkpoint = save_quant_outputs(
        qat_model, f"qat_{tag}", nc=num_classes, meta=best_meta, scale=scale
    )[0]
    return qat_checkpoint, best_fitness, best_meta


def compare_precision(quant_method=DEFAULT_QUANT_METHOD, batch_size=8, num_classes=NUM_CLASSES,
                      num_workers=2, max_eval_batches=None, scale=det.DEFAULT_SCALE):
    set_quant_method(quant_method)
    tag = quant_method
    print(f"========== Float vs QAT precision ({det.model_name(scale)}-pose / coco8-pose / {tag}) ==========")
    print(f"Device: {device}")

    base = det.base_name(scale, 'pose')
    # 优先 _best.pth，fallback 到无后缀
    float_checkpoint = os.path.join(MODEL_DIR, f"{base}_best.pth")
    if not os.path.exists(float_checkpoint):
        float_checkpoint = os.path.join(MODEL_DIR, f"{base}.pth")
    qat_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_{base}_best.pth")
    if not os.path.exists(qat_checkpoint):
        qat_checkpoint = os.path.join(MODEL_DIR, f"qat_{tag}_{base}.pth")
    if not os.path.exists(float_checkpoint):
        raise FileNotFoundError(f"找不到浮点权重，请先运行 float_train(): {float_checkpoint}")
    if not os.path.exists(qat_checkpoint):
        raise FileNotFoundError(
            f"找不到 {tag} 的 QAT 权重，请先运行 QAT_training('{tag}'): {qat_checkpoint}"
        )

    _, val_loader, data = get_dataloaders(batch_size, num_workers)

    float_model = FloatYOLO26Pose(num_classes, scale=scale).to(device)
    float_model, float_meta = load_checkpoint(float_model, float_checkpoint, return_meta=True)
    float_metrics = evaluate(
        float_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )

    qat_model = QuantYOLO26Pose(num_classes, scale=scale).to(device)
    qat_model, qat_meta = load_checkpoint(qat_model, qat_checkpoint, return_meta=True)
    saved_method = qat_meta.get("quant_method")
    if saved_method is not None and saved_method != tag:
        print(f"      [warn] QAT checkpoint 记录的方法为 {saved_method}，当前为 {tag}")
    saved_scale = qat_meta.get("scale")
    if saved_scale is not None and saved_scale != det.get_scale(scale):
        print(f"      [warn] QAT checkpoint 记录的尺度为 yolo26{saved_scale}-pose，当前为 yolo26{det.get_scale(scale)}-pose")
    quant_pkg.freeze_batch_init(qat_model)
    qat_metrics = evaluate(
        qat_model, val_loader, data, batch_size=batch_size, max_batches=max_eval_batches
    )

    print(
        f"[Compare] Float | best epoch:{float_meta.get('epoch', '-')}/"
        f"{float_meta.get('total_epochs', '-')} | "
        f"box mAP50:{float_metrics['map50']:.4f} | "
        f"pose P:{float_metrics['pose_precision']:.4f} R:{float_metrics['pose_recall']:.4f} "
        f"mAP50:{float_metrics['pose_map50']:.4f} mAP50-95:{float_metrics['pose_map']:.4f}"
    )
    print(
        f"[Compare] QAT   | best epoch:{qat_meta.get('epoch', '-')}/"
        f"{qat_meta.get('total_epochs', '-')} | "
        f"box mAP50:{qat_metrics['map50']:.4f} | "
        f"pose P:{qat_metrics['pose_precision']:.4f} R:{qat_metrics['pose_recall']:.4f} "
        f"mAP50:{qat_metrics['pose_map50']:.4f} mAP50-95:{qat_metrics['pose_map']:.4f}"
    )
    print(
        f"[Compare] delta | box mAP50:{qat_metrics['map50'] - float_metrics['map50']:+.4f} | "
        f"pose mAP50:{qat_metrics['pose_map50'] - float_metrics['pose_map50']:+.4f} "
        f"mAP50-95:{qat_metrics['pose_map'] - float_metrics['pose_map']:+.4f}"
    )
    return {"float": float_metrics, "qat": qat_metrics, "float_meta": float_meta, "qat_meta": qat_meta}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="YOLO26n-pose: 浮点训练(加载 yolo26n-pose.pt) -> PTQ -> QAT -> box/pose mAP 对比，模型尺度与量化方法可选"
    )
    parser.add_argument(
        "--model",
        choices=det.MODEL_CHOICES,
        default=f"yolo26{det.DEFAULT_SCALE}",
        help="模型尺度（默认 yolo26n；仅 yolo26n-pose 随仓库提供 .pt 预训练权重）",
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
        default=COCO8_POSE_YAML,
        help="数据集 yaml 路径（默认 coco8-pose.yaml）",
    )
    parser.add_argument("--resume", action="store_true",
                        help="从 _last.pth checkpoint 接续训练")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    # 根据 --data yaml 动态覆盖全局 NUM_CLASSES 和 COCO8_POSE_YAML
    COCO8_POSE_YAML = args.data  # 复用变量名；函数内部引用它
    try:
        _tmp = det.check_det_dataset(args.data)
        NUM_CLASSES = len(_tmp["names"])
    except Exception:
        pass  # yaml 解析失败则保留默认 1
    print(f"数据集: {args.data} | 类别数: {NUM_CLASSES}")

    scale = det.get_scale(args.model)
    print(f"Python: {sys.executable}")
    print(f"torch: {torch.__version__} | Device: {device}")
    print(f"模型: {det.model_name(scale)}-pose | 量化方法: {args.quant} | 阶段: {args.stage}")

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
