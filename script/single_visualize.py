#!/usr/bin/env python3
"""单张图片或目录批量推理 + 可视化脚本（yolo26 六任务：detect/seg/pose/cls/obb/depth）。
/ Single-image or directory batch inference + visualization script (yolo26 six tasks: detect/seg/pose/cls/obb/depth).

加载任意 checkpoint（float best/last、PTQ、QAT 的 *.pth），对单张图片或目录中所有图片推理并保存可视化结果。
/ Load any checkpoint (float best/last, PTQ, QAT *.pth), run single-image or batch inference and save visualization.

用法（在 QAT_training 目录下运行）： / Usage (run from QAT_training directory):
    # 单张图片 / Single image
    python script/single_visualize.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth --image ultralytics/ultralytics/data/datasets/coco128-seg/images/train2017/000000000009.jpg

    # 目录批量处理 / Batch process directory
    python script/single_visualize.py --task seg --ckpt model/yolo26-seg/n/lsqplus_v1/qat_lsqplus_v1_yolo26n-seg_best.pth --image ultralytics/ultralytics/data/datasets/coco128-seg/images/train2017 --out results/seg_batch

    # 更多任务示例 / More task examples
    python script/single_visualize.py --task pose --ckpt model/yolo26-pose/n/yolo26n-pose_best.pth --image <图片或目录>
    python script/single_visualize.py --task cls --ckpt model/yolo26-cls/n/yolo26n-cls_best.pth --image <图片或目录>
    python script/single_visualize.py --task obb --ckpt model/yolo26-obb/n/yolo26n-obb_best.pth --image <DOTA 图片或目录>
    python script/single_visualize.py --task depth --ckpt model/yolo26-depth/n/yolo26n-depth_best.pth --image ultralytics/ultralytics/data/datasets/depth8-png/images/train

输出 / Output:
    单张模式 / Single-image mode:
        {out_dir}/{prefix}_{task}_result.jpg
        默认同 checkpoint 目录 / Default: same dir as ckpt
    目录批量模式 / Directory batch mode:
        {out_dir}/{image_name}_result.jpg（每张图一个文件） / One file per image
        默认 {ckpt_dir}/{prefix}_batch_results/ / Default: {ckpt_dir}/{prefix}_batch_results/
"""

import argparse
import importlib.util
import os
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 让脚本可以从仓库根目录直接运行 / Allow script to run directly from repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

TASKS = ["detect", "seg", "pose", "cls", "obb", "depth"]

# 各任务的 Float / Quant 模型类名 / Float / Quant model class names per task
FLOAT_CLS = {
    "detect": "FloatYOLO26",
    "seg": "FloatYOLO26Seg",
    "pose": "FloatYOLO26Pose",
    "cls": "FloatYOLO26Cls",
    "obb": "FloatYOLO26OBB",
    "depth": "FloatYOLO26Depth",
}
QUANT_CLS = {
    "detect": "QuantYOLO26",
    "seg": "QuantYOLO26Seg",
    "pose": "QuantYOLO26Pose",
    "cls": "QuantYOLO26Cls",
    "obb": "QuantYOLO26OBB",
    "depth": "QuantYOLO26Depth",
}

# 各任务的默认类别数 / Default number of classes per task
DEFAULT_NC = {
    "detect": 80,
    "seg": 80,
    "pose": 1,
    "cls": 10,
    "obb": 15,
    "depth": 1,
}

# 各任务的默认输入尺寸 / Default input size per task
DEFAULT_IMGSZ = {
    "detect": 640,
    "seg": 640,
    "pose": 640,
    "cls": 224,
    "obb": 640,
    "depth": 640,
}

# COCO 80 类名（detect/seg 默认）/ COCO 80 class names (default for detect/seg)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# DOTA 15 类名（obb 默认）/ DOTA 15 class names (default for obb)
DOTA_NAMES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool",
]

# imagenet10 类名（cls 默认）/ imagenet10 class names (default for cls)
IMAGENET10_NAMES = [
    "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
    "electric_ray", "stingray", "cock", "hen", "ostrich",
]

# 各任务默认类别名 / Default class names per task
DEFAULT_NAMES = {
    "detect": COCO_NAMES,
    "seg": COCO_NAMES,
    "pose": ["person"],
    "cls": IMAGENET10_NAMES,
    "obb": DOTA_NAMES,
    "depth": [],
}


def load_task_module(task):
    """按任务动态加载 networks_yolo26-{task}.py / Dynamically load networks_yolo26-{task}.py."""
    path = os.path.join(BASE_DIR, f"networks_yolo26-{task}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未知任务 {task}，找不到 {path}")
    spec = importlib.util.spec_from_file_location(f"networks_yolo26_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peek_checkpoint_meta(ckpt_path):
    """只读 checkpoint 的 meta 字段 / Read-only peek at checkpoint's meta field."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "meta" in ckpt and isinstance(ckpt["meta"], dict):
        return ckpt["meta"]
    return {}


def derive_prefix(ckpt_path):
    """从 checkpoint 文件名推导输出前缀 / Derive output prefix from checkpoint filename."""
    base = os.path.basename(ckpt_path)
    return re.sub(r"\.(pth|pt|ckpt)$", "", base, flags=re.IGNORECASE)


def build_model(module, task, meta, ckpt_path, quant_method, scale, nc, device):
    """根据 checkpoint meta 判型并构建模型 / Build float or quant model based on checkpoint meta."""
    stage = meta.get("stage", "float")

    if stage in ("ptq", "qat"):
        module.set_quant_method(quant_method)
        model_cls = getattr(module, QUANT_CLS[task])
        model = model_cls(nc=nc, scale=scale).to(device)
        print(f"[single_visualize] 量化模型 {QUANT_CLS[task]} (method={quant_method})")
    else:
        model_cls = getattr(module, FLOAT_CLS[task])
        model = model_cls(nc=nc, scale=scale).to(device)
        print(f"[single_visualize] 浮点模型 {FLOAT_CLS[task]}")

    _, load_meta = module.load_checkpoint(model, ckpt_path, return_meta=True)
    print(f"[single_visualize] checkpoint stage={load_meta.get('stage')} epoch={load_meta.get('epoch')}"
          f" fitness={load_meta.get('fitness')}")
    model.eval()
    return model


def preprocess(image_path, imgsz):
    """用 ultralytics LetterBox 做标准预处理 / Standard preprocessing using ultralytics LetterBox.

    Returns:
        img0: 原始 BGR 图片 (H, W, 3) / original BGR image
        input_tensor: 模型输入 (1, 3, H, W) float [0, 1] / model input tensor
        ratio_pad: ((ratio_h, ratio_w), (pad_w, pad_h)) for scale_boxes
    """
    from ultralytics.data.augment import LetterBox

    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(f"图片不存在或无法读取 / Image not found or unreadable: {image_path}")

    lb = LetterBox(new_shape=(imgsz, imgsz), auto=True, scaleup=True, stride=32)
    img = lb(image=img0)  # BGR, padded

    # 从 LetterBox 内部推算 ratio 和 padding / Derive ratio and padding from LetterBox internals
    orig_h, orig_w = img0.shape[:2]
    new_h, new_w = img.shape[:2]
    r = min(new_h / orig_h, new_w / orig_w)
    r = min(r, 1.0) if not lb.scaleup else r
    new_unpad_w, new_unpad_h = round(orig_w * r), round(orig_h * r)
    pad_w = (new_w - new_unpad_w) / 2.0
    pad_h = (new_h - new_unpad_h) / 2.0
    ratio_pad = ((r, r), (pad_w, pad_h))

    # BGR -> RGB, HWC -> CHW, to float tensor
    img_rgb = img[:, :, ::-1].transpose(2, 0, 1)
    img_rgb = np.ascontiguousarray(img_rgb)
    input_tensor = torch.from_numpy(img_rgb).float().unsqueeze(0) / 255.0

    return img0, input_tensor, ratio_pad


def get_class_names(task, module, nc, meta):
    """获取类别名列表 / Get class names list.

    优先级：checkpoint meta > 模块 data > 硬编码默认 / Priority: checkpoint meta > module data > hardcoded default.
    """
    # 尝试从 checkpoint meta / Try checkpoint meta
    names = meta.get("names")
    if names is not None:
        if isinstance(names, dict):
            return [names.get(i, f"class_{i}") for i in range(len(names))]
        return list(names)

    # 尝试从模块数据 / Try module data
    try:
        data = module.check_det_dataset(module.DATASET) if hasattr(module, "DATASET") else None
        if data is not None:
            if hasattr(data, "names"):
                names = data.names
            elif isinstance(data, dict):
                names = data.get("names")
            if names is not None:
                if isinstance(names, dict):
                    return [names.get(i, f"class_{i}") for i in range(len(names))]
                return list(names)
    except Exception:
        pass

    # 硬编码默认 / Hardcoded defaults
    defaults = DEFAULT_NAMES.get(task, [])
    if len(defaults) >= nc:
        return defaults[:nc]
    # 补齐 / Pad
    return defaults + [f"class_{i}" for i in range(len(defaults), nc)]


def _draw_obb(img, cx, cy, w, h, angle_rad, color, label=None):
    """在原图上画旋转框 / Draw rotated bounding box on original image."""
    corners = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(np.degrees(angle_rad))))
    corners = np.intp(corners)
    cv2.polylines(img, [corners], isClosed=True, color=color, thickness=2)
    if label:
        # 在左上角写标签 / Put label at top-left corner
        x_min = int(corners[:, 0].min())
        y_min = int(corners[:, 1].min())
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(img, (x_min, y_min - th - 6), (x_min + tw, y_min), color, -1)
        cv2.putText(img, label, (x_min, y_min - 4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


@torch.no_grad()
def run_detect(model, input_tensor, img0, ratio_pad, args, names, device):
    """detect 推理 + 可视化 / detect inference + visualization."""
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    from ultralytics.utils.plotting import Annotator, colors

    input_shape = input_tensor.shape[2:]  # (H, W)
    pred_raw = model(input_tensor.to(device))  # (1, 4+nc, N)

    predictions = non_max_suppression(
        pred_raw,
        conf_thres=args.conf,
        iou_thres=args.iou,
        nc=len(names),
        multi_label=True,
        agnostic=False,
        max_det=300,
    )
    pred = predictions[0]  # (N, 6) xyxy, conf, cls

    # 坐标还原到原图 / Scale boxes back to original image
    if pred.shape[0]:
        pred[:, :4] = scale_boxes(input_shape, pred[:, :4], img0.shape[:2], ratio_pad=ratio_pad)

    annotator = Annotator(img0, line_width=2, example=str(names))
    for det in pred:
        xyxy, conf, cls_id = det[:4], det[4].item(), int(det[5].item())
        label = f"{names[cls_id]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=colors(cls_id, True))

    return annotator.result()


@torch.no_grad()
def run_seg(model, input_tensor, img0, ratio_pad, args, names, device):
    """seg 推理 + 可视化 / seg inference + visualization."""
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes, process_mask
    from ultralytics.utils.plotting import Annotator, colors

    input_shape = input_tensor.shape[2:]  # (H, W)
    det_raw, protos = model(input_tensor.to(device))  # det: (1, 4+nc+nm, N), protos: (1, nm, mh, mw)

    nm = protos.shape[1]  # mask coefficient count

    predictions = non_max_suppression(
        det_raw,
        conf_thres=args.conf,
        iou_thres=args.iou,
        nc=len(names),
        multi_label=True,
        agnostic=False,
        max_det=300,
    )
    pred = predictions[0]  # (N, 6+nm) xyxy, conf, cls, mask_coeff

    if pred.shape[0] == 0:
        return img0.copy()

    # 处理 mask / Process masks
    mask_coeffs = pred[:, 6:6 + nm]
    boxes_in = pred[:, :4]
    masks = process_mask(protos[0], mask_coeffs, boxes_in, shape=input_shape, upsample=True)
    # (N, input_h, input_w) binary

    # 坐标还原 / Scale boxes to original image
    pred[:, :4] = scale_boxes(input_shape, pred[:, :4], img0.shape[:2], ratio_pad=ratio_pad)

    # Mask 也要还原到原图 / Scale masks to original image
    masks_np = masks.cpu().numpy()  # (N, input_h, input_w)
    orig_h, orig_w = img0.shape[:2]
    (_, _), (pad_w, pad_h) = ratio_pad
    # 去掉 padding / Remove padding
    pad_w_int, pad_h_int = int(round(pad_w)), int(round(pad_h))
    input_h, input_w = input_shape
    new_unpad_w = int(round(input_w - 2 * pad_w))
    new_unpad_h = int(round(input_h - 2 * pad_h))
    # Crop then resize
    masks_cropped = masks_np[:, pad_h_int:pad_h_int + new_unpad_h, pad_w_int:pad_w_int + new_unpad_w]
    if masks_cropped.shape[1] != orig_h or masks_cropped.shape[2] != orig_w:
        masks_cropped = np.stack([
            cv2.resize(m.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            for m in masks_cropped
        ])
    masks_binary = torch.from_numpy(masks_cropped > 0.5)

    annotator = Annotator(img0, line_width=2, example=str(names))

    # 画 mask overlay / Draw mask overlay
    if masks_binary.shape[0] > 0:
        mask_colors = [colors(int(pred[i, 5]), True) for i in range(pred.shape[0])]
        annotator.masks(masks_binary, colors=mask_colors, alpha=0.35)

    # 画 bbox + label / Draw bbox + label
    for det in pred:
        xyxy, conf, cls_id = det[:4], det[4].item(), int(det[5].item())
        label = f"{names[cls_id]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=colors(cls_id, True))

    return annotator.result()


@torch.no_grad()
def run_pose(model, input_tensor, img0, ratio_pad, args, names, device):
    """pose 推理 + 可视化 / pose inference + visualization."""
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes, scale_coords
    from ultralytics.utils.plotting import Annotator, colors

    input_shape = input_tensor.shape[2:]  # (H, W)
    pred_raw = model(input_tensor.to(device))  # (1, 4+nc+nk, N)

    # nkpt*ndim: 17*3 = 51
    nkpt_ndim = 51

    predictions = non_max_suppression(
        pred_raw,
        conf_thres=args.conf,
        iou_thres=args.iou,
        nc=len(names),
        multi_label=True,
        agnostic=False,
        max_det=300,
    )
    pred = predictions[0]  # (N, 6+51) xyxy, conf, cls, kpts

    if pred.shape[0] == 0:
        return img0.copy()

    # 坐标还原 / Scale boxes to original
    pred[:, :4] = scale_boxes(input_shape, pred[:, :4], img0.shape[:2], ratio_pad=ratio_pad)

    # 关键点还原 / Scale keypoints to original
    kpts = pred[:, 6:6 + nkpt_ndim].view(-1, 17, 3)  # (N, 17, 3)
    orig_shape_2d = img0.shape[:2]
    input_shape_2d = input_shape

    annotator = Annotator(img0, line_width=2, example=str(names))
    for i, det in enumerate(pred):
        xyxy, conf, cls_id = det[:4], det[4].item(), int(det[5].item())
        label = f"{names[cls_id]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=colors(cls_id, True))

        # 关键点 / Keypoints
        kp = kpts[i].clone()  # (17, 3)
        kp_scaled = scale_coords(input_shape_2d, kp[:, :2], orig_shape_2d, ratio_pad=ratio_pad)
        kp[:, :2] = kp_scaled
        annotator.kpts(kp, shape=orig_shape_2d, conf_thres=args.conf)

    return annotator.result()


@torch.no_grad()
def run_obb(model, input_tensor, img0, ratio_pad, args, names, device):
    """obb 推理 + 可视化 / obb inference + visualization."""
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.plotting import colors

    pred_raw = model(input_tensor.to(device))  # (1, 4+nc+1, N)

    predictions = non_max_suppression(
        pred_raw,
        conf_thres=args.conf,
        iou_thres=args.iou,
        nc=len(names),
        multi_label=True,
        agnostic=False,
        max_det=300,
        rotated=True,
    )
    pred = predictions[0]  # (N, 7) x,y,w,h,conf,cls,angle

    if pred.shape[0] == 0:
        return img0.copy()

    # OBB 结果: [x, y, w, h, conf, cls, angle]，中心点在 letterbox 坐标系 / OBB result in letterbox space
    # 坐标还原 / Scale to original image
    (r_h, r_w), (pad_w, pad_h) = ratio_pad

    vis_img = img0.copy()
    for det in pred:
        cx, cy, w, h = det[0].item(), det[1].item(), det[2].item(), det[3].item()
        conf = det[4].item()
        cls_id = int(det[5].item())
        angle = det[6].item()  # 弧度 / radians

        # 去 padding + 缩放 / Remove padding + scale
        cx = (cx - pad_w) / r_w
        cy = (cy - pad_h) / r_h
        w = w / r_w
        h = h / r_h

        color = colors(cls_id, True)
        label = f"{names[cls_id]} {conf:.2f}"
        _draw_obb(vis_img, cx, cy, w, h, angle, color, label)

    return vis_img


@torch.no_grad()
def run_depth(model, input_tensor, img0, device):
    """depth 推理 + 可视化（独立热图） / depth inference + visualization (standalone heatmap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = model(input_tensor.to(device))  # (1, 1, H/4, W/4)
    depth = pred[0, 0].float().cpu().numpy()  # (H/4, W/4)

    # Upsample 到原图尺寸 / Upsample to original image size
    orig_h, orig_w = img0.shape[:2]
    depth_up = F.interpolate(
        torch.from_numpy(depth)[None, None],
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    # 渲染为热图 / Render as heatmap
    fig, ax = plt.subplots(1, 1, figsize=(orig_w / 100, orig_h / 100), dpi=100)
    ax.imshow(depth_up, cmap="plasma")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    # 转为 BGR numpy / Convert to BGR numpy
    buf = fig.canvas.buffer_rgba()
    vis = np.asarray(buf)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    return vis


@torch.no_grad()
def run_cls(model, input_tensor, img0, names, device):
    """cls 推理 + 可视化 / cls inference + visualization."""
    from ultralytics.utils.plotting import Annotator

    pred = model(input_tensor.to(device))  # (1, nc) softmax
    probs = pred[0].cpu()

    topk = min(5, len(names))
    top_probs, top_indices = probs.topk(topk)

    annotator = Annotator(img0, line_width=2)
    y_offset = 30
    for i in range(topk):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        text = f"{i+1}. {names[idx]} {prob:.3f}"
        annotator.text((10, y_offset), text, txt_color=(255, 255, 255), anchor="top",
                       box_color=(0, 0, 0))
        y_offset += 30

    return annotator.result()


def main():
    parser = argparse.ArgumentParser(
        description="yolo26 六任务单张图片推理 + 可视化 / yolo26 six-task single-image inference + visualization"
    )
    parser.add_argument("--task", required=True, choices=TASKS,
                        help="任务类型 / Task type")
    parser.add_argument("--ckpt", required=True,
                        help="checkpoint 路径（.pth，支持 float / PTQ / QAT） / Checkpoint path (.pth, float/PTQ/QAT)")
    parser.add_argument("--image", required=True,
                        help="输入图片路径或目录（目录则批量处理所有图片） / Input image path or directory (batch process all images in dir)")
    parser.add_argument("--out", default=None,
                        help="输出路径或目录；默认同 ckpt 目录 / Output path or directory; default: same dir as ckpt")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入尺寸（默认按任务：detect/seg/pose/obb/depth=640, cls=224） / Input size (default per task)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 / Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值 / NMS IoU threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备 / Device")
    parser.add_argument("--show", action="store_true",
                        help="弹窗显示结果 / Show result in popup window")
    parser.add_argument("--quant-method", default=None,
                        help="量化后端（meta 缺失时手动指定） / Quant backend (manual if meta missing)")
    args = parser.parse_args()

    # --- 路径归一化 / Path normalization ---
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(BASE_DIR, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    image_path = args.image if os.path.isabs(args.image) else os.path.join(BASE_DIR, args.image)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    # --- 输出路径 / Output path ---
    prefix = derive_prefix(ckpt_path)
    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)
    else:
        out_dir = os.path.dirname(ckpt_path)
        out_path = os.path.join(out_dir, f"{prefix}_{args.task}_result.jpg")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- 加载任务模块 / Load task module ---
    module = load_task_module(args.task)

    # --- 读 checkpoint meta 判型 / Peek meta to determine model type ---
    meta = peek_checkpoint_meta(ckpt_path)
    stage = meta.get("stage", "float")

    # quant_method 优先级：命令行 > meta / quant_method priority: cli > meta
    quant_method = args.quant_method or meta.get("quant_method")
    if stage in ("ptq", "qat") and quant_method is None:
        raise RuntimeError(
            f"checkpoint stage={stage} 但 meta 中无 quant_method，请用 --quant-method 指定"
        )

    # scale: meta > 默认 n / scale: meta > default n
    scale = meta.get("scale") or "n"
    if scale and not scale.startswith("yolo26"):
        scale = scale.lstrip("yolo26")

    # nc: meta > 模块默认 / nc: meta > module default
    nc = meta.get("nc") or meta.get("num_classes") or getattr(module, "NUM_CLASSES", DEFAULT_NC[args.task])

    # imgsz: 命令行 > 任务默认 / imgsz: cli > task default
    imgsz = args.imgsz or DEFAULT_IMGSZ[args.task]

    # device
    device = torch.device(args.device)

    print(f"[single_visualize] task={args.task} stage={stage} scale=yolo26{scale} nc={nc} imgsz={imgsz}")
    print(f"[single_visualize] checkpoint : {ckpt_path}")
    print(f"[single_visualize] image      : {image_path}")
    print(f"[single_visualize] output     : {out_path}")
    print(f"[single_visualize] device     : {device}")

    # --- 获取类别名 / Get class names ---
    names = get_class_names(args.task, module, nc, meta)

    # --- 构建模型并加载权重 / Build model and load weights ---
    model = build_model(module, args.task, meta, ckpt_path, quant_method, scale, nc, device)

    # --- 批量处理 / Batch processing ---
    if os.path.isdir(image_path):
        # 目录模式：遍历所有图片 / Directory mode: loop all images
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = sorted([f for f in os.listdir(image_path)
                             if os.path.splitext(f)[1].lower() in image_exts])
        if not image_files:
            raise RuntimeError(f"目录中无图片 / No images in directory: {image_path}")

        # 输出目录：--out 指定则用之，否则同 ckpt 目录下新建子目录 / Output dir: use --out or create subdir in ckpt dir
        if args.out:
            out_dir = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)
        else:
            out_dir = os.path.join(os.path.dirname(ckpt_path), f"{prefix}_batch_results")
        os.makedirs(out_dir, exist_ok=True)

        print(f"[single_visualize] 批量处理 {len(image_files)} 张图片到 {out_dir} / Batch processing {len(image_files)} images to {out_dir}")

        for idx, img_name in enumerate(image_files, 1):
            img_path = os.path.join(image_path, img_name)
            out_name = os.path.splitext(img_name)[0] + "_result.jpg"
            out_path_i = os.path.join(out_dir, out_name)

            print(f"\n[{idx}/{len(image_files)}] {img_name}")

            # 预处理 / Preprocess
            img0, input_tensor, ratio_pad = preprocess(img_path, imgsz)

            # 推理 + 可视化 / Inference + visualization
            vis_img = _run_task(args.task, model, input_tensor, img0, ratio_pad, args, names, device)

            # 保存 / Save
            cv2.imwrite(out_path_i, vis_img)
            print(f"  → {out_path_i}")

        print(f"\n[single_visualize] 批量完成 / Batch done: {len(image_files)} images → {out_dir}")

    else:
        # 单张模式 / Single-image mode
        img0, input_tensor, ratio_pad = preprocess(image_path, imgsz)
        print(f"[single_visualize] input: {img0.shape} -> {input_tensor.shape}")

        vis_img = _run_task(args.task, model, input_tensor, img0, ratio_pad, args, names, device)

        cv2.imwrite(out_path, vis_img)
        print(f"[single_visualize] 结果已保存 / Result saved to: {out_path}")

        # 弹窗显示 / Popup display
        if args.show:
            cv2.imshow(f"{args.task} result", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def _run_task(task, model, input_tensor, img0, ratio_pad, args, names, device):
    """分发到各任务的推理 + 可视化函数 / Dispatch to task-specific inference + visualization."""
    if task == "detect":
        return run_detect(model, input_tensor, img0, ratio_pad, args, names, device)
    elif task == "seg":
        return run_seg(model, input_tensor, img0, ratio_pad, args, names, device)
    elif task == "pose":
        return run_pose(model, input_tensor, img0, ratio_pad, args, names, device)
    elif task == "obb":
        return run_obb(model, input_tensor, img0, ratio_pad, args, names, device)
    elif task == "depth":
        return run_depth(model, input_tensor, img0, device)
    elif task == "cls":
        return run_cls(model, input_tensor, img0, names, device)
    else:
        raise ValueError(f"未知任务 / Unknown task: {task}")


if __name__ == "__main__":
    '''
    # 单张图片 / Single image
    python3 script/single_visualize.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth --image ultralytics/ultralytics/data/datasets/coco128-seg/images/train2017/000000000009.jpg

    # 目录批量 / Directory batch
    python3 script/single_visualize.py --task seg --ckpt model/yolo26-seg/n/lsqplus_v1/qat_lsqplus_v1_yolo26n-seg_best.pth --image ultralytics/ultralytics/data/datasets/coco128-seg/images/train2017 --out results/seg
    python3 script/single_visualize.py --task depth --ckpt model/yolo26-depth/n/yolo26n-depth_best.pth --image ultralytics/ultralytics/data/datasets/depth8-png/images/train
    python3 script/single_visualize.py --task cls --ckpt model/yolo26-cls/n/yolo26n-cls_best.pth --image <imagenet10 目录> --out results/cls
    '''
    main()
