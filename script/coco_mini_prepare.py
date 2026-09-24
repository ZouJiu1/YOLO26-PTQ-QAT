#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coco_mini_prepare.py — 从完整 COCO2017 构造 1/20 规模的迷你数据集
================================================================

目的
----
用完整 COCO 的 **二十分之一** 数据，快速验证 YOLO26 的三类任务
（detect 检测 / segment 分割 / pose 姿态）在
    float 浮点训练 -> PTQ 后训练量化 -> QAT 量化感知训练
全流程上能否跑通，而不必动用 11.8 万张训练图。

采样规则（确定性，可复现）
--------------------------
* 把每个 split 的候选图片 id 排序后，每隔 20 张取 1 张（ids[::20]），
  不使用随机数，因此任何人、任何机器跑出来的子集完全一致。
* detect / segment 的候选池 = instances_*.json 中的全部图片
      train: 118287 -> 约 5915 张； val: 5000 -> 250 张
* pose 只有 person 类，候选池 = person_keypoints_*.json 中
  「至少含 1 个非 crowd 人物标注」的图片（与官方 coco-pose 口径一致）。

产物布局
--------
<COCO根>/mini/
├── detect/
│   ├── images/{train2017,val2017}/xxx.jpg   # 软链接，指向原始大图（不复制，省磁盘）
│   ├── labels/{train2017,val2017}/xxx.txt   # YOLO 检测框: cls cx cy w h（归一化）
│   ├── train.txt / val.txt                  # 图片清单（相对 path）
├── seg/   （同构；labels 为多边形: cls x1 y1 x2 y2 ...）
├── pose/  （同构；labels 为: cls cx cy w h + 17×(x y v)）
├── coco_mini_detect.yaml
├── coco_mini_seg.yaml
└── coco_mini_pose.yaml

用法
----
    python script/coco_mini_prepare.py                 # 用默认 COCO 根
    python script/coco_mini_prepare.py --coco-root /path/to/coco --ratio 20

注意
----
* 图片一律用 **相对软链接**，目录整体迁移后仍有效；不复制图片，几乎不占额外空间。
* iscrowd=1 的标注跳过（与官方 coco2017labels 转换口径一致）。
* COCO 原始 category id 不连续（1..90），脚本会重映射为连续的 0..79。
"""

import argparse
import json
import os
from collections import defaultdict


# --------------------------------------------------------------------------- #
# COCO 17 个关键点的标准信息（仅 pose 任务使用）
# --------------------------------------------------------------------------- #
# 水平翻转时关键点的交换索引（官方 coco-pose.yaml 固定值）
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
# 17 个关键点名称
KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def load_json(path):
    """读取一个 COCO json 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_every_nth(ids, ratio):
    """确定性等间隔采样：排序后每 ratio 个取 1 个。"""
    return sorted(ids)[::ratio]


def ensure_dir(path):
    """递归创建目录（已存在则跳过）。"""
    os.makedirs(path, exist_ok=True)


def link_image(src_abs, link_abs):
    """为图片创建相对软链接；已存在则先删掉重建，保证指向正确。"""
    if os.path.islink(link_abs) or os.path.exists(link_abs):
        os.remove(link_abs)
    # 计算相对路径，使整个 mini 目录可搬迁
    rel = os.path.relpath(src_abs, os.path.dirname(link_abs))
    os.symlink(rel, link_abs)


def write_list_file(path, lines):
    """写图片清单 txt（每行一条相对路径）。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_category_maps(instances):
    """
    根据 instances json 建立类别映射。

    返回:
        id_to_yolo : COCO 原始 category id -> 连续 0..79
        yolo_names : {连续 id: 类别名}
    """
    sorted_cats = sorted(instances["categories"], key=lambda c: c["id"])
    id_to_yolo = {c["id"]: i for i, c in enumerate(sorted_cats)}
    yolo_names = {i: c["name"] for i, c in enumerate(sorted_cats)}
    return id_to_yolo, yolo_names


def group_annotations(annotations):
    """把标注按 image_id 聚合成字典: image_id -> [ann, ...]。"""
    grouped = defaultdict(list)
    for ann in annotations:
        grouped[ann["image_id"]].append(ann)
    return grouped


# --------------------------------------------------------------------------- #
# 标签生成
# --------------------------------------------------------------------------- #
def make_detect_labels(image_info, anns, id_to_yolo):
    """
    生成单张图片的 YOLO detect 标签行。

    COCO bbox = [x, y, w, h]（左上角 + 宽高，单位像素）
    YOLO 格式 = cls cx cy w h（全部按图片宽高归一化的中心点格式）
    """
    W, H = image_info["width"], image_info["height"]
    lines = []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue  # 跳过 crowd
        x, y, w, h = ann["bbox"]
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        nw, nh = w / W, h / H
        cls = id_to_yolo[ann["category_id"]]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def make_seg_labels(image_info, anns, id_to_yolo):
    """
    生成单张图片的 YOLO segment 标签行。

    每个多边形实例写一行（一个 ann 可能含多个多边形，则写多行）：
        cls x1 y1 x2 y2 ...   （坐标全部归一化）
    crowd 标注的分割是 RLE（非多边形列表），直接跳过。
    """
    W, H = image_info["width"], image_info["height"]
    lines = []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        seg = ann.get("segmentation", [])
        cls = id_to_yolo[ann["category_id"]]
        # segmentation 为 [[x,y,x,y,...], ...]，一个元素就是一个多边形
        if not isinstance(seg, list):
            continue  # RLE 等非多边形格式，跳过
        for polygon in seg:
            if len(polygon) < 6:  # 至少 3 个点
                continue
            coords = []
            for i in range(0, len(polygon), 2):
                coords.append(f"{polygon[i] / W:.6f}")
                coords.append(f"{polygon[i + 1] / H:.6f}")
            lines.append(f"{cls} " + " ".join(coords))
    return lines


def make_pose_labels(image_info, anns):
    """
    生成单张图片的 YOLO pose 标签行。

    每行: cls cx cy w h  x1 y1 v1  x2 y2 v2 ... x17 y17 v17
        * bbox 与关键点坐标均归一化；
        * 关键点 v 为可见性标志（0 未标 / 1 被遮挡 / 2 可见），保持原值；
        * 约定 v==0 的点把 x,y 也清零（符合常见 YOLO pose 训练约定）。
    """
    W, H = image_info["width"], image_info["height"]
    lines = []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        nw, nh = w / W, h / H
        kpts = ann["keypoints"]  # 长度 51 = 17 × (x, y, v)
        tokens = []
        for i in range(0, len(kpts), 3):
            kx, ky, v = kpts[i], kpts[i + 1], kpts[i + 2]
            if v == 0:
                tokens.extend(["0.000000", "0.000000", "0"])
            else:
                tokens.append(f"{kx / W:.6f}")
                tokens.append(f"{ky / H:.6f}")
                tokens.append(str(int(v)))
        # pose 只有 person 一类，cls 固定 0
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(tokens))
    return lines


# --------------------------------------------------------------------------- #
# 单个任务的 mini 数据集构造
# --------------------------------------------------------------------------- #
def build_task(task, split, selected_ids, id_by_img, anns_by_img,
               src_image_dir, task_dir, id_to_yolo):
    """
    为某个任务、某个 split 建立 images 软链接 + labels + 图片清单。

    参数:
        task          : 'detect' / 'seg' / 'pose'
        split         : 'train2017' / 'val2017'
        selected_ids  : 采样得到的图片 id 列表
        id_by_img     : image_id -> 图片信息 dict
        anns_by_img   : image_id -> 标注列表
        src_image_dir : 原始图片所在目录
        task_dir      : 该任务的产物根目录（mini/<task>）
        id_to_yolo    : 类别 id 映射（pose 不用）
    返回:
        list_lines    : 供 train.txt/val.txt 使用的相对路径行
    """
    img_out_dir = os.path.join(task_dir, "images", split)
    lbl_out_dir = os.path.join(task_dir, "labels", split)
    ensure_dir(img_out_dir)
    ensure_dir(lbl_out_dir)

    list_lines = []
    n_empty = 0
    for img_id in selected_ids:
        info = id_by_img[img_id]
        file_name = info["file_name"]                       # 如 000000000001.jpg
        stem = os.path.splitext(file_name)[0]

        # 1) 图片软链接（仅当原始文件确实存在时）
        src_img = os.path.join(src_image_dir, file_name)
        if not os.path.exists(src_img):
            continue  # 原始图片缺失则跳过，避免死链
        link_image(src_img, os.path.join(img_out_dir, file_name))
        list_lines.append(f"./images/{split}/{file_name}")  # ./ 前缀：ultralytics 据此拼绝对路径

        # 2) 生成标签
        anns = anns_by_img.get(img_id, [])
        if task == "detect":
            label_lines = make_detect_labels(info, anns, id_to_yolo)
        elif task == "seg":
            label_lines = make_seg_labels(info, anns, id_to_yolo)
        else:  # pose
            label_lines = make_pose_labels(info, anns)

        # 没有有效标注的图片写空 txt（保持图片/标签一一对应）
        if not label_lines:
            n_empty += 1
        with open(os.path.join(lbl_out_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

    # 3) 图片清单
    write_list_file(os.path.join(task_dir, f"{split}.txt"), list_lines)
    print(f"    [{task}/{split}] 图片 {len(list_lines)} 张，其中空标签 {n_empty} 张")
    return list_lines


# --------------------------------------------------------------------------- #
# yaml 生成
# --------------------------------------------------------------------------- #
def dump_names_block(names_dict, indent=2):
    """把 {id: name} 渲染成 yaml 的 names 块。"""
    pad = " " * indent
    return "\n".join(f"{pad}{k}: {v}" for k, v in sorted(names_dict.items()))


def write_detect_seg_yaml(yaml_path, task_dir_name, names_dict, task_title):
    """写 detect / seg 的数据集 yaml。"""
    content = (
        f"# 自动生成：COCO 1/20 迷你数据集 — {task_title}\n"
        f"# 由 script/coco_mini_prepare.py 生成，请勿手改\n"
        f"path: {task_dir_name}        # 数据集根目录（相对本 yaml 所在目录）\n"
        f"train: train2017.txt         # 训练图片清单（相对 path）\n"
        f"val: val2017.txt             # 验证图片清单\n"
        f"\n"
        f"# 80 个类别\n"
        f"names:\n"
        f"{dump_names_block(names_dict)}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


def write_pose_yaml(yaml_path, task_dir_name):
    """写 pose 的数据集 yaml（含 kpt_shape / flip_idx / kpt_names）。"""
    kpt_names_block = "\n".join(f"    - {n}" for n in KPT_NAMES)
    content = (
        "# 自动生成：COCO 1/20 迷你数据集 — pose\n"
        "# 由 script/coco_mini_prepare.py 生成，请勿手改\n"
        f"path: {task_dir_name}\n"
        "train: train2017.txt\n"
        "val: val2017.txt\n"
        "\n"
        "# 关键点配置：17 个关键点，坐标维度 3 = x,y,visible\n"
        "kpt_shape: [17, 3]\n"
        f"flip_idx: {FLIP_IDX}\n"
        "\n"
        "# 类别（pose 只有 person）\n"
        "names:\n"
        "  0: person\n"
        "\n"
        "# 关键点名称\n"
        "kpt_names:\n"
        "  0:\n"
        f"{kpt_names_block}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="从完整 COCO2017 构造 1/20 的 detect/seg/pose 迷你数据集"
    )
    default_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..",
                     "ultralytics", "ultralytics", "data", "datasets", "coco")
    )
    parser.add_argument("--coco-root", default=default_root,
                        help="COCO 数据集根目录（内含 train2017/ val2017/ annotations/）")
    parser.add_argument("--ratio", type=int, default=100,
                        help="抽样比例，默认 20（取 1/20）")
    args = parser.parse_args()

    coco_root = args.coco_root
    ratio = args.ratio
    mini_root = os.path.join(coco_root, "mini")
    ensure_dir(mini_root)
    print(f"COCO 根目录 : {coco_root}")
    print(f"mini 输出   : {mini_root}")
    print(f"抽样比例    : 1/{ratio}\n")

    splits = [("train2017", "train"), ("val2017", "val")]

    # ---- 载入 detect/seg 用的 instances 标注 ---- #
    inst_data = {}
    for split, _tag in splits:
        inst_data[split] = load_json(
            os.path.join(coco_root, "annotations", f"instances_{split}.json"))
    # 以 train 的类别表为准建立映射（train/val 类别一致）
    id_to_yolo, yolo_names = build_category_maps(inst_data["train2017"])

    # ---- 载入 pose 用的关键点标注 ---- #
    pose_data = {}
    for split, _tag in splits:
        pose_data[split] = load_json(
            os.path.join(coco_root, "annotations", f"person_keypoints_{split}.json"))

    # ---- 逐 split、逐任务构造 ---- #
    for split, _tag in splits:
        print(f"=== {split} ===")
        src_image_dir = os.path.join(coco_root, split)

        # instances：图片信息 + 按图聚合的标注
        inst = inst_data[split]
        inst_img_by_id = {im["id"]: im for im in inst["images"]}
        inst_anns_by_img = group_annotations(inst["annotations"])

        # detect / seg：候选池为全部图片
        selected_all = pick_every_nth(list(inst_img_by_id.keys()), ratio)

        # pose：候选池 = 至少有 1 个非 crowd person 标注的图片
        pose = pose_data[split]
        pose_img_by_id = {im["id"]: im for im in pose["images"]}
        person_imgs = {
            ann["image_id"]
            for ann in pose["annotations"]
            if not ann.get("iscrowd", 0)
        }
        selected_pose = pick_every_nth(sorted(person_imgs), ratio)
        pose_anns_by_img = group_annotations(pose["annotations"])

        # 三个任务分别建数据
        build_task("detect", split, selected_all, inst_img_by_id,
                   inst_anns_by_img, src_image_dir,
                   os.path.join(mini_root, "detect"), id_to_yolo)
        build_task("seg", split, selected_all, inst_img_by_id,
                   inst_anns_by_img, src_image_dir,
                   os.path.join(mini_root, "seg"), id_to_yolo)
        build_task("pose", split, selected_pose, pose_img_by_id,
                   pose_anns_by_img, src_image_dir,
                   os.path.join(mini_root, "pose"), id_to_yolo)
        print()

    # ---- 生成 3 个 yaml（path 用绝对路径，避免 ultralytics 解析到默认数据集目录）---- #
    write_detect_seg_yaml(
        os.path.join(mini_root, "coco_mini_detect.yaml"),
        os.path.join(mini_root, "detect"), yolo_names, "detect")
    write_detect_seg_yaml(
        os.path.join(mini_root, "coco_mini_seg.yaml"),
        os.path.join(mini_root, "seg"), yolo_names, "segment")
    write_pose_yaml(
        os.path.join(mini_root, "coco_mini_pose.yaml"),
        os.path.join(mini_root, "pose"))

    print("全部完成！生成的 yaml：")
    for name in ("coco_mini_detect.yaml", "coco_mini_seg.yaml", "coco_mini_pose.yaml"):
        print(f"  {os.path.join(mini_root, name)}")


if __name__ == "__main__":
    main()
