#!/usr/bin/env python3
"""统一的 yolo26 六任务验证集可视化脚本（detect/seg/pose/cls/obb/depth）。 / Unified yolo26 six-task val-set visualization script (detect/seg/pose/cls/obb/depth).

加载任意 checkpoint（float best/last、PTQ、QAT 的 *.pth），在验证集上推理并保存可视化图。
复用各任务模块已有的 evaluate()（内部调 ultralytics plot_images 保存 GT/预测拼图到
{viz_dir}/fvisualize/），不重新实现绘图。 / Load any checkpoint (float best/last, PTQ, QAT *.pth),
run inference on val set and save visualization images. Reuses each task module's existing
evaluate() (internally calls ultralytics plot_images to save GT/prediction mosaics into
{viz_dir}/fvisualize/), no re-implementation of plotting.

用法（在 QAT_training 目录下运行）： / Usage (run from QAT_training directory):
    # detect：float best / float best
    python script/visualize.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth

    # seg：QAT best / QAT best
    python script/visualize.py --task seg --ckpt model/yolo26-seg/n/lsqplus_v1/qat_lsqplus_v1_yolo26n-seg_best.pth

    # pose：QAT best / QAT best
    python script/visualize.py --task pose --ckpt model/yolo26-pose/n/lsqplus_v1/qat_lsqplus_v1_yolo26n-pose_best.pth

    # cls：float best / float best
    python script/visualize.py --task cls --ckpt model/yolo26-cls/n/yolo26n-cls_best.pth

    # obb：float best / float best
    python script/visualize.py --task obb --ckpt model/yolo26-obb/n/yolo26n-obb_best.pth

    # depth：float best / float best
    python script/visualize.py --task depth --ckpt model/yolo26-depth/n/yolo26n-depth_best.pth

输出 / Output:
    {ckpt_dir}/fvisualize/{viz_prefix}_batch{N}_labels.jpg  — GT 拼图 / GT mosaic
    {ckpt_dir}/fvisualize/{viz_prefix}_batch{N}_pred.jpg    — 预测拼图 / Prediction mosaic
    viz_prefix 自动从 checkpoint 文件名派生（去掉 .pth），如 float_best、qat_lsqplus_v1_yolo26n-seg_best。
    viz_prefix is auto-derived from checkpoint filename (strip .pth).
"""

import argparse
import glob
import importlib.util
import os
import re
import sys

import torch

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
    """从 checkpoint 文件名推导 viz_prefix：去掉 .pth 扩展名 / Derive viz_prefix from checkpoint filename."""
    base = os.path.basename(ckpt_path)
    return re.sub(r"\.(pth|pt|ckpt)$", "", base, flags=re.IGNORECASE)


def build_model(module, task, meta, quant_method, scale, nc):
    """根据 checkpoint meta 判型并构建模型 / Build float or quant model based on checkpoint meta."""
    stage = meta.get("stage", "float")
    device = module.device if hasattr(module, "device") else torch.device("cpu")

    if stage in ("ptq", "qat"):
        # 量化模型：先 set_quant_method 再构建 / Quant model: set_quant_method BEFORE constructing
        module.set_quant_method(quant_method)
        model_cls = getattr(module, QUANT_CLS[task])
        model = model_cls(nc=nc, scale=scale).to(device)
        print(f"[visualize] 量化模型 {QUANT_CLS[task]} (method={quant_method})")
    else:
        model_cls = getattr(module, FLOAT_CLS[task])
        model = model_cls(nc=nc, scale=scale).to(device)
        print(f"[visualize] 浮点模型 {FLOAT_CLS[task]}")

    _, load_meta = module.load_checkpoint(model, ckpt_path_global, return_meta=True)
    print(f"[visualize] checkpoint stage={load_meta.get('stage')} epoch={load_meta.get('epoch')}"
          f" fitness={load_meta.get('fitness')}")
    return model


def build_val_loader(module, task, batch_size, num_workers, split, data_override):
    """构建 val/train dataloader / Build val or train dataloader.

    无 --data 时直接用模块 get_dataloaders()；有 --data 时手动 check_dataset + _build_loaders。 /
    Without --data, use module get_dataloaders(); with --data, manually check_dataset + _build_loaders.
    """
    if data_override:
        # 覆盖数据集 yaml / Override dataset yaml
        if task == "cls":
            data = module.check_cls_dataset(data_override)
            cfg = module._make_cfg(num_workers)
            split_key = "train" if split == "train" else "val"
            augment = split == "train"
            loader = module._build_loader(cfg, data[split_key], data, batch_size, num_workers,
                                          augment, augment)
        else:
            data = module.check_det_dataset(data_override)
            train_loader, val_loader, _, _ = module._build_loaders(batch_size, num_workers, data)
            loader = train_loader if split == "train" else val_loader
        return loader, data
    else:
        # 默认数据管道 / Default data pipeline
        train_loader, val_loader, data = module.get_dataloaders(
            batch_size=batch_size, num_workers=num_workers)
        loader = train_loader if split == "train" else val_loader
        return loader, data


def run_evaluate(module, task, model, val_loader, data, batch_size, max_batches,
                 viz_dir, viz_max, viz_prefix):
    """调用模块级 evaluate() / Call module-level evaluate()."""
    if task == "cls":
        # cls evaluate 签名不同：无 data / batch_size / cls evaluate has different signature: no data / batch_size
        return module.evaluate(model, val_loader, max_batches=max_batches,
                               viz_dir=viz_dir, viz_max=viz_max, viz_prefix=viz_prefix)
    else:
        return module.evaluate(model, val_loader, data, batch_size=batch_size,
                               max_batches=max_batches, viz_dir=viz_dir,
                               viz_max=viz_max, viz_prefix=viz_prefix)


# 全局变量，供 build_model 使用 / Global for build_model to access
ckpt_path_global = None


def main():
    global ckpt_path_global

    parser = argparse.ArgumentParser(
        description="yolo26 六任务验证集可视化（detect/seg/pose/cls/obb/depth） / "
                    "yolo26 six-task val-set visualization (detect/seg/pose/cls/obb/depth)"
    )
    parser.add_argument("--task", required=True, choices=TASKS,
                        help="任务类型 / Task type")
    parser.add_argument("--ckpt", required=True,
                        help="checkpoint 路径（.pth，支持 float / PTQ / QAT） / Checkpoint path (.pth, float/PTQ/QAT)")
    parser.add_argument("--data", default=None,
                        help="覆盖数据集 yaml（可选） / Override dataset yaml (optional)")
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="数据集 split / Dataset split")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="batch size / batch size")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="最大推理 batch 数（默认全部） / Max inference batches (default: all)")
    parser.add_argument("--viz-max", type=int, default=30,
                        help="可视化最大图片数 / Max visualization images")
    parser.add_argument("--out-dir", default=None,
                        help="输出目录；默认 checkpoint 所在目录 / Output dir; default: checkpoint dir")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="dataloader workers / dataloader workers")
    parser.add_argument("--viz-prefix", default=None,
                        help="可视化文件名前缀；默认从 ckpt 文件名派生 / Viz filename prefix; default derived from ckpt")
    parser.add_argument("--quant-method", default=None,
                        help="量化后端（meta 缺失时手动指定） / Quant backend (manual if meta missing)")
    args = parser.parse_args()

    # --- 路径归一化 / Path normalization ---
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(BASE_DIR, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    ckpt_path_global = ckpt_path

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(ckpt_path)
    elif not os.path.isabs(out_dir):
        out_dir = os.path.join(BASE_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    viz_prefix = args.viz_prefix or derive_prefix(ckpt_path)

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
    nc = meta.get("nc") or meta.get("num_classes") or getattr(module, "NUM_CLASSES", 80)

    print(f"[visualize] task={args.task} stage={stage} scale=yolo26{scale} nc={nc}")
    print(f"[visualize] checkpoint : {ckpt_path}")
    print(f"[visualize] output dir : {out_dir}")
    print(f"[visualize] viz_prefix : {viz_prefix}")
    print(f"[visualize] split={args.split} batch_size={args.batch_size}"
          f" max_batches={args.max_batches} viz_max={args.viz_max}")

    # --- 构建模型并加载权重 / Build model and load weights ---
    model = build_model(module, args.task, meta, quant_method, scale, nc)

    # --- 构建 dataloader / Build dataloader ---
    val_loader, data = build_val_loader(
        module, args.task, args.batch_size, args.num_workers, args.split, args.data)

    # --- 评估 + 可视化 / Evaluate + visualize ---
    metrics = run_evaluate(
        module, args.task, model, val_loader, data,
        batch_size=args.batch_size, max_batches=args.max_batches,
        viz_dir=out_dir, viz_max=args.viz_max, viz_prefix=viz_prefix)

    # --- 收集生成的可视化文件 / Collect generated visualization files ---
    viz_out = os.path.join(out_dir, "fvisualize")
    viz_files = sorted(glob.glob(os.path.join(viz_out, f"{viz_prefix}_*"))) if os.path.isdir(viz_out) else []

    print(f"\n[visualize] 评估指标 / Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n[visualize] 可视化文件 ({len(viz_files)}) / Visualization files ({len(viz_files)}):")
    for f in viz_files:
        print(f"  {f}")


if __name__ == "__main__":
    '''
    python3 script/visualize.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth
    python3 script/visualize.py --task seg --ckpt model/yolo26-seg/n/lsqplus_v1/qat_lsqplus_v1_yolo26n-seg_best.pth
    python3 script/visualize.py --task depth --ckpt model/yolo26-depth/n/yolo26n-depth_best.pth --viz-max 30
    '''
    main()
