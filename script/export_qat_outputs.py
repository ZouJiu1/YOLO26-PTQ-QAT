"""把 QAT / PTQ 训练保存的量化 checkpoint（best / last / PTQ 初值）导出全套部署产物。 / Export full deployment artifacts from QAT / PTQ training quantization checkpoints (best / last / PTQ init).

背景 / Background:
    networks_yolo26-*.py 训练流程里，save_quant_outputs() 会在 PTQ 标定结束、QAT 最佳
    模型产出后被调用，一次性导出：量化参数 .json/.pth、干净浮点权重 float.pth、
    干净浮点 ONNX（带 onnxsim 简化）以及做 onnxruntime 验证。但若 checkpoint 是在
    中间阶段手动中断保存的，或用户想单独从某个已有的 QAT checkpoint 补发这些产物，
    本脚本即可独立完成 save_quant_outputs 的全部工作，无需重新跑训练。 /
    In the networks_yolo26-*.py training pipeline, save_quant_outputs() is called after
    PTQ calibration and after QAT best model is produced, exporting in one shot: quant params
    .json/.pth, clean float weights float.pth, clean float ONNX (with onnxsim simplification),
    and onnxruntime verification. But if a checkpoint was saved mid-training-interrupt, or if
    you want to retroactively generate these artifacts from an existing QAT checkpoint, this
    script independently performs the full save_quant_outputs workflow without re-running training.

关键约束 / Critical constraint:
    量化后端必须在构建 QuantYOLO26(Cls/Seg/Pose) 之前确定，因此本脚本第一步先用
    torch.load 只读 checkpoint meta（含 quant_method），随后 set_quant_method()，
    再构建量化模型、灌权、freeze_batch_init、最后调 save_quant_outputs()。 /
    The quant backend must be determined BEFORE constructing QuantYOLO26(Cls/Seg/Pose).
    Therefore this script first torch.loads only the checkpoint meta (containing quant_method),
    then calls set_quant_method(), builds the quant model, loads weights, freeze_batch_init,
    and finally invokes save_quant_outputs().

用法（在 QAT_training 目录下运行）： / Usage (run from QAT_training directory):
    # detect：QAT best / last / PTQ checkpoint / detect: QAT best / last / PTQ checkpoint
    python script/export_qat_outputs.py --task detect --ckpt model/yolo26-detect/lsqplus_v1/qat_lsqplus_v1_yolo26n_best.pth
    python script/export_qat_outputs.py --task detect --ckpt model/yolo26-detect/lsqplus_v1/qat_lsqplus_v1_yolo26n_last.pth
    python script/export_qat_outputs.py --task detect --ckpt model/yolo26-detect/lsqplus_v1/ptq_lsqplus_v1_yolo26n.pth

    # seg / pose / cls / obb / depth 同理 / seg / pose / cls / obb / depth similar
    python script/export_qat_outputs.py --task seg  --ckpt model/yolo26-seg/lsqplus_v1/qat_lsqplus_v1_yolo26n_best.pth
    python script/export_qat_outputs.py --task pose --ckpt model/yolo26-pose/lsqplus_v1/qat_lsqplus_v1_yolo26n_best.pth
    python script/export_qat_outputs.py --task cls  --ckpt model/yolo26-cls/lsqplus_v1/qat_lsqplus_v1_yolo26n_best.pth
    python script/export_qat_outputs.py --task obb  --ckpt model/yolo26-obb/lsqplus_v1/qat_lsqplus_v1_yolo26n-obb_best.pth
    python script/export_qat_outputs.py --task depth --ckpt model/yolo26-depth/lsqplus_v1/qat_lsqplus_v1_yolo26n-depth_best.pth

输出产物（默认写到 checkpoint 所在目录） / Output artifacts (written to checkpoint dir by default):
    {prefix}_quant_params.json      —— 量化参数（scale / zero_point，可读版） / Quant params (scale / zero_point, human-readable)
    {prefix}_quant_params.pth       —— 量化参数（torch.save 格式，可再灌） / Quant params (torch.save format, reloadable)
    {prefix}_{base}_float.pth       —— 反量化后的干净浮点权重 / Dequantized clean float weights
    {prefix}_{base}_float.onnx      —— 干净浮点 ONNX（含 onnxsim） / Clean float ONNX (with onnxsim)

    prefix 默认取 checkpoint 文件名去掉 _best / _last / .pth 后的主干，例如 /
    prefix defaults to checkpoint filename stripped of _best / _last / .pth, e.g.
    qat_lsqplus_v1_yolo26n_best.pth → prefix = qat_lsqplus_v1

说明 / Notes:
    * 导出只需 CPU，与正在进行的 GPU 训练互不干扰； / * Export only needs CPU, no interference with ongoing GPU training;
    * 自动从 checkpoint meta 推断 quant_method 与 scale，无需手动指定； / * Auto-infer quant_method and scale from checkpoint meta, no manual input needed;
    * checkpoint 中必须含 quant_method 字段（QAT / PTQ 阶段保存的 checkpoint 默认都带）。 / * Checkpoint must contain quant_method field (default for QAT / PTQ stage).
"""

import argparse
import importlib.util
import os
import re
import sys

import torch

# 让脚本可以从仓库根目录直接运行（python script/export_qat_outputs.py） / Allow script to run directly from repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


# 各任务的 Quant 模型类名（模块内定义的统一命名） / Quant model class names for each task
QUANT_CLS = {
    "detect": "QuantYOLO26",
    "seg": "QuantYOLO26Seg",
    "pose": "QuantYOLO26Pose",
    "cls": "QuantYOLO26Cls",
    "obb": "QuantYOLO26OBB",
    "depth": "QuantYOLO26Depth",
}


def load_task_module(task):
    """按任务动态加载 networks_yolo26-{task}.py（文件名含 '-'，用文件路径加载） / Dynamically load networks_yolo26-{task}.py by task name."""
    path = os.path.join(BASE_DIR, f"networks_yolo26-{task}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未知任务 {task}，找不到 {path}")
    spec = importlib.util.spec_from_file_location(f"networks_yolo26_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def peek_checkpoint_meta(ckpt_path):
    """只读 checkpoint 的 meta 字段（含 quant_method / scale / nc 等） / Read-only peek at checkpoint's meta field (quant_method / scale / nc etc.)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "meta" in ckpt and isinstance(ckpt["meta"], dict):
        return ckpt["meta"]
    return {}


def derive_prefix(ckpt_path):
    """从 checkpoint 文件名推导 prefix：去掉扩展名、_best / _last 后缀 / Derive prefix from checkpoint filename: strip extension and _best / _last suffix."""
    base = os.path.basename(ckpt_path)
    # 去掉 .pth / .pt / .ckpt 等扩展名 / Strip .pth / .pt / .ckpt etc.
    stem = re.sub(r"\.(pth|pt|ckpt)$", "", base, flags=re.IGNORECASE)
    # 去掉 _best / _last 尾部 / Strip trailing _best / _last
    stem = re.sub(r"_(best|last)$", "", stem, flags=re.IGNORECASE)
    return stem


def main():
    parser = argparse.ArgumentParser(
        description="QAT/PTQ checkpoint → 全套部署产物（quant_params.json + pth + float.pth + float.onnx） / "
                    "QAT/PTQ checkpoint → full deployment artifacts (quant_params.json + pth + float.pth + float.onnx)"
    )
    parser.add_argument("--task", required=True, choices=["detect", "seg", "pose", "cls", "obb", "depth"],
                        help="任务类型，决定加载哪个 networks_yolo26-*.py / Task type, determines which networks_yolo26-*.py to load")
    parser.add_argument("--ckpt", required=True,
                        help="QAT / PTQ checkpoint 路径（相对 QAT_training 目录或绝对路径） / QAT / PTQ checkpoint path")
    parser.add_argument("--prefix", default=None,
                        help="输出文件前缀；默认从 checkpoint 文件名推导 / Output file prefix; default derived from checkpoint filename")
    parser.add_argument("--out-dir", default=None,
                        help="输出目录；默认 checkpoint 所在目录 / Output directory; default checkpoint directory")
    parser.add_argument("--scale", default=None,
                        help="模型尺度（n/s/m/l/x）；默认从 checkpoint meta 或文件名推导 / Model scale; default derived from checkpoint meta or filename")
    parser.add_argument("--nc", type=int, default=None,
                        help="类别数；默认从 checkpoint meta 或模块默认值 / Number of classes; default from checkpoint meta or module default")
    parser.add_argument("--quant-method", default=None,
                        help="量化后端；默认从 checkpoint meta 读取，缺失时需手动指定（如 lsqplus_v1 / minmax / pact / dorefa） / "
                             "Quant backend; default read from checkpoint meta, must be specified manually if missing (e.g. lsqplus_v1 / minmax / pact / dorefa)")
    args = parser.parse_args()

    # --- 路径归一化 / Path normalization ---
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(BASE_DIR, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(ckpt_path)
    elif not os.path.isabs(out_dir):
        out_dir = os.path.join(BASE_DIR, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    prefix = args.prefix if args.prefix is not None else derive_prefix(ckpt_path)

    # --- 动态加载任务模块 / Dynamically load task module ---
    module = load_task_module(args.task)
    quant_cls = getattr(module, QUANT_CLS[args.task])

    # --- 先读 checkpoint meta，拿到 quant_method / scale / nc / First peek meta ---
    meta = peek_checkpoint_meta(ckpt_path)
    # quant_method 优先级：命令行 > checkpoint meta > 父目录名 > 报错 / quant_method priority: cli > meta > parent dir > error
    quant_method = args.quant_method or meta.get("quant_method")
    if quant_method is None:
        parent_dir = os.path.basename(os.path.dirname(ckpt_path))
        backend_names = getattr(module.quant_pkg, 'QUANT_METHODS', [])
        quant_method = parent_dir if parent_dir in backend_names else None
    if quant_method is None:
        raise RuntimeError(
            "checkpoint 中找不到 quant_method 字段，也无法从目录名推断。\n"
            "请确认这是 QAT / PTQ 阶段保存的 checkpoint，或通过 --quant-method 手动指定。"
        )

    # scale 优先 checkpoint meta，其次命令行，再次文件名最后 n / scale: meta > cmdline > filename > n
    scale = args.scale or meta.get("scale") or "n"
    if scale and not scale.startswith("yolo26"):
        scale = scale.lstrip("yolo26")  # 若 meta 里存的是 "yolo26n" 则去掉前缀

    # nc: 命令行 > checkpoint meta > 模块默认值 / nc: cmdline > meta > module default
    nc = args.nc
    if nc is None:
        nc = meta.get("nc") or meta.get("num_classes") or getattr(module, "NUM_CLASSES", 80)

    print(f"[export_qat] task={args.task} quant_method={quant_method} scale=yolo26{scale} nc={nc}")
    print(f"[export_qat] checkpoint : {ckpt_path}")
    print(f"[export_qat] output dir : {out_dir}")
    print(f"[export_qat] prefix     : {prefix}")

    # --- 关键：先设置量化后端，再构建量化模型 / CRITICAL: set backend BEFORE constructing quant model ---
    module.set_quant_method(quant_method)
    print(f"[export_qat] quant backend set to: {module.QUANT_METHOD}")

    # --- 构建量化模型、加载 checkpoint、freeze_batch_init / Build quant model, load checkpoint, freeze_batch_init ---
    device = module.device if hasattr(module, "device") else torch.device("cpu")
    quant_model = quant_cls(nc=nc, scale=scale).to(device)
    _, load_meta = module.load_checkpoint(quant_model, ckpt_path, return_meta=True)
    module.quant_pkg.freeze_batch_init(quant_model)
    print(f"[export_qat] 模型加载完成，stage={load_meta.get('stage')} epoch={load_meta.get('epoch')}"
          f" fitness={load_meta.get('fitness')}")

    # --- 复用 save_quant_outputs 一键导出全部产物 + onnxruntime 验证 / Reuse save_quant_outputs ---
    quant_ckpt, json_path, pth_path, float_ckpt, onnx_path = module.save_quant_outputs(
        quant_model, prefix,
        nc=nc,
        meta=load_meta,
        scale=scale,
        model_dir=out_dir,
    )

    print("\n[export_qat] 全部产物已生成 ✅")
    print(f"  量化 checkpoint : {quant_ckpt}")
    print(f"  量化参数 JSON   : {json_path}")
    print(f"  量化参数 PTH    : {pth_path}")
    print(f"  干净浮点权重    : {float_ckpt}")
    print(f"  干净浮点 ONNX   : {onnx_path}")


if __name__ == "__main__":
    main()
