"""把 float 训练保存的 checkpoint（best / last）导出为 ONNX（带 onnxsim 简化）。

背景：
    networks_yolo26-*.py 训练流程里，export_onnx() 只在 PTQ/QAT 部署阶段被调用
    （导出的是量化参数回灌后的"干净浮点"模型）。float 训练阶段保存的
    {base}_best.pth / {base}_last.pth 本身不会自动转 ONNX，本脚本补上这一步。

用法（在 QAT_training 目录下运行）：
    # detect：best（老命名为 yolo26n.pth，无 _best 后缀）和 last
    python script/export_float_onnx.py --task detect --ckpt model/yolo26n.pth
    python script/export_float_onnx.py --task detect --ckpt model/yolo26n_last.pth

    # seg / pose / cls 同理（按各自 base_name 规则）
    python script/export_float_onnx.py --task seg  --ckpt model/yolo26n-seg_best.pth
    python script/export_float_onnx.py --task pose --ckpt model/yolo26n-pose_best.pth
    python script/export_float_onnx.py --task cls  --ckpt model/yolo26n-cls_best.pth

输出命名规则：
    不指定 --out 时，输出为 {ckpt去掉.pth}_float.onnx（与 PTQ/QAT 导出的
    *_float.onnx 命名风格保持一致）。

说明：
    * 导出只需 CPU，与正在进行的 GPU 训练互不干扰；
    * export_onnx 内部已带 onnxsim 简化（未安装 onnxsim 时自动跳过）；
    * checkpoint 保存的是 EMA 权重（与验证/部署口径一致）。
"""

import argparse
import importlib.util
import os
import sys

# 让脚本可以从仓库根目录直接运行（python script/export_float_onnx.py）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_task_module(task):
    """按任务动态加载 networks_yolo26-{task}.py（文件名含 '-'，用文件路径加载）。"""
    path = os.path.join(BASE_DIR, f"networks_yolo26-{task}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未知任务 {task}，找不到 {path}")
    spec = importlib.util.spec_from_file_location(f"networks_yolo26_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 各任务的 Float 模型类名（模块内定义的统一命名）
FLOAT_CLS = {
    "detect": "FloatYOLO26",
    "seg": "FloatYOLO26Seg",
    "pose": "FloatYOLO26Pose",
    "cls": "FloatYOLO26Cls",
}


def main():
    parser = argparse.ArgumentParser(description="float checkpoint -> ONNX（含 onnxsim）")
    parser.add_argument("--task", required=True, choices=["detect", "seg", "pose", "cls"],
                        help="任务类型，决定加载哪个 networks_yolo26-*.py 及 Float 模型类")
    parser.add_argument("--ckpt", required=True,
                        help="float checkpoint 路径（相对 QAT_training 目录或绝对路径）")
    parser.add_argument("--out", default=None,
                        help="输出 onnx 路径；默认 {ckpt去掉.pth}_float.onnx")
    parser.add_argument("--scale", default="yolo26n",
                        help="模型尺度，默认 yolo26n（与 mini 训练一致）")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset，默认 16")
    args = parser.parse_args()

    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(BASE_DIR, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    onnx_path = args.out
    if onnx_path is None:
        onnx_path = os.path.splitext(ckpt_path)[0] + "_float.onnx"
    elif not os.path.isabs(onnx_path):
        onnx_path = os.path.join(BASE_DIR, onnx_path)

    module = load_task_module(args.task)
    float_cls = getattr(module, FLOAT_CLS[args.task])

    # 构建浮点模型并加载 checkpoint（load_checkpoint 自带 shape mismatch 过滤，
    # 且 checkpoint 里存的就是 EMA 权重，直接加载即可）
    model = float_cls(scale=args.scale).to(module.device)
    _, meta = module.load_checkpoint(model, ckpt_path, return_meta=True)
    print(f"[export] 加载 {ckpt_path}")
    if meta:
        print(f"[export] meta: stage={meta.get('stage')} epoch={meta.get('epoch')}"
              f" fitness={meta.get('fitness')}")

    module.export_onnx(model, onnx_path, opset=args.opset)
    print(f"[export] ONNX 已写出: {onnx_path}")


if __name__ == "__main__":
    main()
