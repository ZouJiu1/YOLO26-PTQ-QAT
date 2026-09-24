# YOLO26 / MobileNetV3 / CIFAR-CNN 量化感知训练（QAT）

> English version: [README.md](README.md)

The idea is mine, and the implementation is mainly handled by Trae.

本工程在 PyTorch 上为以下网络提供统一的「float 训练 → PTQ 校准 → QAT 微调 →
精度对比 → 部署导出（ONNX + JSON 量化参数）」全流程：

| 文件 | 任务 | 数据集 |
|------|------|--------|
| `networks_yolo26-detect.py` | YOLO26 目标检测 | coco128 / COCO mini |
| `networks_yolo26-seg.py` | YOLO26 实例分割 | coco128-seg / COCO mini |
| `networks_yolo26-pose.py` | YOLO26 关键点姿态 | coco8-pose / COCO mini |
| `networks_yolo26-cls.py` | YOLO26 图像分类 | imagenet10 |
| `networks_mobileNetv3.py` | MobileNetV3 分类 | CIFAR-10 |
| `networks_cifarCNN.py` | 自定义 CIFAR-CNN | CIFAR-10 |
| `networks_example.py` | 最小量化算子示例 | CIFAR-10（无 CLI） |

## 安装

```bash
pip3 install -r requirements.txt
```

`requirements.txt` 固定了本机验证过的版本（torch 2.x + CUDA 12.6、ultralytics、
onnx / onnxruntime / onnxsim 等）。CPU-only 环境把 torch/torchvision 换成 `+cpu` 版本即可。

**训练必须使用 GPU（CUDA），请勿在 CPU 上跑训练任务。**

## 统一训练流程

所有 `networks_*.py` 都按同一管线执行：

```
float_train -> PTQ_calibration -> QAT_training -> compare_precision
  (浮点训练)    (训练后量化校准)    (量化感知训练)     (浮点 vs QAT 精度对比)
```

- YOLO 系 float 训练默认加载 `ultralytics/yolo26n*.pt` 预训练权重；
- PTQ 阶段统计出的量化参数用于初始化 QAT 的量化参数；
- QAT 使用 STE（直通估计）微调。

每个 PTQ / QAT 阶段结束自动写出五件套（`save_quant_outputs`）：

| 产物 | 说明 |
|------|------|
| `model/{prefix}_{name}.pth` | 量化模型权重 |
| `model/{prefix}_quant_params.json` | 每层 scale / zero_point（部署用） |
| `model/{prefix}_quant_params.pth` | 同上参数的 .pth 二进制形式 |
| `model/{prefix}_{name}_float.pth` | 量化参数回灌后的干净浮点权重 |
| `model/{prefix}_{name}_float.onnx` | 上述浮点权重的 ONNX（opset=16） |

## 统一特性（所有 networks_*.py 一致）

- **best 后缀**：训练期间最优 checkpoint 保存为 `*_best.pth`，同时维护 `*_last.pth`；
  float 老权重（无后缀）仍被 compare 阶段兼容（优先 `_best`，找不到再回退）。
- **保存 best .pth 的同时同步导出同名 .onnx**（`_try_export_onnx`）；
  导出失败仅告警，绝不影响训练。
- **ONNX 一律经过 onnxsim simplify**（未安装 onnxsim 时自动跳过）。
- **resume 接续训练**：加 `--resume` 会从 `*_last.pth` 恢复模型 / optimizer /
  scheduler / epoch 继续训练。
- 数据加载器的 collect 一律使用 `torch.concat`，不做量化拼接；
  `evaluate()` 接收 `check_det_dataset()` 返回的原始 data 字典。

## 量化后端与推荐

`--quant` 可切换 7 种后端：

| `--quant` | 后端模块 | 说明 |
|-----------|----------|------|
| `lsqplus_v1` | `quantization/lsqplus_quantize_V1.py` | LSQ+（可学习 scale + beta） |
| `lsqplus_v2` | `quantization/lsqplus_quantize_V2.py` | LSQ+ V2 |
| `lsq_v1` | `quantization/lsqquantize_V1.py` | LSQ（对称，可学习 scale） |
| `lsq_v2` | `quantization/lsqquantize_V2.py` | LSQ V2 |
| `minmax` | `quantization/minmax.py` | MinMax（running min/max，标准非对称） |
| `dorefa` | `quantization/dorefa.py` | DoReFa + 可学习激活尺度 |
| `pact` | `quantization/pact.py` | PACT（可学习 clip 阈值 alpha） |

### 推荐

**默认推荐 `lsqplus_v1`**（所有网络文件的默认值）：

1. **验证最充分**：在 1/20 COCO mini 上完整跑通过 detect / seg / pose 的
   float → PTQ → QAT 全流程（detect QAT best mAP50 = 0.5958）；
2. **精度最优**：可学习 scale + beta（偏移）对非负激活（SiLU 之后）更友好，
   PTQ 阶段经浮点前向 min-max 初始化后，量化精度可追平浮点基线；
3. **稳定性已加固**：dummy 输入用 `randn * 0.1` 避免全零前向导致 s=0 / NaN。

其他后端的适用场景：

- `minmax`：最简单稳定、无可学习参数，适合做 baseline 对照；
- `lsq_v1`：只需对称量化时的轻量选择；
- `dorefa` / `pact`：研究对比用（dorefa 已修复无界激活坍缩问题，加了可学习尺度 s）；
- `*_v2`：对应算法的实验变体，接口与 V1 一致。

每个后端实现同一套量化算子：`QuantConv2d` / `QuantConvTranspose2d` / `QuantLinear` /
`QuantAdd` / `QuantSub` / `QuantMultiply` / `QuantDiv` / `QuantConcat` / `QuantMaxPool` 等。
`quantization/__init__.py` 负责后端加载（`load_quant_backend`）、量化器状态
freeze / reset，以及跨后端的 scale / zero_point 提取（`collect_quant_params`）。
`quantization/constants.py` 中的 `INIT_STATE_FROZEN` 为量化器冻结哨兵常量。

## 快速开始

单个网络完整管线（以检测为例）：

```bash
python3 networks_yolo26-detect.py --model yolo26n --stage all --quant lsqplus_v1
```

只跑某个阶段：

```bash
python3 networks_cifarCNN.py --quant minmax --stage qat
python3 networks_cifarCNN.py --quant minmax --stage compare
```

冒烟测试（每个 epoch 只跑 1 个 batch）：

```bash
python3 networks_mobileNetv3.py --quant lsqplus_v1 \
    --float-epochs 1 --max-train-batches 1 --max-eval-batches 1
```

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--quant` | `lsqplus_v1` | 量化后端 |
| `--stage` | `all` | `all / float / ptq / qat / compare` |
| `--float-epochs` | 100 | 浮点训练轮数 |
| `--qat-epochs` | float 的 1/5 | QAT 轮数 |
| `--float-lr` / `--qat-lr` | — | 学习率（QAT 默认 float 的 1/10） |
| `--float-batch-size` / `--ptq-batch-size` / `--qat-batch-size` | 16 (yolo) / 128 (cls) | 各阶段 batch size |
| `--calibration-batches` | 20 | PTQ 校准批数 |
| `--data` | — | 覆盖数据集 yaml（yolo26 系） |
| `--resume` | False | 从 `*_last.pth` 接续训练 |
| `--max-train-batches` / `--max-eval-batches` | None | 限制 batch 数（冒烟测试） |

## COCO mini 数据集（1/20 抽样验证）

用完整 COCO2017 的二十分之一快速验证 detect / seg / pose 全流程：

```bash
python3 script/coco_mini_prepare.py                 # 默认 COCO 根
python3 script/coco_mini_prepare.py --coco-root /path/to/coco --ratio 20
```

采样规则：图片 id 排序后每隔 20 张取 1 张（`ids[::20]`），确定性可复现；
图片用相对软链接不复制；iscrowd 标注跳过；category id 重映射为 0..79。
产物在 `<COCO根>/mini/` 下，含 detect / seg / pose 三个任务的
images / labels / train.txt / val.txt 与对应 yaml。

一键串跑三个任务（float40 → PTQ → QAT8，共享 GPU 故串行）：

```bash
bash script/run_mini_all.sh        # 日志分别写入 mini_{detect,seg,pose}.log
```

## 工具脚本（script/）

- `script/coco_mini_prepare.py` — 构造 1/20 COCO mini 数据集（见上节）。
- `script/run_mini_all.sh` — 依次串行跑 detect / seg / pose 的 mini 全流程。
- `script/run_mini_pipeline.py` — mini 三任务 GPU 串跑的 Python 版编排器。
- `script/export_float_onnx.py` — 把 float 训练保存的 checkpoint 单独转 ONNX
  （含 onnxsim 简化，CPU 即可）：

```bash
python3 script/export_float_onnx.py --task detect --ckpt model/yolo26n_best.pth
python3 script/export_float_onnx.py --task seg    --ckpt model/yolo26n-seg_best.pth
python3 script/export_float_onnx.py --task pose   --ckpt model/yolo26n-pose_best.pth
python3 script/export_float_onnx.py --task cls    --ckpt model/yolo26n-cls_best.pth
```

输出默认 `{ckpt去掉.pth}_float.onnx`。

## 目录结构

```
QAT_training/
├── quantization/                # 7 个量化后端 + __init__.py 后端选择器 + constants.py
├── networks_yolo26-detect.py    # YOLO26 检测（seg/pose/cls 复用其 backbone/neck）
├── networks_yolo26-seg.py       # YOLO26 实例分割
├── networks_yolo26-pose.py      # YOLO26 关键点姿态
├── networks_yolo26-cls.py       # YOLO26 分类
├── networks_mobileNetv3.py      # MobileNetV3 分类
├── networks_cifarCNN.py         # CIFAR-CNN 分类
├── networks_example.py          # 最小量化算子示例
├── script/                      # 数据准备 / 编排 / ONNX 导出工具
├── requirements.txt
├── README.md                    # English
├── README_CN.md                 # 本文件
├── datas/                       # 数据集（运行时自动下载，git 不跟踪）
├── model/                       # checkpoint + 量化参数 JSON + ONNX（git 不跟踪）
└── ultralytics/                 # ultralytics 包与 yolo26n*.pt 预训练权重
```

## 部署导出说明

- **ONNX**：导出的是「量化参数回灌后的干净浮点模型」，不含 `QuantizeLinear` /
  `DequantizeLinear` 节点；导出自带 onnxsim 简化，并用 onnxruntime 与 PyTorch
  做数值比对（分类网误差阈值 1e-4，检测网按输出幅度取相对阈值）。
- **JSON**：`*_quant_params.json` 记录每个量化张量的 scale / zero_point，
  供下游部署工具链（如地平线模型编译）消费；同名 `.pth` 为二进制形式。

## 注意事项与经验教训

- 必须用 GPU 训练；请在系统终端 / VSCode 终端运行长任务。
- PTQ 量化参数必须用于初始化 QAT 量化参数（`copy_float_to_quant` 后校准）。
- pose 的 `flow_model`（RealNVP）仅在训练损失中使用，永不量化，保持浮点 `nn.Linear`。
- cls 的全局平均池化保持浮点 `nn.AdaptiveAvgPool2d`，不参与量化。
- yolo26 的 seg / pose / cls 通过 importlib 复用 detect 的 backbone/neck，
  切换量化后端时会同步切换 det 模块内绑定的算子，改动时四个文件需保持一致。
- checkpoint 保存的是 EMA 权重（与验证/部署口径一致）。

## 参考资料

- [mqbench](https://github.com/modeltc/mqbench) — QAT / PTQ 参考
- [ultralytics](https://ultralytics.com/) — YOLO26 预训练权重
- [horizon PTQ/QAT 部署指南](https://doc.oe.horizon.auto/3.8.1/guide/model_compile.html)
- [horizon developer portal](https://developer.horizon.auto/)
- [Trae](https://www.trae.cn/)

## Credits

The idea is mine, and the implementation is mainly handled by Trae.
If there are issues with Trae's implementation, I will point them out and request fixes.
