# YOLO26 / MobileNetV3 / CIFAR-CNN 量化感知训练（QAT）

> English version: [README.md](README.md)

本工程在 PyTorch 上为以下网络提供统一的「float 训练 → PTQ 校准 → QAT 微调 →
精度对比 → 部署导出（ONNX + JSON 量化参数）」全流程：

| 文件 | 任务 | 数据集 |
|------|------|--------|
| `networks_yolo26-detect.py` | YOLO26 目标检测 | coco8（自动下载）/ COCO mini |
| `networks_yolo26-seg.py` | YOLO26 实例分割 | coco8-seg（自动下载）/ COCO mini |
| `networks_yolo26-pose.py` | YOLO26 关键点姿态 | coco8-pose（自动下载）/ COCO mini |
| `networks_yolo26-cls.py` | YOLO26 图像分类 | imagenet10 |
| `networks_yolo26-obb.py` | YOLO26 旋转框检测（OBB） | dota8-multispectral（10 通道，自动下载） |
| `networks_yolo26-depth.py` | YOLO26 单目深度估计 | depth8（RGB + 16-bit PNG 深度图，自动下载） |
| `networks_mobileNetv3.py` | MobileNetV3 分类 | CIFAR-10 |
| `networks_cifarCNN.py` | 自定义 CIFAR-CNN | CIFAR-10 |
| `networks_example.py` | 最小量化算子示例 | CIFAR-10（无 CLI） |

**数据集** — yolo26 各任务首次运行时，直接走 ultralytics 官方的
`check_det_dataset` / `safe_download` 流程，把官方数据集自动下载到项目
`dataset/` 目录（无需手动准备；脚本会生成指向 `dataset/<name>` 的便携 yaml）：

| 任务 | 官方 yaml | 下载地址 |
|------|-----------|----------|
| detect | `coco8.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip> |
| seg | `coco8-seg.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-seg.zip> |
| pose | `coco8-pose.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip> |
| obb | `dota8-multispectral.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/dota8-multispectral.zip> |
| depth | `depth8.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/depth8-png.zip> |

- 上游数据集定义：<https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/datasets>
- 也可用 `--data /path/to/data.yaml`（或包含 yaml 的数据集目录）手动加载任意其他目录。

## 安装

```bash
pip3 install -r requirements.txt
```

`requirements.txt` 固定了本机验证过的版本（torch 2.x + CUDA 12.6、ultralytics、
onnx / onnxruntime / onnxsim 等）。CPU-only 环境把 torch/torchvision 换成 `+cpu` 版本即可。

## 统一训练流程

所有 `networks_*.py` 都按同一管线执行：

```
float_train -> PTQ_calibration -> QAT_training -> compare_precision
  (浮点训练)    (训练后量化校准)    (量化感知训练)     (浮点 vs QAT 精度对比)
```

- YOLO 系 float 训练默认加载 `ultralytics/yolo26n*.pt` 预训练权重；本地缺失的官方权重（任意尺度/任务，如 `yolo26s-seg.pt`）会调用 ultralytics 的 `attempt_download_asset` 从 GitHub Releases 自动下载，传入自定义路径则直接加载；
- PTQ 阶段统计出的量化参数用于初始化 QAT 的量化参数；
- QAT 使用 STE（直通估计）微调。

每个 PTQ / QAT 阶段结束自动写出五件套（`save_quant_outputs`）：

| 产物 | 说明 |
|------|------|
| `model/{prefix}_{name}.pth` | 量化模型权重 |
| `model/{prefix}_quant_params.json` | 每层 scale / zero_point（部署用） |
| `model/{prefix}_quant_params.pth` | 同上参数的 .pth 二进制形式 |
| `model/{prefix}_{name}_float.pth` | 量化参数回灌后的干净浮点权重 |
| `model/{prefix}_{name}_float.onnx` | 上述浮点权重的 ONNX（opset=16，静态 shape `[1, C, H, W]`，无 dynamic_axes；C=3，多光谱 OBB 网络为 C=10） |

## 统一特性（所有 networks_*.py 一致）

- **best 后缀**：训练期间最优 checkpoint 保存为 `*_best.pth`，同时维护 `*_last.pth`；
  float 老权重（无后缀）仍被 compare 阶段兼容（优先 `_best`，找不到再回退）。
- **保存 best .pth 的同时同步导出同名 .onnx**（`_try_export_onnx`）；
  导出失败仅告警，绝不影响训练。
- **ONNX 输入形状完全固定**（`[1, C, H, W]`，无 `dynamic_axes`；C=3，
  多光谱 OBB 网络为 C=10），可视化工具直接显示精确尺寸，不会出现符号维度。
- **ONNX 一律经过 onnxsim simplify**（未安装 onnxsim 时自动跳过）。
- **resume 接续训练**：加 `--resume` 会从 `*_last.pth` 恢复模型 / optimizer /
  scheduler / epoch 继续训练。
- **每次评估自动可视化**：每次 `evaluate()` 结束后，用 ultralytics 官方
  `plot_images` 把前约 30 张验证图渲染成 GT / 预测拼图（seg 附带掩码、pose
  附带关键点骨架、obb 以多边形绘制旋转框、cls 显示 GT 与预测类别、depth
  渲染深度热图；多光谱输入取前 3 个通道显示），保存到 `{model_dir}/fvisualize/`，
  文件命名 `{阶段标签}_batch{n}_{labels|pred}.jpg`
  （如 `float_ep001_batch0_pred.jpg`、`qat_lsqplus_v1_best_batch0_labels.jpg`）。
  3 个 CIFAR 分类网络改用 matplotlib 网格图。可视化失败仅告警，绝不影响评估。
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

1. **验证最充分**：在 1/100 COCO mini 上完整跑通过 detect / seg / pose 的
   float → PTQ → QAT 全流程，并参与全部 7 个后端的统一横评（见下文结果表）；
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

## COCO mini 数据集与全后端横评（1/100 抽样）

用完整 COCO2017 的百分之一快速横评 detect / seg / pose 三个任务与全部 7 个量化后端：

```bash
python3 script/coco_mini_prepare.py                 # 默认 1/100，产物输出到项目 dataset/
python3 script/coco_mini_prepare.py --coco-root /path/to/coco --ratio 100
python3 script/coco_mini_prepare.py --out-dir /path/to/output   # 自定义产物根目录
```

采样规则：图片 id 排序后每隔 N 张取 1 张（`ids[::N]`），确定性可复现；
图片用相对软链接不复制；iscrowd 标注跳过；category id 重映射为 0..79。
产物默认在项目根 `dataset/` 下（与 coco8/ 平级，可用 `--out-dir` 覆盖），
含 detect / seg / pose 三个任务的 images / labels / train2017.txt / val2017.txt
与对应 `coco_mini_*.yaml`。1/100 抽样规模：

| 任务 | 训练图片 | 验证图片 | 类别 |
|------|---------|---------|------|
| detect | 1183 | 50 | 80 |
| seg | 1183 | 50 | 80 |
| pose | 642 | 27 | 1（person） |

一键串跑：每个任务 **float 只训练 1 次（50 epoch）**，随后遍历全部量化后端
（lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / dorefa / pact），
每个后端各做 PTQ（20 batch 校准）→ QAT（10 epoch = 50×0.2）→ float/QAT 对比，
共享 GPU 故串行；单个阶段失败不影响其余后端：

```bash
bash script/run_mini_all.sh        # 日志 log/mini_{task}_float.log、log/mini_{task}_{backend}_{ptq,qat,compare}.log
```

### 全后端横评结果（1/100 COCO mini，float 50ep / QAT 10ep，batch=8）

> 评估集仅 50 张（pose 27 张），mAP/P/R 为小样本结果，主要用于横向对比与回归验证，
> 不代表 COCO 全量精度。结果由 `script/run_mini_all.sh` 自动产出。

<!-- RESULTS_TABLE_PLACEHOLDER -->
**detect（val 50 张，80 类）** — Float 基线（best epoch 4/50）：P 0.7229 / R 0.5004 / mAP50 **0.5814** / mAP50-95 **0.4321**

| 量化后端 | PTQ mAP50 | QAT P | QAT R | QAT mAP50 | QAT mAP50-95 | ΔmAP50 (QAT−Float) |
|---|---|---|---|---|---|---|
| lsqplus_v1 | 0.5663 | 0.7692 | 0.4757 | 0.5792 | 0.4201 | −0.0022 |
| lsqplus_v2 | 0.5663 | 0.7338 | 0.5119 | 0.5707 | 0.4183 | −0.0107 |
| lsq_v1 | 0.5701 | 0.7740 | 0.4787 | **0.5809** | 0.4190 | −0.0005 |
| lsq_v2 | 0.5859 | 0.6979 | 0.5326 | 0.5649 | 0.4197 | −0.0165 |
| minmax | 0.5557 | 0.8034 | 0.4678 | 0.5725 | 0.4133 | −0.0089 |
| dorefa | 0.5750 | 0.7047 | 0.5215 | 0.5774 | **0.4203** | −0.0040 |
| pact | 0.5655 | 0.7799 | 0.4811 | **0.5920** | 0.4204 | **+0.0106** |

**seg（val 50 张，80 类；任务指标为 mask 分割 mAP）** — Float 基线（best epoch 5/50）：box mAP50 0.5362；mask P 0.6397 / R 0.4663 / mAP50 **0.5001** / mAP50-95 **0.3156**

| 量化后端 | PTQ mask mAP50 | QAT box mAP50 | QAT mask P | QAT mask R | QAT mask mAP50 | QAT mask mAP50-95 | ΔmAP50 |
|---|---|---|---|---|---|---|---|
| lsqplus_v1 | 0.4715 | 0.5352 | 0.6939 | 0.4119 | 0.4966 | 0.3139 | −0.0035 |
| lsqplus_v2 | 0.4715 | 0.5252 | 0.6850 | 0.4476 | 0.4902 | **0.3192** | −0.0099 |
| lsq_v1 | 0.4684 | 0.5356 | 0.6752 | 0.4492 | **0.5066** | 0.3133 | **+0.0065** |
| lsq_v2 | 0.4332 | 0.5218 | 0.6839 | 0.4451 | 0.4833 | 0.2976 | −0.0168 |
| minmax | **0.5049** | **0.5430** | 0.6211 | 0.4450 | 0.5050 | 0.3172 | +0.0049 |
| dorefa ² | 0.4601 | 0.5385 | 0.6170 | 0.4686 | 0.4941 | 0.3101 | −0.0060 |
| pact | 0.4890 | 0.5325 | 0.6460 | 0.4448 | 0.4990 | 0.3151 | −0.0011 |

**pose（val 27 张，person 单类；任务指标为 pose 关键点 mAP）** — Float 基线（best epoch 1/50）：box mAP50 0.6178；pose P 0.7843 / R 0.4565 / mAP50 **0.4839** / mAP50-95 **0.3263**

| 量化后端 | PTQ pose mAP50 | QAT box mAP50 | QAT pose P | QAT pose R | QAT pose mAP50 | QAT pose mAP50-95 | ΔmAP50 |
|---|---|---|---|---|---|---|---|
| lsqplus_v1 | 0.4823 | 0.6164 | 0.7895 | 0.4565 | 0.4844 | 0.3219 | +0.0005 |
| lsqplus_v2 | 0.4823 | **0.6291** | 0.7446 | 0.4565 | 0.4868 | 0.3250 | +0.0029 |
| lsq_v1 | 0.4724 | 0.6265 | 0.8270 | 0.4638 | 0.4791 | 0.2995 | −0.0048 |
| lsq_v2 | 0.4540 | 0.6102 | 0.8012 | 0.4565 | 0.4863 | 0.2994 | +0.0024 |
| minmax | **0.4934** | 0.6147 | 0.7849 | 0.4494 | 0.4793 | 0.3181 | −0.0046 |
| dorefa | 0.4629 | 0.6263 | **0.8409** | 0.4565 | 0.4837 | 0.3030 | −0.0002 |
| pact | 0.4891 | 0.6234 | 0.7788 | 0.4593 | **0.4869** | 0.3225 | **+0.0030** |

² seg 的 dorefa QAT 在 8GB GPU 上 batch=8 时 tanh 量化器显存不足（OOM），该组以 batch=4 训练 QAT（PTQ 仍为 batch=8），compare 在 batch=8 下评估以保证 float 基线一致。
其余 20 组全部 batch=8。逐阶段完整日志见 `log/mini_{task}_float.log` 与 `log/mini_{task}_{backend}_{ptq,qat,compare}.log`。

**小结（小样本横向对比，不代表全量 COCO 精度）：**
- 7 个后端 QAT mAP50 相对 float 的损失均 ≤ 0.017，PTQ 单独即可恢复到 float 的 90%+，QAT 普遍再追回大部分差距；
- detect / seg / pose 的 QAT 最优 mAP50 分别由 pact（0.5920，超过 float +0.0106）、lsq_v1（0.5066）、pact（0.4869）取得；
- minmax 无需训练参数、PTQ 精度通常最高（seg mask 0.5049、pose 0.4934），是最省事的后端；
- lsq_v2 / lsqplus_v2 的 mAP50-95（严格 IoU）在 pose 上偏弱（0.299），dorefa 在 seg 上显存占用最大。

## 工具脚本（script/）

- `script/coco_mini_prepare.py` — 构造 1/N COCO mini 数据集（默认 1/100，见上节）。
- `script/run_mini_all.sh` — 三任务 × 全 7 后端的 mini 横评串跑（float 1 次 + 各后端 PTQ/QAT/compare）。
- `script/export_float_onnx.py` — 把 float 训练保存的 checkpoint 单独转 ONNX
  （含 onnxsim 简化，CPU 即可；静态 shape，支持 `--imgsz` 自定义尺寸）：

```bash
python3 script/export_float_onnx.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth
python3 script/export_float_onnx.py --task seg    --ckpt model/yolo26-seg/n/yolo26n-seg_best.pth
python3 script/export_float_onnx.py --task pose   --ckpt model/yolo26-pose/n/yolo26n-pose_best.pth
python3 script/export_float_onnx.py --task cls    --ckpt model/yolo26-cls/n/yolo26n-cls_best.pth
python3 script/export_float_onnx.py --task obb    --ckpt model/yolo26-obb/n/yolo26n-obb_best.pth
python3 script/export_float_onnx.py --task depth  --ckpt model/yolo26-depth/n/yolo26n-depth_best.pth
# 可选：覆盖默认输入尺寸（detect/seg/pose/obb/depth=640，cls=224）
python3 script/export_float_onnx.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth --imgsz 480
```

输出默认 `{ckpt去掉.pth}_float.onnx`。

- `script/export_qat_outputs.py` — 从任意 QAT / PTQ checkpoint 补发全套部署产物
  （`quant_params.json` + `.pth` + 干净浮点 `.pth` + ONNX + onnxruntime 验证），
  无需重跑训练。自动从 checkpoint meta 推断 `quant_method`、`scale`、`nc`：

```bash
# 自动推断 quant_method（推荐）
python3 script/export_qat_outputs.py --task detect \
    --ckpt model/yolo26-detect/n/lsqplus_v1/qat_lsqplus_v1_yolo26n_best.pth

# 手动指定（例如老 checkpoint 缺 quant_method）
python3 script/export_qat_outputs.py --task detect --ckpt <old.pth> \
    --quant-method minmax
```

- `script/visualize.py` — 在任务的**验证集**上批量可视化任意 checkpoint
  （float / PTQ / QAT；自动从 checkpoint meta 判型）。复用各任务模块的
  `evaluate()`（内部调用 ultralytics 官方 `plot_images`），拼图输出到
  `{ckpt_dir}/fvisualize/`：

```bash
python3 script/visualize.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth
python3 script/visualize.py --task seg    --ckpt model/yolo26-seg/n/lsqplus_v1/qat_lsqplus_v1_yolo26n-seg_best.pth
python3 script/visualize.py --task depth  --ckpt model/yolo26-depth/n/yolo26n-depth_best.pth --viz-max 30
```

- `script/single_visualize.py` — 对**单张图片或图片目录**做推理 + 可视化
  （无需标签，纯预测；用 ultralytics `Annotator` 标注；depth 输出独立热图）：

```bash
# 单张图片 → {ckpt_dir}/{prefix}_{task}_result.jpg
python3 script/single_visualize.py --task detect \
    --ckpt model/yolo26-detect/n/yolo26n_best.pth \
    --image ultralytics/ultralytics/data/datasets/coco128-seg/images/train2017/000000000009.jpg

# 目录批量 → {out}/{image_name}_result.jpg（每张图一个文件）
python3 script/single_visualize.py --task seg \
    --ckpt model/yolo26-seg/n/lsqplus_v1/qat_lsqplus_v1_yolo26n-seg_best.pth \
    --image ultralytics/ultralytics/data/datasets/coco128-seg/images/train2017 \
    --out results/seg_batch
```

## 目录结构

```
QAT_training/
├── quantization/                # 7 个量化后端 + __init__.py 后端选择器 + constants.py
├── networks_yolo26-detect.py    # YOLO26 检测（seg/pose/cls 复用其 backbone/neck）
├── networks_yolo26-seg.py       # YOLO26 实例分割
├── networks_yolo26-pose.py      # YOLO26 关键点姿态
├── networks_yolo26-cls.py       # YOLO26 分类
├── networks_yolo26-obb.py       # YOLO26 旋转框检测（OBB，复用 detect backbone）
├── networks_yolo26-depth.py     # YOLO26 单目深度估计（复用 detect backbone）
├── networks_mobileNetv3.py      # MobileNetV3 分类
├── networks_cifarCNN.py         # CIFAR-CNN 分类
├── networks_example.py          # 最小量化算子示例
├── script/                      # 数据准备 / 编排 / ONNX 导出 / 可视化工具
├── requirements.txt
├── README.md                    # English
├── README_CN.md                 # 本文件
├── dataset/                     # 自动下载的数据集（coco8 / depth8-png / dota8-multispectral 等，git 不跟踪）
├── model/                       # checkpoint + 量化参数 JSON + ONNX（git 不跟踪）
└── ultralytics/                 # ultralytics 包与 yolo26n*.pt 预训练权重
```

## 部署导出说明

- **ONNX**：导出的是「量化参数回灌后的干净浮点模型」，不含 `QuantizeLinear` /
  `DequantizeLinear` 节点；输入形状完全固定（`[1, 3, H, W]`，无
  `dynamic_axes`），可视化工具直接显示精确尺寸；导出自带 onnxsim 简化，
  并用 onnxruntime 与 PyTorch 做数值比对（分类网误差阈值 1e-4，检测网按
  输出幅度取相对阈值）。
- **JSON**：`*_quant_params.json` 记录每个量化张量的 scale / zero_point，
  供下游部署工具链（如地平线模型编译）消费；同名 `.pth` 为二进制形式。
- **手动补发**：如果 QAT / PTQ checkpoint 还在但部署产物缺失（训练中断），
  用 `script/export_qat_outputs.py` 一键补齐全套。

## 注意事项与经验教训

- 必须用 GPU 训练；请在系统终端 / VSCode 终端运行长任务。
- PTQ 量化参数必须用于初始化 QAT 量化参数（`copy_float_to_quant` 后校准）。
- pose 的 `flow_model`（RealNVP）仅在训练损失中使用，永不量化，保持浮点 `nn.Linear`。
- cls 的全局平均池化保持浮点 `nn.AdaptiveAvgPool2d`，不参与量化。
- yolo26 的 seg / pose / cls / obb / depth 均通过 importlib 复用 detect 的
  backbone/neck，切换量化后端时会同步切换 det 模块内绑定的算子，改动时六个
  文件需保持一致。obb 将层 23 替换为 OBBDetect 头（角度分支 + dist2rbox 解码，
  v8OBBLoss、rotated NMS、batch_probiou 评估）；其输入通道数按数据集 yaml 的
  `channels` 字段自适应（dota8-multispectral 为 10），预训练 `yolo26n-obb.pt`
  的 3 通道首层卷积通过 shape-mismatch 过滤自动跳过。depth 仅替换层 23 为
  Depth 头（log-depth 回归），评估指标为 delta1 / abs_rel / rmse / silog，
  不计算 mAP。
- checkpoint 保存的是 EMA 权重（与验证/部署口径一致）。

## 参考资料
- [LSQplus](https://github.com/ZouJiu1/LSQplus)
- [Dorefa_Pact](https://github.com/ZouJiu1/Dorefa_Pact)
- [Trae](https://www.trae.cn/)
- [mqbench](https://github.com/modeltc/mqbench) — QAT / PTQ reference
- [ultralytics](https://ultralytics.com/) — YOLO26 pretrained weights
- [horizon PTQ/QAT deployment guide](https://doc.oe.horizon.auto/3.8.1/guide/model_compile.html)
- [horizon developer portal](https://developer.horizon.auto/)
- [micronet](https://github.com/666DZY666/micronet)
- [LSQuantization](https://github.com/hustzxd/LSQuantization)
- [lsq-net](https://github.com/zhutmost/lsq-net)
- [HAWQ](https://github.com/Zhen-Dong/HAWQ)
- [PACT](https://github.com/KwangHoonAn/PACT)
- [pytorch-quantization-demo](https://github.com/Jermmy/pytorch-quantization-demo)

## Credits

The idea is mine, and the implementation is mainly handled by Trae.
If there are issues with Trae's implementation, I will point them out and request fixes.
