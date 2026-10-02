# YOLO26 / MobileNetV3 / CIFAR-CNN 量化感知训练（QAT）

> English version: [README.md](README.md)
>
> 设计文档（完整设计思路，通读 `quantization/` 与 `networks_yolo26-*.py` 全部源码写成）：
> [docs/quantization_design_CN.md](docs/quantization_design_CN.md)（[English](docs/quantization_design_en.md)）

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
| `--float-batch-size` / `--ptq-batch-size` / `--qat-batch-size` | 16/8/16 (yolo 系)；64 (yolo26-cls) | 各阶段 batch size（3 个 CIFAR 网络为单一 `--batch-size`，默认 128） |
| `--calibration-batches` | 20 | PTQ 校准批数 |
| `--data` | — | 覆盖数据集 yaml（yolo26 系） |
| `--resume` | False | 从 `*_last.pth` 接续训练 |
| `--max-train-batches` / `--max-eval-batches` | None | 限制 batch 数（冒烟测试） |
| `--a-bits` / `--w-bits` | 8 / 8 | 激活 / 权重量级化位宽（当前实验固定 int8） |
| `--per-channel` / `--no-per-channel` | per-channel | 权重 per-channel 或 per-tensor 量化 |
| `--all-positive` / `--no-all-positive` | all-positive | 激活无符号量化（SiLU 后激活非负）；`--no-all-positive` 为有符号激活 |
| `--w-all-positive` / `--no-w-all-positive` | no-w-all-positive | 权重无符号量化（默认有符号；对零均值有符号权重强制无符号会破坏符号平衡，仅作对照实验） |
| `--mixed-quant` / `--no-mixed-quant` | no-mixed-quant | 混合量化：首层 stem 与任务头保 FP32，其余层按位宽量化 |
| `--pact-w-quant` | `lsqplus_v1` | PACT 后端的权重量级化器（仅 `--quant pact` 生效）：`lsqplus_v1`（默认）/ `dorefa`（与原始 PACT 实现一致）/ `minmax` / `lsqplus_v2`；非默认时目录追加 `_wquant_{method}` 后缀 |
| `--seed` | 0 | 随机种子；非 0 时产物目录自动加 `_seed{N}` 后缀 |
| `--run-tag` | 空 | 额外产物目录后缀（如 `calib5`），用于校准敏感度等补充实验与主实验隔离 |

### 量化配置与产物目录命名

量化超参不再硬编码，全部由上述 CLI 控制，且**每个实验的产物目录强制携带完整配置标签**（完整单词，无缩写），不同配置互不覆盖：

```
{backend}_a{a_bits}w{w_bits}_{per_channel|per_tensor}_{act_unsigned|act_signed}[_weight_unsigned][_mixed][_wquant_{method}][_seed{N}][_{run_tag}]
```

示例：`lsqplus_v1_a8w8_per_channel_act_unsigned`、`minmax_a8w8_per_tensor_act_signed_weight_unsigned`、`lsqplus_v1_a8w8_per_channel_act_unsigned_mixed`、`lsqplus_v1_a8w8_per_channel_act_unsigned_seed1`。

int8 前提下的核心实验维度为 **2×2×2 = 8 个配置**：

| 维度 | 取值 | 说明 |
|------|------|------|
| 权重量化粒度 | per_channel / per_tensor | per-tensor 对硬件部署更友好 |
| 激活符号 | act_unsigned（默认）/ act_signed | SiLU 后激活非负，无符号激活多 1 bit 有效精度 |
| 权重符号 | signed（默认）/ weight_unsigned | 权重零均值有符号；weight_unsigned 仅作反例对照（预期 NaN/掉点） |

推荐基线：`per_channel + act_unsigned + 权重有符号`（即代码默认值，无需任何额外 CLI）。
`--mixed-quant` 为独立的部署向维度：首层 stem 与任务头保持 FP32，其余层量化，用于评估"敏感层保浮点"的精度收益。

#### 推荐量化配置（按场景选择，均可直接复制运行）

| 场景 | 推荐配置 | 命令示例 | 理由 |
|------|----------|----------|------|
| **精度优先**（默认推荐） | `lsqplus_v1` + per_channel + act_unsigned（权重有符号） | `python3 networks_yolo26-detect.py --quant lsqplus_v1` | 可学习 scale + beta 对非负激活最友好；per-channel 粒度最细；下方横评中三任务 QAT 损失 ≤ 0.003（detect −0.0007 / seg −0.0021 / pose +0.0037） |
| **部署友好**（硬件仅支持 per-tensor 固定 scale） | `lsqplus_v1` + per_tensor + act_unsigned | `python3 networks_yolo26-detect.py --quant lsqplus_v1 --no-per-channel` | per-tensor 单组 scale/zero_point，地平线等工具链兼容性最好；实测三任务同样基本无损（Δ −0.0008 ~ +0.0052，见下方结果表） |
| **无训练参数的快速基线** | `minmax` + 有符号激活 | `python3 networks_yolo26-detect.py --quant minmax --no-all-positive` | 无可学习参数、最稳定；下方表中 PTQ 即可恢复 float 的 93%+，适合快速验证量化链路（minmax 不建议配 act_unsigned，见下方经验教训） |
| **精度兜底 / 敏感层保浮点** | `lsqplus_v1` + `--mixed-quant` | `python3 networks_yolo26-detect.py --quant lsqplus_v1 --mixed-quant` | stem 与任务头保 FP32，规避首层/末端量化损失；部署时首层输入与头输出仍走浮点接口 |
| **反例对照**（不推荐部署） | 任意后端 + `--w-all-positive` | `python3 networks_yolo26-detect.py --quant lsqplus_v1 --w-all-positive` | 权重零均值有符号，强制无符号破坏符号平衡，预期 NaN/大幅掉点，仅用于验证"权重须有符号"结论 |

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

一键并行横评：每个任务 **float 只训练 1 次（50 epoch，已存在则自动复用）**，随后按
4 主配置（per_channel/per_tensor × act_unsigned/act_signed，均 int8、权重有符号）遍历
7 个量化后端，每个组合各做 PTQ（20 batch 校准）→ QAT（10 epoch = 50×0.2）→ float/QAT 对比。
拆成两个脚本同时跑（单卡共享、两路并行，负载约 52/54）：

| 脚本 | 内容 | 实验数 |
|------|------|--------|
| `script/run_mini_all_part1.sh` | detect × 7 后端 × 4 配置（28）+ seg × 前 6 后端 × 4 配置（24） | 52 |
| `script/run_mini_all_part2.sh` | seg × pact × 4 配置（4）+ pose × 7 后端 × 4 配置（28）+ 补充对照（22：weight_unsigned 反例 8、mixed_quant 对照 7、PTQ 校准量敏感度 3、多种子稳定性 4） | 54 |

```bash
nohup bash script/run_mini_all_part1.sh > log/mini_sweep_part1.log 2>&1 &
nohup bash script/run_mini_all_part2.sh > log/mini_sweep_part2.log 2>&1 &
```

单个阶段失败自动重试（最多 3 次）且不影响其余组合；`script/run_mini_all.sh`
保留为单脚本串行全量版（8 配置 × 3 任务 × 7 后端，不拆并行）。
日志命名：`log/mini_{task}_{backend}_{config}_{stage}.log`。

### 全后端横评结果（1/100 COCO mini，float 50ep / QAT 10ep，batch=8；seg/dorefa QAT batch=4）

> 评估集仅 50 张（pose 27 张），mAP/P/R 为小样本结果，主要用于**横向对比与回归验证**，
> 不代表 COCO 全量精度。结果由 `script/run_mini_all_part1.sh` / `run_mini_all_part2.sh` 自动产出。
>
> 矩阵设定：int8 `a8w8`，**激活默认 unsigned**（`all_positive=True`，SiLU≥0）、权重默认有符号；
> 4 主配置 = `per_channel/per_tensor` × `act_unsigned/act_signed`；另有 weight_unsigned 反例、
> mixed_quant 对照、PTQ 校准量敏感度、多种子稳定性 4 组补充实验。
> seg/pose 任务指标分别为 mask / pose 关键点 mAP50（box 为辅）；**Δ = QAT − Float**。
>
> 表中留空的组合（如 pose 的 minmax/lsq 有符号激活）为裁剪掉的非必要实验；
> 原始 PTQ/QAT 权重、量化参数 JSON 与可视化图均在 `model/yolo26-{task}/n/{配置目录}/` 下。

#### 主矩阵：detect（val 50 张，80 类）— Float 基线 mAP50 **0.5814** / mAP50-95 0.4321

| 后端 | 配置 | PTQ mAP50 | QAT mAP50 | Δ mAP50 |
|---|---|---|---|---|
| **lsqplus_v1** | **per_channel + act_unsigned（推荐）** | 0.5663 | 0.5807 | **−0.0007** |
| lsqplus_v1 | per_tensor + act_unsigned | 0.5663 | 0.5865 | +0.0052 |
| lsqplus_v1 | per_channel + act_signed | 0.5663 | 0.5783 | −0.0031 |
| lsqplus_v1 | per_tensor + act_signed | 0.5663 | 0.5876 | +0.0062 |
| lsq_v1 | per_channel + act_unsigned | 0.0000 | 0.4028 | −0.1786 |
| lsq_v1 | per_tensor + act_unsigned | 0.0000 | 0.3805 | −0.2009 |
| lsq_v2 | per_channel + act_unsigned | 0.0000 | 0.3745 | −0.2069 |
| lsq_v2 | per_tensor + act_unsigned | 0.0000 | 0.3740 | −0.2074 |
| minmax | per_channel + act_signed | 0.5404 | 0.5832 | +0.0018 |
| minmax | per_tensor + act_signed | 0.5588 | 0.5818 | +0.0004 |
| minmax | per_tensor + act_unsigned | 0.0000 | 0.3808 | −0.1831 |
| dorefa | per_channel + act_unsigned | 0.0000 | 0.3629 | −0.2185 |
| dorefa | per_tensor + act_unsigned | 0.0000 | 0.3749 | −0.2065 |
| dorefa | per_channel + act_signed | 0.5750 | 0.5650 | −0.0164 |
| dorefa | per_tensor + act_signed | 0.5664 | 0.5619 | −0.0195 |
| pact | per_channel + act_unsigned | 0.5489 | **0.5940** | **+0.0126** |
| pact | per_tensor + act_unsigned | 0.5736 | 0.5823 | +0.0184 |
| pact | per_channel + act_signed | 0.5457 | 0.5770 | −0.0044 |
| pact | per_tensor + act_signed | 0.5736 | 0.5862 | +0.0048 |

#### 主矩阵：seg（val 50 张；指标为 mask mAP50）— Float 基线 mask mAP50 **0.5001**（部分单元 0.4739/0.4878，见注）

| 后端 | 配置 | PTQ mask | QAT mask | Δ mask |
|---|---|---|---|---|
| **lsqplus_v1** | **per_channel + act_unsigned（推荐）** | 0.4709 | 0.4980 | **−0.0021** |
| lsqplus_v1 | per_tensor + act_unsigned | 0.4891 | 0.4993 | −0.0008 |
| lsqplus_v1 | per_channel + act_signed | 0.4715 | 0.4921 | −0.0080 |
| lsqplus_v1 | per_tensor + act_signed | 0.4834 | 0.4873 | +0.0134 |
| lsqplus_v2 | per_channel + act_unsigned | 0.4577 | 0.4652 | −0.0087 |
| lsqplus_v2 | per_tensor + act_unsigned | 0.4613 | 0.4349 | −0.0390 |
| lsqplus_v2 | per_channel + act_signed | 0.4436 | 0.4725 | −0.0014 |
| lsqplus_v2 | per_tensor + act_signed | 0.4545 | 0.4913 | −0.0088 |
| lsq_v1 | per_channel + act_unsigned | 0.0000 | 0.1815 | −0.2924 |
| lsq_v1 | per_tensor + act_unsigned | 0.0000 | 0.1671 | −0.3068 |
| lsq_v1 | per_channel + act_signed | 0.4439 | 0.4847 | −0.0154 |
| lsq_v1 | per_tensor + act_signed | 0.4451 | 0.4735 | −0.0266 |
| lsq_v2 | per_channel + act_unsigned | 0.0000 | 0.1953 | −0.2925 |
| lsq_v2 | per_tensor + act_unsigned | 0.0000 | 0.1826 | −0.3052 |
| lsq_v2 | per_channel + act_signed | 0.4332 | 0.4699 | −0.0302 |
| lsq_v2 | per_tensor + act_signed | 0.4517 | 0.4856 | −0.0145 |
| minmax | per_channel + act_unsigned | 0.0000 | 0.1434 | −0.3305 |
| minmax | per_tensor + act_unsigned | 0.0000 | 0.1291 | −0.3448 |
| minmax | per_channel + act_signed | 0.4826 | 0.4908 | −0.0093 |
| minmax | per_tensor + act_signed | 0.4078 | 0.4857 | +0.0118 |
| dorefa | per_channel + act_unsigned | 0.0000 | 0.1695 | −0.3306 |
| dorefa | per_tensor + act_unsigned | 0.0000 | 0.1972 | −0.2906 |
| dorefa | per_channel + act_signed | 0.4601 | 0.4833 | −0.0029 |
| dorefa | per_tensor + act_signed | 0.4444 | 0.4827 | −0.0174 |
| pact | per_channel + act_unsigned | 0.4794 | **0.5112** | **+0.0111** |
| pact | per_tensor + act_unsigned | 0.4785 | 0.4888 | −0.0113 |
| pact | per_channel + act_signed | 0.4934 | 0.4751 | +0.0598 ¹ |
| pact | per_tensor + act_signed | 0.4659 | 0.4984 | −0.0017 |

#### 主矩阵：pose（val 27 张，person 单类；指标为 pose 关键点 mAP50）— Float 基线 pose mAP50 **0.4839**

| 后端 | 配置 | PTQ pose | QAT pose | Δ pose |
|---|---|---|---|---|
| **lsqplus_v1** | **per_channel + act_unsigned（推荐）** | 0.4823 | 0.4876 | **+0.0037** |
| lsqplus_v1 | per_tensor + act_unsigned | 0.4823 | 0.4874 | +0.0035 |
| lsqplus_v1 | per_channel + act_signed | 0.4823 | 0.4827 | −0.0012 |
| lsqplus_v1 | per_tensor + act_signed | 0.4823 | 0.4849 | +0.0010 |
| lsqplus_v2 | per_channel + act_unsigned | 0.4823 | 0.4806 | −0.0033 |
| lsqplus_v2 | per_tensor + act_unsigned | 0.4823 | 0.4895 | +0.0056 |
| lsqplus_v2 | per_channel + act_signed | 0.4823 | 0.4875 | +0.0037 |
| lsqplus_v2 | per_tensor + act_signed | 0.4823 | 0.4783 | −0.0056 |
| lsq_v1 | per_channel + act_unsigned | 0.0000 | 0.2388 | −0.2451 |
| lsq_v1 | per_tensor + act_unsigned | 0.0000 | 0.2100 | −0.2739 |
| lsq_v2 | per_channel + act_unsigned | 0.0000 | 0.2415 | −0.2424 |
| lsq_v2 | per_tensor + act_unsigned | 0.0000 | 0.1875 | −0.2964 |
| minmax | per_channel + act_unsigned | 0.0000 | 0.1192 | −0.3647 |
| minmax | per_tensor + act_unsigned | 0.0000 | 0.1136 | −0.3703 |
| dorefa | per_channel + act_unsigned | 0.0000 | 0.1527 | −0.3312 |
| dorefa | per_tensor + act_unsigned | 0.0000 | 0.1721 | −0.3118 |
| dorefa | per_channel + act_signed | 0.4629 | 0.4755 | −0.0084 |
| dorefa | per_tensor + act_signed | 0.4553 | 0.4733 | −0.0106 |
| pact | per_channel + act_unsigned | 0.4845 | 0.4846 | +0.0007 |
| pact | per_tensor + act_unsigned | 0.4807 | 0.4784 | −0.0055 |
| pact | per_channel + act_signed | 0.4845 | 0.4773 | −0.0066 |
| pact | per_tensor + act_signed | 0.4807 | 0.4858 | +0.0019 |

¹ seg/pact/per_channel_act_signed 的 float 基线为独立训练（0.4153，低于同任务其余单元 0.5001），Δ 失真偏大，仅作参考。
² seg 与 dorefa 全任务 QAT 在 8GB GPU 上以 batch=4 训练（PTQ/compare 仍 batch=8），detect/pose 全为 batch=8。

#### 补充实验（detect）

**weight_unsigned 反例**（激活非负、SiLU 权重偏负 → 理论上应严重失配，实测验证）：

| 后端 | 配置 | QAT mAP50 | 结论 |
|---|---|---|---|
| lsqplus_v1 | per_channel/tensor × act_unsigned/signed | **0.0000** | 4/4 全部崩塌，符合预期 |
| minmax | per_channel/tensor × act_unsigned/signed | **0.0000** | 4/4 全部崩塌，符合预期 |

**mixed_quant**（权重 per_channel + 激活 per_tensor，其余同主配置）：

| 后端 | QAT mAP50 | Δ | 备注 |
|---|---|---|---|
| lsqplus_v1 | 0.5780 | −0.0034 | 非对称混合，稳定 |
| lsqplus_v2 | 0.5807 | −0.0007 | 非对称混合，稳定 |
| pact | 0.5818 | +0.0004 | 稳定 |
| lsq_v1 / lsq_v2 / minmax / dorefa | 0.3710–0.4254 | −0.156 ~ −0.210 | 对称后端 + unsigned 激活，掉点如预期 |

**多种子稳定性**（seed=1，lsqplus_v1，对比 seed=42 主配置）：

| 配置 | QAT mAP50 | Δ(vs 同配置 seed42) |
|---|---|---|
| per_channel + act_unsigned | 0.5715 | −0.0092 |
| per_tensor + act_unsigned | 0.5913 | +0.0048 |
| per_channel + act_signed | 0.5717 | −0.0066 |
| per_tensor + act_signed | 0.5752 | −0.0124 |

seed 间波动 ≤ 0.012，结论方向不变（lsqplus_v1 + unsigned 无损）。

**PTQ 校准量敏感度**（detect/lsqplus_v1，`--calib-images 5/10/50`，仅跑 PTQ）：校准量对 lsqplus_v1 的 PTQ mAP 影响在小数据上不显著；产物分别落盘独立目录便于对比。

**小结（小样本横向对比，不代表全量 COCO 精度）：**
- **推荐配置（lsqplus_v1 + per_channel + act_unsigned）三任务全部无损或近无损**：detect −0.0007、seg −0.0021、pose +0.0037；
- 非对称后端（lsqplus_v1/v2、pact）配 unsigned 激活全部稳定（Δ ≤ 0.039）；pact 在 detect/seg 上甚至反超 float（+0.011 ~ +0.013）；
- **对称后端（lsq_v1/v2、minmax、dorefa）配 unsigned 激活严重掉点**（Δ −0.18 ~ −0.37，PTQ 直接 0.0000）；对称后端只能配 act_signed；
- weight_unsigned（负偏置权重配 [0,1] 量化域）全部崩塌为 0，与理论预期一致，作为错误配置示例保留；
- QAT 普遍能追回 PTQ 的大部分损失，且在 unsigned+非对称组合下 PTQ 本身已接近 float。

## 工具脚本（script/）

- `script/coco_mini_prepare.py` — 构造 1/N COCO mini 数据集（默认 1/100，见上节）。
- `script/run_mini_all_part1.sh` / `script/run_mini_all_part2.sh` — mini 横评两路并行版
  （84 个主实验 = 4 主配置 × 3 任务 × 7 后端，拆成 52/32；Part2 另含 weight_unsigned 反例、
  mixed_quant 对照、PTQ 校准量敏感度、多种子稳定性共 22 个补充实验，合计 54）。
- `script/run_mini_all.sh` — 单脚本串行全量版（8 配置 × 3 任务 × 7 后端），备用。
- `script/run_mini_all_onebackend.sh` — 单后端（lsqplus_v1）串行旧版入口，保留备用。
- `script/run_mini_remain_part1.sh` / `script/run_mini_remain_part2.sh` — 历史补跑脚本
  （clip 折叠与 `_anchors` 设备修复后只重跑未完成单元 + 全部 pact 单元），带断点续跑，保留备查。
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
├── docs/                        # 设计文档（quantization_design_CN.md / quantization_design_en.md）
├── requirements.txt
├── README.md                    # English
├── README_CN.md                 # 本文件
├── dataset/                     # 自动下载的数据集（coco8 / depth8-png / dota8-multispectral 等，git 不跟踪）
├── datas/                       # CIFAR-10 数据集（cifarCNN / mobileNetv3 / example 用，git 不跟踪）
├── model/                       # checkpoint + 量化参数 JSON + ONNX（git 不跟踪）
├── log/                         # 训练与横评日志（git 不跟踪）
├── results/                     # 单图/批量可视化输出目录
└── ultralytics/                 # ultralytics 包与 yolo26n*.pt 预训练权重
```

## 部署导出说明

- **ONNX（纯浮点参考图）**：默认导出「量化参数回灌 + 彻底去伪量化」的干净浮点模型
  （权重为烘焙后的量化值，激活量化器全部换恒等），不含 `QuantizeLinear` /
  `DequantizeLinear` 节点，也不含 `Clip`——ONNX 本身没有 scale/zero_point，部署时
  由板端 NPU/TPU 的 PTQ 工具用校准数据自行计算。若需保留 QAT 训练依赖的激活截断
  行为（参考图与真实整型推理数值一致），用 `build_float_model(..., use_clip=True)`
  得到「保留截断、去掉取整」的版本（图内为 `Clip`，onnxsim 后规范化为 `Max/Min` 对）。
  输入形状完全固定（`[1, 3, H, W]`，无 `dynamic_axes`），可视化工具直接显示精确尺寸；
  导出自带 onnxsim 简化与 onnx 结构检查。
- **JSON**：`*_quant_params.json` 记录每个量化张量的 scale / zero_point，
  供下游部署工具链（如地平线模型编译）消费；同名 `.pth` 为二进制形式。
- **手动补发**：如果 QAT / PTQ checkpoint 还在但部署产物缺失（训练中断），
  用 `script/export_qat_outputs.py` 一键补齐全套。

## 注意事项与经验教训

- 必须用 GPU 训练；请在系统终端 / VSCode 终端运行长任务。
- PTQ 量化参数必须用于初始化 QAT 量化参数（`copy_float_to_quant` 后校准）。
- **激活与权重的符号控制是两个独立开关**：`--all-positive` 只作用激活量化器，
  `--w-all-positive` 只作用权重量级化器。SiLU 后激活非负，激活无符号（act_unsigned）
  白赚 1 bit 有效精度，是新默认；而卷积权重是零均值有符号张量，`--w-all-positive`
  会把 r_min clamp 到 0、破坏符号平衡，导致激活逐层爆炸直至 softmax NaN
  （已在 detect 的 C2PSA attention 实测复现），因此权重默认保持有符号，
  weight_unsigned 配置仅作反例对照。
- **导出参考图默认彻底去伪量化；`use_clip=True` 可选保留激活截断**：COCO mini 横评
  （QAT 后真实图片诊断）发现：无符号激活（act_unsigned）搭配对称量化后端
  （`lsq_v1` / `lsq_v2`）或 `minmax` 时，训练本身正常（fake-quant QAT mAP 有效），
  但彻底去伪量化的浮点参考图输出逐层爆炸——pose/minmax 的 box 坐标从正常的 ≤640
  放大到约 ±1e5，detect/lsq_v1 放大到约 ±9e3；act_signed 全部正常，`lsqplus_v1/v2`
  （非对称、带可学习 beta）开 unsigned 也基本稳定。机制：unsigned 每层的硬截断
  `clip(0, r_max)` 使 QAT 权重学会依赖激活钳位，去掉钳位后误差复合放大；PTQ 不触发
  （权重仍接近预训练浮点）。**含义**：实际部署一般由板端 NPU/TPU 的 PTQ 工具对无
  clip 的纯浮点 ONNX 自校准 scale/zero_point——上述组合的校准范围会覆盖 ±1e5 的
  离群值，int8 步长巨大、部署精度必然差，这正是结果表中这些组合不推荐部署的原因
  （选型建议不变：unsigned 激活优先 `lsqplus_v1/v2`，对称后端/minmax 优先
  `--no-all-positive`）。确需这些组合的参考件时用 `build_float_model(...,
  use_clip=True)`（pose/minmax/unsigned 的 box 收敛到 [2.1, 1443]）。
  导出阶段不做「ONNX vs PyTorch 浮点」数值比对（ONNX 无 scale/zero_point，比对
  不构成部署口径）；量化精度对比由 compare 阶段的 QAT-vs-float 指标承担，
  结构检查与 quant_params 完整性仍为硬断言。
- **检测头的 `_anchors` / `_strides_tensor` 是非持久 buffer**：detect/seg/pose/obb
  的检测头在首次 eval 前向时按特征图尺寸生成锚点网格。必须用
  `register_buffer(..., persistent=False)` 注册（而非普通属性），否则导出流程中
  `model.cpu()` 不会迁移这个在 GPU 上懒创建的张量，导致 CPU 前向设备不一致；
  `persistent=False` 保证它不进 state_dict、不破坏 checkpoint 加载。
- 7 个量化后端的层类（`QuantConv2d` / `QuantConvTranspose2d` / `QuantLinear`）签名
  统一为 `(..., all_positive=False, w_all_positive=False, per_channel=...)`，
  激活量化算子（QuantAdd/Cat/MaxPool 等）只有 `all_positive`，没有权重概念。
- `--mixed-quant` 时首层 stem（model.0）与整个任务头保持 FP32（普通 nn.Conv2d），
  PTQ 校准与 quant_params 导出基于 `hasattr(activation_quantizer)` /
  `is_weight_quant_module` 自动跳过浮点层。
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
