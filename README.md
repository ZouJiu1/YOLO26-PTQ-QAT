# YOLO26 / MobileNetV3 / CIFAR-CNN Quantization-Aware Training (QAT)

> 中文版：[README_CN.md](README_CN.md)
>
> Design document (complete design rationale, written from a full read-through of
> `quantization/` and `networks_yolo26-*.py`):
> [docs/quantization_design_en.md](docs/quantization_design_en.md) ([中文](docs/quantization_design_CN.md))

A PyTorch QAT framework providing a unified pipeline — **float training → PTQ
calibration → QAT fine-tuning → precision comparison → deployment export
(ONNX + JSON quant params)** — for:

| file | task | dataset |
|------|------|---------|
| `networks_yolo26-detect.py` | YOLO26 object detection | coco8 (auto-download) / COCO mini |
| `networks_yolo26-seg.py` | YOLO26 instance segmentation | coco8-seg (auto-download) / COCO mini |
| `networks_yolo26-pose.py` | YOLO26 pose estimation | coco8-pose (auto-download) / COCO mini |
| `networks_yolo26-cls.py` | YOLO26 image classification | imagenet10 |
| `networks_yolo26-obb.py` | YOLO26 rotated detection (OBB) | dota8-multispectral (10-ch, auto-download) |
| `networks_yolo26-depth.py` | YOLO26 monocular depth estimation | depth8 (RGB + 16-bit PNG depth, auto-download) |
| `networks_mobileNetv3.py` | MobileNetV3 classification | CIFAR-10 |
| `networks_cifarCNN.py` | custom CIFAR-CNN | CIFAR-10 |
| `networks_example.py` | minimal quantized-op example | CIFAR-10 (no CLI) |

**Datasets** — on first run each yolo26 task auto-downloads its official dataset
into the project `dataset/` directory via ultralytics' own `check_det_dataset`
/ `safe_download` pipeline (no manual setup; the generated portable yaml points
at `dataset/<name>`):

| task | official yaml | download |
|------|---------------|----------|
| detect | `coco8.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip> |
| seg | `coco8-seg.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-seg.zip> |
| pose | `coco8-pose.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip> |
| obb | `dota8-multispectral.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/dota8-multispectral.zip> |
| depth | `depth8.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/depth8-png.zip> |
| cls | `imagenet10.yaml` | <https://github.com/ultralytics/assets/releases/download/v0.0.0/imagenet10.zip> |

- Upstream dataset definitions: <https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/datasets>
- Any other dataset directory / yaml can be used manually with `--data /path/to/data.yaml`
  (or a dataset directory containing a yaml).

## Installation

```bash
pip3 install -r requirements.txt
```

`requirements.txt` pins the versions verified in this environment (torch 2.x +
CUDA 12.6, ultralytics, onnx / onnxruntime / onnxsim). For CPU-only, switch
torch/torchvision to the `+cpu` wheels.

## Unified pipeline

Every `networks_*.py` runs the same pipeline:

```
float_train -> PTQ_calibration -> QAT_training -> compare_precision
```

- YOLO nets load `ultralytics/yolo26n*.pt` pretrained weights by default; official weights missing locally (any scale/task, e.g. `yolo26s-seg.pt`) are auto-downloaded from GitHub Releases via ultralytics' `attempt_download_asset`, while a user-supplied path is loaded directly;
- PTQ statistics initialize the QAT quantizers;
- QAT fine-tunes with straight-through estimators (STE).

Each PTQ / QAT stage writes five artifacts (`save_quant_outputs`):

| artifact | description |
|----------|-------------|
| `model/{prefix}_{name}.pth` | quantized model weights |
| `model/{prefix}_quant_params.json` | per-tensor scale / zero_point (for deployment) |
| `model/{prefix}_quant_params.pth` | same params in binary .pth form |
| `model/{prefix}_{name}_float.pth` | clean float weights with quant params folded back |
| `model/{prefix}_{name}_float.onnx` | ONNX export of the above (opset=16, static shape `[1, C, H, W]`, no dynamic_axes; C=3, C=10 for the multispectral OBB net) |

## Unified features (all networks_*.py)

- **best suffix**: the best checkpoint is saved as `*_best.pth` alongside
  `*_last.pth`; legacy unsuffixed float weights are still accepted by the
  compare stage (prefers `_best`, falls back otherwise).
- **ONNX is exported alongside the best .pth** (`_try_export_onnx`); export
  failures only warn and never interrupt training.
- **ONNX input shape is fully static** (`[1, C, H, W]`, no `dynamic_axes`; C=3,
  or C=10 for the multispectral OBB net); visualization tools display exact
  dimensions without symbolic axes.
- **ONNX is always simplified with onnxsim** (skipped automatically if not installed).
- **resume**: pass `--resume` to continue training from `*_last.pth`
  (model / optimizer / scheduler / epoch are restored).
- **Per-eval visualization**: after every `evaluate()` call, the first ~30 val
  images are rendered with ultralytics' official `plot_images` (GT / prediction
  mosaics; seg adds masks, pose adds keypoints, obb draws rotated boxes as
  polygons, cls shows GT vs predicted label, depth renders depth heatmaps;
  multispectral inputs are displayed with their first 3 channels) and saved to
  `{model_dir}/fvisualize/`, named `{stage_tag}_batch{n}_{labels|pred}.jpg`
  (e.g. `float_ep001_batch0_pred.jpg`, `qat_lsqplus_v1_best_batch0_labels.jpg`).
  The 3 CIFAR classification nets render a matplotlib grid instead. Visualization
  failures only warn and never affect evaluation.
- The dataloader collate always uses `torch.concat` (no quantized concat);
  `evaluate()` receives the raw data dict from `check_det_dataset()`.

## Quantization backends & recommendation

`--quant` switches among 7 backends:

| `--quant` | backend module | description |
|-----------|----------------|-------------|
| `lsqplus_v1` | `quantization/lsqplus_quantize_V1.py` | LSQ+ (learnable scale + beta) |
| `lsqplus_v2` | `quantization/lsqplus_quantize_V2.py` | LSQ+ V2 |
| `lsq_v1` | `quantization/lsqquantize_V1.py` | LSQ (symmetric, learnable scale) |
| `lsq_v2` | `quantization/lsqquantize_V2.py` | LSQ V2 |
| `minmax` | `quantization/minmax.py` | MinMax (running min/max, standard asymmetric) |
| `dorefa` | `quantization/dorefa.py` | DoReFa + learnable activation scale |
| `pact` | `quantization/pact.py` | PACT (learnable clip threshold alpha) |

### Recommendation

**Use `lsqplus_v1` by default** (the default in every network file):

1. **Most thoroughly validated**: the full float → PTQ → QAT pipeline for
   detect / seg / pose passed end-to-end on 1/100 COCO mini, and it takes part
   in the unified cross-backend comparison across all 7 backends (results table below);
2. **Best accuracy**: learnable scale + beta (offset) suits non-negative
   activations (post-SiLU); after float-forward min-max initialization in PTQ,
   quantized accuracy matches the float baseline;
3. **Hardened stability**: dummy inputs use `randn * 0.1` to avoid the
   all-zero-forward s=0 / NaN pitfall.

When to pick the others:

- `minmax`: simplest and most stable, no learnable parameters — a good baseline;
- `lsq_v1`: a lighter choice when only symmetric quantization is needed;
- `dorefa` / `pact`: for research comparison (dorefa's unbounded-activation
  collapse was fixed by adding a learnable scale s);
- `*_v2`: experimental variants of the corresponding algorithms, same interface as V1.

Each backend implements the same set of quantized operators: `QuantConv2d`,
`QuantConvTranspose2d`, `QuantLinear`, `QuantAdd`, `QuantSub`, `QuantMultiply`,
`QuantDiv`, `QuantConcat`, `QuantMaxPool`, etc. `quantization/__init__.py`
provides backend loading (`load_quant_backend`), quantizer state freeze/reset,
and cross-backend scale / zero_point extraction (`collect_quant_params`).
`quantization/constants.py` holds the `INIT_STATE_FROZEN` sentinel.

## Quick start

Full pipeline for one network (detection example):

```bash
python3 networks_yolo26-detect.py --model yolo26n --stage all --quant lsqplus_v1
```

Run a single stage:

```bash
python3 networks_cifarCNN.py --quant minmax --stage qat
python3 networks_cifarCNN.py --quant minmax --stage compare
```

Smoke test (1 train batch + 1 eval batch per epoch):

```bash
python3 networks_mobileNetv3.py --quant lsqplus_v1 \
    --float-epochs 1 --max-train-batches 1 --max-eval-batches 1
```

### Common arguments

| argument | default | description |
|----------|---------|-------------|
| `--quant` | `lsqplus_v1` | quantization backend |
| `--stage` | `all` | `all / float / ptq / qat / compare` |
| `--float-epochs` | 100 | float-training epochs |
| `--qat-epochs` | float/5 | QAT epochs |
| `--float-lr` / `--qat-lr` | — | learning rates (QAT defaults to float/10) |
| `--float-batch-size` / `--ptq-batch-size` / `--qat-batch-size` | 16/8/16 (yolo nets); 64 (yolo26-cls) | per-stage batch sizes (the 3 CIFAR nets use a single `--batch-size`, default 128) |
| `--calibration-batches` | 20 | PTQ calibration batches |
| `--data` | — | override dataset yaml (yolo26 nets) |
| `--resume` | False | resume training from `*_last.pth` |
| `--max-train-batches` / `--max-eval-batches` | None | limit batches (smoke test) |
| `--a-bits` / `--w-bits` | 8 / 8 | activation / weight quantization bit-width (current sweeps fix int8) |
| `--per-channel` / `--no-per-channel` | per-channel | weight per-channel or per-tensor quantization |
| `--all-positive` / `--no-all-positive` | all-positive | unsigned activation quantization (post-SiLU activations are non-negative); `--no-all-positive` = signed activations |
| `--w-all-positive` / `--no-w-all-positive` | no-w-all-positive | unsigned weight quantization (default signed; forcing zero-mean signed weights unsigned breaks sign balance — ablation only) |
| `--mixed-quant` / `--no-mixed-quant` | no-mixed-quant | mixed quantization: stem (layer 0) and task head stay FP32, other layers quantized |
| `--pact-w-quant` | `lsqplus_v1` | weight quantizer for the pact backend (only with `--quant pact`): `lsqplus_v1` (default) / `dorefa` (matches the original PACT impl) / `minmax` / `lsqplus_v2`; non-default adds a `_wquant_{method}` directory suffix |
| `--seed` | 0 | random seed; non-zero adds a `_seed{N}` suffix to the artifact directory |
| `--run-tag` | empty | extra artifact directory suffix (e.g. `calib5`) to isolate supplementary runs (calibration sensitivity etc.) from main runs |

### Quantization configs & artifact directory naming

Quantization hyperparameters are no longer hardcoded — all are CLI-driven, and
**every experiment directory carries an explicit full-word config tag** (no
abbreviations) so different configs never overwrite each other:

```
{backend}_a{a_bits}w{w_bits}_{per_channel|per_tensor}_{act_unsigned|act_signed}[_weight_unsigned][_mixed][_wquant_{method}][_seed{N}][_{run_tag}]
```

Examples: `lsqplus_v1_a8w8_per_channel_act_unsigned`,
`minmax_a8w8_per_tensor_act_signed_weight_unsigned`,
`lsqplus_v1_a8w8_per_channel_act_unsigned_mixed`,
`lsqplus_v1_a8w8_per_channel_act_unsigned_seed1`.

The core experiment matrix under int8 is **2×2×2 = 8 configs**:

| axis | values | notes |
|------|--------|-------|
| weight granularity | per_channel / per_tensor | per-tensor is hardware-deployment friendly |
| activation signedness | act_unsigned (default) / act_signed | post-SiLU activations are non-negative; unsigned gains 1 effective bit |
| weight signedness | signed (default) / weight_unsigned | weights are zero-mean signed; weight_unsigned is a counterexample ablation (NaN/accuracy drop expected) |

Recommended baseline: `per_channel + act_unsigned + signed weights` (the code
defaults — no extra CLI needed). `--mixed-quant` is an independent
deployment-oriented axis: stem and task head stay FP32 to measure the accuracy
recovery from keeping sensitive layers in float.

#### Recommended quantization configs (pick by scenario, commands copy-paste ready)

| scenario | config | command | rationale |
|----------|--------|---------|-----------|
| **Accuracy first** (default recommendation) | `lsqplus_v1` + per_channel + act_unsigned (signed weights) | `python3 networks_yolo26-detect.py --quant lsqplus_v1` | learnable scale + beta suits non-negative activations; per-channel is the finest granularity; in the sweep below the QAT loss is ≤ 0.003 on all three tasks (detect −0.0007 / seg −0.0021 / pose +0.0037) |
| **Deployment friendly** (hardware supports per-tensor fixed scale only) | `lsqplus_v1` + per_tensor + act_unsigned | `python3 networks_yolo26-detect.py --quant lsqplus_v1 --no-per-channel` | single scale/zero_point per tensor; best toolchain compatibility (e.g. Horizon); also near-lossless in practice (Δ −0.0008 ~ +0.0052, see tables below) |
| **Parameter-free quick baseline** | `minmax` + signed activations | `python3 networks_yolo26-detect.py --quant minmax --no-all-positive` | no learnable params, most stable; PTQ alone recovers 93%+ of float in the tables below — good for pipeline validation (minmax + act_unsigned is not advised, see Notes) |
| **Accuracy fallback / sensitive layers in float** | `lsqplus_v1` + `--mixed-quant` | `python3 networks_yolo26-detect.py --quant lsqplus_v1 --mixed-quant` | stem + head stay FP32, avoiding first/last-layer quantization loss; deployment keeps float interfaces at input/output |
| **Counterexample** (do NOT deploy) | any backend + `--w-all-positive` | `python3 networks_yolo26-detect.py --quant lsqplus_v1 --w-all-positive` | zero-mean signed weights forced unsigned break sign balance; NaN / large accuracy drop expected — validates the "weights must stay signed" conclusion |

## COCO mini dataset & cross-backend comparison (1/100 sampling)

Benchmark all 7 quantization backends on detect / seg / pose using 1/100 of COCO2017:

```bash
python3 script/coco_mini_prepare.py                 # default 1/100, outputs to project dataset/
python3 script/coco_mini_prepare.py --coco-root /path/to/coco --ratio 100
python3 script/coco_mini_prepare.py --out-dir /path/to/output   # custom output root dir
```

Sampling is deterministic (every Nth sorted image id, `ids[::N]`); images are
relative symlinks (no copying); iscrowd annotations are skipped; category ids
are remapped to 0..79. Output defaults to the project-root `dataset/` directory
(alongside coco8/, override with `--out-dir`), with per-task
images / labels / train2017.txt / val2017.txt and the corresponding
`coco_mini_*.yaml` files. 1/100 split sizes:

| task | train images | val images | classes |
|------|-------------|-----------|---------|
| detect | 1183 | 50 | 80 |
| seg | 1183 | 50 | 80 |
| pose | 642 | 27 | 1 (person) |

The sweep trains **float once per task (50 epochs, auto-reused if it exists)**,
then loops over the 4 main configs (per_channel/per_tensor × act_unsigned/act_signed,
all int8 with signed weights) × 7 backends, running PTQ (20 calibration batches) →
QAT (10 epochs = 50×0.2) → float/QAT compare for each. Split into two scripts run
in parallel on the shared GPU (load ≈ 52/54):

| script | contents | runs |
|--------|----------|------|
| `script/run_mini_all_part1.sh` | detect × 7 backends × 4 configs (28) + seg × first 6 backends × 4 configs (24) | 52 |
| `script/run_mini_all_part2.sh` | seg × pact × 4 configs (4) + pose × 7 backends × 4 configs (28) + supplementary runs (22: weight_unsigned counterexamples 8, mixed_quant ablation 7, PTQ calibration-size sensitivity 3, multi-seed stability 4) | 54 |

```bash
nohup bash script/run_mini_all_part1.sh > log/mini_sweep_part1.log 2>&1 &
nohup bash script/run_mini_all_part2.sh > log/mini_sweep_part2.log 2>&1 &
```

A failed stage is retried up to 3 times and never blocks the remaining combos;
`script/run_mini_all.sh` is kept as the serialized all-in-one fallback
(8 configs × 3 tasks × 7 backends, no parallel split).
Log naming: `log/mini_{task}_{backend}_{config}_{stage}.log`.

### Cross-backend results (1/100 COCO mini, float 50ep / QAT 10ep, batch=8 unified across all tasks and backends; seg shows anomalous loss on a few combos — mask bce init issue, not memory)

> The val set has only 50 images (27 for pose), so mAP/P/R are small-sample
> numbers meant for cross-backend comparison and regression checks, not full-COCO
> accuracy. Produced automatically by `script/run_mini_all_part1.sh` / `run_mini_all_part2.sh`.
>
> Matrix setup: int8 `a8w8`, **activations default unsigned** (`all_positive=True`, SiLU≥0), weights default signed;
> 4 main configs = `per_channel/per_tensor` × `act_unsigned/act_signed`; plus weight_unsigned counterexamples,
> mixed_quant ablation, PTQ calibration-size sensitivity, and multi-seed stability.
> seg/pose task metric = mask / keypoint pose mAP50 (box secondary); **Δ = QAT − Float**.
>
> Combos left blank in the tables (e.g. pose minmax/lsq with signed activations)
> were pruned as non-essential; raw PTQ/QAT weights, quant-param JSONs and
> visualizations live under `model/yolo26-{task}/n/{config_dir}/`.

#### Main matrix: detect (50 val images, 80 classes) — Float baseline mAP50 **0.5814** / mAP50-95 0.4321

Each row corresponds to three consecutive stages. Replace `{BACKEND}` and `{FLAGS}` from the table below:

```bash
# PTQ calibration (20 batches, no training — run once per config)
python3 networks_yolo26-detect.py \
  --model yolo26n --stage ptq --quant {BACKEND} \
  --data dataset/coco_mini_detect.yaml \
  --float-epochs 50 --qat-epochs 10 --calibration-batches 20 \
  --float-batch-size 8 --ptq-batch-size 8 --qat-batch-size 8 --num-workers 0 \
  {FLAGS}

# QAT fine-tuning (10 epochs, loads PTQ checkpoint from model/yolo26-detect/n/{dir}/)
python3 networks_yolo26-detect.py \
  --model yolo26n --stage qat --quant {BACKEND} \
  --data dataset/coco_mini_detect.yaml \
  --float-epochs 50 --qat-epochs 10 --calibration-batches 20 \
  --float-batch-size 8 --ptq-batch-size 8 --qat-batch-size 8 --num-workers 0 \
  {FLAGS}

# Compare float vs QAT (loads float best and QAT checkpoint, evaluates both)
python3 networks_yolo26-detect.py \
  --model yolo26n --stage compare --quant {BACKEND} \
  --data dataset/coco_mini_detect.yaml \
  --float-epochs 50 --qat-epochs 10 --calibration-batches 20 \
  --float-batch-size 8 --ptq-batch-size 8 --qat-batch-size 8 --num-workers 0 \
  {FLAGS}
```

| Backend | Config | PTQ mAP50 | QAT mAP50 | Δ mAP50 | CLI Flags |
|---|---|---|---|---|---|
| **lsqplus_v1** | **per_channel + act_unsigned (recommended)** | 0.5663 | 0.5807 | **−0.0007** | *(default)* |
| lsqplus_v1 | per_tensor + act_unsigned | 0.5663 | 0.5865 | +0.0052 | `--no-per-channel` |
| lsqplus_v1 | per_channel + act_signed | 0.5663 | 0.5783 | −0.0031 | `--no-all-positive` |
| lsqplus_v1 | per_tensor + act_signed | 0.5663 | 0.5876 | +0.0062 | `--no-per-channel --no-all-positive` |
| lsq_v1 | per_channel + act_unsigned | 0.0000 | 0.4028 | −0.1786 | *(default)* |
| lsq_v1 | per_tensor + act_unsigned | 0.0000 | 0.3805 | −0.2009 | `--no-per-channel` |
| lsq_v2 | per_channel + act_unsigned | 0.0000 | 0.3745 | −0.2069 | *(default)* |
| lsq_v2 | per_tensor + act_unsigned | 0.0000 | 0.3740 | −0.2074 | `--no-per-channel` |
| minmax | per_channel + act_signed | 0.5404 | 0.5832 | +0.0018 | `--no-all-positive` |
| minmax | per_tensor + act_signed | 0.5588 | 0.5818 | +0.0004 | `--no-per-channel --no-all-positive` |
| minmax | per_tensor + act_unsigned | 0.0000 | 0.3808 | −0.1831 | `--no-per-channel` |
| dorefa | per_channel + act_unsigned | 0.5631 | 0.5723 | −0.0091 | *(default)* |
| dorefa | per_tensor + act_unsigned | 0.5607 | 0.5779 | −0.0034 | `--no-per-channel` |
| dorefa | per_channel + act_signed | 0.5753 | 0.5756 | −0.0058 | `--no-all-positive` |
| dorefa | per_tensor + act_signed | 0.5609 | 0.5736 | −0.0078 | `--no-per-channel --no-all-positive` |
| pact | per_channel + act_unsigned | 0.5489 | **0.5940** | **+0.0126** | *(default)* |
| pact | per_tensor + act_unsigned | 0.5736 | 0.5823 | +0.0184 | `--no-per-channel` |
| pact | per_channel + act_signed | 0.5457 | 0.5770 | −0.0044 | `--no-all-positive` |
| pact | per_tensor + act_signed | 0.5736 | 0.5862 | +0.0048 | `--no-per-channel --no-all-positive` |

#### Main matrix: seg (50 val images; metric = mask mAP50) — Float baseline mask mAP50 **0.5001** (some cells 0.4739/0.4878, see notes)

Three stages per row (same template as detect above, change `--data` and script name):

```bash
# PTQ → QAT → Compare for seg: replace {BACKEND} and {FLAGS} from the table
python3 networks_yolo26-seg.py --model yolo26n --stage ptq --quant {BACKEND} \
  --data dataset/coco_mini_seg.yaml \
  --float-epochs 50 --qat-epochs 10 --calibration-batches 20 \
  --float-batch-size 8 --ptq-batch-size 8 --qat-batch-size 8 --num-workers 0 \
  {FLAGS}
# then same with --stage qat, then --stage compare
```

| Backend | Config | PTQ mask | QAT mask | Δ mask | CLI Flags |
|---|---|---|---|---|---|
| **lsqplus_v1** | **per_channel + act_unsigned (recommended)** | 0.4709 | 0.4980 | **−0.0021** | *(default)* |
| lsqplus_v1 | per_tensor + act_unsigned | 0.4891 | 0.4993 | −0.0008 | `--no-per-channel` |
| lsqplus_v1 | per_channel + act_signed | 0.4715 | 0.4921 | −0.0080 | `--no-all-positive` |
| lsqplus_v1 | per_tensor + act_signed | 0.4834 | 0.4873 | +0.0134 | `--no-per-channel --no-all-positive` |
| lsqplus_v2 | per_channel + act_unsigned | 0.4577 | 0.4652 | −0.0087 | *(default)* |
| lsqplus_v2 | per_tensor + act_unsigned | 0.4613 | 0.4349 | −0.0390 | `--no-per-channel` |
| lsqplus_v2 | per_channel + act_signed | 0.4436 | 0.4725 | −0.0014 | `--no-all-positive` |
| lsqplus_v2 | per_tensor + act_signed | 0.4545 | 0.4913 | −0.0088 | `--no-per-channel --no-all-positive` |
| lsq_v1 | per_channel + act_unsigned | 0.0000 | 0.1815 | −0.2924 | *(default)* |
| lsq_v1 | per_tensor + act_unsigned | 0.0000 | 0.1671 | −0.3068 | `--no-per-channel` |
| lsq_v1 | per_channel + act_signed | 0.4439 | 0.4847 | −0.0154 | `--no-all-positive` |
| lsq_v1 | per_tensor + act_signed | 0.4451 | 0.4735 | −0.0266 | `--no-per-channel --no-all-positive` |
| lsq_v2 | per_channel + act_unsigned | 0.0000 | 0.1953 | −0.2925 | *(default)* |
| lsq_v2 | per_tensor + act_unsigned | 0.0000 | 0.1826 | −0.3052 | `--no-per-channel` |
| lsq_v2 | per_channel + act_signed | 0.4332 | 0.4699 | −0.0302 | `--no-all-positive` |
| lsq_v2 | per_tensor + act_signed | 0.4517 | 0.4856 | −0.0145 | `--no-per-channel --no-all-positive` |
| minmax | per_channel + act_unsigned | 0.0000 | 0.1434 | −0.3305 | *(default)* |
| minmax | per_tensor + act_unsigned | 0.0000 | 0.1291 | −0.3448 | `--no-per-channel` |
| minmax | per_channel + act_signed | 0.4826 | 0.4908 | −0.0093 | `--no-all-positive` |
| minmax | per_tensor + act_signed | 0.4078 | 0.4857 | +0.0118 | `--no-per-channel --no-all-positive` |
| dorefa | per_channel + act_unsigned | 0.4945 | 0.4942 | −0.0060 | *(default)* |
| dorefa | per_tensor + act_unsigned | 0.4732 | 0.5073 | +0.0072 | `--no-per-channel` |
| dorefa | per_channel + act_signed | 0.4871 | 0.5038 | +0.0037 | `--no-all-positive` |
| dorefa | per_tensor + act_signed | 0.4722 | 0.4964 | −0.0037 | `--no-per-channel --no-all-positive` |
| pact | per_channel + act_unsigned | 0.4794 | **0.5112** | **+0.0111** | *(default)* |
| pact | per_tensor + act_unsigned | 0.4785 | 0.4888 | −0.0113 | `--no-per-channel` |
| pact | per_channel + act_signed | 0.4934 | 0.4751 | +0.0598 ¹ | `--no-all-positive` |
| pact | per_tensor + act_signed | 0.4659 | 0.4984 | −0.0017 | `--no-per-channel --no-all-positive` |

#### Main matrix: pose (27 val images, single person class; metric = keypoint pose mAP50) — Float baseline pose mAP50 **0.4839**

Three stages per row (same template, change `--data` and script name):

```bash
# PTQ → QAT → Compare for pose: replace {BACKEND} and {FLAGS} from the table
python3 networks_yolo26-pose.py --model yolo26n --stage ptq --quant {BACKEND} \
  --data dataset/coco_mini_pose.yaml \
  --float-epochs 50 --qat-epochs 10 --calibration-batches 20 \
  --float-batch-size 8 --ptq-batch-size 8 --qat-batch-size 8 --num-workers 0 \
  {FLAGS}
# then same with --stage qat, then --stage compare
```

| Backend | Config | PTQ pose | QAT pose | Δ pose | CLI Flags |
|---|---|---|---|---|---|
| **lsqplus_v1** | **per_channel + act_unsigned (recommended)** | 0.4823 | 0.4876 | **+0.0037** | *(default)* |
| lsqplus_v1 | per_tensor + act_unsigned | 0.4823 | 0.4874 | +0.0035 | `--no-per-channel` |
| lsqplus_v1 | per_channel + act_signed | 0.4823 | 0.4827 | −0.0012 | `--no-all-positive` |
| lsqplus_v1 | per_tensor + act_signed | 0.4823 | 0.4849 | +0.0010 | `--no-per-channel --no-all-positive` |
| lsqplus_v2 | per_channel + act_unsigned | 0.4823 | 0.4806 | −0.0033 | *(default)* |
| lsqplus_v2 | per_tensor + act_unsigned | 0.4823 | 0.4895 | +0.0056 | `--no-per-channel` |
| lsqplus_v2 | per_channel + act_signed | 0.4823 | 0.4875 | +0.0037 | `--no-all-positive` |
| lsqplus_v2 | per_tensor + act_signed | 0.4823 | 0.4783 | −0.0056 | `--no-per-channel --no-all-positive` |
| lsq_v1 | per_channel + act_unsigned | 0.0000 | 0.2388 | −0.2451 | *(default)* |
| lsq_v1 | per_tensor + act_unsigned | 0.0000 | 0.2100 | −0.2739 | `--no-per-channel` |
| lsq_v2 | per_channel + act_unsigned | 0.0000 | 0.2415 | −0.2424 | *(default)* |
| lsq_v2 | per_tensor + act_unsigned | 0.0000 | 0.1875 | −0.2964 | `--no-per-channel` |
| minmax | per_channel + act_unsigned | 0.0000 | 0.1192 | −0.3647 | *(default)* |
| minmax | per_tensor + act_unsigned | 0.0000 | 0.1136 | −0.3703 | `--no-per-channel` |
| dorefa | per_channel + act_unsigned | 0.4942 | 0.4843 | +0.0004 | *(default)* |
| dorefa | per_tensor + act_unsigned | 0.4856 | 0.4900 | +0.0061 | `--no-per-channel` |
| dorefa | per_channel + act_signed | 0.4836 | 0.4792 | −0.0047 | `--no-all-positive` |
| dorefa | per_tensor + act_signed | 0.4844 | 0.4758 | −0.0081 | `--no-per-channel --no-all-positive` |
| pact | per_channel + act_unsigned | 0.4845 | 0.4846 | +0.0007 | *(default)* |
| pact | per_tensor + act_unsigned | 0.4807 | 0.4784 | −0.0055 | `--no-per-channel` |
| pact | per_channel + act_signed | 0.4845 | 0.4773 | −0.0066 | `--no-all-positive` |
| pact | per_tensor + act_signed | 0.4807 | 0.4858 | +0.0019 | `--no-per-channel --no-all-positive` |

¹ seg/pact/per_channel_act_signed uses an independently-trained float baseline (0.4153, lower than the other seg cells' 0.5001); its Δ is skewed and for reference only.
² seg shows anomalous loss on a few combos (e.g. lsqplus_v2/per_tensor/act_unsigned, possibly a mask-bce init issue); the new dorefa (2026-10) removed the tanh nonlinearity and all tasks now run at batch=8 uniformly — no more per-task downgrades.

#### Supplementary experiments (detect)

**weight_unsigned counterexample** (non-negative activations, SiLU weights biased negative → theoretically a severe mismatch, confirmed):

| Backend | Config | QAT mAP50 | Conclusion |
|---|---|---|---|
| lsqplus_v1 | per_channel/tensor × act_unsigned/signed | **0.0000** | 4/4 all collapse, as expected |
| minmax | per_channel/tensor × act_unsigned/signed | **0.0000** | 4/4 all collapse, as expected |

**mixed_quant** (weights per_channel + activations per_tensor, rest same as main config):

```bash
# Mixed quant for detect: replace {BACKEND} from the table
python3 networks_yolo26-detect.py --model yolo26n --stage compare --quant {BACKEND} \
  --data dataset/coco_mini_detect.yaml \
  --float-epochs 50 --qat-epochs 10 --calibration-batches 20 \
  --float-batch-size 8 --ptq-batch-size 8 --qat-batch-size 8 --num-workers 0 \
  --mixed-quant
```

| Backend | QAT mAP50 | Δ | Note |
|---|---|---|---|
| lsqplus_v1 | 0.5780 | −0.0034 | asymmetric mix, stable |
| lsqplus_v2 | 0.5807 | −0.0007 | asymmetric mix, stable |
| pact | 0.5818 | +0.0004 | stable |
| dorefa | 0.5778 | −0.0036 | linear-grid dorefa (new), stable |
| lsq_v1 / lsq_v2 / minmax | 0.3710–0.4254 | −0.156 ~ −0.210 | symmetric backends + unsigned act, expected drop |

**Multi-seed stability** (seed=1, lsqplus_v1, vs seed=42 main config):

| Config | QAT mAP50 | Δ (vs same-config seed42) |
|---|---|---|
| per_channel + act_unsigned | 0.5715 | −0.0092 |
| per_tensor + act_unsigned | 0.5913 | +0.0048 |
| per_channel + act_signed | 0.5717 | −0.0066 |
| per_tensor + act_signed | 0.5752 | −0.0124 |

Seed-to-seed variation ≤ 0.012; the conclusion direction is unchanged (lsqplus_v1 + unsigned is lossless).

**PTQ calibration-size sensitivity** (detect/lsqplus_v1, `--calib-images 5/10/50`, PTQ only): calibration size has no significant effect on lsqplus_v1 PTQ mAP on this small dataset; artifacts are written to separate directories for comparison.

**Takeaways (small-sample cross-backend comparison — not full-COCO accuracy):**
- **Recommended config (lsqplus_v1 + per_channel + act_unsigned) is lossless or near-lossless on all three tasks**: detect −0.0007, seg −0.0021, pose +0.0037;
- asymmetric backends (lsqplus_v1/v2, pact) are all stable with unsigned activations (Δ ≤ 0.039); pact even exceeds float on detect/seg (+0.011 ~ +0.013);
- **symmetric backends (lsq_v1/v2, minmax) degrade severely with unsigned activations** (Δ −0.18 ~ −0.37, PTQ collapses to 0.0000); symmetric backends must pair with act_signed. (**2026-10**: dorefa removed from this list — the new dorefa v2 uses asymmetric activations, now stable with unsigned; see v1→v2 migration notes in `docs/blog_beginner_guide_to_QAT.md` §10.4)
- weight_unsigned (negative-biased weights in a [0,1] quantization domain) collapses to 0 across the board, as theory predicts — kept as a misconfiguration example;
- QAT generally closes most of the PTQ gap, and for unsigned+asymmetric combos PTQ alone is already near float.

## Utility scripts (script/)

- `script/coco_mini_prepare.py` — build the 1/N COCO mini dataset (default 1/100, see above).
- `script/run_mini_all_part1.sh` / `script/run_mini_all_part2.sh` — two-way parallel
  mini sweep (84 main runs = 4 main configs × 3 tasks × 7 backends, split 52/32;
  part2 also carries 22 supplementary runs — weight_unsigned counterexamples,
  mixed_quant ablation, PTQ calibration-size sensitivity, multi-seed stability —
  for 54 in total).
- `script/run_mini_all.sh` — serialized all-in-one fallback (8 configs × 3 tasks × 7 backends).
- `script/run_mini_all_onebackend.sh` — legacy single-backend (lsqplus_v1) serial entry, kept as fallback.
- `script/run_mini_remain_part1.sh` / `script/run_mini_remain_part2.sh` — historical
  rerun scripts (after the clip-folding and `_anchors` device fixes, reran only
  unfinished units + all pact units), with breakpoint resume, kept for reference.
- `script/export_float_onnx.py` — export a float checkpoint to ONNX
  (onnxsim included, CPU is enough; static shape, optional `--imgsz` override):

```bash
python3 script/export_float_onnx.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth
python3 script/export_float_onnx.py --task seg    --ckpt model/yolo26-seg/n/yolo26n-seg_best.pth
python3 script/export_float_onnx.py --task pose   --ckpt model/yolo26-pose/n/yolo26n-pose_best.pth
python3 script/export_float_onnx.py --task cls    --ckpt model/yolo26-cls/n/yolo26n-cls_best.pth
python3 script/export_float_onnx.py --task obb    --ckpt model/yolo26-obb/n/yolo26n-obb_best.pth
python3 script/export_float_onnx.py --task depth  --ckpt model/yolo26-depth/n/yolo26n-depth_best.pth
# optional: override default input size (detect/seg/pose/obb/depth=640, cls=224)
python3 script/export_float_onnx.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth --imgsz 480
```

Default output: `{ckpt without .pth}_float.onnx`.

- `script/export_qat_outputs.py` — retroactively generate the full deployment
  artifact set (`quant_params.json` + `.pth` + clean float `.pth` + ONNX +
  onnxruntime verify) from any QAT / PTQ checkpoint, without re-running
  training. Auto-infers `quant_method`, `scale`, and `nc` from checkpoint meta:

```bash
# auto-infer quant_method from checkpoint meta (recommended)
python3 script/export_qat_outputs.py --task detect \
    --ckpt model/yolo26-detect/n/lsqplus_v1/qat_lsqplus_v1_yolo26n_best.pth

# or manually specify (e.g. old checkpoint missing quant_method)
python3 script/export_qat_outputs.py --task detect --ckpt <old.pth> \
    --quant-method minmax
```

- `script/visualize.py` — batch visualization on the task's **val dataset** for any
  checkpoint (float / PTQ / QAT; auto-detects model type from checkpoint meta).
  Reuses each task module's `evaluate()` (ultralytics `plot_images` under the
  hood) and writes mosaics into `{ckpt_dir}/fvisualize/`:

```bash
python3 script/visualize.py --task detect --ckpt model/yolo26-detect/n/yolo26n_best.pth
python3 script/visualize.py --task seg    --ckpt model/yolo26-seg/n/lsqplus_v1/qat_lsqplus_v1_yolo26n-seg_best.pth
python3 script/visualize.py --task depth  --ckpt model/yolo26-depth/n/yolo26n-depth_best.pth --viz-max 30
```

- `script/single_visualize.py` — inference + visualization on a **single image
  or an image directory** (no labels needed; pure prediction, annotated with
  ultralytics `Annotator`; depth renders a standalone heatmap):

```bash
# single image → {ckpt_dir}/{prefix}_{task}_result.jpg
python3 script/single_visualize.py --task detect \
    --ckpt model/yolo26-detect/n/yolo26n_best.pth \
    --image ultralytics/ultralytics/data/datasets/coco128-seg/images/train2017/000000000009.jpg

# directory batch → {out}/{image_name}_result.jpg per image
python3 script/single_visualize.py --task seg \
    --ckpt model/yolo26-seg/n/lsqplus_v1/qat_lsqplus_v1_yolo26n-seg_best.pth \
    --image ultralytics/ultralytics/data/datasets/coco128-seg/images/train2017 \
    --out results/seg_batch
```

## Project structure

```
QAT_training/
├── quantization/                # 7 backends + __init__.py selector + constants.py
├── networks_yolo26-detect.py    # YOLO26 detection (seg/pose/cls reuse its backbone/neck)
├── networks_yolo26-seg.py       # YOLO26 instance segmentation
├── networks_yolo26-pose.py      # YOLO26 pose estimation
├── networks_yolo26-cls.py       # YOLO26 classification
├── networks_yolo26-obb.py       # YOLO26 rotated detection (OBB, reuses detect backbone)
├── networks_yolo26-depth.py     # YOLO26 monocular depth estimation (reuses detect backbone)
├── networks_mobileNetv3.py      # MobileNetV3 classification
├── networks_cifarCNN.py         # CIFAR-CNN classification
├── networks_example.py          # minimal quantized-op example
├── script/                      # data prep / orchestration / ONNX export / visualization tools
├── docs/                        # design documents (quantization_design_en.md / quantization_design_CN.md)
├── requirements.txt
├── README.md                    # this file
├── README_CN.md                 # 中文版
├── dataset/                     # auto-downloaded datasets (coco8 / depth8-png / dota8-multispectral ..., not tracked)
├── datas/                       # CIFAR-10 dataset (cifarCNN / mobileNetv3 / example, not tracked)
├── model/                       # checkpoints + quant-param JSONs + ONNX (not tracked)
├── log/                         # training and sweep logs (not tracked)
├── results/                     # single-image / batch visualization output
└── ultralytics/                 # ultralytics package + yolo26n*.pt pretrained weights
```

## Deployment export notes

- **ONNX (pure-float reference graph)**: by default, exports the clean float model with
  quant params folded back AND all activation fake-quants removed (weights are baked
  quantized values; activation quantizers become identity) — no `QuantizeLinear` /
  `DequantizeLinear` nodes and no `Clip`. The ONNX carries no scale/zero_point;
  board-side NPU/TPU PTQ tools calibrate them from calibration data. To keep the
  activation clipping that QAT training relies on (reference graph numerically
  consistent with real integer inference), use `build_float_model(..., use_clip=True)`
  for a "keep-clip, drop-rounding" variant (`Clip` in the graph, canonicalized to
  `Max/Min` pairs by onnxsim). Input shape is fully static (`[1, 3, H, W]`, no
  `dynamic_axes`) so visualization tools show exact dimensions. Export includes onnxsim
  simplification and an onnx structural check.
- **JSON**: `*_quant_params.json` records scale / zero_point for every
  quantized tensor, consumed by downstream deployment toolchains (e.g. the
  Horizon model compiler); the same-named `.pth` is the binary form.
  **Important: the scale / zero_point values in the JSON are for
  cross-validation reference only — the embedded board's PTQ tool
  recalculates scale and zero_point from its own calibration data, and
  the board-side recomputed values take precedence**.
- **Manual re-export**: if a QAT / PTQ checkpoint exists but deployment
  artifacts are missing (interrupted training), use
  `script/export_qat_outputs.py` to regenerate the full set in one shot.

## Notes & lessons learned

- Train on GPU; run long jobs in a system terminal / VSCode terminal.
- PTQ quant params must initialize the QAT quantizers (calibrate after
  `copy_float_to_quant`).
- **Activation and weight signedness are two independent switches**:
  `--all-positive` applies only to activation quantizers, `--w-all-positive`
  only to weight quantizers. Post-SiLU activations are non-negative, so unsigned
  activations (act_unsigned) gain 1 effective bit and are the new default.
  Conv weights are zero-mean signed tensors — `--w-all-positive` clamps r_min to
  0, breaks sign balance, and blows activations up layer by layer until softmax
  NaN (reproduced in detect's C2PSA attention), so weights stay signed by default
  and weight_unsigned configs serve only as counterexample ablations.
- **The export reference graph removes fake-quant entirely by default; `use_clip=True`
  optionally keeps activation clipping.** Real-image diagnostics after QAT in the COCO
  mini sweep found that unsigned activations (act_unsigned) with symmetric backends
  (`lsq_v1` / `lsq_v2`) or with `minmax` trained fine (fake-quant QAT mAP valid), but the
  fully de-fake-quantized float reference graph blew up layer by layer on real images:
  pose/minmax box coordinates grew from the normal ≤640 to ~±1e5, detect/lsq_v1 to ~±9e3.
  All act_signed runs stayed fine, and `lsqplus_v1/v2` (asymmetric with learnable beta)
  remained stable with unsigned. Mechanism: the per-layer hard clip `clip(0, r_max)` lets
  QAT-trained weights rely on activation clipping; removing it compounds the error. It
  does not occur at PTQ (weights still near pretrained float). **Implication**: real
  deployment usually hands the clip-free pure-float ONNX to the board NPU/TPU PTQ tool,
  which calibrates scale/zero_point itself — for the combos above the calibration range
  covers ±1e5 outliers, so the int8 step is huge and deployed accuracy must be poor. That
  is exactly why those combos are marked not-recommended in the result tables (selection
  advice unchanged: prefer `lsqplus_v1/v2` for unsigned activations; prefer
  `--no-all-positive` for symmetric lsq / minmax). If you truly need a reference artifact
  for these combos, use `build_float_model(..., use_clip=True)` (pose/minmax/unsigned
  reference boxes settle at [2.1, 1443]). The export stage does not compare exported-ONNX
  vs PyTorch-float numerics (the ONNX has no scale/zero_point, so the comparison is not
  deployment-meaningful); quant accuracy is compared by the compare stage's QAT-vs-float
  metrics, while structural checks and quant_params completeness remain hard asserts.
- **Detection-head `_anchors` / `_strides_tensor` are non-persistent buffers**: the
  detect/seg/pose/obb heads build the anchor grid lazily on the first eval forward for
  the feature-map size. They must be registered via
  `register_buffer(..., persistent=False)` rather than plain attributes — otherwise
  `model.cpu()` in the export path does not move the GPU-created lazy tensor and the CPU
  forward hits a device mismatch; `persistent=False` keeps them out of state_dict so
  checkpoint loading is unaffected.
- Layer classes of all 7 backends (`QuantConv2d` / `QuantConvTranspose2d` /
  `QuantLinear`) share the signature `(..., all_positive=False,
  w_all_positive=False, per_channel=...)`; activation-only ops (QuantAdd/Cat/
  MaxPool etc.) take `all_positive` only (no weight concept).
- With `--mixed-quant`, the stem (model.0) and the whole task head stay FP32
  (plain nn.Conv2d); PTQ calibration and quant_params export automatically skip
  float layers via `hasattr(activation_quantizer)` / `is_weight_quant_module`.
- pose's `flow_model` (RealNVP) is used only by the training loss — never
  quantized, kept as float `nn.Linear`.
- cls keeps global average pooling as float `nn.AdaptiveAvgPool2d`.
- yolo26 seg / pose / cls / obb / depth all reuse detect's backbone/neck via
  importlib; switching the quant backend also switches the ops bound inside the
  det module, so keep the six files consistent when making changes. obb replaces
  layer 23 with an OBBDetect head (angle branch + dist2rbox decode, v8OBBLoss,
  rotated NMS, batch_probiou evaluation); its input channel count adapts to the
  dataset yaml `channels` field (10 for dota8-multispectral), and the 3-channel
  first conv of the pretrained `yolo26n-obb.pt` is skipped automatically via
  shape-mismatch filtering. depth replaces layer 23 with a Depth head
  (log-depth regression) and evaluates with delta1 / abs_rel / rmse / silog
  instead of mAP.
- Checkpoints store EMA weights (consistent with validation/deployment).

## References
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
