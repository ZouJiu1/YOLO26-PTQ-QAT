# YOLO26 / MobileNetV3 / CIFAR-CNN Quantization-Aware Training (QAT)

> 中文版：[README_CN.md](README_CN.md)

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
| `--float-batch-size` / `--ptq-batch-size` / `--qat-batch-size` | 16 (yolo) / 128 (cls) | per-stage batch sizes |
| `--calibration-batches` | 20 | PTQ calibration batches |
| `--data` | — | override dataset yaml (yolo26 nets) |
| `--resume` | False | resume training from `*_last.pth` |
| `--max-train-batches` / `--max-eval-batches` | None | limit batches (smoke test) |

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

The sweep trains **float once per task (50 epochs)**, then loops over all
backends (lsqplus_v1 / lsqplus_v2 / lsq_v1 / lsq_v2 / minmax / dorefa / pact),
running PTQ (20 calibration batches) → QAT (10 epochs = 50×0.2) → float/QAT
compare for each. Serialized because they share the GPU; a failed stage does
not block the remaining backends:

```bash
bash script/run_mini_all.sh        # logs: log/mini_{task}_float.log, log/mini_{task}_{backend}_{ptq,qat,compare}.log
```

### Cross-backend results (1/100 COCO mini, float 50ep / QAT 10ep, batch=8)

> The val set has only 50 images (27 for pose), so mAP/P/R are small-sample
> numbers meant for cross-backend comparison and regression checks, not full-COCO
> accuracy. Produced automatically by `script/run_mini_all.sh`.

<!-- RESULTS_TABLE_PLACEHOLDER -->
**detect (50 val images, 80 classes)** — Float baseline (best epoch 4/50): P 0.7229 / R 0.5004 / mAP50 **0.5814** / mAP50-95 **0.4321**

| Backend | PTQ mAP50 | QAT P | QAT R | QAT mAP50 | QAT mAP50-95 | ΔmAP50 (QAT−Float) |
|---|---|---|---|---|---|---|
| lsqplus_v1 | 0.5663 | 0.7692 | 0.4757 | 0.5792 | 0.4201 | −0.0022 |
| lsqplus_v2 | 0.5663 | 0.7338 | 0.5119 | 0.5707 | 0.4183 | −0.0107 |
| lsq_v1 | 0.5701 | 0.7740 | 0.4787 | **0.5809** | 0.4190 | −0.0005 |
| lsq_v2 | 0.5859 | 0.6979 | 0.5326 | 0.5649 | 0.4197 | −0.0165 |
| minmax | 0.5557 | 0.8034 | 0.4678 | 0.5725 | 0.4133 | −0.0089 |
| dorefa | 0.5750 | 0.7047 | 0.5215 | 0.5774 | **0.4203** | −0.0040 |
| pact | 0.5655 | 0.7799 | 0.4811 | **0.5920** | 0.4204 | **+0.0106** |

**seg (50 val images, 80 classes; task metric = mask mAP)** — Float baseline (best epoch 5/50): box mAP50 0.5362; mask P 0.6397 / R 0.4663 / mAP50 **0.5001** / mAP50-95 **0.3156**

| Backend | PTQ mask mAP50 | QAT box mAP50 | QAT mask P | QAT mask R | QAT mask mAP50 | QAT mask mAP50-95 | ΔmAP50 |
|---|---|---|---|---|---|---|---|
| lsqplus_v1 | 0.4715 | 0.5352 | 0.6939 | 0.4119 | 0.4966 | 0.3139 | −0.0035 |
| lsqplus_v2 | 0.4715 | 0.5252 | 0.6850 | 0.4476 | 0.4902 | **0.3192** | −0.0099 |
| lsq_v1 | 0.4684 | 0.5356 | 0.6752 | 0.4492 | **0.5066** | 0.3133 | **+0.0065** |
| lsq_v2 | 0.4332 | 0.5218 | 0.6839 | 0.4451 | 0.4833 | 0.2976 | −0.0168 |
| minmax | **0.5049** | **0.5430** | 0.6211 | 0.4450 | 0.5050 | 0.3172 | +0.0049 |
| dorefa ² | 0.4601 | 0.5385 | 0.6170 | 0.4686 | 0.4941 | 0.3101 | −0.0060 |
| pact | 0.4890 | 0.5325 | 0.6460 | 0.4448 | 0.4990 | 0.3151 | −0.0011 |

**pose (27 val images, single person class; task metric = keypoint pose mAP)** — Float baseline (best epoch 1/50): box mAP50 0.6178; pose P 0.7843 / R 0.4565 / mAP50 **0.4839** / mAP50-95 **0.3263**

| Backend | PTQ pose mAP50 | QAT box mAP50 | QAT pose P | QAT pose R | QAT pose mAP50 | QAT pose mAP50-95 | ΔmAP50 |
|---|---|---|---|---|---|---|---|
| lsqplus_v1 | 0.4823 | 0.6164 | 0.7895 | 0.4565 | 0.4844 | 0.3219 | +0.0005 |
| lsqplus_v2 | 0.4823 | **0.6291** | 0.7446 | 0.4565 | 0.4868 | 0.3250 | +0.0029 |
| lsq_v1 | 0.4724 | 0.6265 | 0.8270 | 0.4638 | 0.4791 | 0.2995 | −0.0048 |
| lsq_v2 | 0.4540 | 0.6102 | 0.8012 | 0.4565 | 0.4863 | 0.2994 | +0.0024 |
| minmax | **0.4934** | 0.6147 | 0.7849 | 0.4494 | 0.4793 | 0.3181 | −0.0046 |
| dorefa | 0.4629 | 0.6263 | **0.8409** | 0.4565 | 0.4837 | 0.3030 | −0.0002 |
| pact | 0.4891 | 0.6234 | 0.7788 | 0.4593 | **0.4869** | 0.3225 | **+0.0030** |

² The seg dorefa QAT OOMs at batch=8 on the 8 GB GPU (tanh quantizer memory); this single cell trained QAT at batch=4
(PTQ stayed at batch=8), and compare ran at batch=8 to keep the float baseline consistent.
All other 20 cells used batch=8. Full per-stage logs: `log/mini_{task}_float.log` and
`log/mini_{task}_{backend}_{ptq,qat,compare}.log`.

**Takeaways (small-sample cross-backend comparison — not full-COCO accuracy):**
- QAT mAP50 loss vs float stays within 0.017 for all 7 backends; PTQ alone recovers 90%+ of float mAP, and QAT typically closes most of the remaining gap.
- Best QAT mAP50 per task: detect → pact (0.5920, +0.0106 over float), seg → lsq_v1 (0.5066), pose → pact (0.4869).
- minmax has no trainable parameters and usually gives the strongest PTQ accuracy (seg mask 0.5049, pose 0.4934) — the cheapest option.
- lsq_v2 / lsqplus_v2 are weaker on pose mAP50-95 (stricter IoU, 0.299); dorefa has the largest memory footprint on seg.

## Utility scripts (script/)

- `script/coco_mini_prepare.py` — build the 1/N COCO mini dataset (default 1/100, see above).
- `script/run_mini_all.sh` — run the 3 tasks × 7 backends mini sweep (one float run + PTQ/QAT/compare per backend).
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
├── requirements.txt
├── README.md                    # this file
├── README_CN.md                 # 中文版
├── dataset/                     # auto-downloaded datasets (coco8 / depth8-png / dota8-multispectral ..., not tracked)
├── model/                       # checkpoints + quant-param JSONs + ONNX (not tracked)
└── ultralytics/                 # ultralytics package + yolo26n*.pt pretrained weights
```

## Deployment export notes

- **ONNX**: exports the clean float model with quant params folded back — no
  `QuantizeLinear` / `DequantizeLinear` nodes. Input shape is fully static
  (`[1, 3, H, W]`, no `dynamic_axes`) so visualization tools show exact
  dimensions. Export includes onnxsim simplification and an onnxruntime-vs-PyTorch
  numeric check (1e-4 for classification; a magnitude-relative threshold for
  detection).
- **JSON**: `*_quant_params.json` records scale / zero_point for every
  quantized tensor, consumed by downstream deployment toolchains (e.g. the
  Horizon model compiler); the same-named `.pth` is the binary form.
- **Manual re-export**: if a QAT / PTQ checkpoint exists but deployment
  artifacts are missing (interrupted training), use
  `script/export_qat_outputs.py` to regenerate the full set in one shot.

## Notes & lessons learned

- Train on GPU; run long jobs in a system terminal / VSCode terminal.
- PTQ quant params must initialize the QAT quantizers (calibrate after
  `copy_float_to_quant`).
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
