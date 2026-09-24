# YOLO26 / MobileNetV3 / CIFAR-CNN Quantization-Aware Training (QAT)

> 中文版：[README_CN.md](README_CN.md)

The idea is mine, and the implementation is mainly handled by Trae.

A PyTorch QAT framework providing a unified pipeline — **float training → PTQ
calibration → QAT fine-tuning → precision comparison → deployment export
(ONNX + JSON quant params)** — for:

| file | task | dataset |
|------|------|---------|
| `networks_yolo26-detect.py` | YOLO26 object detection | coco128 / COCO mini |
| `networks_yolo26-seg.py` | YOLO26 instance segmentation | coco128-seg / COCO mini |
| `networks_yolo26-pose.py` | YOLO26 pose estimation | coco8-pose / COCO mini |
| `networks_yolo26-cls.py` | YOLO26 image classification | imagenet10 |
| `networks_mobileNetv3.py` | MobileNetV3 classification | CIFAR-10 |
| `networks_cifarCNN.py` | custom CIFAR-CNN | CIFAR-10 |
| `networks_example.py` | minimal quantized-op example | CIFAR-10 (no CLI) |

## Installation

```bash
pip3 install -r requirements.txt
```

`requirements.txt` pins the versions verified in this environment (torch 2.x +
CUDA 12.6, ultralytics, onnx / onnxruntime / onnxsim). For CPU-only, switch
torch/torchvision to the `+cpu` wheels.

**Training must run on GPU (CUDA); do not run training on CPU.**

## Unified pipeline

Every `networks_*.py` runs the same pipeline:

```
float_train -> PTQ_calibration -> QAT_training -> compare_precision
```

- YOLO nets load `ultralytics/yolo26n*.pt` pretrained weights by default;
- PTQ statistics initialize the QAT quantizers;
- QAT fine-tunes with straight-through estimators (STE).

Each PTQ / QAT stage writes five artifacts (`save_quant_outputs`):

| artifact | description |
|----------|-------------|
| `model/{prefix}_{name}.pth` | quantized model weights |
| `model/{prefix}_quant_params.json` | per-tensor scale / zero_point (for deployment) |
| `model/{prefix}_quant_params.pth` | same params in binary .pth form |
| `model/{prefix}_{name}_float.pth` | clean float weights with quant params folded back |
| `model/{prefix}_{name}_float.onnx` | ONNX export of the above (opset=16) |

## Unified features (all networks_*.py)

- **best suffix**: the best checkpoint is saved as `*_best.pth` alongside
  `*_last.pth`; legacy unsuffixed float weights are still accepted by the
  compare stage (prefers `_best`, falls back otherwise).
- **ONNX is exported alongside the best .pth** (`_try_export_onnx`); export
  failures only warn and never interrupt training.
- **ONNX is always simplified with onnxsim** (skipped automatically if not installed).
- **resume**: pass `--resume` to continue training from `*_last.pth`
  (model / optimizer / scheduler / epoch are restored).
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
   detect / seg / pose passed end-to-end on 1/20 COCO mini
   (detect QAT best mAP50 = 0.5958);
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

## COCO mini dataset (1/20 sampling)

Quickly validate the full detect / seg / pose pipeline on 1/20 of COCO2017:

```bash
python3 script/coco_mini_prepare.py                 # default COCO root
python3 script/coco_mini_prepare.py --coco-root /path/to/coco --ratio 20
```

Sampling is deterministic (every 20th sorted image id, `ids[::20]`); images are
relative symlinks (no copying); iscrowd annotations are skipped; category ids
are remapped to 0..79. Output lives under `<COCO root>/mini/` with per-task
images / labels / train.txt / val.txt and the corresponding yamls.

Run all three tasks sequentially (float40 → PTQ → QAT8; serialized because they
share the GPU):

```bash
bash script/run_mini_all.sh        # logs: mini_{detect,seg,pose}.log
```

## Utility scripts (script/)

- `script/coco_mini_prepare.py` — build the 1/20 COCO mini dataset (see above).
- `script/run_mini_all.sh` — run detect / seg / pose mini pipelines sequentially.
- `script/run_mini_pipeline.py` — Python orchestrator for the mini GPU pipeline.
- `script/export_float_onnx.py` — export a float checkpoint to ONNX
  (onnxsim included, CPU is enough):

```bash
python3 script/export_float_onnx.py --task detect --ckpt model/yolo26n_best.pth
python3 script/export_float_onnx.py --task seg    --ckpt model/yolo26n-seg_best.pth
python3 script/export_float_onnx.py --task pose   --ckpt model/yolo26n-pose_best.pth
python3 script/export_float_onnx.py --task cls    --ckpt model/yolo26n-cls_best.pth
```

Default output: `{ckpt without .pth}_float.onnx`.

## Project structure

```
QAT_training/
├── quantization/                # 7 backends + __init__.py selector + constants.py
├── networks_yolo26-detect.py    # YOLO26 detection (seg/pose/cls reuse its backbone/neck)
├── networks_yolo26-seg.py       # YOLO26 instance segmentation
├── networks_yolo26-pose.py      # YOLO26 pose estimation
├── networks_yolo26-cls.py       # YOLO26 classification
├── networks_mobileNetv3.py      # MobileNetV3 classification
├── networks_cifarCNN.py         # CIFAR-CNN classification
├── networks_example.py          # minimal quantized-op example
├── script/                      # data prep / orchestration / ONNX export tools
├── requirements.txt
├── README.md                    # this file
├── README_CN.md                 # 中文版
├── datas/                       # datasets (auto-downloaded, not tracked)
├── model/                       # checkpoints + quant-param JSONs + ONNX (not tracked)
└── ultralytics/                 # ultralytics package + yolo26n*.pt pretrained weights
```

## Deployment export notes

- **ONNX**: exports the clean float model with quant params folded back — no
  `QuantizeLinear` / `DequantizeLinear` nodes. Export includes onnxsim
  simplification and an onnxruntime-vs-PyTorch numeric check (1e-4 for
  classification; a magnitude-relative threshold for detection).
- **JSON**: `*_quant_params.json` records scale / zero_point for every
  quantized tensor, consumed by downstream deployment toolchains (e.g. the
  Horizon model compiler); the same-named `.pth` is the binary form.

## Notes & lessons learned

- Train on GPU; run long jobs in a system terminal / VSCode terminal.
- PTQ quant params must initialize the QAT quantizers (calibrate after
  `copy_float_to_quant`).
- pose's `flow_model` (RealNVP) is used only by the training loss — never
  quantized, kept as float `nn.Linear`.
- cls keeps global average pooling as float `nn.AdaptiveAvgPool2d`.
- yolo26 seg / pose / cls reuse detect's backbone/neck via importlib; switching
  the quant backend also switches the ops bound inside the det module, so keep
  the four files consistent when making changes.
- Checkpoints store EMA weights (consistent with validation/deployment).

## References

- [mqbench](https://github.com/modeltc/mqbench) — QAT / PTQ reference
- [ultralytics](https://ultralytics.com/) — YOLO26 pretrained weights
- [horizon PTQ/QAT deployment guide](https://doc.oe.horizon.auto/3.8.1/guide/model_compile.html)
- [horizon developer portal](https://developer.horizon.auto/)
- [Trae](https://www.trae.cn/)

## Credits

The idea is mine, and the implementation is mainly handled by Trae.
If there are issues with Trae's implementation, I will point them out and request fixes.
