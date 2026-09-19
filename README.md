# YOLO26 float Train + PTQ + QAT

PyTorch quantization-aware training (QAT) framework for **YOLO26n** (detect / seg / pose / cls),
**MobileNetV3**, **CIFAR-CNN**, and a minimal **example** network.

Every `networks_*.py` script runs the same end-to-end pipeline:

```
float_train  ->  PTQ_calibration  ->  QAT_training  ->  compare_precision
   (浮点训练)      (训练后量化校准)     (量化感知训练)       (浮点 vs QAT 精度对比)
```

Seven quantization backends are supported and can be switched with `--quant`:

| `--quant`   | backend module                  | description                                         |
|-------------|---------------------------------|-----------------------------------------------------|
| `lsqplus_v1`| `quantization.lsqplus_quantize_V1` | LSQ+ (learnable scale + beta, V1)               |
| `lsqplus_v2`| `quantization.lsqplus_quantize_V2` | LSQ+ (learnable scale + beta, V2)               |
| `lsq_v1`    | `quantization.lsqquantize_V1`   | LSQ (learnable scale, symmetric)                    |
| `lsq_v2`    | `quantization.lsqquantize_V2`   | LSQ (learnable scale, symmetric, V2)                |
| `minmax`    | `quantization.minmax`           | MinMax (running min/max, standard asymmetric)       |
| `dorefa`    | `quantization.dorefa`           | DoReFa (tanh weight grid + learnable activation scale) |
| `pact`      | `quantization.pact`             | PACT (learnable clip threshold alpha)               |

Each backend independently implements the same set of quantized operators:
`QuantConv2d`, `QuantConvTranspose2d`, `QuantLinear`, `QuantAdd`, `QuantSub`,
`QuantMultiply`, `QuantDiv`, `QuantConcat`, `QuantMaxPool`, `QuantCat`.
`quantization/__init__.py` provides backend loading (`load_quant_backend`),
quantizer freeze/reset, and cross-backend `scale` / `zero_point` extraction.

## Network files

| file | task | dataset | default `--quant` |
|------|------|---------|-------------------|
| `networks_yolov26n-detect.py`  | object detection   | coco128         | `lsqplus_v1` |
| `networks_yolov26n-seg.py`     | instance segmentation | coco128-seg  | `lsqplus_v1` |
| `networks_yolov26n-pose.py`    | pose estimation    | coco8-pose      | `lsqplus_v1` |
| `networks_yolov26n-cls.py`     | image classification | imagenet10   | `lsqplus_v1` |
| `networks_mobileNetv3.py`      | image classification | CIFAR-10     | `lsqplus_v1` |
| `networks_cifarCNN.py`         | image classification | CIFAR-10     | `minmax` |
| `networks_example.py`          | example (no CLI)   | CIFAR-10       | `lsqplus_v1` (hardcoded) |

> YOLO26n detect / seg / pose / cls load pretrained weights from `ultralytics/yolo26n*.pt`
> and reuse ultralytics' data pipeline, loss, NMS, and mAP metrics.
> The `cls` head differs from the detect backbone (no SPPF); seg/pose reuse the detect
> backbone/neck via `importlib` and append their own heads.

## Installation

```bash
pip3 install -r requirements.txt
```

`requirements.txt` pins the versions verified in this environment
(torch 2.11 / torchvision 0.26 with CUDA 12.6, ultralytics 8.4.155, numpy 2.4.4).
For a CPU-only setup, replace the `+cu126` suffixes with `+cpu` and drop the
`--extra-index-url` line.

## Usage

All `networks_*.py` scripts (except `networks_example.py`) share the same CLI:

```bash
python3 networks_yolov26n-detect.py --quant lsqplus_v1 --stage all
```

### Common arguments

| argument | type | default | description |
|----------|------|---------|-------------|
| `--quant` | str | per file | quantization method, one of `lsqplus_v1 lsqplus_v2 lsq_v1 lsq_v2 minmax dorefa pact` |
| `--stage` | str | `all` | pipeline stage to run: `all float ptq qat compare` |
| `--float-batch-size` | int | 16 (yolo) / 128 (cls) | float-training batch size |
| `--qat-batch-size`   | int | 16 (yolo) / 128 (cls) | QAT batch size |
| `--ptq-batch-size`   | int | 8 (yolo) / 128 (cls) | PTQ calibration batch size |
| `--batch-size`       | int | 128 | (cls networks only) single batch size for all stages |
| `--num-workers`      | int | 2 | dataloader workers |
| `--float-epochs`     | int | 100 | float-training epochs |
| `--qat-epochs`       | int | `max(1, float_epochs*0.2)` | QAT epochs |
| `--float-lr`         | float | 1e-3 (cls) / auto (yolo) | float-training learning rate |
| `--qat-lr`           | float | `float_lr * 0.1` | QAT learning rate |
| `--calibration-batches` | int | 20 | number of batches for PTQ calibration |
| `--max-train-batches` | int | None | limit batches per epoch (smoke test) |
| `--max-eval-batches`  | int | None | limit batches per evaluation (smoke test) |

### Stage semantics

- `float`  — train the float model (loads `yolo26n*.pt` pretrained weights for YOLO nets).
- `ptq`    — run PTQ calibration on `--calibration-batches` batches and save the
             calibrated quantized checkpoint.
- `qat`    — fine-tune the quantized model with straight-through estimators.
- `compare`— evaluate float vs. QAT models and print the accuracy/mAP delta.
- `all`    — run `float -> ptq -> qat -> compare` in sequence.

### Examples

```bash
# Full pipeline, default backend
python3 networks_yolov26n-detect.py

# Switch to minmax quantization, run only QAT + compare
python3 networks_cifarCNN.py --quant minmax --stage qat
python3 networks_cifarCNN.py --quant minmax --stage compare

# Quick smoke test (1 train batch, 1 eval batch per epoch)
python3 networks_mobileNetv3.py --quant lsqplus_v1 \
    --float-epochs 1 --max-train-batches 1 --max-eval-batches 1
```

## Project structure

```
QAT_training/
├── quantization/
│   ├── __init__.py              # backend selector + freeze/reset + scale/zp extraction
│   ├── constants.py             # INIT_STATE_* sentinels
│   ├── lsqplus_quantize_V1.py   # LSQ+ backend (V1)
│   ├── lsqplus_quantize_V2.py   # LSQ+ backend (V2)
│   ├── lsqquantize_V1.py        # LSQ backend (V1)
│   ├── lsqquantize_V2.py        # LSQ backend (V2)
│   ├── minmax.py                # MinMax backend
│   ├── dorefa.py                # DoReFa backend
│   └── pact.py                  # PACT backend
├── networks_yolov26n-detect.py  # YOLO26n detection pipeline
├── networks_yolov26n-seg.py     # YOLO26n segmentation pipeline
├── networks_yolov26n-pose.py    # YOLO26n pose pipeline
├── networks_yolov26n-cls.py     # YOLO26n classification pipeline
├── networks_mobileNetv3.py      # MobileNetV3 classification pipeline
├── networks_cifarCNN.py         # CIFAR-CNN classification pipeline
├── networks_example.py          # minimal quantized-op example (no CLI)
├── requirements.txt
└── README.md
```

Expected runtime directories (created on first run, not tracked by git):

```
QAT_training/
├── datas/            # CIFAR-10 / ImageNet / COCO data (downloaded automatically)
├── model/            # float / PTQ / QAT checkpoints (*.pth) + quant params (*.json)
└── ultralytics/      # ultralytics package + yolo26n*.pt pretrained weights
```

## Quantization basics

To quantize an op like `conv`, `linear`, `add`, `concat`, etc., wrap it with the
corresponding `Quant*` class from the active backend. Each weight-quant layer
(`QuantConv2d`, `QuantConvTranspose2d`, `QuantLinear`) carries a `weight_quantizer`,
and element-wise ops (`QuantAdd`, `QuantSub`, ...) quantize their inputs before
computing. Example:

```python
from quantization.lsqplus_quantize_V1 import (
    QuantConv2d, QuantLinear, QuantAdd, QuantConcat,
)

# QuantConv2d has the same signature as nn.Conv2d
conv = QuantConv2d(3, 16, kernel_size=3, padding=1, a_bits=8, w_bits=8)

# QuantAdd / QuantConcat quantize each input path
add = QuantAdd(a_bits=8, all_positive=False)
out = add(x, y)
```

After PTQ calibration or QAT, `quantization.collect_quant_params(model)` extracts
per-layer `scale` / `zero_point` for both weights and activations across all seven
backends, so the model can be exported for deployment.

## References

- [mqbench](https://github.com/modeltc/mqbench) — QAT / PTQ reference

- [ultralytics](https://ultralytics.com/) — YOLO26n pretrained weights

- [horizon PTQ QAT Deployment guide](https://doc.oe.horizon.auto/3.8.1/guide/model_compile.html)

- [Trae](https://www.trae.cn/)

- [developer portal horizon](https://developer.horizon.auto/)

## Credits

The idea is mine, and the implementation is mainly handled by Trae.
If there are issues with Trae's implementation, I will point them out and request fixes.
