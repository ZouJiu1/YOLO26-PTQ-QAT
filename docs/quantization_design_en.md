# YOLO26 Full-Operator Quantization Training Framework — Complete Design

> 中文版：[quantization_design_CN.md](quantization_design_CN.md)

> Scope: YOLO26 detect / segment / pose / classify / OBB / depth (6 tasks),
> full float → PTQ → QAT → compare pipeline, targeting real deployment on
> embedded NPUs / TPUs / Horizon-style accelerators.
>
> Core idea in one sentence: **quantize on the training side exactly what the
> on-device runtime executes** — quantization covers nearly every operator,
> not just conv + linear + relu.
>
> Written from a full read-through of all 8 files under `quantization/` and the
> `networks_yolo26-*.py` network scripts.

---

## Table of contents

1. [Why quantize nearly every operator](#1-why-quantize-nearly-every-operator)
2. [The full quantized-operator set and its classification](#2-the-full-quantized-operator-set-and-its-classification)
3. [Overall architecture: three layers](#3-overall-architecture-three-layers)
4. [The 7 quantization backends and quantizer math](#4-the-7-quantization-backends-and-quantizer-math)
5. [Exact forward behavior of activation-quantized ops](#5-exact-forward-behavior-of-activation-quantized-ops)
6. [PTQ and QAT lifecycle (freeze/reset state machine)](#6-ptq-and-qat-lifecycle-freezereset-state-machine)
7. [Deployment artifacts: the three outputs and clip folding](#7-deployment-artifacts-the-three-outputs-and-clip-folding)
8. [unsigned/signed and per_channel/per_tensor design](#8-unsignedsigned-and-per_channelper_tensor-design)
9. [Network-script integration](#9-network-script-integration)
10. [Key engineering decisions and pitfalls](#10-key-engineering-decisions-and-pitfalls)
11. [Experimental conclusions](#11-experimental-conclusions)
12. [Limitations and future work](#12-limitations-and-future-work)

---

## 1. Why quantize nearly every operator

### 1.1 Textbook vs. real on-device execution

Many introductory quantization examples only quantize `conv + linear + relu`.
That is enough for PTQ/QAT research on GPU, but it **diverges sharply from how
embedded NPUs/TPUs actually run**:

| Aspect | Textbook approach | On-device reality |
|---|---|---|
| Operator scope | conv/linear quantized, everything else FP32 | **every** op on the graph has int input/output scales |
| Element-wise | add/mul/sub/div computed in FP32, re-quantized | computed directly in int8 (requant fused into hardware) |
| concat | FP32 concatenation | each input branch carries its own scale; output uses a unified scale |
| activations | silu/sigmoid/softmax in FP32 | LUT (look-up table) quantized implementation, int in / int out |

If the training side only quantizes conv+linear, the quantization error of
add/concat/silu etc. on device is **something the model never saw during
training** — the single most common root cause of "great in training, broken on
the board".

### 1.2 This project's approach

Insert fake-quant at **every** operator position that introduces quantization
error, so the QAT loss fully perceives on-device noise:

- Quantize both the **weights** and the **input activations** of conv/linear
- Quantize both inputs and the output of residual `add`, shortcut `sub`,
  element-wise `mul`/`div`
- Quantize each `concat` input branch independently (each branch already has its
  own scale on device)
- Quantize the inputs of non-linearities like `silu`/`sigmoid`/`softmax`
  (reproduce the on-device LUT clipping)
- Quantize `maxpool` output

The cost is implementation complexity (each backend implements a full set of
~15 operator classes), but this is the only correct path to deployment fidelity.

---

## 2. The full quantized-operator set and its classification

**Key fact: every one of the 7 backends independently implements the same full
set of ~15 operator classes in its own module file** (no shared base class, no
cross-file reuse — only PACT's weight quantizer is a pluggable factory). The
operators fall into three categories:

### 2.1 Weight-carrying ops

| Operator | Description | What is quantized |
|---|---|---|
| `QuantConv2d` | 2D convolution | input activation + weight (per_channel/per_tensor) |
| `QuantConvTranspose2d` | transposed conv (seg proto head upsampling) | same |
| `QuantLinear` | fully-connected | same |
| `QuantMatMul` | matrix multiply (attention/head) | input activation ×2 |

Duck-typing marker: `hasattr(module, "weight_quantizer")`
([quantization/__init__.py](../quantization/__init__.py) `is_weight_quant_module`).
During training the weight is quantized every forward pass to produce gradients;
at inference the baked quantized weight lives in `.weight` and the weight
quantizer is short-circuited by `quant_inference=True`.

### 2.2 Activation-only ops

No weights, only activation quantizers, only the `all_positive` flag (no weight
concept):

| Operator | Typical location | Quantization points |
|---|---|---|
| `QuantAdd` / `QuantSub` | residual/shortcut, feature diff | both inputs (one quantizer each) + output |
| `QuantMultiply` / `QuantDiv` | gating/attention, normalization | same; `QuantDiv` denominator is `clamp(min=1e-6)` to avoid 0/0 |
| `QuantConcat` | FPN two-way concat | both inputs via `activation_quantizer0/1` |
| `QuantCat` | FPN multi-scale concat | **one quantizer per input branch** (`self.quantizers = nn.ModuleList`) |
| `QuantMaxPool` | downsampling | input |
| `QuantSiLU` / `QuantSigmoid` / `QuantReLU` / `QuantSoftmax` | activation/normalization | input |

**Why each concat branch gets its own quantizer**: on device each input tensor
carries its own scale; the concat kernel moves data per-branch scale and unifies
the output. `QuantCat` reproduces this exactly with a `ModuleList` of per-branch
quantizers.

**Why two-input ops quantize each input**: in the int domain, two inputs must be
aligned to a common scale (requant), and that requant is itself a noise source.
Quantizing the inputs during training lets QAT learn weights robust to it.

### 2.3 The quantizers themselves

Low-level modules separate from the operators that actually perform fake-quant
(`round + clamp + dequant`):

- **Activation quantizers**: `*ActivationQuantizer`, holding learnable
  scale / beta / alpha, signed/unsigned controlled by `all_positive`
- **Weight quantizers**: `*WeightQuantizer`, controlled by `w_all_positive` /
  `per_channel`

---

## 3. Overall architecture: three layers

```
┌─────────────────────────────────────────────────┐
│ Network scripts networks_yolo26-{detect,seg,...} │
│  · set_quant_method()  binds backend ops as names│
│  · build the net directly with QuantConv2d/...   │
│  · float/ptq/qat/compare stage CLIs              │
└──────────────────┬──────────────────────────────┘
                   │ quant_pkg.load_quant_backend(method)
┌──────────────────▼──────────────────────────────┐
│ quantization/__init__.py (cross-backend common)  │
│  · _BACKEND_MODULES / QUANT_METHODS             │
│  · freeze_batch_init / reset_quantizer_states   │
│  · collect_quant_params (export scale/zp set)   │
│  · build_float_model (clean float graph + clip) │
└──────────────────┬──────────────────────────────┘
                   │ importlib dynamic loading
┌──────────────────▼──────────────────────────────┐
│ 7 backend modules quantization/{backend}.py      │
│  · each file independently implements ~15 ops    │
│  · each file's Activation/Weight quantizers      │
└─────────────────────────────────────────────────┘
```

Key points:
- **Pluggable backends**: `load_quant_backend("lsqplus_v1")` uses `importlib` to
  return the module; the operator attribute names the network script sees are
  identical across backends → switching backend is a one-string change
- **Centralized common logic**: quantizer state management, parameter export, and
  float-graph construction live in `__init__.py` so 7 backends don't re-implement
- **Duck typing over inheritance**: roles are detected via
  `hasattr(m, "weight_quantizer")` / `hasattr(m, "clip_bounds")`. The 7 backends
  each implement the full op set independently (**no** shared base class); duck
  typing provides the minimal cross-backend contract

---

## 4. The 7 quantization backends and quantizer math

| Backend | File | Activation symmetry | Quantizers | Recommended for |
|---|---|---|---|---|
| **lsqplus_v1** | `lsqplus_quantize_V1.py` | **asymmetric** (learns s+β) | `LSQPlusActivationQuantizer` / `LSQPlusWeightQuantizer` | **deployment default** |
| lsqplus_v2 | `lsqplus_quantize_V2.py` | asymmetric | same structure | ablation |
| lsq_v1 | `lsqquantize_V1.py` | symmetric | `LSQActivationQuantizer` / `LSQWeightQuantizer` | lightweight, symmetric HW |
| lsq_v2 | `lsqquantize_V2.py` | symmetric | same (activation s init = const 1) | ablation |
| minmax | `minmax.py` | asymmetric | `MinMaxActivationQuantizer` / `MinMaxWeightQuantizer` | baseline |
| dorefa | `dorefa.py` | symmetric | `DorefaActivationQuantizer` / `DorefaWeightQuantizer` | research comparison |
| pact | `pact.py` | asymmetric (learns clip α) | `PactActivationQuantizer` + `build_weight_quantizer()` | comparison / partial deploy |

### 4.1 Quantizer forward math differences

- **dorefa**: the activation quantizer learns a scale `s`, quantizes onto a fixed
  grid (signed `[-1,1]` / unsigned `[0,1]`), `all_positive` selects the range;
  the weight uses the `tanh(weight)` domain for magnitude statistics,
  `per_channel` per output channel.
- **lsq_v1/v2**: learns scale `s`; `all_positive` sets `Qn/Qp` (negative/positive
  levels); the first 20 batches do scale init/smoothing (`batch_init` state
  machine), then fixed. Classic LSQ STE.
- **lsqplus_v1/v2**: learns both `s` and zero-point offset `beta` (initialized to
  `-1e-9`), executes `ALSQPlus`/`WLSQPlus` asymmetric quantization; weight
  initializes `s` from mean/std.
- **minmax**: standard asymmetric `q = clamp(round(x/scale)+zp, qmin, qmax)`;
  supports `percent/cluster` outlier collection; `all_positive` truncates
  `r_min` to 0. Gradient passes through inside the range, zeroed outside.
- **pact**: learns a clipping threshold `alpha` (`PactActivationQuantizer`);
  `all_positive` selects unsigned/signed. The weight quantizer is dispatched by
  the `build_weight_quantizer(method, ...)` factory over
  `dorefa/minmax/lsqplus_v1/lsqplus_v2` (default `lsqplus_v1`).

### 4.2 Unified interface (`__init__.py`)

`load_quant_backend(method)` looks up `_BACKEND_MODULES` and returns the module;
`_WEIGHT_OP_NAMES` (weight-carrying op class names) and `_ELTWISE_OP_NAMES`
(activation op class names) are enumerated so `weight_quant_ops(backend)` /
`all_quant_ops(backend)` collect the operator class tuples for freeze/reset/export
traversal.

---

## 5. Exact forward behavior of activation-quantized ops

The `quant_inference` flag's universal semantics (every operator class obeys):
- **`quant_inference=False` (training)**: perform fake-quant/dequant (produces
  gradients), then the operation
- **`quant_inference=True` (inference)**: skip quantization, run the raw float op

> Note: this flag means "is quantization noise active". Training uses `False` =
> quantized; float-graph export sets `True` = remove quantization (together with
> §7 clip folding, the activation quantizer is replaced entirely). A few ops like
> `QuantCat` hardcode `quant_inference=True` (multi-input concat always
> quantizes each branch).

Under the quantized path (`False`), the per-op quantization points are listed in
the §2.2 table: two-input ops quantize each input, single-input activations
quantize only the input, `QuantDiv` clamps the denominator against 0/0. The PACT
backend additionally makes its `QuantSiLU/Sigmoid/ReLU/Softmax/MatMul` quantizers
PACT-style (learning α).

---

## 6. PTQ and QAT lifecycle

### 6.1 Three stages

| Stage | What happens | Quantizer state |
|---|---|---|
| float | full-precision training, produce baseline best.pth | inactive |
| PTQ | load float weights, forward a few calibration images, **only calibrate quantizer params**, no weight training | calibrate → freeze |
| QAT | load PTQ result, **continue training weights under quantization noise** (STE backward) | weights trainable, quantizers backend-dependent |

### 6.2 freeze / reset state machine (`quantization/__init__.py`)

- `freeze_batch_init(model)`: walk all quantizers, mark the current scale/zp as
  frozen (the `INIT_STATE_FROZEN` sentinel in `quantization/constants.py`),
  locking them after PTQ calibration to prevent early-QAT drift. The lsq family's
  `batch_init` state machine also relies on it to stop statistics
- `reset_quantizer_states(model)`: unfreeze / reset to batch 0 so learnable
  quantizers (lsqplus/pact) keep fine-tuning during QAT

### 6.3 `quant_inference` and the deployment inference path

Training (`False`) conv forward runs `weight_quantizer(self.weight)` to quantize
the weight live and produce gradients; inference export (`True`) uses the baked
quantized weight already stored in `.weight`, short-circuiting the weight
quantizer. This flag is the key that lets `build_float_model` safely construct a
clean float graph (see §7.3).

---

## 7. Deployment artifacts: the three outputs and clip folding

### 7.1 Three outputs

| Artifact | Consumer | Content |
|---|---|---|
| `*_quant_params.json/.pth` | toolchain compiler | per-layer activation/weight scale + zero_point |
| `*_float.onnx` | compiler input | clean float graph (no Q/DQ nodes), weights are post-quantization values |
| quant `.pth` | training/re-export | fake-quant trained model checkpoint |

Export order: `collect_quant_params` saves the JSON first, then
`build_float_model` produces the ONNX. Both come from the same quantizer state in
one save, so they are consistent by construction.

### 7.2 `collect_quant_params` (`__init__.py`)

Walk the quantized model and, for each quantizer, extract scale/zero_point with
`activation_scale_zp` / `weight_scale_zp` (internally handling per-backend
differences: lsqplus β, minmax zp, pact α, dorefa s), packing into
`{layer: {scale, zero_point}}`. `required_quant_keys` validates export
completeness (hard assert).

### 7.3 `build_float_model` and clip folding

**Background**: with `act_unsigned`, the activation quantizer is essentially
"clip to `[0, 255·s]`, then round". QAT makes the weights **depend on that hard
clipping**. Early versions removed the whole activation quantizer when exporting
the float graph (removing the clip too), causing unsigned+symmetric-backend float
ONNX outputs to explode layer by layer (±1e5, where ≤640 is normal).

**Fix (clip folding)**: when exporting the float graph, the activation quantizer
is not deleted but replaced by `_ClipOnlyQuantizer(lower, upper)` — **keep the
clamping, drop the rounding**. Each backend's activation quantizer implements
`clip_bounds()` returning the activation-space bounds:

- unsigned symmetric backends: `(0, 255·s)`
- lsqplus (with β): `(β, β+255·s)`
- signed symmetric: `(-128·s, 127·s)`
- pact: `(0, α)` / `(-α, α)`

Implementation points (`__init__.py` L224-306):
1. `deepcopy(quant_model)` → `model.cpu()` → `freeze_batch_init` / `eval`
2. Set `quant_inference=True` on all modules (short-circuit)
3. For weight-carrying layers, `.weight.copy_(weight_quantizer(weight))` to bake
   the quantized weight, then swap the weight quantizer for `_IdentityQuantizer`
4. Recursive `_replace_activation_quantizers` uses duck typing
   `hasattr(child,'clip_bounds')` to swap every activation quantizer for
   `_ClipOnlyQuantizer`. **Must recurse** — `QuantCat` stores quantizers in
   `self.quantizers = nn.ModuleList` (not an `activation_quantizer*` attribute)
5. `_ClipOnlyQuantizer` stores bounds via `register_buffer` (its forward **must
   not use `.to()`** — tracing produces `aten::copy_` and breaks ONNX export),
   and bounds are `.detach()`-ed

**Default `use_clip=False`**: in real deployment the on-device PTQ tool computes
scale/zp itself, so the float ONNX usually **does not need** the clip (the int
domain clips in hardware anyway). `use_clip=True` is only for generating a
reference graph exactly equivalent to QAT for verification.

### 7.4 The correct compare-stage protocol

QAT vs Float accuracy is compared **only via the two checkpoints' val-set
metrics**, never via "exported ONNX vs float outputs" — the latter is
meaningless because the clean float ONNX has no scale/zp; the on-device PTQ tool
computes those itself. This project's compare only reports Float mAP vs QAT mAP
(`[Compare] Float/QAT/delta`).

---

## 8. unsigned/signed and per_channel/per_tensor design

### 8.1 Unsigned activations (`all_positive=True`, default)

The YOLO backbone uses **SiLU/ReLU** heavily, so activations are naturally ≥ 0.
Unsigned ([0,255]) uses the full positive range better than signed
([-128,127]) — the key to near-lossless int8.

- **Asymmetric backends** (lsqplus_v1/v2, pact, minmax): natively supported,
  stable and lossless
- **Symmetric backends** (lsq_v1/v2, dorefa): the zero point is forced outside
  the bound, unsigned activations get severely clipped → **0.18–0.37 drop, PTQ
  collapses**. Symmetric backends must pair with `act_signed`

### 8.2 Signed weights (`w_all_positive=False`, default)

Conv weights have both signs and must be signed. **weight_unsigned is a
misconfiguration** (negative weights cannot be represented in a [0,1] domain);
all 8 measured cells collapse to 0 — kept as a counterexample.

### 8.3 per_channel (default True)

Weights get one scale set **per output channel**, more accurate than per_tensor;
activations are usually per_tensor. `mixed_quant` (weights per_channel +
activations per_tensor) is a common on-device compromise, verified stable on
asymmetric backends.

---

## 9. Network-script integration

### 9.1 Backend injection (`set_quant_method`)

`networks_yolo26-detect.py` L367-383: `set_quant_method(method)` calls
`quant_pkg.load_quant_backend(method)` and binds `QuantAdd/QuantCat/QuantConcat/
QuantConv2d/QuantMaxPool/QuantSiLU/...` to that backend's exported operator
classes (module-level global names). seg.py (L108) and pose.py (L119) have their
**own** `set_quant_method` (do not reuse det's, or head quantizer keys go
missing).

### 9.2 Build-time replacement (not runtime traversal)

YOLO26 does **not** traverse and swap nn layers at runtime; it builds the
quantized network directly with backend operator classes: `Conv` (L418-439) uses
`QuantConv2d(...)` when `quant=True`, the activation becomes `QuantSiLU(...)`,
residuals use `QuantAdd(..., quant_inference=True)`, concats use
`QuantCat/QuantConcat`. `_quant_layer_kwargs()` (L406-415) uniformly builds
`a_bits/w_bits/per_channel/all_positive/w_all_positive` from `QUANT_CFG` (plus
`w_quant` for PACT) and passes them to every quantized layer.

> Note: the `quantization` package does have `add_quant_op`-style traversal
> helpers, but the YOLO26 main flow builds quantized layers directly; traversal
> is only for auxiliary cases. depthwise/group conv on the traversal path must be
> recognized as `nn.Conv2d`.

### 9.3 Single source of config (`QUANT_CFG`)

`networks_yolo26-detect.py` L222-232: `DEFAULT_QUANT_CFG` fields
`a_bits=8 / w_bits=8 / per_channel=True / all_positive=True / w_all_positive=False /
mixed_quant=False / pact_w_quant='lsqplus_v1'`. CLI flags `--a-bits/--w-bits/
--per-channel/--all-positive/--w-all-positive/--mixed-quant/--pact-w-quant/
--seed/--run-tag` override; `_quant_cfg_tag()` emits full-word tags (no
abbreviations).

### 9.4 Stages and checkpoints

`load_checkpoint()`/`save_checkpoint()` load/save `.pth`; best.pth uses
`ck['state_dict']`, while PTQ `.pth` may be a bare state_dict or `ck['model']`
(handle both). `save_quant_outputs()` (L1323-1331) calls `freeze_batch_init()`
then saves the quantized model and exports the three artifacts;
`build_float_model()` (L1216-1230) and `collect_quant_params()` (L1233-1235)
both delegate to `quant_pkg`; `verify()` (L1297-1320) hard-asserts the exported
ONNX has no quant nodes and quant_params are complete. `meta` passes through via
`**(meta or {})` to record quant method/config/seed.

---

## 10. Key engineering decisions and pitfalls

| Decision/pitfall | Root cause | Solution |
|---|---|---|
| `_anchors` device mismatch | detection-head lazily-created `_anchors`/`_strides_tensor` are plain attributes; `model.cpu()` doesn't move them, deepcopy leaves GPU/CPU mixed | register as **non-persistent buffers** (`persistent=False`: move with device, excluded from state_dict, not treated as quantizers by `freeze_batch_init`) |
| ONNX export `aten::copy_` failure | `_ClipOnlyQuantizer.forward` used `.to()` | store bounds with `register_buffer`, no device conversion in forward |
| QuantCat quantizers missed on replace | quantizers live in a `ModuleList` under a non-standard attribute name | recurse + duck-type `hasattr(child,'clip_bounds')` traversal |
| bash resume never triggering | `already_ok` matched `[OK] task=` but log lines are `[OK] date task=` | regex anchored on dated `[OK]` lines |
| seg/dorefa OOM | 640 resolution + tanh quantizer memory | seg and all-task dorefa QAT always batch=4 |
| LSQ v1/pact all-zero dummy | all-zero input → scale=0 → NaN | dummy input `randn*0.1` |
| seg/pose head quantizer missing keys | reusing det's set_quant_method across tasks | seg/pose each have their own set_quant_method |
| dorefa unsigned activation collapse | unbounded activation in tanh domain | fixed by adding a learnable scale s |

**Why duck typing over a registry/inheritance**: the 7 backends' quantizer
internals differ a lot (pact learns α, lsqplus learns β, minmax has no params,
dorefa uses the tanh domain). A forced common base class would constrain
implementations. The `hasattr` contract (`weight_quantizer`/`clip_bounds`/
`quant_inference`) gives a minimal cross-backend surface, and each backend
implements the full op set independently in its own file.

---

## 11. Experimental conclusions

Full 79-cell sweep (4 main configs × 3 tasks × 7 backends + weight_unsigned
counterexamples + mixed_quant + multi-seed + calibration size); data in
[README_CN.md](../README_CN.md) / [README.md](../README.md) results section.
Core conclusions:

- **Recommended config (lsqplus_v1 + per_channel + act_unsigned) is lossless or
  near-lossless on all three tasks**: detect −0.0007 / seg −0.0021 / pose +0.0037
- asymmetric backends + unsigned all stable (Δ ≤ 0.039)
- symmetric backends + unsigned degrade severely (must use act_signed)
- weight_unsigned collapses entirely (a theoretically-expected misconfiguration)
- multi-seed variation ≤ 0.012, conclusion direction stable
- QAT recovers most of the PTQ gap; for unsigned+asymmetric, PTQ is already near
  float

---

## 12. Limitations and future work

**Current limitations**:
- If a custom head has an operator not covered by `_ELTWISE_OP_NAMES`, its
  on-device quantization protocol still needs confirmation
- LUT non-linearities (silu/sigmoid/softmax) are approximated by fake-quant in
  training and may differ slightly from the board's actual LUT width; re-validate
  accuracy with the real compiler after deployment
- Validation uses a 1/100 COCO mini small sample; absolute accuracy does not
  represent full COCO

**Future extensions**:
- New backend: implement the same ~15 op classes + quantizer + `clip_bounds()`,
  register in `_BACKEND_MODULES` — no network-script changes needed
- Mixed precision (sensitive layers at int16/FP16) can be annotated per-layer at
  `collect_quant_params` export
- Exact modeling of on-device int-domain element-wise requant (currently
  approximated by fake-quant)

---

*Document matches the code at the current git commit. Operator implementations:
`quantization/{backend}.py`; common mechanisms: `quantization/__init__.py`;
network integration: `networks_yolo26-*.py`.*
