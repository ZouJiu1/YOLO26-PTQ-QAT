"""量化后端选择与跨后端通用工具 / Quantization backend selection and cross-backend utilities.

7 种量化方法（每个文件各自实现 QuantConv2d / QuantConvTranspose2d / QuantLinear /
QuantAdd / QuantSub / QuantMultiply / QuantDiv / QuantConcat / QuantMaxPool / QuantCat）：
/ 7 quantization methods (each file implements QuantConv2d / QuantConvTranspose2d / QuantLinear /
QuantAdd / QuantSub / QuantMultiply / QuantDiv / QuantConcat / QuantMaxPool / QuantCat):

    lsqplus_v1 -> quantization.lsqplus_quantize_V1
    lsqplus_v2 -> quantization.lsqplus_quantize_V2
    lsq_v1     -> quantization.lsqquantize_V1
    lsq_v2     -> quantization.lsqquantize_V2
    minmax     -> quantization.minmax
    dorefa     -> quantization.dorefa
    pact       -> quantization.pact

网络脚本通过 load_quant_backend(method) 拿到统一算子集合，其余
冻结/复位/scale/zero_point 提取等逻辑全部在本文件做跨后端兼容，
网络侧不需要再关心各量化器内部属性差异。
/ Network scripts obtain a unified operator set via load_quant_backend(method). All other
logic — freeze/reset/scale/zero_point extraction — is handled here with cross-backend
compatibility, so network code does not need to care about per-backend attribute differences.
"""

import copy
import datetime
import importlib
import re
import sys

import torch

from .constants import INIT_STATE_FROZEN, INIT_STATE_TRAINING, INIT_STATE_UNINIT


def install_print_timestamp():
    """给每行 print 输出加分钟级时间戳前缀 / Prepend a minute-precision timestamp to every printed line.

    效果 / Effect: ``hello`` -> ``[2026-09-25 19:42] hello``。
    仅包装控制台 print（stdout），不写日志文件 / Only wraps console print (stdout), writes no log files；
    ``\\r`` 开头的 tqdm 进度条不加戳，避免进度条被拆烂 / lines driven by ``\\r`` (tqdm bars) are not prefixed。
    重复调用安全 / Safe to call multiple times.
    """
    if getattr(sys.stdout, "_minute_timestamped", False):
        return
    real_stdout = sys.stdout

    class _MinuteTimestampedStdout:
        _minute_timestamped = True

        def __init__(self, stream):
            self._stream = stream
            self._need_prefix = True  # 下一个非控制字符片段是否需要前缀 / whether next text piece needs a prefix

        def write(self, text):
            if not text:
                return 0
            for piece in re.split(r"(\r|\n)", text):
                if piece == "\n":
                    self._stream.write("\n")
                    self._need_prefix = True
                elif piece == "\r":
                    # 回车原地刷新（tqdm）：直接透传，且后续片段不补时间戳 / Carriage-return in-place refresh (tqdm): pass through, no prefix
                    self._stream.write("\r")
                    self._need_prefix = False
                elif piece:
                    if self._need_prefix:
                        self._stream.write(
                            datetime.datetime.now().strftime("[%Y-%m-%d %H:%M] ")
                        )
                        self._need_prefix = False
                    self._stream.write(piece)
            return len(text)

        def flush(self):
            self._stream.flush()

        def __getattr__(self, name):
            return getattr(self._stream, name)

    sys.stdout = _MinuteTimestampedStdout(real_stdout)

QUANT_METHODS = (
    "lsqplus_v1",
    "lsqplus_v2",
    "lsq_v1",
    "lsq_v2",
    "minmax",
    "dorefa",
    "pact",
)

_BACKEND_MODULES = {
    "lsqplus_v1": "quantization.lsqplus_quantize_V1",
    "lsqplus_v2": "quantization.lsqplus_quantize_V2",
    "lsq_v1": "quantization.lsqquantize_V1",
    "lsq_v2": "quantization.lsqquantize_V2",
    "minmax": "quantization.minmax",
    "dorefa": "quantization.dorefa",
    "pact": "quantization.pact",
}

_WEIGHT_OP_NAMES = ("QuantConv2d", "QuantConvTranspose2d", "QuantLinear")
_ELTWISE_OP_NAMES = (
    "QuantAdd",
    "QuantSub",
    "QuantMultiply",
    "QuantDiv",
    "QuantConcat",
    "QuantMaxPool",
    "QuantCat",
)


def load_quant_backend(method):
    """按方法名返回量化后端模块（属性即统一算子集合） / Return quantization backend module by method name (attributes are the unified operator set)."""
    if method not in _BACKEND_MODULES:
        raise ValueError(
            f"未知量化方法: {method}，可选: {', '.join(QUANT_METHODS)}"
        )
    return importlib.import_module(_BACKEND_MODULES[method])


def weight_quant_ops(backend):
    return tuple(getattr(backend, name) for name in _WEIGHT_OP_NAMES)


def all_quant_ops(backend):
    return tuple(getattr(backend, name) for name in _WEIGHT_OP_NAMES + _ELTWISE_OP_NAMES)


def is_weight_quant_module(module):
    """带 weight_quantizer 的卷积/反卷积/全连接层（各后端统一鸭子类型） / Conv/ConvTranspose/Linear layers with weight_quantizer (uniform duck-typing across backends)."""
    return hasattr(module, "weight_quantizer")


# ----------------------------------------------------------------------------
# 量化器状态：冻结（校准/训练结束）与复位（灌入新权重后重新校准）
# / Quantizer state: freeze (after calibration/training) and reset (recalibrate after new weights)
# ----------------------------------------------------------------------------
def freeze_batch_init(model):
    """停止 lsq/lsq+ 量化器前 batch_init 批的 scale/beta 滑动统计。
    / Stop lsq/lsq+ quantizers' scale/beta sliding stats for the first batch_init batches.

    init_state 在 yolo 脚本里可能被注册成 buffer（Tensor），其余脚本里是普通 int，
    两种情况都处理；minmax/pact 没有该属性，不受影响。
    / init_state may be registered as a buffer (Tensor) in YOLO scripts or a plain int elsewhere;
    both cases are handled. minmax/pact do not have this attribute and are unaffected.
    """
    for module in model.modules():
        init_state = getattr(module, "init_state", None)
        if isinstance(init_state, torch.Tensor):
            init_state.fill_(INIT_STATE_FROZEN)
        elif isinstance(init_state, int):
            module.init_state = INIT_STATE_FROZEN
        # minmax/pact 激活量化器：标记为已初始化，
        # 避免载入 PTQ checkpoint 后首批数据把校准好的 r_min/r_max、alpha 冲掉
        # / minmax/pact activation quantizers: mark as initialized so the first batch
        #   after loading a PTQ checkpoint does not overwrite calibrated r_min/r_max or alpha
        init_flag = getattr(module, "init", None)
        if type(init_flag) is int:
            module.init = 1


def reset_quantizer_states(model):
    """权重复位后让所有量化器从第 0 批重新统计。
    / Reset all quantizers to start collecting stats from batch 0 after weight reset.

    - lsq/lsq+：init_state 清 0，首批前向时各量化器用自己的原生公式重置 s/beta；
    - minmax/pact 激活量化器：init 清 0，首批直接重新赋值 r_min/r_max 或 alpha。
    / - lsq/lsq+: clear init_state to 0, each quantizer resets s/beta with its native formula on first forward;
    / - minmax/pact activation quantizers: clear init to 0, directly reassign r_min/r_max or alpha on first batch.
    """
    for module in model.modules():
        init_state = getattr(module, "init_state", None)
        if isinstance(init_state, torch.Tensor):
            init_state.fill_(INIT_STATE_UNINIT)
        elif isinstance(init_state, int):
            module.init_state = INIT_STATE_UNINIT
        init_flag = getattr(module, "init", None)
        if type(init_flag) is int:
            module.init = 0


def dequantized_weight(module):
    """对带权量化层做一次 weight_quantizer 前向，拿到反量化后的权重 / Run weight_quantizer forward once on a weight-quantized layer to obtain dequantized weights."""
    with torch.no_grad():
        return module.weight_quantizer(module.weight).detach().clone()


class _IdentityQuantizer(torch.nn.Module):
    """恒等量化器（占位）：权重已反量化并原地灌回，前向直通 /
    Identity placeholder: weights are already dequantized in place, forward is a pass-through."""

    def forward(self, x):
        return x


class _ClipOnlyQuantizer(torch.nn.Module):
    """仅保留量化器的截断（clip）、去掉舍入（round）的激活算子 /
    Activation op that keeps only the quantizer's clip and removes rounding.

    硬件整型量化推理必然先把激活截断到 [Qn,Qp] 对应的区间再取整；导出纯浮点参考
    ONNX 时保留截断、去掉取整，可使参考图与真实部署数值行为一致，避免 QAT 后的权重
    "依赖激活钳位"在去伪量化图中逐层放大（unsigned + 对称量化后端上曾观测到输出爆炸）。
    / Hardware integer quantization always clips activations to the interval mapped
    from [Qn,Qp] before rounding. Keeping the clip and dropping rounding in the exported
    pure-float reference ONNX makes the reference graph numerically consistent with real
    deployment, and avoids post-QAT weights that "rely on activation clipping" blowing up
    layer by layer in the de-fake-quantized graph (observed with unsigned + symmetric backends).
    """

    def __init__(self, lower, upper):
        super().__init__()
        self.register_buffer('lower', lower.detach().reshape(1).clone())
        self.register_buffer('upper', upper.detach().reshape(1).clone())

    def forward(self, x):
        # 不在前向里做 .to()（tracing 会产生 aten::copy_，无法导出 ONNX）；
        # buffer 随模型 .to(device)/.half() 一起迁移，(1,) 形状自动按通道广播 /
        # No .to() in forward (tracing emits aten::copy_, breaking ONNX export);
        # buffers move with the model, the (1,) shape broadcasts over channels.
        return torch.clamp(x, self.lower, self.upper)


def build_float_model(quant_model, use_clip=False):
    """从量化模型构建纯浮点参考模型（深拷贝，不改原模型） /
    Build a pure-float reference model from a quantized model (deep copy; untouched).

    三步 / Three steps:
      1) 所有带 quant_inference 开关的模块置 True（走"权重已烘焙"的推理路径）/
         Set quant_inference=True everywhere (baked-weight inference path);
      2) 带权层用反量化权重原地覆盖 .weight，weight_quantizer 换成恒等（必须在替换前完成反量化）/
         Overwrite .weight in place with dequantized weights and replace weight_quantizer by
         identity (dequantization must happen before replacement);
      3) 激活量化器按 use_clip 处理 / activation quantizers by use_clip:
         - use_clip=False（默认）：全部换成恒等（彻底去伪量化）。导出的 ONNX 无 scale/zero_point，
           部署时由板端 NPU/TPU 的 PTQ 工具自行校准，这是最常见的部署交付物。
           注意：unsigned 激活 + 对称量化后端（lsq_v1/v2、minmax）时，QAT 权重依赖激活钳位，
           无 clip 的参考图在真实图片上数值发散（box 可达 ±1e5），仅作权重交付/指标参考，
           不建议直接喂给板端校准 / replace all by identity (fully de-fake-quantized).
           The exported ONNX has no scale/zero_point; board-side NPU/TPU PTQ tools calibrate
           them — the usual deployment deliverable. CAUTION: with unsigned activations +
           symmetric backends (lsq_v1/v2, minmax), QAT weights rely on activation clipping,
           and a clip-free reference graph diverges on real images (boxes up to ±1e5);
         - use_clip=True：换成 _ClipOnlyQuantizer（保留截断、去掉舍入），参考图数值行为与
           真实量化部署一致（QAT 权重依赖激活钳位时不会爆炸）/
           replace by _ClipOnlyQuantizer (keep clip, drop rounding): the reference graph
           stays numerically consistent with real quantized deployment (no blow-up when
           QAT weights rely on activation clipping).
    """
    _orig_device = next(quant_model.parameters()).device
    # 训练时部分量化层会把带梯度的中间张量存成普通属性（如 lsqplus QuantConv2d 的
    # input / quant_input / quant_weight，见 lsqplus_quantize_V1.py），deepcopy 遇到
    # 非叶张量（requires_grad 且有 grad_fn）会直接 RuntimeError，这正是 QAT 训完后
    # 导出处 exit=1 的元凶。深拷贝前统一 detach 这些普通属性（仅实例属性，不碰
    # _parameters/_buckets 注册项；导出只读权重与量化参数，detach 无副作用）/
    # Training caches grad-carrying intermediates as plain module attributes (e.g.
    # lsqplus QuantConv2d input/quant_input/quant_weight in lsqplus_quantize_V1.py);
    # deepcopy raises RuntimeError on non-leaf tensors — the cause of exit=1 at export
    # right after QAT finishes. Detach plain tensor attributes before copying (instance
    # attrs only; registered params/buffers untouched; export only reads weights/params).
    for module in quant_model.modules():
        for attr_name, value in list(vars(module).items()):
            if isinstance(value, torch.Tensor) and value.requires_grad:
                setattr(module, attr_name, value.detach())
    model = copy.deepcopy(quant_model.cpu())  # CPU 上深拷贝，避免整图复制占用显存 / deep-copy on CPU
    quant_model.to(_orig_device)  # 原模型搬回，调用方契约不变 / restore caller's model device
    model.cpu()
    freeze_batch_init(model)
    model.eval()

    with torch.no_grad():
        # 1) 推理路径开关 / inference-path switch
        for module in model.modules():
            if hasattr(module, 'quant_inference'):
                module.quant_inference = True

        # 2) 反量化权重烘焙 + 权重量级化器恒等 / bake dequantized weights + identity weight quantizers
        for module in list(model.modules()):
            if is_weight_quant_module(module) and hasattr(module, 'weight'):
                module.weight.copy_(module.weight_quantizer(module.weight).detach())
                module.weight_quantizer = _IdentityQuantizer()

        # 3) 激活量化器按 use_clip 替换。递归遍历（不依赖属性名）：各后端挂载方式不同，
        #    可能是 activation_quantizer / activation_quantizer0/1，
        #    也可能是 QuantCat 里的 self.quantizers = ModuleList([...])。
        #    鸭子类型：只有激活量化器实现了 clip_bounds()，权重量级化器没有。
        # / activation quantizers → identity (default) or clip-only (use_clip=True).
        #   Recurse by duck type instead of attr names: backends mount them as
        #   activation_quantizer / activation_quantizer0/1, or inside QuantCat as
        #   self.quantizers = ModuleList([...]). Only activation quantizers implement
        #   clip_bounds(); weight quantizers do not.
        def _replace_activation_quantizers(parent):
            for child_name, child in list(parent.named_children()):
                if hasattr(child, 'clip_bounds'):
                    if use_clip:
                        lower, upper = child.clip_bounds()
                        setattr(parent, child_name, _ClipOnlyQuantizer(lower, upper))
                    else:
                        setattr(parent, child_name, _IdentityQuantizer())
                else:
                    _replace_activation_quantizers(child)

        _replace_activation_quantizers(model)

    return model


# ----------------------------------------------------------------------------
# scale / zero_point 提取（导出 JSON 用）
# / scale / zero_point extraction (for JSON export)
# ----------------------------------------------------------------------------
def _pack(scale, zero_point):
    scale = scale.detach().cpu().flatten().to(torch.float64)
    zero_point = zero_point.detach().cpu().flatten().to(torch.int64)
    if scale.numel() == 1:
        return {"scale": float(scale.item()), "zero_point": int(zero_point.item())}
    return {"scale": scale.tolist(), "zero_point": zero_point.tolist()}


def activation_scale_zp(quantizer):
    """从任意后端的激活量化器提取 scale / zero_point / Extract scale / zero_point from an activation quantizer of any backend."""
    # lsqplus：可学习 s + beta（非对称） / lsqplus: learnable s + beta (asymmetric)
    beta = getattr(quantizer, "beta", None)
    if hasattr(quantizer, "s") and beta is not None and beta.numel() > 0:
        scale = quantizer.s.detach()
        beta = beta.detach()
        zero_point = torch.round(-beta / scale).to(torch.int64).clamp(
            quantizer.Qn, quantizer.Qp
        )
        return _pack(scale, zero_point)

    # lsq：仅可学习 s（对称，zero_point=0） / lsq: only learnable s (symmetric, zero_point=0)
    if hasattr(quantizer, "s"):
        scale = quantizer.s.detach().to(torch.float64).flatten()
        return _pack(scale, torch.zeros_like(scale, dtype=torch.int64))

    # minmax：运行期统计出的 scale / zero_point buffer / minmax: runtime-collected scale / zero_point buffers
    if hasattr(quantizer, "zero_point") and hasattr(quantizer, "scale"):
        return _pack(quantizer.scale, quantizer.zero_point.to(torch.int64))

    # pact：可学习截断阈值 alpha / pact: learnable clipping threshold alpha
    if hasattr(quantizer, "alpha"):
        if getattr(quantizer, "all_positive", False):
            scale = quantizer.alpha.detach() / quantizer.q_range
            zero_point = torch.zeros_like(scale, dtype=torch.int64)
        else:
            scale = quantizer.alpha.detach() * 2.0 / quantizer.q_range
            zero_point = torch.round(
                torch.full_like(quantizer.alpha.detach(), quantizer.q_range / 2.0)
            ).to(torch.int64)
        return _pack(scale, zero_point)

    # dorefa：激活先除以可学习尺度 s 归一化到固定网格，再乘回 s；
    # 输出尺度 = s / Qp（对称，zero_point=0）
    # / dorefa: activation first normalized to fixed grid by learnable scale s, then multiplied back by s;
    #   output scale = s / Qp (symmetric, zero_point=0)
    s = getattr(quantizer, "s", None)
    if s is not None:
        scale = (s.detach().abs().flatten().to(torch.float64) / quantizer.Qp).clamp(min=1e-12)
    else:
        scale = torch.tensor(1.0 / quantizer.Qp, dtype=torch.float64)
    return _pack(scale, torch.zeros_like(scale, dtype=torch.int64))


def weight_scale_zp(module):
    """从任意后端的带权量化层提取权重 scale / zero_point / Extract weight scale / zero_point from a weight-quantized layer of any backend."""
    wq = module.weight_quantizer

    # lsq / lsq+：可学习 s（权重恒对称，zero_point=0） / lsq / lsq+: learnable s (weights are always symmetric, zero_point=0)
    if hasattr(wq, "s"):
        scale = wq.s.detach().cpu().flatten().to(torch.float64)
        return _pack(scale, torch.zeros_like(scale, dtype=torch.int64))

    # minmax：按当前权重复算标准非对称 scale/zero_point / minmax: recompute standard asymmetric scale/zero_point from current weights
    if hasattr(wq, "qmin"):
        weight = module.weight.detach()
        if getattr(wq, "per_channel", False):
            w_tmp = weight.reshape(weight.size(0), -1)
            r_min = w_tmp.min(dim=1).values.view(
                -1, *([1] * (weight.dim() - 1))
            )
            r_max = w_tmp.max(dim=1).values.view(
                -1, *([1] * (weight.dim() - 1))
            )
        else:
            r_min = weight.min()
            r_max = weight.max()
        if getattr(wq, "all_positive", False):
            r_min = r_min.clamp(min=0)
        eps = torch.finfo(weight.dtype).eps
        scale = torch.clamp(r_max - r_min, min=eps) / (wq.qmax - wq.qmin)
        zero_point = (wq.qmin - r_min / scale).round().clamp(wq.qmin, wq.qmax)
        return _pack(scale, zero_point)

    # dorefa / pact（pact 权重量化器就是 DorefaWeightQuantizer）：
    # tanh 域非线性网格没有严格统一的 weight 域 scale，
    # 这里用反量化权重的幅度近似一个对称 scale，仅用于参数文件完整性。
    # / dorefa / pact (pact weight quantizer is DorefaWeightQuantizer):
    #   tanh-domain non-linear grid has no strict uniform weight-domain scale.
    #   We approximate a symmetric scale using the magnitude of dequantized weights, only for param-file completeness.
    with torch.no_grad():
        w_dq = wq(module.weight).detach()
    qp = wq.Qp
    if getattr(wq, "per_channel", False):
        w_tmp = w_dq.reshape(w_dq.size(0), -1)
        if getattr(wq, "all_positive", False):
            amax = w_tmp.max(dim=1).values
        else:
            amax = w_tmp.abs().max(dim=1).values
        scale = (amax if getattr(wq, "all_positive", False) else 2.0 * amax) / qp
    else:
        amax = w_dq.max() if getattr(wq, "all_positive", False) else w_dq.abs().max()
        scale = (amax if getattr(wq, "all_positive", False) else 2.0 * amax) / qp
    scale = scale.to(torch.float64).flatten()
    return _pack(scale, torch.zeros_like(scale, dtype=torch.int64))


def _add_quantizer_param(params, quantizer, tensor_name):
    params[tensor_name] = activation_scale_zp(quantizer)


def collect_quant_params(quant_model):
    """遍历量化模型，收集全部权重/激活的 scale/zero_point / Traverse quantized model and collect all weight/activation scale/zero_point."""
    params = {}
    for name, module in quant_model.named_modules():
        if is_weight_quant_module(module):
            params[f"{name}.weight"] = weight_scale_zp(module)
            # pact 的量化卷积没有 activation_quantizer（激活在 PACT-ReLU 处量化）
            # / pact quantized conv has no activation_quantizer (activation is quantized at PACT-ReLU)
            if hasattr(module, "activation_quantizer"):
                _add_quantizer_param(
                    params, module.activation_quantizer, f"{name}.input"
                )
            continue

        if not getattr(module, "quant_inference", False):
            continue

        # QuantCat：任意路输入，每一路一个量化器 / QuantCat: arbitrary number of inputs, each with its own quantizer
        if hasattr(module, "quantizers"):
            for index, quantizer in enumerate(module.quantizers):
                _add_quantizer_param(params, quantizer, f"{name}.input{index}")
            continue

        if hasattr(module, "activation_quantizer"):
            _add_quantizer_param(
                params, module.activation_quantizer, f"{name}.input"
            )
        for suffix in ("0", "1"):
            quantizer = getattr(module, f"activation_quantizer{suffix}", None)
            if quantizer is not None:
                _add_quantizer_param(params, quantizer, f"{name}.input{suffix}")
    return params


def required_quant_keys(quant_model):
    """导出后必须存在的量化参数 key（与 collect_quant_params 的产出对应） / Quant param keys that must exist after export (correspond to collect_quant_params output)."""
    keys = []
    for name, module in quant_model.named_modules():
        if is_weight_quant_module(module):
            keys.append(f"{name}.weight")
            if hasattr(module, "activation_quantizer"):
                keys.append(f"{name}.input")
        elif getattr(module, "quant_inference", False):
            if hasattr(module, "quantizers"):
                keys.extend(
                    f"{name}.input{index}"
                    for index in range(len(module.quantizers))
                )
            elif hasattr(module, "activation_quantizer"):
                keys.append(f"{name}.input")
            else:
                keys.extend(f"{name}.input{suffix}" for suffix in ("0", "1"))
    return keys
