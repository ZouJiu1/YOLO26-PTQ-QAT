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
