import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .constants import INIT_STATE_FROZEN, INIT_STATE_TRAINING, INIT_STATE_UNINIT


# ********************* quantizers（量化器，量化） *********************
# 取整 (STE) / rounding (STE)
class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class quantizek(Function):
    @staticmethod
    def forward(ctx, values, q_range):
        ctx.save_for_backward(values)
        ctx.other = q_range
        values = Round.apply(values * q_range) / q_range
        return values

    @staticmethod
    def backward(ctx, grad_weight):
        return grad_weight, None

# A(特征)量化 / A(activation) quantization
class DorefaActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False):
        super(DorefaActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        if self.all_positive:
            # 无符号激活量化到 [0, 2^b-1] / unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            # 有符号激活量化到 [-2^(b-1), 2^(b-1)-1] / signed activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        # scale (步长) 与 beta (零点偏移) 均可学习，遵循标准非对称量化公式：
        #   q = round((x - beta) / s).clamp(Qn, Qp)
        #   x_q = q * s + beta
        # / Learnable scale s and zero-point beta following the standard asymmetric formula:
        #   q = round((x - beta) / s).clamp(Qn, Qp)
        #   x_q = q * s + beta
        self.s = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.init_state = INIT_STATE_UNINIT  # UNINIT / TRAINING / FROZEN

    def _set_init_state(self, value):
        # init_state 在 YOLO 网络中会被注册成持久化 tensor buffer，直接赋 int 会报错
        # / init_state may be registered as a persistent tensor buffer in YOLO nets; direct int assignment would error
        if isinstance(self.init_state, torch.Tensor):
            self.init_state.fill_(value)
        else:
            self.init_state = value

    # 量化/反量化 / quantize/dequantize
    def forward(self, activation):
        if self.init_state == INIT_STATE_UNINIT:
            # 首批用真实激活的 min/max 初始化 s 与 beta
            # / Initialize s and beta from real activation min/max on first batch
            x = activation.detach()
            cur_min = x.min()
            cur_max = x.max()
            eps = torch.finfo(x.dtype).eps
            cur_s = torch.clamp(cur_max - cur_min, min=eps) / (self.Qp - self.Qn)
            cur_beta = cur_min - cur_s * self.Qn
            self.s.data.copy_(cur_s.reshape(1).clamp(min=1e-6))
            self.beta.data.copy_(cur_beta.reshape(1))
            self._set_init_state(INIT_STATE_TRAINING)
        elif self.init_state != INIT_STATE_FROZEN:
            # 训练阶段 EMA 平滑更新 s 与 beta / EMA-smooth update of s and beta during training
            x = activation.detach()
            cur_min = x.min()
            cur_max = x.max()
            eps = torch.finfo(x.dtype).eps
            cur_s = torch.clamp(cur_max - cur_min, min=eps) / (self.Qp - self.Qn)
            cur_beta = cur_min - cur_s * self.Qn
            self.s.data.mul_(0.9).add_(cur_s.reshape(1).clamp(min=1e-6), alpha=0.1)
            self.beta.data.mul_(0.9).add_(cur_beta.reshape(1), alpha=0.1)

        # 标准非对称量化 / standard asymmetric quantization
        q_a = Round.apply((activation - self.beta) / self.s).clamp(self.Qn, self.Qp)
        return q_a * self.s + self.beta

    def clip_bounds(self):
        """激活空间内的截断边界 (Qn·s+beta, Qp·s+beta)；导出"截断保留、舍入去除"的纯浮点参考图时使用 /
        Clip bounds in activation space (Qn·s+beta, Qp·s+beta): used when exporting a clip-only (rounding-free) float reference graph."""
        return (self.Qn * self.s + self.beta).detach(), (self.Qp * self.s + self.beta).detach()

# W(权重)量化 / W(weight) quantization
class DorefaWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False):
        super(DorefaWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.per_channel = per_channel
        if self.all_positive:
            # 无符号量化等级范围 [0, 2^b-1] / unsigned level range [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** w_bits - 1
        else:
            # 有符号量化等级范围 [-2^(b-1), 2^(b-1)-1] / signed level range [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (w_bits - 1)
            self.Qp = 2 ** (w_bits - 1) - 1

    # 量化/反量化 / quantize/dequantize
    def forward(self, weight):
        # 标准对称量化：直接在权重域计算 scale，不再使用 tanh 非线性变换
        # / Standard symmetric quantization: compute scale in weight domain directly, no tanh nonlinearity
        if self.per_channel:
            maxvalue = weight.detach().abs().reshape(weight.size(0), -1).max(dim=1).values
            maxvalue = maxvalue.view(-1, *([1] * (weight.dim() - 1)))
        else:
            maxvalue = weight.detach().abs().max()
        maxvalue = maxvalue.clamp(min=1e-8)

        if self.all_positive:
            # 无符号：[0, maxvalue] 映射到 [0, Qp] / unsigned: [0, maxvalue] map to [0, Qp]
            tmp = weight / maxvalue
            tmp = Round.apply(tmp * self.Qp).clamp(self.Qn, self.Qp) / self.Qp
        else:
            # 有符号：[-maxvalue, maxvalue] 映射到 [Qn, Qp] / signed: [-maxvalue, maxvalue] map to [Qn, Qp]
            tmp = weight / maxvalue
            tmp = Round.apply(tmp * self.Qp).clamp(self.Qn, self.Qp) / self.Qp

        q_w = maxvalue * tmp
        return q_w

class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False,
                 w_all_positive=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits, all_positive=w_all_positive, per_channel=per_channel)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        # print('input:',input.size(),self.quant_inference)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                          self.groups)
        return output


class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False,
                 w_all_positive=False):
        # 注意: ConvTranspose2d 的参数顺序为 (..., output_padding, groups, bias, dilation, padding_mode)
        # / Note: ConvTranspose2d parameter order is (..., output_padding, groups, bias, dilation, padding_mode)
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                                   groups, bias, dilation, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits, all_positive=w_all_positive, per_channel=per_channel)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.output_padding,
                                    self.groups, self.dilation)
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 a_bits=8,
                 w_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False,
                 w_all_positive=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits, all_positive=w_all_positive, per_channel=per_channel)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output


class QuantAdd(nn.Module):
    def __init__(self,
                 a_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False,
                 w_all_positive=False):
        super(QuantAdd, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, A, C):
        if not self.quant_inference:
            return A + C
        else:
            Q_A = self.activation_quantizer0(A)
            Q_C = self.activation_quantizer1(C)
            return Q_A + Q_C

class QuantSub(nn.Module):
    def __init__(self,
                 a_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False,
                 w_all_positive=False):
        super(QuantSub, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, A, C):
        if not self.quant_inference:
            return A - C
        else:
            Q_A = self.activation_quantizer0(A)
            Q_C = self.activation_quantizer1(C)
            return Q_A - Q_C

class QuantMultiply(nn.Module):
    def __init__(self,
                 a_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False,
                 w_all_positive=False):
        super(QuantMultiply, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, A, C):
        if not self.quant_inference:
            return A * C
        else:
            Q_A = self.activation_quantizer0(A)
            Q_C = self.activation_quantizer1(C)
            return Q_A * Q_C

class QuantDiv(nn.Module):
    def __init__(self,
                 a_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False,
                 w_all_positive=False):
        super(QuantDiv, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, A, C):
        if not self.quant_inference:
            return A / C
        else:
            Q_A = self.activation_quantizer0(A)
            Q_C = self.activation_quantizer1(C)
            # 分母反量化网格可能恰好落在 0，钳到小正数防止 0/0 产生 NaN
            # / Dequantized denominator grid may land exactly on 0; clamp to a small positive number to avoid 0/0 → NaN
            return Q_A / torch.clamp(Q_C, min=1e-6)

class QuantConcat(nn.Module):
    def __init__(self,
                 a_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False,
                 w_all_positive=False):
        super(QuantConcat, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, A, C, dim):
        if not self.quant_inference:
            return torch.concat([A, C], dim=dim)
        else:
            Q_A = self.activation_quantizer0(A)
            Q_C = self.activation_quantizer1(C)
            return torch.concat([Q_A, Q_C], dim=dim)

class QuantMaxPool(nn.Module):
    def __init__(self,
                kernel_size: any,
                stride=None,
                padding=0,
                dilation=1,
                return_indices: bool = False,
                ceil_mode: bool = False,
                a_bits=8,
                quant_inference=False,
                all_positive=False,
                per_channel=False):
        super(QuantMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, x):
        if not self.quant_inference:
            Q_A = x
        else:
            Q_A = self.activation_quantizer(x)
        return nn.functional.max_pool2d(Q_A, kernel_size=self.kernel_size, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation,
                                            return_indices=self.return_indices, ceil_mode=self.ceil_mode)

class QuantCat(nn.Module):
    """对多个输入张量分别做激活伪量化后再 concat（每一路各持有一个激活量化器）。
    / Apply activation pseudo-quantization to each input tensor separately, then concat (each input has its own activation quantizer).
    """

    def __init__(self, num_inputs, a_bits=8):
        super().__init__()
        self.quant_inference = True
        self.quantizers = nn.ModuleList(
            [DorefaActivationQuantizer(a_bits=a_bits) for _ in range(num_inputs)]
        )

    def forward(self, tensors, dim=1):
        return torch.cat([q(t) for q, t in zip(self.quantizers, tensors)], dim=dim)


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8, quant_inference=False,
                 all_positive=False, per_channel=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1:  # 第一层也量化 / quantize the first layer too
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ConvTranspose2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1:  # 第一层也量化 / quantize the first layer too
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups,
                                                                bias=True,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel)
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels,
                                                                child.out_channels,
                                                                child.kernel_size,
                                                                stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation,
                                                                groups=child.groups, bias=False,
                                                                padding_mode=child.padding_mode,
                                                                a_bits=a_bits,
                                                                w_bits=w_bits,
                                                                quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel)
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] >= 1:  # 第一层也量化 / quantize the first layer too
                if child.bias is not None:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=True, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel)
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=False, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel)
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        else:
            add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits,
                         quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel)

def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False,
            all_positive=False, per_channel=False):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits,
                 quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel)
    return model

class QuantSiLU(nn.Module):
    """量化 SiLU (x * sigmoid(x)) — 单输入, 只有 input quantizer。
    / Quantized SiLU (x * sigmoid(x)) — single input, only input quantizer.
    """
    def __init__(self, a_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantSiLU, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, x):
        if not self.quant_inference:
            return F.silu(x)
        else:
            return F.silu(self.activation_quantizer(x))



class QuantSigmoid(nn.Module):
    """量化 Sigmoid — 单输入, 只有 input quantizer。
    / Quantized Sigmoid — single input, only input quantizer.
    """
    def __init__(self, a_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantSigmoid, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, x):
        if not self.quant_inference:
            return torch.sigmoid(x)
        else:
            return torch.sigmoid(self.activation_quantizer(x))



class QuantReLU(nn.Module):
    """量化 ReLU — 单输入, 只有 input quantizer。all_positive=True 更合理。
    / Quantized ReLU — single input, only input quantizer. all_positive=True is more appropriate.
    """
    def __init__(self, a_bits=8, quant_inference=False, all_positive=True,
                 per_channel=False, batch_init=20):
        super(QuantReLU, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, x):
        if not self.quant_inference:
            return F.relu(x)
        else:
            return F.relu(self.activation_quantizer(x))



class QuantSoftmax(nn.Module):
    """量化 Softmax — 单输入, 只有 input quantizer。逐行量化/反量化后做 softmax。
    / Quantized Softmax — single input, only input quantizer. Softmax applied after row-wise quantize/dequantize.
    """
    def __init__(self, a_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantSoftmax, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, x, dim=-1):
        if not self.quant_inference:
            return torch.softmax(x, dim=dim)
        else:
            return torch.softmax(self.activation_quantizer(x), dim=dim)

class QuantMatMul(nn.Module):
    """量化矩阵乘法 — 两个输入 (A, B)。
    / Quantized matrix multiplication — two inputs (A, B).
    """
    def __init__(self, a_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantMatMul, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, A, B):
        if not self.quant_inference:
            return torch.matmul(A, B)
        else:
            Q_A = self.activation_quantizer0(A)
            Q_B = self.activation_quantizer1(B)
            return torch.matmul(Q_A, Q_B)
