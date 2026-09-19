import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .constants import INIT_STATE_FROZEN, INIT_STATE_TRAINING, INIT_STATE_UNINIT


# ********************* quantizers（量化器，量化） *********************
# 取整(ste)
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

# A(特征)量化
class DorefaActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False):
        super(DorefaActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            # signed activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        # 可学习尺度 s：把无界激活映射到 DoReFa 的固定网格区间，再反量化回原尺度。
        # 没有 s 的话，SiLU/ReLU 等 >1 的激活会全部被 clamp 到 ±1，信号坍缩。
        # 与权重路径 tanh(weight)/maxvalue 的尺度归一设计保持一致。
        self.s = nn.Parameter(torch.ones(1))
        self.init_state = INIT_STATE_UNINIT  # UNINIT/TRAINING/FROZEN

    def _set_init_state(self, value):
        # init_state 在 YOLO 网络中会被注册成持久化 tensor buffer，直接赋 int 会报错
        if isinstance(self.init_state, torch.Tensor):
            self.init_state.fill_(value)
        else:
            self.init_state = value

    # 量化/反量化
    def forward(self, activation):
        if self.init_state == INIT_STATE_UNINIT:
            # 首批用真实激活幅度初始化尺度（对称：取绝对值最大；无符号：取最大值）
            if self.all_positive:
                cur_s = activation.detach().max()
            else:
                cur_s = activation.detach().abs().max()
            self.s.data.copy_(cur_s.reshape(1).clamp(min=1e-6))
            self._set_init_state(INIT_STATE_TRAINING)
        elif self.init_state != INIT_STATE_FROZEN:
            # 训练阶段 EMA 平滑更新尺度
            if self.all_positive:
                cur_s = activation.detach().max()
            else:
                cur_s = activation.detach().abs().max()
            self.s.data.mul_(0.9).add_(
                cur_s.reshape(1).clamp(min=1e-6), alpha=0.1
            )

        # 归一化到固定网格区间 -> 量化 -> 反量化回原尺度
        bounded = activation / self.s
        if self.all_positive:
            bounded = bounded.clamp(0.0, 1.0)
        else:
            bounded = bounded.clamp(-1.0, 1.0)
        q_a = Round.apply(bounded * self.Qp).clamp(self.Qn, self.Qp) / self.Qp
        return q_a * self.s

# W(权重)量化
class DorefaWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False):
        super(DorefaWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.per_channel = per_channel
        if self.all_positive:
            # unsigned level range [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** w_bits - 1
        else:
            # signed level range [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (w_bits - 1)
            self.Qp = 2 ** (w_bits - 1) - 1

    # 量化/反量化
    def forward(self, weight):
        if self.per_channel: # 按输出通道统计tanh域的幅度
            maxvalue = torch.tanh(weight).abs().reshape(weight.size(0), -1).max(dim=1).values
            maxvalue = maxvalue.view(-1, *([1] * (weight.dim() - 1)))
        else:
            maxvalue = torch.tanh(weight).abs().max()
        if self.all_positive:
            # [0,1] map, [0, 2^b-1] levels (原dorefa行为)
            tmp = torch.tanh(weight) / maxvalue * 0.5 + 0.5
            tmp = Round.apply(tmp * self.Qp).clamp(self.Qn, self.Qp) / self.Qp
            tmp = 2 * tmp - 1
        else:
            # [-1,1] map, [-2^(b-1), 2^(b-1)-1] levels
            tmp = torch.tanh(weight) / maxvalue
            tmp = Round.apply(tmp * self.Qp).clamp(self.Qn, self.Qp) / self.Qp

        #for my opinion，need to restore the original weight range
        tmp = maxvalue * tmp
        q_w = torch.arctanh(tmp)
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
                 per_channel=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel)

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
                 per_channel=False):
        # 注意: ConvTranspose2d的参数顺序为(..., output_padding, groups, bias, dilation, padding_mode)
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                                   groups, bias, dilation, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel)

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
                 per_channel=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = DorefaActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel)

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
                 per_channel=False):
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
                 per_channel=False):
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
                 per_channel=False):
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
                 per_channel=False):
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
            return Q_A / torch.clamp(Q_C, min=1e-6)

class QuantConcat(nn.Module):
    def __init__(self,
                 a_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False):
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
    """对多个输入张量分别做激活伪量化后再 concat（每一路各持有一个激活量化器）。"""

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
            if layer_counter[0] >= 1: #第一层也量化
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
            if layer_counter[0] >= 1: #第一层也量化
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
            if layer_counter[0] >= 1: #第一层也量化
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
