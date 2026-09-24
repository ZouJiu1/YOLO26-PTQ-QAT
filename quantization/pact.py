import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from quantization.dorefa import DorefaWeightQuantizer


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

class quantize_pact(Function):
    @staticmethod
    def forward(ctx, values, alpha, q_range, all_positive):
        ctx.save_for_backward(values, alpha)
        ctx.other = q_range, all_positive
        tmp = values.clone()
        tmp[tmp > alpha] = alpha
        if all_positive: # 无符号: [0, alpha]
            tmp[tmp < 0] = 0
            tmp = Round.apply(tmp * q_range / alpha) / q_range * alpha
        else: # 有符号: [-alpha, alpha]
            tmp[tmp < -alpha] = -alpha
            tmp = Round.apply((tmp + alpha) * q_range / (2 * alpha)) / q_range * 2 * alpha - alpha
        return tmp

    @staticmethod
    def backward(ctx, grad_weight):
        values, alpha = ctx.saved_tensors
        _, all_positive = ctx.other
        copyvalue = values.clone()
        if all_positive:
            double = values.clone()
            double[double < alpha] = 0
            double[double >= alpha] = 1
            grad_alpha = torch.sum(grad_weight * double).unsqueeze(dim=0)

            copyvalue[copyvalue < 0] = -1
            copyvalue[copyvalue >= alpha] = -1
            copyvalue[copyvalue >= 0] = 1 #copyvalue >= 0 & copyvalue < alpha  == copyvalue >= 0
            copyvalue[copyvalue < 0] = 0
            grad_weight = grad_weight * copyvalue
        else:
            # 上界处 dq/dalpha=1, 下界处 dq/dalpha=-1, 区间内STE
            upper = values.clone()
            upper[upper < alpha] = 0
            upper[upper >= alpha] = 1
            lower = values.clone()
            lower[lower > -alpha] = 0
            lower[lower <= -alpha] = 1
            grad_alpha = torch.sum(grad_weight * (upper - lower)).unsqueeze(dim=0)

            mask = ((copyvalue > -alpha) & (copyvalue < alpha)).to(copyvalue.dtype)
            grad_weight = grad_weight * mask
        return grad_weight, grad_alpha, None, None

# A(特征)量化
class PactActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False):
        super(PactActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.q_range = 2 ** self.a_bits - 1
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.init = 0

    # 量化/反量化
    def forward(self, activation):
        if self.init==0: #initization
            # copy_ 保持 alpha 的 (1,) 形状，便于 state_dict 重载
            self.alpha.data.copy_(activation.detach().abs().max().to(self.alpha))
            self.init = 1
        q_a = quantize_pact.apply(activation, self.alpha, self.q_range, self.all_positive)
        return q_a

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
        # self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel)

    def forward(self, inputs):
        # quant_input = self.activation_quantizer(inputs)
        # print('inputs:',inputs.size(),self.quant_inference)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv2d(inputs, quant_weight, self.bias, self.stride, self.padding, self.dilation,
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
        # self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel)

    def forward(self, inputs):
        # quant_input = self.activation_quantizer(inputs)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(inputs, quant_weight, self.bias, self.stride, self.padding, self.output_padding,
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
        # self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = DorefaWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel)

    def forward(self, inputs):
        # quant_input = self.activation_quantizer(inputs)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(inputs, quant_weight, self.bias)
        return output


class QuantAdd(nn.Module):
    def __init__(self,
                 a_bits=8,
                 quant_inference=False,
                 all_positive=False,
                 per_channel=False):
        super(QuantAdd, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer0 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer0 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer0 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer0 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
            [PactActivationQuantizer(a_bits=a_bits) for _ in range(num_inputs)]
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
        elif isinstance(child, nn.ReLU):
            Relu = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
            module._modules[name] = Relu
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

class QuantSiLU(nn.Module):
    """量化 SiLU (x * sigmoid(x)) — 单输入, 只有 input quantizer。"""
    def __init__(self, a_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantSiLU, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, x):
        if not self.quant_inference:
            return F.silu(x)
        else:
            return F.silu(self.activation_quantizer(x))



class QuantSigmoid(nn.Module):
    """量化 Sigmoid — 单输入, 只有 input quantizer。"""
    def __init__(self, a_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantSigmoid, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, x):
        if not self.quant_inference:
            return torch.sigmoid(x)
        else:
            return torch.sigmoid(self.activation_quantizer(x))



class QuantReLU(nn.Module):
    """量化 ReLU — 单输入, 只有 input quantizer。all_positive=True 更合理。"""
    def __init__(self, a_bits=8, quant_inference=False, all_positive=True,
                 per_channel=False, batch_init=20):
        super(QuantReLU, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, x):
        if not self.quant_inference:
            return F.relu(x)
        else:
            return F.relu(self.activation_quantizer(x))



class QuantSoftmax(nn.Module):
    """量化 Softmax — 单输入, 只有 input quantizer。逐行量化/反量化后做 softmax。"""
    def __init__(self, a_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantSoftmax, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, x, dim=-1):
        if not self.quant_inference:
            return torch.softmax(x, dim=dim)
        else:
            return torch.softmax(self.activation_quantizer(x), dim=dim)

class QuantMatMul(nn.Module):
    """量化矩阵乘法 — 两个输入 (A, B)。"""
    def __init__(self, a_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantMatMul, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = PactActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, A, B):
        if not self.quant_inference:
            return torch.matmul(A, B)
        else:
            Q_A = self.activation_quantizer0(A)
            Q_B = self.activation_quantizer1(B)
            return torch.matmul(Q_A, Q_B)
