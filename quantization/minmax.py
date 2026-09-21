import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


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

# min-max量化(标准非对称量化: scale/zero_point, 范围内梯度直通, 范围外梯度置0)
# 量化: q = clamp(Round(x/scale) + zero_point, qmin, qmax)   反量化: x' = (q - zero_point) * scale
class quantize_minmax(Function):
    @staticmethod
    def forward(ctx, values, scale, zero_point, qmin, qmax):
        ctx.save_for_backward(values, scale, zero_point)
        ctx.other = qmin, qmax
        q = Round.apply(values / scale) + zero_point
        q = q.clamp(min=qmin, max=qmax)
        return (q - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):
        values, scale, zero_point = ctx.saved_tensors
        qmin, qmax = ctx.other
        r_min = (qmin - zero_point) * scale
        r_max = (qmax - zero_point) * scale
        mask = ((values >= r_min) & (values <= r_max)).to(values.dtype)
        grad_values = grad_output * mask
        return grad_values, None, None, None, None

# A(特征)量化
class MinMaxActivationQuantizer(nn.Module):
    def __init__(self, a_bits, collect_method='percent', percentile=99.9, cluster_ratio=0.1, all_positive=False):
        super(MinMaxActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.q_range = 2 ** self.a_bits - 1
        self.collect_method = collect_method # 收集方式: 'percent'百分位截断 / 'cluster'间隙聚类去离群
        self.percentile = percentile         # collect_method='percent'时使用的百分位
        self.cluster_ratio = cluster_ratio   # collect_method='cluster'时间隙阈值占整体范围的比例
        self.all_positive = all_positive     # 无符号: 量化范围下限截断为0
        self.init = 0
        self.register_buffer('r_min', torch.zeros(1))
        self.register_buffer('r_max', torch.zeros(1))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    # 间隙聚类去离群点(一维版欧式聚类): 排序后按间隙分簇, 取最大簇的min/max
    def collect_cluster(self, values):
        if values.numel() <= 2:
            return values[0], values[-1]
        gaps = values[1:] - values[:-1]
        threshold = (values[-1] - values[0]) * self.cluster_ratio
        split = torch.nonzero(gaps > threshold).flatten() + 1
        if split.numel() == 0: # 间隙均小于阈值, 全部属于一个簇
            return values[0], values[-1]
        bounds = torch.cat((torch.zeros(1, device=values.device, dtype=split.dtype),
                            split,
                            torch.full((1,), values.numel(), device=values.device, dtype=split.dtype)))
        sizes = bounds[1:] - bounds[:-1]
        main = int(torch.argmax(sizes))
        return values[bounds[main]], values[bounds[main + 1] - 1]

    # 百分位截断去离群点: 取低分位为min、高分位为max
    def collect_percent(self, values):
        n = values.numel()
        low = (100.0 - self.percentile) / 100.0
        r_min = values[min(n - 1, int(low * (n - 1)))]
        r_max = values[min(n - 1, int(self.percentile / 100.0 * (n - 1)))]
        return r_min, r_max

    # 收集min/max + 量化/反量化
    def forward(self, activation):
        values = torch.sort(activation.detach().reshape(-1))[0]
        if self.collect_method == 'cluster':
            batch_min, batch_max = self.collect_cluster(values)
        else:
            batch_min, batch_max = self.collect_percent(values)
        if self.init == 0: # initization
            # copy_ 保持 buffer 的 (1,) 形状，便于 state_dict 重载
            self.r_min.copy_(batch_min.reshape(1))
            self.r_max.copy_(batch_max.reshape(1))
            self.init = 1
        else: # collect running min/max
            self.r_min.copy_(torch.min(self.r_min, batch_min).reshape(1))
            self.r_max.copy_(torch.max(self.r_max, batch_max).reshape(1))
        if self.all_positive: # 无符号: 量化范围下限截断为0
            self.r_min.clamp_(min=0)
        # 由收集到的[r_min, r_max]计算标准的scale/zero_point
        if self.all_positive: # 无符号整型层: [0, 2^b-1]
            qmin, qmax = 0, self.q_range
        else: # 有符号整型层: [-2^(b-1), 2^(b-1)-1]
            qmin, qmax = -(2 ** (self.a_bits - 1)), 2 ** (self.a_bits - 1) - 1
        eps = torch.finfo(activation.dtype).eps
        # 用全新局部张量参与前向/反传：同一量化器实例可能在一次前向中被多次调用
        # （如 SPPF 中共享的 QuantMaxPool 连串 3 次），若直接把会被 copy_ 原地改写的
        # buffer 传给 save_for_backward，后续调用会让先前调用保存的张量版本号失效。
        scale = (torch.clamp(self.r_max - self.r_min, min=eps) / (qmax - qmin)).reshape(1)
        zero_point = (qmin - self.r_min / scale).round().clamp(qmin, qmax).reshape(1)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        q_a = quantize_minmax.apply(activation, scale, zero_point, qmin, qmax)
        return q_a

# W(权重)量化
class MinMaxWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False):
        super(MinMaxWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.per_channel = per_channel
        if self.all_positive: # 无符号整型层: [0, 2^b-1]
            self.qmin, self.qmax = 0, 2 ** w_bits - 1
        else: # 有符号整型层: [-2^(b-1), 2^(b-1)-1]
            self.qmin, self.qmax = -(2 ** (w_bits - 1)), 2 ** (w_bits - 1) - 1

    # 量化/反量化
    def forward(self, weight):
        if self.per_channel: # 按输出通道(dim 0)统计min/max
            w_tmp = weight.detach().reshape(weight.size(0), -1)
            r_min = w_tmp.min(dim=1).values.view(-1, *([1] * (weight.dim() - 1)))
            r_max = w_tmp.max(dim=1).values.view(-1, *([1] * (weight.dim() - 1)))
        else:
            r_min = weight.detach().min()
            r_max = weight.detach().max()
        if self.all_positive: # 无符号: 量化范围下限截断为0
            r_min = r_min.clamp(min=0)
        # 由[r_min, r_max]计算标准的scale/zero_point
        eps = torch.finfo(weight.dtype).eps
        scale = torch.clamp(r_max - r_min, min=eps) / (self.qmax - self.qmin)
        zero_point = (self.qmin - r_min / scale).round().clamp(self.qmin, self.qmax)
        q_w = quantize_minmax.apply(weight, scale, zero_point, self.qmin, self.qmax)
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
        self.activation_quantizer = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = MinMaxWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel)

    def forward(self, inputs):
        quant_input = self.activation_quantizer(inputs)
        # print('inputs:',inputs.size(),self.quant_inference)
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
        self.activation_quantizer = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = MinMaxWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel)

    def forward(self, inputs):
        quant_input = self.activation_quantizer(inputs)
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
        self.activation_quantizer = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.weight_quantizer = MinMaxWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel)

    def forward(self, inputs):
        quant_input = self.activation_quantizer(inputs)
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
        self.activation_quantizer0 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer0 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer0 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer0 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer0 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
            [MinMaxActivationQuantizer(a_bits=a_bits) for _ in range(num_inputs)]
        )

    def forward(self, tensors, dim=1):
        return torch.cat([q(t) for q, t in zip(self.quantizers, tensors)], dim=dim)


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8, quant_inference=False,
                 all_positive=False, per_channel=False,
                 collect_method='percent', percentile=99.9, cluster_ratio=0.1):
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
            Relu = MinMaxActivationQuantizer(a_bits=a_bits, collect_method=collect_method,
                                             percentile=percentile, cluster_ratio=cluster_ratio,
                                             all_positive=all_positive)
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
                         quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel,
                         collect_method=collect_method, percentile=percentile, cluster_ratio=cluster_ratio)

def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False,
            all_positive=False, per_channel=False,
            collect_method='percent', percentile=99.9, cluster_ratio=0.1):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits,
                 quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel,
                 collect_method=collect_method, percentile=percentile, cluster_ratio=cluster_ratio)
    return model

class QuantSiLU(nn.Module):
    """量化 SiLU (x * sigmoid(x)) — 单输入, 只有 input quantizer。"""
    def __init__(self, a_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantSiLU, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

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
        self.activation_quantizer0 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)
        self.activation_quantizer1 = MinMaxActivationQuantizer(a_bits=a_bits, all_positive=all_positive)

    def forward(self, A, B):
        if not self.quant_inference:
            return torch.matmul(A, B)
        else:
            Q_A = self.activation_quantizer0(A)
            Q_B = self.activation_quantizer1(B)
            return torch.matmul(Q_A, Q_B)
