import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .constants import INIT_STATE_FROZEN, INIT_STATE_UNINIT


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

class FunLSQ(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel=False):
        #根据论文里LEARNED STEP SIZE QUANTIZATION第2节的公式
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        #根据论文里LEARNED STEP SIZE QUANTIZATION第2.1节
        #分为三部分：位于量化区间的、小于下界的、大于上界的
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float() #bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float() #bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller -bigger #得到位于量化区间的index
        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * Round.apply(q_w) - between * q_w)*grad_weight * g)
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp + 
                between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0) #?
        #在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

# A(特征)量化
class LSQActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False, batch_init = 20):
        #activations 没有per-channel这个选项的
        super(LSQActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        # self.s = torch.nn.Parameter(torch.ones(0.01), requires_grad=True)
        # self.register_parameter('Ascale', self.s)
        self.init_state = INIT_STATE_UNINIT

    # 量化/反量化
    def forward(self, activation):
        '''
        For this work, each layer of weights and each layer of activations has a distinct step size, represented
as an fp32 value, initialized to 2h|v|i/√OP , computed on either the initial weights values or the first
batch of activations, respectively
        '''
        #V1
        if not hasattr(self, "g"):
            # g 只依赖张量大小，载入 checkpoint 后 init_state 会被冻结，
            # 不能只放在 init_state==INIT_STATE_UNINIT 分支里初始化
            self.g = 1.0/math.sqrt(activation.numel() * self.Qp)
        if self.init_state==INIT_STATE_UNINIT:
            cur_s = torch.mean(torch.abs(activation.detach()))*2/(math.sqrt(self.Qp))
            self.s.data.copy_(cur_s.to(self.s))
            self.init_state += 1
        elif self.init_state<self.batch_init:
            cur_s = torch.mean(torch.abs(activation.detach()))*2/(math.sqrt(self.Qp))
            self.s.data.mul_(0.9).add_(cur_s.to(self.s), alpha=0.1)
            self.init_state += 1
        elif self.init_state==self.batch_init:
            # self.s = torch.nn.Parameter(self.s)
            self.init_state += 1
        if self.a_bits == 32:
            output = activation
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            # print(self.s, self.g)
            q_a = FunLSQ.apply(activation, self.s, self.g, self.Qn, self.Qp)

            # alpha = grad_scale(self.s, g)
            # q_a = Round.apply((activation/alpha).clamp(Qn, Qp)) * alpha
        return q_a

# W(权重)量化
class LSQWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False, batch_init = 20, num_channels=None):
        super(LSQWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** w_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (w_bits - 1)
            self.Qp = 2 ** (w_bits - 1) - 1
        self.per_channel = per_channel
        scale_shape = (num_channels,) if per_channel and num_channels is not None else (1,)
        self.s = torch.nn.Parameter(torch.ones(scale_shape), requires_grad=True)
        # self.register_parameter('Wscale', self.s)
        self.init_state = INIT_STATE_UNINIT

    # 量化/反量化
    def forward(self, weight):
        if not hasattr(self, "g"):
            self.g = 1.0/math.sqrt(weight.numel() * self.Qp)
        if self.init_state==INIT_STATE_UNINIT:
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                self.s.data = torch.mean(torch.abs(weight_tmp), dim=1)*2/(math.sqrt(self.Qp))
            else:
                cur_s = torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.Qp))
                self.s.data.copy_(cur_s.to(self.s))
            self.init_state += 1
        elif self.init_state<self.batch_init:
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                self.s.data = 0.9*self.s.data + 0.1*torch.mean(torch.abs(weight_tmp), dim=1)*2/(math.sqrt(self.Qp))
            else:
                cur_s = torch.mean(torch.abs(weight.detach()))*2/(math.sqrt(self.Qp))
                self.s.data.mul_(0.9).add_(cur_s.to(self.s), alpha=0.1)
            self.init_state += 1
        elif self.init_state==self.batch_init:
            # self.s = torch.nn.Parameter(self.s)
            self.init_state += 1
        if self.w_bits == 32:
            output = weight
        elif self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1
        else:
            # print(self.s, self.g)
            w_q = FunLSQ.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)

            # alpha = grad_scale(self.s, g)
            # w_q = Round.apply((weight/alpha).clamp(Qn, Qp)) * alpha
        return w_q

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
                 batch_init = 20):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive,batch_init = batch_init)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel,batch_init = batch_init, num_channels=out_channels)

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
                 batch_init = 20):
        # 注意: ConvTranspose2d的参数顺序为(..., output_padding, groups, bias, dilation, padding_mode)
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                                   groups, bias, dilation, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive,batch_init = batch_init)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel,batch_init = batch_init, num_channels=in_channels)

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
                 batch_init = 20):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive,batch_init = batch_init)
        self.weight_quantizer = LSQWeightQuantizer(w_bits=w_bits, all_positive=all_positive, per_channel=per_channel,batch_init = batch_init, num_channels=out_features)

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
                 batch_init=20):
        super(QuantAdd, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)
        self.activation_quantizer1 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)

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
                 batch_init=20):
        super(QuantSub, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)
        self.activation_quantizer1 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)

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
                 batch_init=20):
        super(QuantMultiply, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)
        self.activation_quantizer1 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)

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
                 batch_init=20):
        super(QuantDiv, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)
        self.activation_quantizer1 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)

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
                 per_channel=False,
                 batch_init=20):
        super(QuantConcat, self).__init__()
        self.quant_inference = quant_inference
        self.activation_quantizer0 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)
        self.activation_quantizer1 = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)

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
                per_channel=False,
                batch_init=20):
        super(QuantMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQActivationQuantizer(a_bits=a_bits, all_positive=all_positive, batch_init=batch_init)

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

    def __init__(self, num_inputs, a_bits=8, batch_init=20):
        super().__init__()
        self.quant_inference = True
        self.quantizers = nn.ModuleList(
            [LSQActivationQuantizer(a_bits=a_bits, batch_init=batch_init) for _ in range(num_inputs)]
        )

    def forward(self, tensors, dim=1):
        return torch.cat([q(t) for q, t in zip(self.quantizers, tensors)], dim=dim)


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8,
                 quant_inference=False, all_positive=False, per_channel=False, 
                 batch_init = 20):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
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
                                                                quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
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
                                                                quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] >= 1: #第一层也量化
                if child.bias is not None:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=True, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=False, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        else:
            add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits,
                         quant_inference=quant_inference, all_positive=all_positive, per_channel=per_channel, batch_init = batch_init)


def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False,
            all_positive=False, per_channel=False, batch_init = 20):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits,
                 quant_inference=quant_inference, all_positive=all_positive, 
                 per_channel=per_channel, batch_init = batch_init)
    return model
