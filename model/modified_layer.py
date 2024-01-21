import torch
from torch import nn, Tensor

class ModifiedConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None
                 ):
        self.momentum = 0.1
        self.running_mean = None
        self.running_var = None
        self.save_var = False

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias,
                         padding_mode,
                         device,
                         dtype)

    def forward(self, input: Tensor):
        output = super().forward(input)
        shape = len(list(output.shape))
        dim = [i for i in range(shape) if i != 1]
        cur_var = torch.var(output, dim=dim).detach().cpu()
        cur_mean = torch.mean(output, dim=dim).detach().cpu()
        if self.save_var is False:
            self.running_var = cur_var
            self.running_mean = cur_mean
            self.save_var = True
        else:
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * cur_mean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * cur_var
        return output
    
class ModifiedLinear(nn.Linear):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype=None):
        self.running_var = None
        self.running_mean = None
        self.save_var = False
        self.momentum = 0.1
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: Tensor):
        output = super().forward(input)
        shape = len(list(output.shape))
        dim = [i for i in range(shape) if i != 1]
        cur_var = torch.var(output, dim=dim).detach().cpu()
        cur_mean = torch.mean(output, dim=dim).detach().cpu()
        if self.save_var is False:
            self.running_var = cur_var
            self.running_mean = cur_mean
            self.save_var = True
        else:
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * cur_mean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * cur_var
        return output

class ModifiedAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        self.momentum = 0.1
        self.running_mean = None
        self.running_var = None
        self.save_var = False

        super().__init__(output_size=output_size)

    def forward(self, input):
        output = super().forward(input)
        shape = len(list(output.shape))
        dim = [i for i in range(shape) if i != 1]
        cur_var = torch.var(output, dim=dim).detach().cpu()
        cur_mean = torch.mean(output, dim=dim).detach().cpu()
        if self.save_var is False:
            self.running_var = cur_var
            self.running_mean = cur_mean
            self.save_var = True
        else:
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * cur_mean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * cur_var
        return output

class ModifiedMaxPool2d(nn.MaxPool2d):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 dilation=1,
                 return_indices=False,
                 ceil_mode=False):
        self.momentum = 0.1
        self.running_mean = None
        self.running_var = None
        self.save_var = False

        super().__init__(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, input):
        output = super().forward(input)
        shape = len(list(output.shape))
        dim = [i for i in range(shape) if i != 1]
        cur_var = torch.var(output, dim=dim).detach().cpu()
        cur_mean = torch.mean(output, dim=dim).detach().cpu()
        if self.save_var is False:
            self.running_var = cur_var
            self.running_mean = cur_mean
            self.save_var = True
        else:
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * cur_mean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * cur_var
        return output
