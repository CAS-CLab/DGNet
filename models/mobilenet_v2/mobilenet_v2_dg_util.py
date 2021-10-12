import torch
import math
from torch import nn
from ..mask import Mask_s, Mask_c


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class ConvBNReLU_1st(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU_1st, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, input):
        x, norm_1, norm_2, flops = input
        x = super(ConvBNReLU_1st, self).forward(x)
        return x, norm_1, norm_2, flops


class Sequential_DG(nn.Sequential):
    def __init__(self, layers):
        super(Sequential_DG, self).__init__(*layers)
        self._module_num = len(layers)

    def forward(self, input):
        x, mask_c, mask_s1, mask_s2 = input
        i = 0
        for module in self._modules.values():
            if self.training:
                if i == self._module_num-2:
                    x = x * mask_c
                x = module(x)
            else:
                if i == 0:
                    x = module(x) * mask_s1                
                elif i == self._module_num-2:
                    x = x * mask_c * mask_s2
                    x = module(x)
                else:
                    x = module(x)
            i += 1
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, h, w, eta, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.height = conv2d_out_dim(h, kernel_size=3, stride=stride, padding=1)
        self.width  = conv2d_out_dim(w, kernel_size=3, stride=stride, padding=1)
        self.spatial = self.height * self.width
        self.expand = expand_ratio == 1
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        if self.use_res_connect:
            self.conv = Sequential_DG(layers)
            # channel mask
            self.mask_c = Mask_c(inp, hidden_dim, **kwargs)
            flops_mkc = self.mask_c.get_flops()
            # spatial mask
            self.mask_s = Mask_s(self.height, self.width, inp, eta, eta, **kwargs)
            self.upsample = nn.Upsample(size=(h, w), mode='nearest')
            flops_mks = self.mask_s.get_flops()
        else:
            self.conv = nn.Sequential(*layers)
            flops_mkc, flops_mks = 0, 0
            self.norm_c_t = torch.Tensor([hidden_dim])
            self.norm_s_t = torch.Tensor([self.spatial])
        # misc
        self.inp, self.oup = inp, oup
        self.hidden_dim = hidden_dim
        # flops
        flops_dw_full = torch.Tensor([9 * self.spatial * hidden_dim])
        flops_pw_full = torch.Tensor([self.spatial * hidden_dim * oup])
        self.flops_full = flops_dw_full + flops_pw_full
        if expand_ratio != 1:
            self.flops_full = self.flops_full + torch.Tensor([h * w * hidden_dim * inp])
            self.upsample1 = nn.Upsample(size=(h, w), mode='nearest')
        # mask flops        
        self.flops_mask = torch.Tensor([flops_mks + flops_mkc])

    def forward(self, input):
        if not self.use_res_connect:
            x, norm_1, norm_2, flops = input
            x = self.conv(x)
            norm_s = torch.ones((x.shape[0], self.spatial), device=x.device).sum(1)
            norm_c = torch.ones((x.shape[0], self.hidden_dim), device=x.device).sum(1)
            norm_1 = torch.cat((norm_1, torch.cat((norm_s, self.norm_s_t.to(x.device))).unsqueeze(0)))
            norm_2 = torch.cat((norm_2, torch.cat((norm_c, self.norm_c_t.to(x.device))).unsqueeze(0)))
            flops_blk = torch.cat((torch.ones(x.shape[0])*self.flops_full, self.flops_mask, self.flops_full)).to(flops.device)
            flops = torch.cat((flops, flops_blk.unsqueeze(0)))
            return (x, norm_1, norm_2, flops)
        else:
            x_in, norm_1, norm_2, flops = input
            # channel mask
            mask_c, norm_c, norm_c_t = self.mask_c(x_in) # [N, C_out, 1, 1]
            # spatial mask
            mask_s_m, norm_s, norm_s_t = self.mask_s(x_in) # [N, 1, h, w]
            mask_s1 = self.upsample1(mask_s_m) # [N, 1, H1, W1]
            mask_s = self.upsample(mask_s_m) # [N, 1, H, W]
            x = self.conv((x_in, mask_c, mask_s1, mask_s))            
            x = x * mask_s
            # norm
            norm_1 = torch.cat((norm_1, torch.cat((norm_s, norm_s_t)).unsqueeze(0)))
            norm_2 = torch.cat((norm_2, torch.cat((norm_c, norm_c_t)).unsqueeze(0)))
            # flops
            flops_blk = self.get_flops(mask_c, mask_s)
            flops = torch.cat((flops, flops_blk.unsqueeze(0)))
            return (x+x_in, norm_1, norm_2, flops)

    def get_flops(self, mask_c, mask_s_up):
        s_sum = mask_s_up.sum((1,2,3))
        c_sum = mask_c.sum((1,2,3))
        # convdw
        flops_dw = 9 * s_sum * c_sum
        # convpw
        flops_pw = s_sum * c_sum * self.oup
        # conv1x1
        flops = flops_dw + flops_pw
        if not self.expand:
            mask_s_1 = self.upsample1(mask_s_up)
            flops = flops + mask_s_1.sum((1,2,3)) * c_sum * self.inp
        # total
        return torch.cat((flops, self.flops_mask.to(flops.device), self.flops_full.to(flops.device)))
