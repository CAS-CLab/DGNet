import math
import logging
import torch
import torch.nn as nn
from prettytable import PrettyTable
from ..mask import Mask_s, Mask_c

__all__ = ['resdg18', 'resdg34', 'resdg50']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, h, w, eta=8, stride=1, 
                 downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # gating modules
        self.height = conv2d_out_dim(h, kernel_size=3, stride=stride, padding=1)
        self.width  = conv2d_out_dim(w, kernel_size=3, stride=stride, padding=1)
        self.mask_s = Mask_s(self.height, self.width, inplanes, eta, eta, **kwargs)
        self.mask_c = Mask_c(inplanes, planes, **kwargs)
        self.upsample = nn.Upsample(size=(self.height, self.width), mode='nearest')
        # conv 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # conv 2
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # misc
        self.downsample = downsample
        self.inplanes, self.planes = inplanes, planes
        # flops
        flops_conv1_full = torch.Tensor([9 * self.height * self.width * planes * inplanes])
        flops_conv2_full = torch.Tensor([9 * self.height * self.width * planes * planes])
        self.flops_downsample = torch.Tensor([self.height*self.width*planes*inplanes]
                                            )if downsample is not None else torch.Tensor([0])
        self.flops_full = flops_conv1_full + flops_conv2_full + self.flops_downsample
        # mask flops
        flops_mks = self.mask_s.get_flops()
        flops_mkc = self.mask_c.get_flops()
        self.flops_mask = torch.Tensor([flops_mks + flops_mkc])

    def forward(self, input):
        x, norm_1, norm_2, flops = input
        residual = x
        mask_s_m, norm_s, norm_s_t = self.mask_s(x) # [N, 1, h, w]
        mask_c, norm_c, norm_c_t = self.mask_c(x) # [N, C_out, 1, 1]
        mask_s = self.upsample(mask_s_m) # [N, 1, H, W]        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out * mask_c * mask_s if not self.training else out * mask_c
        # conv 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * mask_s
        # identity        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        # norm
        norm_1 = torch.cat((norm_1, torch.cat((norm_s, norm_s_t)).unsqueeze(0)))
        norm_2 = torch.cat((norm_2, torch.cat((norm_c, norm_c_t)).unsqueeze(0)))
        # flops
        flops_blk = self.get_flops(mask_s, mask_c)
        flops = torch.cat((flops, flops_blk.unsqueeze(0)))
        return (out, norm_1, norm_2, flops)
    
    def get_flops(self, mask_s_up, mask_c):
        s_sum = mask_s_up.sum((1,2,3))
        c_sum = mask_c.sum((1,2,3))
        # conv1
        flops_conv1 = 9 * s_sum * c_sum * self.inplanes
        # conv2
        flops_conv2 = 9 * s_sum * c_sum * self.planes
        # total
        flops = flops_conv1 + flops_conv2 + self.flops_downsample.to(flops_conv1.device)
        return torch.cat((flops, self.flops_mask.to(flops.device), self.flops_full.to(flops.device)))


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']
    def __init__(self, inplanes, planes, h, w, eta=8, stride=1,
                 downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # spatial gating module
        self.height_1, self.width_1 = h, w
        self.height_2 = conv2d_out_dim(h, 3, dilation, stride, dilation)
        self.width_2 = conv2d_out_dim(w, 3, dilation, stride, dilation)
        self.mask_s = Mask_s(self.height_2, self.width_2, inplanes, eta, eta, **kwargs)  
        self.upsample_1 = nn.Upsample(size=(self.height_1, self.width_1), mode='nearest')
        self.upsample_2 = nn.Upsample(size=(self.height_2, self.width_2), mode='nearest')
        # conv 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.mask_c1 = Mask_c(inplanes, width, **kwargs)
        # conv 2
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.mask_c2 = Mask_c(width, width, **kwargs)
        # conv 3 
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        # misc
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.inplanes, self.width, self.planes = inplanes, width, planes * self.expansion
        # flops
        flops_conv1_full = torch.Tensor([self.height_1 * self.width_1 * width * inplanes])
        flops_conv2_full = torch.Tensor([9 * self.height_2 * self.width_2 * width * width])
        flops_conv3_full = torch.Tensor([self.height_2 * self.width_2 * width * planes*self.expansion])
        self.flops_downsample = torch.Tensor([self.height_2*self.width_2*planes*self.expansion*inplanes]
                                            ) if self.downsample is not None else torch.Tensor([0])
        self.flops_full = flops_conv1_full+flops_conv2_full+flops_conv3_full+self.flops_downsample
        # mask flops
        flops_mask_s  = self.mask_s.get_flops()
        flops_mask_c1 = self.mask_c1.get_flops()
        flops_mask_c2 = self.mask_c2.get_flops()
        self.flops_mask = torch.Tensor([flops_mask_s + flops_mask_c1 + flops_mask_c2])

    def forward(self, input):
        x, norm_1, norm_2, flops = input
        identity = x
        # spatial mask
        mask_s_m, norm_s, norm_s_t = self.mask_s(x) # [N, 1, h, w]
        mask_c1, norm_c1, norm_c1_t = self.mask_c1(x)
        mask_s1 = self.upsample_1(mask_s_m) # [N, 1, H1, W1]
        mask_s = self.upsample_2(mask_s_m) # [N, 1, H2, W2]
        # conv 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out * mask_c1 * mask_s1 if not self.training else out * mask_c1
        # conv 2
        mask_c2, norm_c2, norm_c2_t = self.mask_c2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out * mask_c2 * mask_s if not self.training else out * mask_c2
        # conv 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = out * mask_s
        # identity
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        # norm
        norm_1 = torch.cat((norm_1, torch.cat((norm_s, norm_s_t)).unsqueeze(0)))
        norm_2 = torch.cat((norm_2, torch.cat((norm_c1, norm_c1_t)).unsqueeze(0)))
        norm_2 = torch.cat((norm_2, torch.cat((norm_c2, norm_c2_t)).unsqueeze(0)))
        # flops
        flops_blk = self.get_flops(mask_s, mask_s1, mask_c1, mask_c2)
        flops = torch.cat((flops, flops_blk.unsqueeze(0)))
        return (out, norm_1, norm_2, flops)
    
    def get_flops(self, mask_s, mask_s1, mask_c1, mask_c2):
        s_sum = mask_s.sum((1,2,3))
        c1_sum, c2_sum = mask_c1.sum((1,2,3)), mask_c2.sum((1,2,3))
        # conv
        s_sum_1 = mask_s1.sum((1,2,3))
        flops_conv1 = s_sum_1 * c1_sum * self.inplanes
        flops_conv2 = 9 * s_sum * c2_sum * c1_sum
        flops_conv3 = s_sum * self.planes * c2_sum
        # total
        flops = flops_conv1+flops_conv2+flops_conv3+self.flops_downsample.to(flops_conv1.device)
        return torch.cat((flops, self.flops_mask.to(flops.device), self.flops_full.to(flops.device)))


class ResDG(nn.Module):

    def __init__(self, block, layers, h=224, w=224, num_classes=1000,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, **kwargs):
        super(ResDG, self).__init__()
        # block
        self.height, self.width = h, w
        # norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # conv1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        h = conv2d_out_dim(h, kernel_size=7, stride=2, padding=3)
        w = conv2d_out_dim(w, kernel_size=7, stride=2, padding=3)
        self.flops_conv1 = torch.Tensor([49 * h * w * self.inplanes * 3])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        h = conv2d_out_dim(h, kernel_size=3, stride=2, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, stride=2, padding=1)
        # residual blocks
        self.layer1, h, w = self._make_layer(block, 64, layers[0], h, w, 8, **kwargs)
        self.layer2, h, w = self._make_layer(block, 128, layers[1], h, w, 4, stride=2,
                                       dilate=replace_stride_with_dilation[0], **kwargs)
        self.layer3, h, w = self._make_layer(block, 256, layers[2], h, w, 2, stride=2,
                                       dilate=replace_stride_with_dilation[1], **kwargs)
        self.layer4, h, w = self._make_layer(block, 512, layers[3], h, w, 1, stride=2,
                                       dilate=replace_stride_with_dilation[2], **kwargs)
        # fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.flops_fc = torch.Tensor([512 * block.expansion * num_classes])
        # criterion
        self.criterion = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, h, w, tile, stride=1, dilate=False, **kwargs):
        norm_layer, downsample, previous_dilation = self._norm_layer, None, self.dilation
        mask_s = torch.ones(blocks)
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, h, w, tile, stride, downsample,
                            self.groups, self.base_width, previous_dilation, norm_layer, **kwargs))
        h = conv2d_out_dim(h, kernel_size=1, stride=stride, padding=0)
        w = conv2d_out_dim(w, kernel_size=1, stride=stride, padding=0)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, h, w, tile, groups=self.groups, 
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,**kwargs))
        return nn.Sequential(*layers), h, w

    def forward(self, x, label, den_target, lbda, gamma, p):
        # See note [TorchScript super()]
        batch_num, _, _, _ = x.shape
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # residual modules
        norm1 = torch.zeros(1, batch_num+1).to(x.device)
        norm2 = torch.zeros(1, batch_num+1).to(x.device)
        flops = torch.zeros(1, batch_num+2).to(x.device)
        x = self.layer1((x, norm1, norm2, flops))
        x = self.layer2(x)
        x = self.layer3(x)
        x, norm1, norm2, flops = self.layer4(x)
        # fc layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # norm and flops
        norm_s = norm1[1:, 0:batch_num].permute(1, 0).contiguous()
        norm_c = norm2[1:, 0:batch_num].permute(1, 0).contiguous()
        norm_s_t = norm1[1:, -1].unsqueeze(0)
        norm_c_t = norm2[1:, -1].unsqueeze(0)
        flops_real = [flops[1:, 0:batch_num].permute(1, 0).contiguous(), 
                      self.flops_conv1.to(x.device), self.flops_fc.to(x.device)]
        flops_mask = flops[1:, -2].unsqueeze(0)
        flops_ori  = flops[1:, -1].unsqueeze(0)
        # get outputs
        outputs = {}
        outputs["closs"], outputs["rloss"], outputs["bloss"] = self.get_loss(
                            x, label, batch_num, den_target, lbda, gamma, p,
                            norm_s, norm_c, norm_s_t, norm_c_t, 
                            flops_real, flops_mask, flops_ori)
        outputs["out"] = x
        outputs["flops_real"] = flops_real
        outputs["flops_mask"] = flops_mask
        outputs["flops_ori"] = flops_ori
        return outputs

    def set_criterion(self, criterion):
        self.criterion = criterion
        return
    
    def get_loss(self, output, label, batch_size, den_target, lbda, gamma, p,
                 mask_norm_s, mask_norm_c, norm_s_t, norm_c_t,
                 flops_real, flops_mask, flops_ori):
        closs, rloss, bloss = self.criterion(output, label, flops_real, flops_mask,
                flops_ori, batch_size, den_target, lbda, mask_norm_s, mask_norm_c,
                norm_s_t, norm_c_t, gamma, p)
        return closs, rloss, bloss

    def record_flops(self, flops_conv, flops_mask, flops_ori, flops_conv1, flops_fc):
        i = 0
        table = PrettyTable(['Layer', 'Conv FLOPs', 'Conv %', 'Mask FLOPs', 'Total FLOPs', 'Total %', 'Original FLOPs'])
        table.add_row(['layer0'] + ['{flops:.2f}K'.format(flops=flops_conv1/1024)] + [' ' for _ in range(5)])
        for name, m in self.named_modules():
            if isinstance(m, (BasicBlock, Bottleneck)):
                table.add_row([name] + ['{flops:.2f}K'.format(flops=flops_conv[i]/1024)] + ['{per_f:.2f}%'.format( 
                    per_f=flops_conv[i]/flops_ori[i]*100)] + ['{mask:.2f}K'.format(mask=flops_mask[i]/1024)] +
                    ['{total:.2f}K'.format(total=(flops_conv[i]+flops_mask[i])/1024)] + ['{per_t:.2f}%'.format(
                    per_t=(flops_conv[i]+flops_mask[i])/flops_ori[i]*100)] +
                    ['{ori:.2f}K'.format(ori=flops_ori[i]/1024)])
                i+=1
        table.add_row(['fc'] + ['{flops:.2f}K'.format(flops=flops_fc/1024)] + [' ' for _ in range(5)])
        table.add_row(['Total'] + ['{flops:.2f}K'.format(flops=(flops_conv[i]+flops_conv1+flops_fc)/1024)] + 
                    ['{per_f:.2f}%'.format(per_f=(flops_conv[i]+flops_conv1+flops_fc)/(flops_ori[i]+flops_conv1+flops_fc)*100)] + 
                    ['{mask:.2f}K'.format(mask=flops_mask[i]/1024)] + ['{total:.2f}K'.format(
                    total=(flops_conv[i]+flops_mask[i]+flops_conv1+flops_fc)/1024)] + ['{per_t:.2f}%'.format(
                    per_t=(flops_conv[i]+flops_mask[i]+flops_conv1+flops_fc)/(flops_ori[i]+flops_conv1+flops_fc)*100)] +
                    ['{ori:.2f}K'.format(ori=(flops_ori[i]+flops_conv1+flops_fc)/1024)])
        logging.info('\n{}'.format(table))


def _resdg(arch, block, layers, **kwargs):
    model = ResDG(block, layers, **kwargs)
    return model


def resdg18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resdg('resdg18', BasicBlock, [2, 2, 2, 2], **kwargs)


def resdg34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resdg('resdg34', BasicBlock, [3, 4, 6, 3], **kwargs)


def resdg50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resdg('resdg50', Bottleneck, [3, 4, 6, 3], **kwargs)
