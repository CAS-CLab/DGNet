import math
import logging
import torch
import torch.nn as nn
from prettytable import PrettyTable
from ..mask import Mask_s, Mask_c


__all__ = ['resdg20_cifar10', 'resdg32_cifar10', 'resdg56_cifar10',
           'resdg110_cifar10']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, h, w, eta=4,
                 stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # gating modules
        self.height = conv2d_out_dim(h, kernel_size=3, stride=stride, padding=1)
        self.width  = conv2d_out_dim(w, kernel_size=3, stride=stride, padding=1)
        self.mask_s = Mask_s(self.height, self.width, inplanes, eta, eta, **kwargs)
        self.mask_c = Mask_c(inplanes, planes, **kwargs)
        self.upsample = nn.Upsample(size=(self.height, self.width), mode='nearest')
        # conv 1
        self.conv1  = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # conv 2
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # misc
        self.downsample = downsample
        self.inplanes, self.planes = inplanes, planes
        self.b = eta * eta
        self.b_reduce = (eta-1) * (eta-1)
        flops_conv1_full = torch.Tensor([9 * self.height * self.width * planes * inplanes])
        flops_conv2_full = torch.Tensor([9 * self.height * self.width * planes * planes])
        # downsample flops
        self.flops_downsample = torch.Tensor([self.height*self.width*planes*inplanes]
                                            )if downsample is not None else torch.Tensor([0])
        # full flops 
        self.flops_full = flops_conv1_full + flops_conv2_full + self.flops_downsample
        # mask flops
        flops_mks = self.mask_s.get_flops()
        flops_mkc = self.mask_c.get_flops()
        self.flops_mask = torch.Tensor([flops_mks + flops_mkc])

    def forward(self, input):
        x, norm_1, norm_2, flops = input
        residual = x
        # spatial mask
        mask_s_m, norm_s, norm_s_t = self.mask_s(x) # [N, 1, h, w]
        mask_s = self.upsample(mask_s_m) # [N, 1, H, W]
        # conv 1 
        mask_c, norm_c, norm_c_t = self.mask_c(x) # [N, C_out, 1, 1]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if not self.training:
            out = out * mask_c * mask_s
        else:
            out = out * mask_c
        # conv 2 
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * mask_s
        # identity
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        # flops
        flops_blk = self.get_flops(mask_s_m, mask_s, mask_c)
        flops = torch.cat((flops, flops_blk.unsqueeze(0)))
        # norm
        norm_1 = torch.cat((norm_1, torch.cat((norm_s, norm_s_t)).unsqueeze(0)))
        norm_2 = torch.cat((norm_2, torch.cat((norm_c, norm_c_t)).unsqueeze(0)))
        return (out, norm_1, norm_2, flops)
    
    def get_flops(self, mask_s, mask_s_up, mask_c):
        s_sum = mask_s.sum((1,2,3))
        c_sum = mask_c.sum((1,2,3))
        # conv1
        flops_conv1 = 9 * self.b * s_sum * c_sum * self.inplanes
        # conv2
        flops_conv2 = 9 * self.b * s_sum * self.planes * c_sum    
        # total
        flops = flops_conv1 + flops_conv2 + self.flops_downsample.to(flops_conv1.device)
        return torch.cat((flops, self.flops_mask.to(flops.device), self.flops_full.to(flops.device)))


class ResNetCifar10(nn.Module):
    def __init__(self, depth, num_classes=10, h=32, w=32, **kwargs):
        super(ResNetCifar10, self).__init__()
        self.height, self.width = h, w
        # Model type specifies number of layers for CIFAR-10 model
        n = (depth - 2) // 6
        block = BasicBlock
        # norm
        self._norm_layer = nn.BatchNorm2d
        # conv1
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # residual blocks
        self.layer1, h, w = self._make_layer(block, 16, n, h, w, 4, **kwargs)
        self.layer2, h, w = self._make_layer(block, 32, n, h, w, 2, stride=2, **kwargs)
        self.layer3, h, w = self._make_layer(block, 64, n, h, w, 2, stride=2, **kwargs)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # flops
        self.flops_conv1 = torch.Tensor([9 * self.height * self.width * 16 * 3])
        self.flops_fc = torch.Tensor([64 * block.expansion * num_classes])
        # criterion
        self.criterion = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None and m.bias is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, h, w, tile, stride=1, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, h, w, tile,
                            stride, downsample, **kwargs))
        h = conv2d_out_dim(h, kernel_size=1, stride=stride, padding=0)
        w = conv2d_out_dim(w, kernel_size=1, stride=stride, padding=0)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, h, w, tile, **kwargs))
        return nn.Sequential(*layers), h, w

    def forward(self, x, label, den_target, lbda, gamma, p):
        batch_num, _, _, _ = x.shape
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32
        # residual blocks
        norm1 = torch.zeros(1, batch_num+1).to(x.device)
        norm2 = torch.zeros(1, batch_num+1).to(x.device)
        flops = torch.zeros(1, batch_num+2).to(x.device)
        x = self.layer1((x, norm1, norm2, flops))  # 32x32
        x = self.layer2(x)  # 16x16
        x, norm1, norm2, flops = self.layer3(x)  # 8x8
        # fc layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # flops
        flops_real = [flops[1:, 0:batch_num].permute(1, 0).contiguous(), 
                      self.flops_conv1.to(x.device), self.flops_fc.to(x.device)]
        flops_mask, flops_ori = flops[1:, -2].unsqueeze(0), flops[1:, -1].unsqueeze(0)
        # norm
        norm_s = norm1[1:, 0:batch_num].permute(1, 0).contiguous()
        norm_c = norm2[1:, 0:batch_num].permute(1, 0).contiguous()
        norm_s_t, norm_c_t = norm1[1:, -1].unsqueeze(0), norm2[1:, -1].unsqueeze(0)
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
            if isinstance(m, BasicBlock):
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


def resdg20_cifar10(**kwargs):
    """
        return a ResNet 20 object for cifar-10.
    """
    return ResNetCifar10(20, **kwargs)


def resdg32_cifar10(**kwargs):
    """
        return a ResNet 32 object for cifar-10.
    """
    return ResNetCifar10(32, **kwargs)


def resdg56_cifar10(**kwargs):
    """
        return a ResNet 56 object for cifar-10.
    """
    return ResNetCifar10(56, **kwargs)


def resdg110_cifar10(**kwargs):
    """
        return a ResNet 110 object for cifar-10.
    """
    return ResNetCifar10(110, **kwargs)
