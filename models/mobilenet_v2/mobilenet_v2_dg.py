import torch
import math
import logging
from torch import nn
from prettytable import PrettyTable
from .mobilenet_v2_dg_util import ConvBNReLU_1st, InvertedResidual


__all__ = ['mobilenet_v2_dg']


def conv2d_out_dim(dim, kernel_size, padding=0, stride=1, dilation=1, ceil_mode=False):
    if ceil_mode:
        return int(math.ceil((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
    else:
        return int(math.floor((dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, 
                 round_nearest=8, in_size=(224, 224), block = InvertedResidual, **kwargs):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 1280
        h, w = in_size

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, tile
                [1, 16, 1, 1, 16],
                [6, 24, 2, 2, 8],
                [6, 32, 3, 2, 4],
                [6, 64, 4, 2, 2],
                [6, 96, 3, 1, 2],
                [6, 160, 3, 2, 2],
                [6, 320, 1, 1, 2],
            ]
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 5-element list, got {}".format(inverted_residual_setting))
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU_1st(3, input_channel, stride=2)]
        h = conv2d_out_dim(h, kernel_size=3, stride=2, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, stride=2, padding=1)
        self.flops_conv1 = torch.Tensor([3 * h * w * 3 * input_channel])
        # building inverted residual blocks
        for t, c, n, s, tile in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, 
                                      expand_ratio=t, h=h, w=w, eta=tile, **kwargs))
                h = conv2d_out_dim(h, kernel_size=3, stride=stride, padding=1)
                w = conv2d_out_dim(w, kernel_size=3, stride=stride, padding=1)
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU_1st(input_channel, self.last_channel, kernel_size=1))
        self.flops_fc = torch.Tensor([input_channel * self.last_channel *h*w])
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        self.flops_fc = self.flops_fc + torch.Tensor([num_classes * self.last_channel])
        # criterion
        self.criterion = None

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, label, den_target, lbda, gamma, p):
        batch_num, _, _, _ = x.shape
        norm1 = torch.zeros(1, batch_num+1).to(x.device)
        norm2 = torch.zeros(1, batch_num+1).to(x.device)
        flops = torch.zeros(1, batch_num+2).to(x.device)
        x, norm1, norm2, flops = self.features((x, norm1, norm2, flops))
        x = x.mean([2, 3])
        x = self.classifier(x)
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
            if isinstance(m, InvertedResidual):
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


def mobilenet_v2_dg(**kwargs):
    return MobileNetV2(block=InvertedResidual, **kwargs)
