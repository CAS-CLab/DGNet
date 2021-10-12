import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class spar_loss(nn.Module):
    def __init__(self):
        super(spar_loss, self).__init__()

    def forward(self, flops_real, flops_mask, flops_ori, batch_size, den_target, lbda):
        # total sparsity
        flops_tensor, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
        # block flops
        flops_conv = flops_tensor[0:batch_size,:].mean(0).sum()
        flops_mask = flops_mask.mean(0).sum()
        flops_ori = flops_ori.mean(0).sum() + flops_conv1.mean() + flops_fc.mean()
        flops_real = flops_conv + flops_mask + flops_conv1.mean() + flops_fc.mean()
        # loss
        rloss = lbda * (flops_real / flops_ori - den_target)**2
        return rloss


class blance_loss(nn.Module):
    def __init__(self):
        super(blance_loss, self).__init__()

    def forward(self, mask_norm_s, mask_norm_c, norm_s_t, norm_c_t, batch_size, 
                den_target, gamma, p):
        norm_s = mask_norm_s
        norm_s_t = norm_s_t.mean(0)
        norm_c = mask_norm_c
        norm_c_t = norm_c_t.mean(0)
        den_s = norm_s[0:batch_size,:].mean(0) / norm_s_t
        den_c = norm_c[0:batch_size,:].mean(0) / norm_c_t
        den_tar = math.sqrt(den_target)
        bloss_s = get_bloss_basic(den_s, den_tar, batch_size, gamma, p)
        bloss_c = get_bloss_basic(den_c, den_tar, batch_size, gamma, p)
        bloss = bloss_s + bloss_c
        return bloss


def get_bloss_basic(spar, spar_tar, batch_size, gamma, p):
    # bound
    bloss_l = (F.relu(p*spar_tar-spar)**2).mean()
    bloss_u = (F.relu(spar-1+p-p*spar_tar)**2).mean()
    bloss = gamma * (bloss_l + bloss_u)
    return bloss 


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss()
        self.spar_loss = spar_loss()
        self.balance_loss = blance_loss()
    
    def forward(self, output, targets, flops_real, flops_mask, flops_ori, batch_size, 
                den_target, lbda, mask_norm_s, mask_norm_c, norm_s_t, norm_c_t,
                gamma, p):
        closs = self.task_loss(output, targets)
        sloss = self.spar_loss(flops_real, flops_mask, flops_ori, batch_size, den_target, lbda)
        bloss = self.balance_loss(mask_norm_s, mask_norm_c, norm_s_t, norm_c_t, batch_size,
                                  den_target, gamma, p)
        return closs, sloss, bloss
