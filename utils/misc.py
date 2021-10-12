import math
import torch


__all__ = ['AverageMeter', 'accuracy', 'analyse_flops', 'ExpAnnealing']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def analyse_flops(flops_real, flops_mask, flops_ori, batch_size):
    def add_sum(data):
        s = data.sum().unsqueeze(0)
        out = torch.cat([data, s])
        return out
    block_flops, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
    flops_mask = flops_mask.mean(0)
    # block flops
    flops_conv = add_sum(block_flops[0:batch_size,:].mean(0))
    flops_mask = add_sum(flops_mask)
    flops_ori = add_sum(flops_ori.mean(0))
    return flops_conv, flops_mask, flops_ori, flops_conv1.mean(), flops_fc.mean()


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from 
       https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ExpAnnealing(object):
    r"""
    Args:
        T_max (int): Maximum number of iterations.
        eta_ini (float): Initial density. Default: 1.
        eta_min (float): Minimum density. Default: 0.
    """

    def __init__(self, T_ini, eta_ini=1, eta_final=0, up=False, alpha=1):
        self.T_ini = T_ini
        self.eta_final = eta_final
        self.eta_ini = eta_ini
        self.up = up
        self.last_epoch = 0
        self.alpha = alpha

    def get_lr(self, epoch):
        if epoch < self.T_ini:
            return self.eta_ini
        elif self.up:
            return self.eta_ini + (self.eta_final-self.eta_ini) * (1-
                   math.exp(-self.alpha*(epoch-self.T_ini)))
        else:
            return self.eta_final + (self.eta_ini-self.eta_final) * math.exp(
                   -self.alpha*(epoch-self.T_ini))

    def step(self):
        self.last_epoch += 1
        return self.get_lr(self.last_epoch)
