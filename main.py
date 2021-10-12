import os
import time
import math
import random
import shutil
import logging
import torch
import torch.nn as nn
import models
import numpy as np
from options import parser
from collections import OrderedDict
from dataloader import getDataLoader
from utils import *
from regularization import *

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print('Parameters:')
for key, value in state.items():
    print('    {key} : {value}'.format(key=key, value=value))

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True

best_acc = 0  # best test accuracy

# Get loggers and save the config information
train_log, test_log, checkpoint_dir, log_dir = get_loggers(args)

def main():
    global best_acc, train_log, test_log, checkpoint_dir, log_dir
    # create model
    logging.info("=" * 89)
    logging.info("=> creating model '{}'".format(args.arch))
    model = models.get_model(pretrained=args.pretrained, dataset = args.dataset,
                             arch = args.arch, bias=args.bias)
    # define loss function (criterion) and optimizer
    criterion = Loss()
    model.set_criterion(criterion)
    # Data loader
    trainloader, testloader = getDataLoader(args.data, args.dataset, args.batch_size,
                                            args.workers)
    # to cuda
    if torch.cuda.is_available() and args.gpu_id != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        model = torch.nn.DataParallel(model).cuda()
        logging.info('=> running the model on gpu{}.'.format(args.gpu_id))
    else:
        logging.info('=> running the model on cpu.')
    # define optimizer
    param_dict = dict(model.named_parameters())
    params = []
    BN_name_pool = []
    for m_name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            BN_name_pool.append(m_name + '.weight')
            BN_name_pool.append(m_name + '.bias')
    for key, value in param_dict.items():
        if (key in BN_name_pool and 'mobilenet' in args.arch) or 'mask' in key:
            params += [{'params': [value], 'lr': args.learning_rate, 'weight_decay': 0.}]
        else:
            params += [{'params':[value]}]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate,weight_decay=args.weight_decay,
                                momentum=args.momentum, nesterov=True)
    p_anneal = ExpAnnealing(0, 1, 0, alpha=args.alpha)
    # ready
    logging.info("=" * 89)
    # Evaluate
    if args.evaluate:
        logging.info('Evaluate model')
        top1, top5 = validate(testloader, model, criterion, 0, use_cuda, 
                              (args.lbda, 0), args.den_target)
        logging.info('Test Acc (Top-1): %.2f, Test Acc (Top-5): %.2f' % (top1, top5))
        return
    # training
    logging.info('\n Train for {} epochs'.format(args.epochs))
    train_process(model, args.epochs, testloader, trainloader, criterion, optimizer,
                  use_cuda, args.lbda, args.gamma, p_anneal, checkpoint_dir, args.den_target)
    train_log.close()
    test_log.close()
    logging.info('Best acc: {}'.format(best_acc))
    return


def train_process(model, total_epochs, testloader, trainloader, criterion, optimizer,
                  use_cuda, lbda, gamma, p_anneal, checkpoint_dir, den_target):
    global best_acc
    for epoch in range(total_epochs):
        p = p_anneal.get_lr(epoch)
        # get target density
        state['den_target'] = den_target
        # update lr
        adjust_learning_rate(optimizer, epoch=epoch)
        # Training
        train(trainloader, model, criterion, optimizer, epoch, use_cuda, (lbda, gamma),
              den_target, p)
        test_acc, _ = validate(testloader, model, criterion, epoch, use_cuda,
                                          (lbda, gamma), den_target, p=p)
        # save checkpoint
        if checkpoint_dir is not None:
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            model_dict = model.module.state_dict() if use_cuda else model.state_dict()
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model_dict,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                },
                is_best=is_best,
                checkpoint_dir=checkpoint_dir)
    return


def train(train_loader, model, criterion, optimizer, epoch, use_cuda, param, 
          den_target, p):
    lbda, gamma = param
    # switch to train mode
    model.train()
    logging.info("=" * 89)

    batch_time, data_time, closses, rlosses, blosses, losses, top1, top5 = getAvgMeter(8)
    
    end = time.time()
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (x, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # get inputs
        if use_cuda:
            x, targets = x.cuda(), targets.cuda()
        x, targets = torch.autograd.Variable(x), torch.autograd.Variable(targets)
        batch_size = x.size(0)
        # inference
        inputs = {"x": x, "label": targets, "den_target": den_target, "lbda": lbda,
                  "gamma": gamma, "p": p}
        outputs= model(**inputs)
        loss = outputs["closs"].mean() + outputs["rloss"].mean() + outputs["bloss"].mean()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs["out"].data, targets.data, topk=(1, 5))
        closses.update(outputs["closs"].mean().item(), batch_size)
        rlosses.update(outputs["rloss"].mean().item(), batch_size)
        blosses.update(outputs["bloss"].mean().item(), batch_size)
        losses.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)
        # compute gradient and do SGD step
        optimizer.zero_grad()            
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | '.format(
            batch=batch_idx+1, size=len(train_loader), data=data_time.val, bt=batch_time.val,
            )+'Total: {total:} | (C,R,B)Loss: {closs:.2f}, {rloss:.2f}, {bloss:.2f}'.format(
            total=bar.elapsed_td, closs=closses.avg, rloss=rlosses.avg, bloss=blosses.avg,
            )+' | Loss: {loss:.2f} | top1: {top1:.2f} | top5: {top5:.2f}'.format(top1=top1.avg,
            top5=top5.avg, loss=losses.avg)
        bar.next()
    bar.finish()
    train_log.write(content="{epoch}\t{top1.avg:.4f}\t{top5.avg:.4f}\t{loss.avg:.4f}\t"
                            "{closs.avg:.4f}\t{rloss.avg:.4f}\t{bloss.avg:.4f}".format(
                            epoch=epoch, top1=top1, top5=top5,loss=losses, closs=closses,
                            rloss=rlosses, bloss=blosses),
                    wrap=True, flush=True)
    return


def validate(val_loader, model, criterion, epoch, use_cuda, param, den_target, p=0):
    global log_dir
    lbda, gamma = param
    # switch to evaluate mode
    model.eval()
    logging.info("=" * 89)

    (batch_time, data_time, closses, rlosses, blosses, losses, 
                                            top1, top5, block_flops)= getAvgMeter(9)

    with torch.no_grad():
        end = time.time()
        bar = Bar('Processing', max=len(val_loader))
        for batch_idx, (x, targets) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # get inputs
            if use_cuda:
                x, targets = x.cuda(), targets.cuda(non_blocking=True)
            x, targets = torch.autograd.Variable(x), torch.autograd.Variable(targets)
            batch_size = x.size(0)
            # inference
            inputs = {"x": x, "label": targets, "den_target": den_target, "lbda": lbda,
                  "gamma": gamma, "p": p}
            outputs= model(**inputs)
            loss = outputs["closs"].mean() + outputs["rloss"].mean() + outputs["bloss"].mean()
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs["out"].data, targets.data, topk=(1, 5))
            closses.update(outputs["closs"].mean().item(), batch_size)
            rlosses.update(outputs["rloss"].mean().item(), batch_size)
            blosses.update(outputs["bloss"].mean().item(), batch_size)
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # get flops
            flops_real = outputs["flops_real"]
            flops_mask = outputs["flops_mask"]
            flops_ori  = outputs["flops_ori"]
            flops_conv, flops_mask, flops_ori, flops_conv1, flops_fc = analyse_flops(
                                              flops_real, flops_mask, flops_ori, batch_size)
            block_flops.update(flops_conv, batch_size)
            # plot progress
            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:}'.format(
                batch=batch_idx+1, size=len(val_loader), bt=batch_time.avg, total=bar.elapsed_td
                )+' |  (C,R,B)Loss: {closs:.2f}, {rloss:.2f}, {bloss:.2f}'.format(
                closs=closses.avg, rloss=rlosses.avg, bloss=blosses.avg,
                )+' | Loss: {loss:.2f} | top1: {top1:.2f} | top5: {top5:.2f}'.format(
                top1=top1.avg, top5=top5.avg,  loss=losses.avg)
            bar.next()
        bar.finish()
        # log
        if use_cuda:
            model.module.record_flops(block_flops.avg, flops_mask, flops_ori, flops_conv1, flops_fc)
        else:
            model.record_flops(block_flops.avg, flops_mask, flops_ori, flops_conv1, flops_fc)
        flops = (block_flops.avg[-1]+flops_mask[-1]+flops_conv1.mean()+flops_fc.mean())/1024
        flops_per = (block_flops.avg[-1]+flops_mask[-1]+flops_conv1.mean()+flops_fc.mean())/(
                         flops_ori[-1]+flops_conv1.mean()+flops_fc.mean())*100
        test_log.write(content="{epoch}\t{top1.avg:.4f}\t{top5.avg:.4f}\t{loss.avg:.4f}\t"
                               "{closs.avg:.4f}\t{rloss.avg:.4f}\t{bloss.avg:.4f}\t"
                               "{flops_per:.2f}%\t{flops:.2f}K\t".format(epoch=epoch, top1=top1,
                               top5=top5, loss=losses, closs=closses, rloss=rlosses,
                               bloss=blosses, flops_per=flops_per, flops=flops),
                       wrap=True, flush=True)
    return (top1.avg, top5.avg)


def getAvgMeter(num):
    return [AverageMeter() for _ in range(num)]


def adjust_learning_rate(optimizer, epoch):
    global state
    if args.lr_mode == 'cosine':
        lr = 0.5*args.learning_rate*(1+math.cos(math.pi*float(epoch)/float(args.epochs)))
        state['learning_rate'] = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.lr_mode == 'step':
        if epoch in args.schedule:
            state['learning_rate'] *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']
    else:
        raise NotImplementedError('can not support lr mode {}'.format(args.lr_mode))
    logging.info("\nEpoch: {epoch:3d} | learning rate = {lr:.6f}".format(
                epoch=epoch, lr=state['learning_rate']))


def save_checkpoint(state,
                    is_best,
                    filename='checkpoint.pth.tar',
                    checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, filename)
    torch.save(state, filename, pickle_protocol=4)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar'))

if __name__ == "__main__":
    main()
