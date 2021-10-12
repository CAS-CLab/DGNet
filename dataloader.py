import os
import logging
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def _getCifarLoader(data, dataset, batch_size, workers):
    traindir = os.path.join(data)
    valdir = os.path.join(data)
    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
    else:
        normalize = transforms.Normalize(mean=(0.507, 0.487, 0.441),
                                         std=(0.267, 0.256, 0.276))

    logging.info('=> Preparing dataset %s' % dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root=traindir,
                          train=True,
                          download=False,
                          transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=workers)

    testset = dataloader(root=valdir,
                         train=False,
                         download=False,
                         transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=workers)
    return trainloader, testloader


def _getImageNetLoader(data, batch_size, workers):
    traindir = os.path.join(data, 'train')
    valdir = os.path.join(data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
        drop_last=True)
    return train_loader, val_loader


def getDataLoader(data, dataset, batch_size, workers):
    if dataset == 'imagenet':
        return _getImageNetLoader(data, batch_size, workers)
    else:
        return _getCifarLoader(data, dataset, batch_size, workers)
