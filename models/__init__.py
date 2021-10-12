import os
import copy
import torch
import logging
import pretrainedmodels
import torchvision.models as torch_models
import torch.backends.cudnn as cudnn
from torchvision.models.utils import load_state_dict_from_url
from . import cifar as cifar_models
from . import imagenet as imagenet_extra_models
from . import mobilenet_v2 as mobilenet_models


SUPPORTED_DATASETS = ('imagenet', 'cifar10')

TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__
                                 if not name.startswith("__")
                                 and callable(torch_models.__dict__[name]))

IMAGENET_MODEL_NAMES = copy.deepcopy(TORCHVISION_MODEL_NAMES)
IMAGENET_MODEL_NAMES.extend(sorted(name for name in imagenet_extra_models.__dict__
                                   if name.islower() and not name.startswith("__")
                                   and callable(imagenet_extra_models.__dict__[name])))

CIFAR_MODEL_NAMES = sorted(name for name in cifar_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(cifar_models.__dict__[name]))

MOBILENET_MODEL_NAMES = sorted(name for name in mobilenet_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(mobilenet_models.__dict__[name]))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(), set(IMAGENET_MODEL_NAMES + CIFAR_MODEL_NAMES
                                                    + MOBILENET_MODEL_NAMES)))

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def get_model(pretrained, dataset, arch, **kwargs):
    """Create a pytorch model based on the model architecture and dataset

    Args:
        pretrained [boolean]: True is you wish to load a pretrained model.
            Some models do not have a pretrained version.
        dataset: dataset name ('imagenet', 'cifar100', and 'cifar10' are supported)
        arch: architecture name
    """
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Dataset {} is not supported'.format(dataset))

    model = None
    cadene = False
    try:
        if dataset == 'imagenet':
            if 'mobilenet' in arch:
                model = _create_mobilenet_model(arch, pretrained, **kwargs)
            else:
                kwargs['num_classes'] = 1000
                model = _create_imagenet_model(arch, pretrained, **kwargs)
        elif dataset == 'cifar10':
            kwargs['num_classes'] = 10
            model = _create_cifar10_model(arch, pretrained, **kwargs)
    except ValueError:
        raise ValueError('Could not recognize dataset {} and model {} pair'.format(dataset, arch))

    logging.info("=> created a %s%s model with the %s dataset" % ('pretrained ' if pretrained else '',
                                                                     arch, dataset))
    logging.info('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    return model


def _create_imagenet_model(arch, pretrained, **kwargs):
    dataset = "imagenet"
    model = None
    pretrained_pytorch = pretrained == 'pytorch'
    pretrained_checkpoint = os.path.isfile(pretrained)
    if arch in TORCHVISION_MODEL_NAMES:
        try:
            model = getattr(torch_models, arch)(pretrained=pretrained_pytorch)
        except NotImplementedError:
            # In torchvision 0.3, trying to download a model that has no
            # pretrained image available will raise NotImplementedError
            if not pretrained_pytorch:
                raise
    if model is None and (arch in imagenet_extra_models.__dict__):
        model = imagenet_extra_models.__dict__[arch](**kwargs)
        if pretrained_pytorch:
            model_dict = model.state_dict()
            # get pretrained model
            if arch.startswith('resdg'):
                arch_pretrained = 'resnet' + arch.lstrip('resdg')
            else:
                raise ValueError("There is no pretrained model for {} in pytorch".format(arch))
            logging.info("=> use a pretrained %s model to initialize." % (arch_pretrained))
            pretrained_model = getattr(torch_models, arch_pretrained)(pretrained=pretrained)
            pretrained_dict = pretrained_model.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            model.load_state_dict(model_dict)
        elif pretrained_checkpoint:
            checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
            logging.info("=> loaded checkpoint (prec {:.2f})".format(checkpoint['best_acc']))
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    if model is None and (arch in pretrainedmodels.model_names):
        model = pretrainedmodels.__dict__[arch](
            num_classes=1000,
            pretrained=(dataset if pretrained else None))

    if model is None:
        error_message = ''
        if arch not in IMAGENET_MODEL_NAMES:
            error_message = "Model {} is not supported for dataset ImageNet".format(arch)
        elif pretrained:
            error_message = "Model {} (ImageNet) does not have a pretrained model".format(arch)
        raise ValueError(error_message or 'Failed to find model {}'.format(arch))
    return model


def _create_cifar10_model(arch, pretrained, **kwargs):
    try:
        model = cifar_models.__dict__[arch](**kwargs)
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    pretrained_path = pretrained
    pretrained = os.path.isfile(pretrained_path)
    # load pretrained model
    if pretrained:
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        logging.info("=> loaded checkpoint (prec {:.2f})".format(checkpoint['best_acc']))
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def _create_mobilenet_model(arch, pretrained, **kwargs):
    model = mobilenet_models.__dict__[arch](**kwargs)
    pretrained_pytorch = pretrained == 'pytorch'
    pretrained_checkpoint = os.path.isfile(pretrained)
    if pretrained_pytorch:
        model_dict = model.state_dict()
        arch_pretrained = arch[:-3]
        logging.info("=> use a pretrained %s model to initialize." % (arch_pretrained))
        if 'mobilenet_v2' in arch:
            pretrained_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=True)
        else:
            pretrained_model = mobilenet_models.__dict__[arch_pretrained](pretrained=True, **kwargs)
            pretrained_dict = pretrained_model.state_dict()
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # load the new state dict
        model.load_state_dict(model_dict)
    elif pretrained_checkpoint:
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        logging.info("=> loaded checkpoint (prec {:.2f})".format(checkpoint['best_acc']))
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        logging.info("=> no checkpoint found at '{}'".format(pretrained))
    return model
