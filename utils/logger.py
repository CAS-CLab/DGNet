from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import datetime
import torch
import logging
import errno
import numpy as np
from logging import handlers

__all__ = ['Logger', 'LoggerMonitor', 'savefig', 'get_loggers']


def savefig(fname, dpi=None):
    dpi = 150 if dpi is None else dpi
    plt.savefig(fname, dpi=dpi)


def plot_overlap(logger, names=None):
    names = logger.names if names is None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]


def get_loggers(args):
    """ 
        Generate loggers

    Args:
        - args : config information
    """
    # log file and checkpoint file
    checkpoint = args.checkpoint
    arch = args.arch
    if(checkpoint == ''):
        dir_name = arch + '_' + datetime.datetime.now().strftime('%m%d_%H%M')
    else:
        dir_name = checkpoint
    log_dir = os.path.join('logs', dir_name)
    checkpoint_dir = log_dir
    print('\n--------------------------------------------------------')
    if not os.path.isdir(checkpoint_dir):
        mkdir_p(log_dir)
        mkdir_p(checkpoint_dir)
        print("=> make directory '{}'".format(log_dir))
    else:
        print("=> directory '{}' exists".format(log_dir))

    train_log = Logger(os.path.join(log_dir, 'train.log'))
    test_log = Logger(os.path.join(log_dir, 'test.log'))
    config_log = Logger(os.path.join(log_dir, 'config.log'))
    if not os.path.isdir(os.path.join(log_dir, 'tb')):
        os.makedirs(os.path.join(log_dir, 'tb'))

    # msg logger
    log_level = logging.INFO
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    logging.basicConfig(level=log_level,
                        filename=os.path.join(log_dir, 'message.log'),
                        filemode='w',
                        format=fmt)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Save the config info
    for k, v in vars(args).items():
        config_log.write(content="{k} : {v}".format(k=k, v=v),
                         wrap=True,
                         flush=True)
    config_log.close()
    
    # logger initialization
    test_log.write(content="epoch\ttop1\ttop5\tloss\tcloss\trloss\tbloss\tdensity\tflops_per\tflops\t",
                   wrap=True,
                   flush=True)
    train_log.write(content="epoch\ttop1\ttop5\tloss\tcloss\trloss\tbloss",
                    wrap=True,
                    flush=True)
    return train_log, test_log, checkpoint_dir, log_dir


def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title is None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names is None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

    def write(self, content, wrap=True, flush=False, verbose=False):
        """
        write file and flush buffer to the disk
        :param content: str
        :param wrap: bool, whether to add '\n' at the end of the content
        :param flush: bool, whether to flush buffer to the disk, default=False
        :param verbose: bool, whether to print the content, default=False
        :return:
            void
        """
        if verbose:
            print(content)
        if wrap:
            content += "\n"
        self.file.write(content)
        if flush:
            self.file.flush()
            os.fsync(self.file)


class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__(self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text,
                   bbox_to_anchor=(1.05, 1),
                   loc=2,
                   borderaxespad=0.)
        plt.grid(True)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def size_to_str(torch_size):
    """Convert a pytorch Size object to a string"""
    assert isinstance(torch_size, (torch.Size, tuple, list))
    return '(' + (', ').join(['%d' % v for v in torch_size]) + ')'


def to_np(var):
    return var.data.cpu().numpy()


def norm_filters(weights, p=1):
    """Compute the p-norm of convolution filters.

    Args:
        weights - a 4D convolution weights tensor.
                  Has shape = (#filters, #channels, k_w, k_h)
        p - the exponent value in the norm formulation
    """
    assert weights.dim() == 4
    return weights.view(weights.size(0), -1).norm(p=p, dim=1)
