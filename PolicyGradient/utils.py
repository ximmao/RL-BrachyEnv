import glob
import os

import torch
import torch.nn as nn

"""
utility method directly from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
"""

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reset_lr(optimizer, initial_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr

def B(value):
    # bonus: MBIE-EB
    #print(value + 0.01)
    return torch.sqrt(1./(value + 0.01)).float()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
