import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import init
from distribution import DiagGaussian

"""
Modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
"""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ActorCriticConv(nn.Module):

    def __init__(self, output_dim, input_channel, input_size, std_scale, net_mode = 0, branch_dim = 512, repeat=1):
        super(ActorCriticConv, self).__init__()
        self.output_dim = output_dim
        self.input_size = input_size
        self.branch_dim = branch_dim
        self.net_mode = net_mode
        """net_mode 0: previous +bn+max
                    1: nomax + bn
                    2: nomax + nobn """
        multi = 8
        init_cnn = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        init_out = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        conv = []
        if self.net_mode == 0:
            conv += [init_cnn(nn.Conv2d(input_channel, multi, kernel_size=(7,7), stride=2, padding=1)),
                  nn.BatchNorm2d(multi),
                  nn.ReLU(True),
                  nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]
            after_size = (input_size - 5) // 2 + 1
            after_size = after_size // 2 + 1
        elif self.net_mode == 1:
            conv += [init_cnn(nn.Conv2d(input_channel, multi, kernel_size=(7,7), stride=2, padding=1)),
                     nn.BatchNorm2d(multi),
                     nn.ReLU(True)]
            after_size = (input_size - 5) // 2 + 1
        elif self.net_mode == 2:
            conv += [init_cnn(nn.Conv2d(input_channel, multi, kernel_size=(7,7), stride=2, padding=1)),
                     nn.ReLU(True)]
            after_size = (input_size - 5) // 2 + 1

        for i in range(repeat):
            if self.net_mode == 0:
                conv += [init_cnn(nn.Conv2d(multi * (2**i), multi * (2**(i+1)), kernel_size=(5,5), stride=1, padding=1)),
                          nn.BatchNorm2d(multi * (2**(i+1))),
                          nn.ReLU(True)]
                after_size = (after_size -3) // 1 + 1
                conv += [init_cnn(nn.Conv2d(multi * (2**(i+1)), multi * (2**(i+2)), kernel_size=(5,5), stride=1, padding=1)),
                          nn.BatchNorm2d(multi * (2**(i+2))),
                          nn.ReLU(True)]
                after_size = (after_size -3) // 1 + 1
            elif self.net_mode == 1:
                conv += [init_cnn(nn.Conv2d(multi * (2**i), multi * (2**(i+1)), kernel_size=(5,5), stride=2, padding=1)),
                         nn.BatchNorm2d(multi * (2**(i+1))),
                         nn.ReLU(True)]
                after_size = (after_size -3) // 2 + 1
            elif self.net_mode == 2:
                conv += [init_cnn(nn.Conv2d(multi * (2**i), multi * (2**(i+1)), kernel_size=(5,5), stride=2, padding=1)),
                         nn.ReLU(True)]
                after_size = (after_size -3) // 2 + 1

        if self.net_mode == 0:
            conv += [nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]
            after_size = after_size // 2 + 1

        if self.net_mode == 0:
            conv += [init_cnn(nn.Conv2d(multi * (2**repeat), multi * (2**(repeat+1)), kernel_size=(3,3), stride=2, padding=1)),
                  nn.BatchNorm2d(multi * (2**(repeat+1))),
                  nn.ReLU(True),
                  nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]
            after_size = (after_size -1) // 2 + 1
            after_size = after_size // 2 + 1
        elif self.net_mode == 1:
            conv += [init_cnn(nn.Conv2d(multi * (2**repeat), multi * (2**(repeat+1)), kernel_size=(3,3), stride=2, padding=1)),
                     nn.BatchNorm2d(multi * (2**(repeat+1))),
                     nn.ReLU(True)]
            after_size = (after_size -1) // 2 + 1
        elif self.net_mode == 2:
            conv += [init_cnn(nn.Conv2d(multi * (2**repeat), multi * (2**(repeat+1)), kernel_size=(3,3), stride=2, padding=1)),
                  nn.ReLU(True)]
            after_size = (after_size -1) // 2 + 1

        self.conv_layers = nn.Sequential(*conv)
        fc_size = (after_size ** 2) * multi * (2**(repeat+1))
        fc = []
        if self.net_mode == 0 or self.net_mode == 1:
            fc += [init_cnn(nn.Linear(fc_size, branch_dim)),
               nn.BatchNorm1d(branch_dim),
               nn.ReLU(True)]
        elif self.net_mode == 2:
            fc += [init_cnn(nn.Linear(fc_size, branch_dim)),
                   nn.ReLU(True)]
        self.fc_layers = nn.Sequential(*fc)
        self.critic_fc = init_out(nn.Linear(branch_dim, 1))
        self.actor_fc = init_out(nn.Linear(branch_dim, output_dim))
        self.actor_dist = DiagGaussian(output_dim, std_scale)

    def forward(self, x):
        assert len(x.size()) == 4
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        critic_out = self.critic_fc(x)
        actor_out = self.actor_dist(self.actor_fc(x))
        # critic output Q value/ actor output Normal distribution parameterised by output mean
        return critic_out, actor_out
