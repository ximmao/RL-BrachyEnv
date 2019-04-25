import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import B

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

"""
Modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
"""
#
# Standardize distribution interfaces
#

FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_outputs, std_scale):
        super(DiagGaussian, self).__init__()
        #self.logstd = nn.Parameter(torch.ones(num_outputs)* 2)
        self.logstd = nn.Parameter(torch.ones(num_outputs)* std_scale)
        #self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = x

        action_logstd = self.logstd.unsqueeze(0).expand_as(action_mean)
        return FixedNormal(action_mean, action_logstd.exp())

ActionNormal = torch.distributions.Normal
log_prob_normal = ActionNormal.log_prob
ActionNormal.probs_sum = lambda self, actions: log_prob_normal(self, actions).exp().sum()

MultiNormal = torch.distributions.MultivariateNormal

class PseudoCnt():
    def __init__(self, num_dwell, logstd, scaler, device, v_min = 1.0, v_max = 200.0):
        super(PseudoCnt, self).__init__()
        #self.discretized_pdf = np.ones((num_dwell, bins_per_pnt)) * (1 / bins_per_pnt)
        self.pdf_mean = torch.zeros(num_dwell)
        self.pdf_cov = torch.eye(num_dwell) * logstd
        self.pdf_mean = self.pdf_mean.to(device)
        self.pdf_cov = self.pdf_cov.to(device)
        #self.bins = bins_per_pnt
        #self.v_min = v_min
        #self.v_max = v_max
        #self.stepsize_floor = (self.v_max - self.v_min) // self.discretized_pdf.shape[1]
        self.logstd = logstd
        self.scaler = scaler
        # for multivariate normal, the scaler for peak to reach 1.0 should be scaler of single variate normal ** num_dim
        self.num_dwell = num_dwell
        self.p_theta = MultiNormal(self.pdf_mean, self.pdf_cov)
        self.curr_step = 0.

    def add_pcnt(self, rollouts, device):
        (num_s, num_p, d) = rollouts.obs[:-1].size()
        step = self.curr_step
        for step_s in range(num_s):
            for step_p in range(num_p):
                step += 1.0
                if not rollouts.bad_masks[step_s + 1, step_p].item() == 0.0:
                    prob_theta = self.p_theta.log_prob(rollouts.obs[step_s, step_p, :].to(device)).exp()
                    delta_vec = rollouts.obs[step_s, step_p, :] - self.pdf_mean
                    assert delta_vec.unsqueeze(0).size() == (1, self.num_dwell)
                    assert delta_vec.unsqueeze(1).size() == (self.num_dwell, 1)
                    self.pdf_cov = (step/(step+1.))*self.pdf_cov + (step/((step+1.)*(step+1.)))*torch.mul(delta_vec.unsqueeze(0),delta_vec.unsqueeze(1))
                    self.pdf_mean = self.pdf_mean + (1./(step+1.))*delta_vec
                    self.pdf_mean = self.pdf_mean.to(device)
                    self.pdf_cov = self.pdf_cov.to(device)
                    self.p_theta = MultiNormal(self.pdf_mean, self.pdf_cov)
                    prob_theta_prime = self.p_theta.log_prob(rollouts.obs[step_s, step_p, :]).exp()
                    # if not retraining can remove the max()
                    #pseudocnt = prob_theta * (1 - prob_theta_prime) / max((prob_theta_prime - prob_theta), 1e-1000)
                    pseudocnt = prob_theta * (1 - prob_theta_prime) / (prob_theta_prime - prob_theta)
                    rollouts.rewards[step_s, step_p] += B(pseudocnt) * self.scaler
        self.curr_step = step
