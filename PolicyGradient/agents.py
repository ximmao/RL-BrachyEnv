import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from MCworlds import MCenv, MCDataGetter, FeatureTransformer
from models import ActorCriticConv

"""
Modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
"""

class PPOAgent():
    def __init__(self, model, num_dwell, horizon, max_grad_norm, coef_value_loss,
                       coef_entropy_loss, clip_param, training_batch, num_epochs, constraint_idx,
                       use_clipped_value_loss = True,  data_folder = "./pg_data"):
        self.prefix = data_folder
        self.num_dwell = num_dwell
        self.horizon = horizon
        self.max_grad_norm = max_grad_norm
        self.ppo_epoch = num_epochs
        self.world = MCenv(data_getter = MCDataGetter(data_folder), n_d = num_dwell, H = horizon, constraint_idx = constraint_idx)
        self.transformer = FeatureTransformer(data_folder)
        self.model = model
        self.coef_value_loss = coef_value_loss
        self.coef_entropy_loss = coef_entropy_loss
        self.training_batch = training_batch
        self.clip_param = clip_param
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        #self.std_scale = std_scale

    def sample_actions(self, inputs, masks, device, deterministic=False):
        value, actor_dist = self.model(self.transformer.get_nn_input(inputs).to(device))

        if deterministic:
            action = actor_dist.mode()
        else:
            action = actor_dist.sample()

        action_log_probs = actor_dist.log_probs(action)
        dist_entropy = actor_dist.entropy().mean()
        #print(action)

        return value, action, action_log_probs

    def get_value(self, inputs, masks, device):
        value, _ = self.model(self.transformer.get_nn_input(inputs).to(device))
        return value

    def evaluate_actions(self, inputs, masks, action, device):
        value, actor_dist = self.model(self.transformer.get_nn_input(inputs).to(device))

        action_log_probs = actor_dist.log_probs(action)
        dist_entropy = actor_dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    def update(self, rollouts, device):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, mini_batch_size = self.training_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample
                #print(obs_batch.size())

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.evaluate_actions(obs_batch, masks_batch, actions_batch, device)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()

                self.optimizer.zero_grad()
                (value_loss * self.coef_value_loss + action_loss -
                 dist_entropy * self.coef_entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.training_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
