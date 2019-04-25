import numpy as np
import os
import glob
import time
from collections import deque
import matplotlib.pyplot as plt
import torch

from models import ActorCriticConv
from agents import PPOAgent
from storage import MCRolloutStorage
from utils import update_linear_schedule
from distribution import PseudoCnt
from pg_data.Utils import matrix_size

"""
Modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
"""


if __name__ =="__main__":
    num_processes = 1
    num_steps = 1000
    num_dwell = 10
    horizon = 100
    coef_value_loss = 0.5
    coef_entropy_loss = 0.01
    max_grad_norm = 0.5
    ppo_epochs = 4
    clip_param = 0.2
    training_batch = 32
    input_size = matrix_size[0]
    input_channel = 1 + 1
    total_supposed_steps = 100000
    initial_lr = 1e-4
    use_linear_lr_decay = True
    use_gae = True
    gamma = 0.99
    gae_lambda = 0.95
    use_proper_time_limits = True
    model_save_dir = "./trained_models"
    model_name = "PPO"
    constraint = 0
    save_model = False
    show_plot = True
    load_pretrain_model = False
    std_scale = 3.0
    sigma = 3.0
    beta = 1.0
    reset_pcnt = True
    using_pcnt = True
    bn_flag = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    actor_critic = ActorCriticConv(num_dwell, input_channel, input_size, std_scale, bn_flag)
    actor_critic.to(device)

    agent = PPOAgent(actor_critic, num_dwell, horizon, max_grad_norm, coef_value_loss,
                     coef_entropy_loss, clip_param, training_batch, ppo_epochs, constraint, data_folder = "./pg_data")


    if load_pretrain_model:
        print("Loading trained model from ", model_save_dir, '{}-all-{}.ckpt'.format(total_supposed_steps, model_name))
        model_path = os.path.join(model_save_dir, '{}-all-{}.ckpt'.format(total_supposed_steps, model_name))
        state_dict = torch.load(model_path)
        agent.model.load_state_dict(state_dict, strict=False)

    if reset_pcnt:
        pcnt_dist = PseudoCnt(num_dwell, sigma, beta, device)
    else:
        pcnt_dist = prev_pcnt_dist

    rollouts = MCRolloutStorage(num_steps, num_processes, num_dwell, num_dwell)

    print("start training")
    obs = agent.world.reset()
    print("initial", obs)
    rollouts.obs[0].copy_(torch.from_numpy(obs))
    rollouts.to(device)

    start = time.time()
    num_updates = int(total_supposed_steps) // num_steps // num_processes
    all_return = []
    all_length = []
    for j in range(num_updates):
        print("runs", j+1)

        if use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(agent.optimizer, j, num_updates, initial_lr)
        cumul_return = []
        episo_length = []
        episode_rewards = []
        episode_lengths = 0.0
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                agent.model.eval()
                value, action, action_log_prob = agent.sample_actions(rollouts.obs[step], rollouts.masks[step], device)

            # Obser reward and next obs
            action_world = action.cpu().numpy().reshape(-1)
            obs, reward, done, success = agent.world.step(action_world)
            episode_lengths += 1.0
            episode_rewards.append(reward)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done else [1.0]])
            bad_masks = torch.FloatTensor(
                [[0.0] if done and not success else [1.0]])
            reward = torch.FloatTensor([[reward]])
            if done:
                # resample the start state were finished
                cumul_return.append(np.sum(episode_rewards))
                episo_length.append(episode_lengths)
                episode_rewards = []
                episode_lengths = 0.0
                obs = agent.world.reset()
            obs = torch.from_numpy(obs)
            rollouts.insert(obs, action,
                            action_log_prob, value, reward, masks, bad_masks)
        with torch.no_grad():
            agent.model.eval()
            next_value = agent.get_value(rollouts.obs[-1], rollouts.masks[-1], device)

        if using_pcnt:
            pcnt_dist.add_pcnt(rollouts, device)

        rollouts.compute_returns(next_value, use_gae, gamma,
                                 gae_lambda, use_proper_time_limits)

        agent.model.train()
        value_loss, action_loss, dist_entropy = agent.update(rollouts, device)

        rollouts.after_update()

        #if j % args.log_interval == 0 and len(episode_rewards) > 1:
        all_return.append(np.mean(cumul_return))
        all_length.append(np.mean(episo_length))
        if True:
            total_num_steps = (j + 1) * num_processes * num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n, entropy loss {}, value_loss {}, action_loss {}"
                .format(j, total_num_steps,
                        len(cumul_return), np.mean(cumul_return),
                        np.median(cumul_return), np.min(cumul_return),
                        np.max(cumul_return), dist_entropy, value_loss,
                        action_loss))

    # Evaluate the final model
    rollouts = MCRolloutStorage(num_steps, num_processes, num_dwell, num_dwell)
    print("testing results")
    obs = agent.world.reset()
    rollouts.obs[0].copy_(torch.from_numpy(obs))
    rollouts.to(device)
    cumul_return = []
    episo_length = []
    episode_rewards = []
    episode_lengths = 0.0
    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            agent.model.eval()
            #print(rollouts.obs[step])
            value, action, action_log_prob = agent.sample_actions(rollouts.obs[step], rollouts.masks[step], device, True)

        # Obser reward and next obs
        action_world = action.cpu().numpy().reshape(-1)
        obs, reward, done, success = agent.world.step(action_world)
        episode_lengths += 1.0
        episode_rewards.append(reward)

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done else [1.0]])
        bad_masks = torch.FloatTensor(
            [[0.0] if done and not success else [1.0]])
        reward = torch.FloatTensor([[reward]])
        if done:
            # resample the start state were finished
            cumul_return.append(np.sum(episode_rewards))
            episo_length.append(episode_lengths)
            episode_rewards = []
            episode_lengths = 0.0
            obs = agent.world.reset()
        obs = torch.from_numpy(obs)
        rollouts.insert(obs, action,
                        action_log_prob, value, reward, masks, bad_masks)
    test_r = np.mean(cumul_return)
    test_l = np.mean(episo_length)
    print("testing results:\n average cumulative returns {:.1f}, average episode lengths {:.1f}".format(np.mean(cumul_return), np.mean(episo_length)))

    # save models
    if save_model:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_path = os.path.join(model_save_dir, '{}-{}.ckpt'.format(total_supposed_steps, model_name))
        #D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
        torch.save(agent.model.state_dict(), model_path)
        #torch.save(self.D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format(model_path))

    if show_plot:
        plt.plot([i+1 for i in range(num_updates)], all_return)
        plt.title("Performance in Rewards of PPO in 2D continuous MC environment")
        plt.xlabel('x1000 Steps')
        plt.ylabel('Average Cumulative Rewards')
        plt.legend(['PPO'])
        plt.show()

        plt.plot([i+1 for i in range(num_updates)], all_length)
        plt.title("Performance in Lengths of PPO in 2D continuous MC environment")
        plt.xlabel('x1000 Steps')
        plt.ylabel('Average Episode Lengths')
        plt.legend(['PPO'])
        plt.show()
