'''
DQN Model

Adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

'''

import time
import json
import random
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple, deque
from itertools import count
from datetime import datetime

from adaptarith.training_simulator import LearnerEnv
from dqn_model.dqn import DQN

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.translation import gettext_lazy as _

dqn_config = settings.ADAPTARITH_TRAINING

steps_done = 0

env = LearnerEnv()
# Get number of actions from gym action space
n_actions = env.action_space.n
state = env.reset()
n_observations = len(state)

episode_durations = []
episode_rewards = []

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

policy_net = DQN(n_observations, n_actions, settings.ADAPTARITH_TRAINING['hidden_dims']).to(device)
target_net = DQN(n_observations, n_actions, settings.ADAPTARITH_TRAINING['hidden_dims']).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=dqn_config['lr'], amsgrad=True)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(dqn_config['replay_memory'] )

def select_action(state):
    global steps_done
    sample = random.random()
    # linear decay
    eps_threshold = max(dqn_config['eps_end'] ,
                        dqn_config['eps_start']
                        - (steps_done / dqn_config['eps_decay'])
                        * (dqn_config['eps_start']
                        - dqn_config['eps_end'] ))
    # exponential decay
    #eps_threshold = dqn_config['eps_end'] \
    #                + (dqn_config['eps_start']
    #                   - dqn_config['eps_end'] ) \
    #                * math.exp(-1. * steps_done / dqn_config['eps_decay'])
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def plot_rewards(show_result=False, save_path=None):
    plt.figure(2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)

def plot_durations(show_result=False, save_path=None):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)

def optimize_model(batch_size):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * dqn_config['gamma']) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

class Command(BaseCommand):
    help = _(u"For training DQN model")
    errors = []

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch_size',
            type=int,
            default=dqn_config['batch_size'],
            help=f"Size of each batch (default: {dqn_config['batch_size']})"
        )
        parser.add_argument(
            '--num_episodes',
            type=int,
            default=dqn_config['num_episodes'],
            help=f"Number of episodes to process (default: {dqn_config['num_episodes']})"
        )

    def handle(self, *args, **options):
        dqn_config['batch_size'] = options['batch_size']
        dqn_config['num_episodes'] = options['num_episodes']
        start_time = time.time()
        for i_episode in range(dqn_config['num_episodes']):
            # Initialize the environment and get its state
            state = env.reset()
            print(f"Episode: {i_episode}")
            env.render()
            total_reward = 0
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = select_action(state)
                observation, reward, done, _  = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                total_reward += reward
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model(dqn_config['batch_size'])

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] \
                                                 * dqn_config['tau'] + target_net_state_dict[key] * (
                                                 1 - dqn_config['tau'])
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    episode_rewards.append(total_reward)
                    env.render()
                    break
        print('Complete')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        dqn_config['runtime'] = elapsed_time
        # write pth, graphs and config
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_dir = os.path.join(settings.BASE_DIR, 'dqn_model', 'results', timestamp)
        os.makedirs(output_dir, exist_ok=True)

        model_output_file = os.path.join(output_dir, "model.pth")
        durations_file = os.path.join(output_dir, "results-durations.png")
        rewards_file = os.path.join(output_dir, "results-rewards.png")
        config_output_file = os.path.join(output_dir, "config.json")

        torch.save(policy_net.state_dict(), model_output_file)

        with open(config_output_file, "w") as file:
            json.dump(dqn_config, file, indent=4)

        plot_durations(show_result=True, save_path=durations_file)
        plot_rewards(show_result=True, save_path=rewards_file)
