'''
RNN DQN Model

'''

import os
import time
import torch
import torch.nn as nn
import random
import pickle
import pandas as pd
from itertools import count
from datetime import datetime
import torch.optim as optim
from tensorboardX import SummaryWriter

from simulators.courses.simulator_v5 import HMMPredictorEnv
from rnn_dqn_model.rnn_dqn import LSTM_DQN

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.translation import gettext_lazy as _
from rnn_dqn_model import rnn_dqn_config
from adaptarith import training_utils

rnn_dqn_config = rnn_dqn_config.ADAPTARITH_TRAINING

rnn_dqn_config["simulator"] = "simulators.courses.simulator_v5"

episode_durations = []
episode_rewards = []

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

"""
memory to save the state, action, reward sequence from the current episode. 
"""
class ReplayBuffer:
    def __init__(self):
        self.capacity = rnn_dqn_config['replaybuffer_capacity']
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, device, batch_size):
        self.gamma = rnn_dqn_config['gamma']
        self.batch_size = rnn_dqn_config['batch_size']
        self.policy_net = LSTM_DQN(input_size, hidden_size, output_size)
        self.target_net = LSTM_DQN(input_size, hidden_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=rnn_dqn_config['lr'])
        self.replay_buffer = ReplayBuffer()
        self.steps_done = 0
        self.device = device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.policy_net.output_size))
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values, _ = self.policy_net(state)
            if q_values.dim() == 1:
                q_values = q_values.unsqueeze(0)
            return torch.argmax(q_values, dim=1).item()

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)

        # Prepare the batch
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)

        # if state is just [0.5] then unsqueeze(-1) if [0.5, 0.6...] then unsqueeze(1)
        states = states.unsqueeze(1)

        hidden_state = self.policy_net.init_hidden(self.batch_size, device)

        # Get current Q-values from policy network
        q_values, next_hidden_state = self.policy_net(states, hidden_state)
        q_values = q_values.squeeze(1)

        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_states = next_states.unsqueeze(1)

        next_q_values, _ = self.target_net(next_states, hidden_state)
        next_q_values = next_q_values.squeeze(1)
        next_state_values = next_q_values.max(1)[0]

        # Expected Q-values
        expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return next_hidden_state

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Training loop example
def train_dqn(agent, env, num_episodes, epsilon_decay, writer):
    epsilon = rnn_dqn_config['epsilon_start']
    for episode in range(num_episodes):
        state = env.reset()
        ep_start_time = time.time()
        total_reward = 0

        for t in count():
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, next_state, reward, done)

            agent.optimize_model()

            state = next_state
            total_reward += reward

            if episode % 10 == 0:
                agent.update_target_network()

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(total_reward)
                break

        # Decay epsilon
        epsilon = max(rnn_dqn_config['epsilon_end'],
                      epsilon - (rnn_dqn_config['epsilon_start'] - rnn_dqn_config['epsilon_end']) / epsilon_decay)
        ep_time = time.time() - ep_start_time
        writer.add_scalar('Episode Reward', total_reward, episode)
        writer.add_scalar('Episode Duration', episode_durations[-1], episode)
        writer.add_scalar('Episode Epsilon', epsilon, episode)
        writer.add_scalar('Episode Time', ep_time, episode)
        print(state)
        print(f"Episode {episode}/{num_episodes}, Duration {episode_durations[-1]}, "
                f"Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}, Time: {ep_time:.2f}")



class Command(BaseCommand):
    help = _(u"For training DDQN model")
    errors = []

    def add_arguments(self, parser):

        parser.add_argument(
            '--batch_size',
            type=int,
            default=rnn_dqn_config['batch_size'],
            help=f"Size of each batch (default: {rnn_dqn_config['batch_size']})"
        )
        parser.add_argument(
            '--num_episodes',
            type=int,
            default=rnn_dqn_config['num_episodes'],
            help=f"Number of episodes to process (default: {rnn_dqn_config['num_episodes']})"
        )

    def handle(self, *args, **options):
        rnn_dqn_config['batch_size'] = options['batch_size']
        rnn_dqn_config['num_episodes'] = options['num_episodes']
        rnn_dqn_config['epsilon_decay'] = options['num_episodes'] * 3/4
        start_time = time.time()

        tb_run_dir = os.path.join(settings.BASE_DIR, 'rnn_dqn_model', 'runs',
                                  datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
        os.makedirs(tb_run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_run_dir)


        env = HMMPredictorEnv(settings.BASE_DIR)

        state = env.reset()

        n_actions = env.action_space.n
        n_observations = len(state)
        hidden_size = rnn_dqn_config['hidden_dims']

        agent = DQNAgent(n_observations,
                         hidden_size,
                         n_actions,
                         device,
                         rnn_dqn_config['batch_size'])
        train_dqn(agent,
                  env,
                  rnn_dqn_config['num_episodes'],
                  rnn_dqn_config['epsilon_decay'],
                  writer)

        writer.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        rnn_dqn_config['runtime'] = elapsed_time

        # write pth, graphs and config
        training_utils.save_results('rnn_dqn_model',
                                    agent.policy_net.state_dict(),
                                    rnn_dqn_config,
                                    episode_durations,
                                    episode_rewards)
