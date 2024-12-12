'''
DDQN Model

Adapted from: https://github.com/fschur/DDQN-with-PyTorch-for-OpenAI-Gym/tree/master

'''

import json
import os
import time
import torch
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
from datetime import datetime
import torch.optim as optim
from adaptarith.training_simulator import LearnerEnv
from rnn_dqn_model.rnn_dqn import LSTM_DQN

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.translation import gettext_lazy as _
from rnn_dqn_model import rnn_dqn_config

rnn_dqn_config = rnn_dqn_config.ADAPTARITH_TRAINING


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
    def __init__(self, capacity):
        self.capacity = capacity
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
    def __init__(self, input_size, hidden_size, output_size, device, batch_size=32, gamma=0.99, lr=1e-3, capacity=100000):
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_net = LSTM_DQN(input_size, hidden_size, output_size)
        self.target_net = LSTM_DQN(input_size, hidden_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity)
        self.steps_done = 0
        self.device = device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(range(self.policy_net.output_size))
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values, _ = self.policy_net(state)
            if q_values.dim() == 1:
                q_values = q_values.unsqueeze(0)
            return torch.argmax(q_values, dim=1).item()

    def optimize_model(self, hidden_state):
        seq_len = 10
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of transitions
        batch = self.replay_buffer.sample(self.batch_size)

        # Prepare the batch
        states, actions, next_states, rewards, dones = zip(*batch)

        # Convert to tensors

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)

        states = states.unsqueeze(-1)

        hidden_state = self.policy_net.init_hidden(self.batch_size, device)

        # Get current Q-values from policy network
        q_values, next_hidden_state = self.policy_net(states, hidden_state)
        q_values = q_values.squeeze(1)

        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_states = next_states.unsqueeze(-1)
        # Get next Q-values from target network
        next_q_values, _ = self.target_net(next_states, hidden_state)
        next_q_values = next_q_values.squeeze(1)
        next_state_values = next_q_values.max(1)[0]

        # Compute expected Q-values
        expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return next_hidden_state

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Training loop example
def train_dqn(agent, env, num_episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
    epsilon = epsilon_start
    hidden_state = agent.policy_net.init_hidden(agent.batch_size, device)
    for episode in range(num_episodes):
        state = env.reset()

        done = False
        total_reward = 0

        for t in count():
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Store the transition in the replay buffer
            agent.replay_buffer.push(state, action, next_state, reward, done)

            # Perform one step of optimization
            hidden_state = agent.optimize_model(hidden_state)

            # Move to the next state
            state = next_state
            total_reward += reward

            # Update the target network
            if episode % 10 == 0:
                agent.update_target_network()

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(total_reward)
                #plot_durations()
                #plot_rewards()
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

        print(
            f"Episode {episode}/{num_episodes}, Duration {episode_durations[-1]}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")


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
        start_time = time.time()

        env = LearnerEnv(rnn_dqn_config['max_steps'])
        # Get number of actions from gym action space

        state = env.reset()

        n_actions = env.action_space.n
        n_observations = len(state)
        hidden_size = rnn_dqn_config['num_episodes']

        agent = DQNAgent(n_observations, hidden_size, n_actions, device, batch_size=rnn_dqn_config['batch_size'])
        train_dqn(agent, env, num_episodes=rnn_dqn_config['num_episodes'])


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        rnn_dqn_config['runtime'] = elapsed_time
        # write pth, graphs and config
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_dir = os.path.join(settings.BASE_DIR, 'rnn_dqn_model', 'results', timestamp)
        os.makedirs(output_dir, exist_ok=True)

        model_output_file = os.path.join(output_dir, "model.pth")
        durations_file = os.path.join(output_dir, "results-durations.png")
        rewards_file = os.path.join(output_dir, "results-rewards.png")
        config_output_file = os.path.join(output_dir, "config.json")

        torch.save(agent.policy_net.state_dict(), model_output_file)

        with open(config_output_file, "w") as file:
            json.dump(rnn_dqn_config, file, indent=4)

        plot_durations(show_result=True, save_path=durations_file)
        plot_rewards(show_result=True, save_path=rewards_file)
