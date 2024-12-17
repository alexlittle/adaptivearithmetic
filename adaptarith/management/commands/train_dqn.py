'''
DQN Model

Adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

'''

import time
import random
import os

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from collections import namedtuple, deque
from itertools import count


from simulators.courses.simulator_v8 import LearningPredictorEnv
from dqn_model.dqn import DQN

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.translation import gettext_lazy as _
from dqn_model import dqn_config
from adaptarith import training_utils

data_file_path = os.path.join(settings.BASE_DIR, 'data','ou','studentassessment_course_bbb2013b_with_activities.csv')

env = LearningPredictorEnv(data_file_path)
n_actions = env.action_space.n
n_observations = len(env.reset())

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


class DQNAgent():

    def __init__(self, dqn_agent_config, writer):
        self.dqn_config = dqn_agent_config
        self.writer = writer
        self.memory = ReplayMemory(self.dqn_config['replaybuffer_capacity'])
        self.device = torch.device(
                        "cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else
                        "cpu"
                    )
        self.policy_net = DQN(n_observations, self.dqn_config['hidden_dims'], n_actions).to(self.device)
        self.target_net = DQN(n_observations, self.dqn_config['hidden_dims'], n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.dqn_config['lr'], amsgrad=True)
        self.epsilon = self.dqn_config['epsilon_lin_start']
        self.episode_durations = []
        self.episode_rewards = []


    def select_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)


    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.dqn_config['gamma']) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        for episode in range(self.dqn_config['num_episodes']):
            # Initialize the environment and get its state
            state = env.reset()
            ep_start_time = time.time()
            total_reward = 0
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, done, _  = env.step(action.item())
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)

                if done:
                    print(state)
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)


                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model(self.dqn_config['batch_size'])

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] \
                                                 * self.dqn_config['tau'] + target_net_state_dict[key] * (
                                                 1 - self.dqn_config['tau'])
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(total_reward)
                    break

            ep_time = time.time() - ep_start_time
            self.writer.add_scalar('Episode Reward', total_reward, episode)
            self.writer.add_scalar('Episode Duration', self.episode_durations[-1], episode)
            self.writer.add_scalar('Episode Epsilon', self.epsilon, episode)
            self.writer.add_scalar('Episode Time', ep_time, episode)

            print(f"Episode {episode}/{self.dqn_config['num_episodes']}, Duration {self.episode_durations[-1]}, "
                f"Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}, Time: {ep_time:.2f}")
            self.epsilon = max(self.dqn_config['epsilon_lin_end'],
                          self.epsilon - (self.dqn_config['epsilon_lin_start']
                                     - self.dqn_config['epsilon_lin_end']) / self.dqn_config['epsilon_decay'])


class Command(BaseCommand):
    help = _(u"For training DQN model")
    errors = []

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch_size',
            type=int,
            default=dqn_config.ADAPTARITH_TRAINING['batch_size'],
            help=f"Size of each batch (default: {dqn_config.ADAPTARITH_TRAINING['batch_size']})"
        )
        parser.add_argument(
            '--num_episodes',
            type=int,
            default=dqn_config.ADAPTARITH_TRAINING['num_episodes'],
            help=f"Number of episodes to process (default: {dqn_config.ADAPTARITH_TRAINING['num_episodes']})"
        )

    def handle(self, *args, **options):
        dqn_config.ADAPTARITH_TRAINING['batch_size'] = options['batch_size']
        dqn_config.ADAPTARITH_TRAINING['num_episodes'] = options['num_episodes']
        dqn_config.ADAPTARITH_TRAINING['epsilon_decay'] = options['num_episodes'] * 3 / 4
        dqn_config.ADAPTARITH_TRAINING['simulator'] = "simulator_v8"
        dqn_config.ADAPTARITH_TRAINING['data_file'] = data_file_path


        start_time = time.time()

        os.makedirs(dqn_config.ADAPTARITH_TRAINING['run_dir'], exist_ok=True)
        writer = SummaryWriter(log_dir=dqn_config.ADAPTARITH_TRAINING['run_dir'])


        agent = DQNAgent(dqn_config.ADAPTARITH_TRAINING, writer)
        agent.train()

        print('Complete')
        writer.close()
        end_time = time.time()
        elapsed_time = end_time - start_time

        # average of all episode rewards
        average_reward = sum(agent.episode_rewards) / len(agent.episode_rewards)
        print(f"Average rewards (overall): {average_reward:.2f} ")

        # average of last half of episode rewards
        last_half = agent.episode_rewards[len(agent.episode_rewards) // 2:]
        average_reward_last_half = sum(last_half) / len(last_half)
        print(f"Average rewards (last half): {average_reward_last_half:.2f} ")

        last_100 = agent.episode_rewards[len(agent.episode_rewards)-100 :]
        average_reward_last_100 = sum(last_100) / len(last_100)
        print(f"Average rewards (last 100): {average_reward_last_100:.2f} ")

        print(f"Total runtime: {elapsed_time:.2f} seconds")
        dqn_config.ADAPTARITH_TRAINING['runtime'] = elapsed_time
        dqn_config.ADAPTARITH_TRAINING['average_reward'] = average_reward
        dqn_config.ADAPTARITH_TRAINING['average_reward_last_half'] = average_reward_last_half
        dqn_config.ADAPTARITH_TRAINING['average_reward_last_100'] = average_reward_last_100

        dqn_config.ADAPTARITH_TRAINING['episode_rewards'] = agent.episode_rewards

        # write pth, graphs and config
        training_utils.save_results('dqn_model',
                                    agent.policy_net.state_dict(),
                                    dqn_config.ADAPTARITH_TRAINING,
                                    agent.episode_durations,
                                    agent.episode_rewards,
                                    path_postfix="")

