'''
DDQN Model

Adapted from: https://github.com/fschur/DDQN-with-PyTorch-for-OpenAI-Gym/tree/master

'''

import json
import os
import time
import torch
import numpy as np

import random
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
from datetime import datetime

from adaptarith.training_simulator import LearnerEnv
from ddqn_model.ddqn import QNetwork

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.translation import gettext_lazy as _

ddqn_config = settings.ADAPTARITH_TRAINING


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
class Memory:
    def __init__(self, len):
        self.rewards = deque(maxlen=len)
        self.state = deque(maxlen=len)
        self.action = deque(maxlen=len)
        self.is_done = deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n-1), batch_size)

        state = np.array(self.state)
        action = np.array(self.action)
        return torch.Tensor(state)[idx].to(device), torch.LongTensor(action)[idx].to(device), \
               torch.Tensor(state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()

def select_action(model, env, state, eps):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(values.cpu().numpy())

    return action

def train(batch_size, current, target, optim, memory, gamma):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = Qmodel(state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _ = env.step(action)
            perform += reward
    Qmodel.train()
    return perform/repeats



def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

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
            default=settings.ADAPTARITH_TRAINING['batch_size'],
            help=f"Size of each batch (default: {settings.ADAPTARITH_TRAINING['batch_size']})"
        )
        parser.add_argument(
            '--num_episodes',
            type=int,
            default=settings.ADAPTARITH_TRAINING['num_episodes'],
            help=f"Number of episodes to process (default: {settings.ADAPTARITH_TRAINING['num_episodes']})"
        )

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        num_episodes = options['num_episodes']
        start_time = time.time()

        lr_step = 100
        lr_gamma = 1
        measure_step = 100
        min_episodes = 20
        measure_repeats = 100
        update_step = 10
        update_repeats = 50
        eps_decay = 0.998
        gamma = settings.ADAPTARITH_TRAINING['gamma']
        eps_min = settings.ADAPTARITH_TRAINING['eps_end']
        eps = settings.ADAPTARITH_TRAINING['eps_start']

        env = LearnerEnv()
        # Get number of actions from gym action space
        n_actions = env.action_space.n
        state = env.reset()
        n_observations = len(state)

        Q_1 = QNetwork(action_dim=n_actions, state_dim=n_observations).to(device)
        Q_2 = QNetwork(action_dim=n_actions, state_dim=n_observations).to(device)

        update_parameters(Q_1, Q_2)

        # we only train Q_1
        for param in Q_2.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(Q_1.parameters(), lr=settings.ADAPTARITH_TRAINING['lr'])
        scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

        memory = Memory(settings.ADAPTARITH_TRAINING['replay_memory'] )
        performance = []

        for episode in range(num_episodes):
            state = env.reset()
            memory.state.append(state)
            print(f"Episode: {episode}")
            env.render()
            total_reward = 0
            for t in count():

                action = select_action(Q_2, env, state, eps)
                state, reward, done, _  = env.step(action)
                total_reward += reward
                # save state, action, reward sequence
                memory.update(state, action, reward, done)

                if done:
                    episode_durations.append(t + 1)
                    episode_rewards.append(total_reward)
                    env.render()
                    break

            if episode >= min_episodes and episode % update_step == 0:
                for _ in range(update_repeats):
                    train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

                # transfer new parameter from Q_1 to Q_2
                update_parameters(Q_1, Q_2)

            # update learning rate and eps
            scheduler.step()
            eps = max(eps * eps_decay, eps_min)

            # display the performance
            if (episode % measure_step == 0) and episode >= min_episodes:
                performance.append([episode, evaluate(Q_1, env, measure_repeats)])
                print("Episode: ", episode)
                print("rewards: ", performance[-1][1])
                print("lr: ", scheduler.get_last_lr()[0])
                print("eps: ", eps)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        ddqn_config['runtime'] = elapsed_time
        # write pth, graphs and config
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_dir = os.path.join(settings.BASE_DIR, 'ddqn_model', 'results', timestamp)
        os.makedirs(output_dir, exist_ok=True)

        model_output_file = os.path.join(output_dir, "model.pth")
        durations_file = os.path.join(output_dir, "results-durations.png")
        rewards_file = os.path.join(output_dir, "results-rewards.png")
        config_output_file = os.path.join(output_dir, "config.json")

        torch.save(Q_1.state_dict(), model_output_file)

        with open(config_output_file, "w") as file:
            json.dump(ddqn_config, file, indent=4)

        plot_durations(show_result=True, save_path=durations_file)
        plot_rewards(show_result=True, save_path=rewards_file)
