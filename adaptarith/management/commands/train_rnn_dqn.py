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
from rnn_dqn_model.rnn_dqn import RNNQNetwork

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


class Memory:
    def __init__(self, len, sequence_length):
        self.sequence_length = sequence_length
        self.rewards = deque(maxlen=len)
        self.states = deque(maxlen=len)
        self.actions = deque(maxlen=len)
        self.is_done = deque(maxlen=len)

    def update(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        idx = random.sample(range(self.sequence_length, len(self.is_done) - 1), batch_size)

        # Convert deque to list for slicing
        states_list = list(self.states)

        sequences = [states_list[i - self.sequence_length:i] for i in idx]

        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.is_done)

        return (
            torch.Tensor(sequences).to(device),
            torch.LongTensor(actions[idx]).to(device),
            torch.Tensor(sequences).to(device),
            torch.Tensor(rewards[idx]).to(device),
            torch.Tensor(dones[idx]).to(device),
        )

    def reset(self):
        self.rewards.clear()
        self.states.clear()
        self.actions.clear()
        self.is_done.clear()

def select_action(model, env, state, eps, hidden_state=None):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state.unsqueeze(0))  # Unsqueeze for RNN to add batch dimension

    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)  # Select a random action
    else:
        q_values = values[0]
        action = np.argmax(q_values.cpu().numpy())  # Choose the action with the highest Q-value

    return action


def train(batch_size, model, optimizer, memory, gamma):
    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    # Compute Q-values for the current states
    q_values = model(states)
    q_values_tensor = q_values[0]  # Adjust based on your model's output

    # Get Q-values for next states
    next_q_values = model(next_states)
    next_q_values_tensor = next_q_values[0]  # Adjust based on your model's output

    # Compute the max Q-values for the next states
    max_next_q_values = torch.max(next_q_values_tensor, dim=1)[0]

    # Compute target Q-values
    target_q_values = rewards + gamma * max_next_q_values * (1 - is_done)

    # Get the Q-values corresponding to the actions taken
    q_value = q_values_tensor.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute the loss
    loss = (q_value - target_q_values.detach()).pow(2).mean()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()  # Set model to evaluation mode
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            # Convert state to a 3D tensor: (batch_size=1, seq_len=1, input_size)
            state = torch.Tensor(state).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                values = Qmodel(state)  # Forward pass through Q-model

            # Get the action with the highest Q-value
            q_values = values[0]
            action = np.argmax(q_values.cpu().numpy())

            # Step the environment
            state, reward, done, _ = env.step(action)
            perform += reward
    Qmodel.train()  # Set model back to training mode
    return perform / repeats



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

        eps = rnn_dqn_config['eps_start']

        env = LearnerEnv(rnn_dqn_config['max_steps'])
        # Get number of actions from gym action space
        n_actions = env.action_space.n
        state = env.reset()
        n_observations = len(state)

        Q_model = RNNQNetwork(action_dim=n_actions, state_dim=n_observations).to(device)

        optimizer = torch.optim.Adam(Q_model.parameters(), lr=rnn_dqn_config['lr'])
        scheduler = StepLR(optimizer, step_size=rnn_dqn_config['lr_step'], gamma=rnn_dqn_config['lr_gamma'])

        memory = Memory(rnn_dqn_config['replay_memory'],
                        rnn_dqn_config['sequence_length'])
        performance = []

        for episode in range(rnn_dqn_config['num_episodes']):
            state = env.reset()
            memory.states.append(state)
            print(f"Episode: {episode}")
            env.render()
            total_reward = 0
            for t in count():

                action = select_action(Q_model, env, state, eps)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                # save state, action, reward sequence
                memory.update(state, action, reward, done)

                if done:
                    episode_durations.append(t + 1)
                    episode_rewards.append(total_reward)
                    env.render()
                    break

            if episode >= rnn_dqn_config['min_episodes'] and episode % rnn_dqn_config['update_step'] == 0:
                for _ in range(rnn_dqn_config['update_repeats']):
                    train(rnn_dqn_config['batch_size'], Q_model, optimizer, memory, rnn_dqn_config['lr_gamma'])

            # update learning rate and eps
            scheduler.step()
            eps = max(eps * rnn_dqn_config['eps_decay'], rnn_dqn_config['eps_end'])

            # display the performance
            if (episode % rnn_dqn_config['measure_step'] == 0) and episode >= rnn_dqn_config['min_episodes']:
                performance.append([episode, evaluate(Q_model, env, rnn_dqn_config['measure_repeats'])])
                print("Episode: ", episode)
                print("rewards: ", performance[-1][1])
                print("lr: ", scheduler.get_last_lr()[0])
                print("eps: ", eps)

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

        torch.save(Q_model.state_dict(), model_output_file)

        with open(config_output_file, "w") as file:
            json.dump(rnn_dqn_config, file, indent=4)

        plot_durations(show_result=True, save_path=durations_file)
        plot_rewards(show_result=True, save_path=rewards_file)
