import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from django.conf import settings


def plot_durations(episode_durations, save_path=None):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Result')
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

def plot_rewards(episode_rewards, save_path=None):
    plt.figure(2)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Result')
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


def save_results(model_dir, state_dict, config, episode_durations, episode_rewards):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = os.path.join(settings.BASE_DIR, model_dir, 'results', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    model_output_file = os.path.join(output_dir, "model.pth")
    durations_file = os.path.join(output_dir, "results-durations.png")
    rewards_file = os.path.join(output_dir, "results-rewards.png")
    config_output_file = os.path.join(output_dir, "config.json")

    torch.save(state_dict, model_output_file)

    with open(config_output_file, "w") as file:
        json.dump(config, file, indent=4)

    plot_durations(episode_durations, save_path=durations_file)
    plot_rewards(episode_rewards, save_path=rewards_file)