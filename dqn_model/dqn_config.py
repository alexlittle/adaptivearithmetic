import os
from django.conf import settings
from datetime import datetime

tb_run_dir = os.path.join(settings.BASE_DIR, 'dqn_model', 'runs',
                                  datetime.now().strftime('%Y-%m-%d-%H_%M_%S') + "")

ADAPTARITH_TRAINING = {
    'hidden_dims': 128, # how many nodes in each hidden layer of NN
    'num_episodes': 300, # number of simulator episodes to run when training
    'batch_size': 64, # BATCH_SIZE is the number of transitions sampled from the replay buffer
    'max_steps': 100, # max no steps per episode whilst training
    'epsilon_fn': 'linear',
    'epsilon_lin_start': 1.0,
    'epsilon_lin_end': 0.01,
    'replaybuffer_capacity': 500000,
    'gamma': 0.99,
    'lr': 1e-4,
    'tau': 0.005,
    'run_dir': tb_run_dir

}