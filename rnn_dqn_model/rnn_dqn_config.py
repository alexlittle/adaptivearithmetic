ADAPTARITH_TRAINING = {
    'hidden_dims': 128, # how many nodes in each hidden layer of NN
    'num_episodes': 300, # number of simulator episodes to run when training
    'batch_size': 32, # BATCH_SIZE is the number of transitions sampled from the replay buffer
    'max_steps': 100, # max no steps per episode whilst training
    'eps_start': 1.0, # EPS_START is the starting value of epsilon
    'eps_end': 0.05, # EPS_END is the final value of epsilon
    'lr': 1e-4, # LR is the learning rate of the ``AdamW`` optimizer
    'replay_memory': 2000000,
    'lr_step' : 500,
    'lr_gamma': 0.95,
    'measure_step': 100,
    'min_episodes': 20,
    'measure_repeats': 100,
    'update_step': 10,
    'update_repeats': 100,
    'eps_decay': 0.999,
    'sequence_length': 20,
}