ADAPTARITH_TRAINING = {
    'hidden_dims': 128, # how many nodes in each hidden layer of NN
    'num_episodes': 300, # number of simulator episodes to run when training
    'batch_size': 64, # BATCH_SIZE is the number of transitions sampled from the replay buffer
    'max_steps': 100, # max no steps per episode whilst training
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'replaybuffer_capacity': 500000,
    'gamma': 0.99,
    'lr': 1e-3,
    'tau': 0.005, # TAU is the update rate of the target network

}