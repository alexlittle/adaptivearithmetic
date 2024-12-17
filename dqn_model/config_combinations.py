from dqn_model import dqn_config

# get the base config
dqn_config = dqn_config.ADAPTARITH_TRAINING

NUM_EPISODES = [50000, 100000, 250000, 500000]
BATCH_SIZES = [32, 64, 128]
MEMORY_BUFFER = [500000, 1000000, 2000000]
LEARNING_RATE = [1e-3, 1e-4]
HIDDEN_DIMS = [128, 256]