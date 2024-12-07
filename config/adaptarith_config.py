ADAPTARITH_LEVELS = ['easy', 'mod', 'hard', 'vhard']

ADAPTARITH_TOPICS = ['add', 'subtract', 'multiply', 'divide']

'''
Settings for model and simulator whilst training
'''
# BATCH_SIZE is the number of transitions sampled from the replay buffer
ADAPTARITH_TRAINING_BATCH_SIZE = 256

# max no steps per episode whilst training
ADAPTARITH_TRAINING_MAX_STEPS = 150

# GAMMA is the discount factor
ADAPTARITH_TRAINING_GAMMA = 0.99

# EPS_START is the starting value of epsilon
ADAPTARITH_TRAINING_EPS_START = 1.0

# EPS_END is the final value of epsilon
ADAPTARITH_TRAINING_EPS_END = 0.05

# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
ADAPTARITH_TRAINING_EPS_DECAY = 5000

# TAU is the update rate of the target network
ADAPTARITH_TRAINING_TAU = 0.005

# LR is the learning rate of the ``AdamW`` optimizer
ADAPTARITH_TRAINING_LR = 1e-4

ADAPTARITH_TRAINING_NUM_EPISODES = 100

'''
Settings for points, difficulty levels etc
'''
# knowledge level required for each topic to complete
ADAPTARITH_PASS_THRESHOLD = 90

# for the pre-test, how many knowledge points per question (used for setting baseline)
ADAPTARITH_INITIAL_KNOWLEDGE_POINTS_PER_QUESTION = 10

# when learning, how many knowledge points for correct/incorrect answer
ADAPTARITH_POINTS_FOR_CORRECT = 5
ADAPTARITH_POINTS_FOR_INCORRECT = 0

# maximum increase in knowledge points per question
ADAPTARITH_MAX_GAIN = 5

ADAPTARITH_LEVEL_EASY_RANGE = range(0, 25)
ADAPTARITH_LEVEL_MOD_RANGE = range(25, 50)
ADAPTARITH_LEVEL_HARD_RANGE = range(50, 75)
ADAPTARITH_LEVEL_VHARD_RANGE = range(75, 100)

# define ranges of values per level to be used in the arithmetic
ADAPTARITH_RANGES = {
    'add':
        {'easy': {
            'first_term': range(1, 10),
            'second_term': range(1, 10)
        },
            'mod': {
                'first_term': range(10, 50),
                'second_term': range(10, 50)
            },
            'hard': {
                'first_term': range(50, 100),
                'second_term': range(50, 100)
            },
            'vhard': {
                'first_term': range(100, 1000),
                'second_term': range(100, 1000)
            }},
    'subtract':
        {'easy': {
            'first_term': range(5, 10),
            'second_term': range(1, 5)
        },
            'mod': {
                'first_term': range(20, 50),
                'second_term': range(1, 20)
            },
            'hard': {
                'first_term': range(50, 100),
                'second_term': range(25, 75)
            },
            'vhard': {
                'first_term': range(100, 1000),
                'second_term': range(50, 500)
            }},
    'multiply':
        {'easy': {
            'first_term': range(1, 5),
            'second_term': range(1, 5)
        },
            'mod': {
                'first_term': range(5, 10),
                'second_term': range(5, 10)
            },
            'hard': {
                'first_term': range(5, 15),
                'second_term': range(5, 15)
            },
            'vhard': {
                'first_term': range(10, 20),
                'second_term': range(10, 20)
            }},
    'divide':
        {'easy': {
            'first_term': range(10, 20),
            'second_term': range(1, 5)
        },
            'mod': {
                'first_term': range(20, 50),
                'second_term': range(5, 10)
            },
            'hard': {
                'first_term': range(50, 100),
                'second_term': range(5, 15)
            },
            'vhard': {
                'first_term': range(100, 500),
                'second_term': range(10, 25)
            }}

}