LEVELS = ['easy', 'mod', 'hard', 'vhard']

TOPICS = ['add', 'subtract', 'multiply', 'divide']

# knowledge level required for each topic to complete
PASS_THRESHOLD = 90

# for the pre-test, how many knowledge points per question (used for setting baseline)
INITIAL_KNOWLEDGE_POINTS_PER_QUESTION = 10

# when learning, how many knowledge points for correct/incorrect answer
POINTS_FOR_CORRECT = 5
POINTS_FOR_INCORRECT = 0

# maximum increase in knowledge points per question
MAX_GAIN = 5

LEVEL_EASY_RANGE = range(0, 25)
LEVEL_MOD_RANGE = range(25, 50)
LEVEL_HARD_RANGE = range(50, 75)
LEVEL_VHARD_RANGE = range(75, 100)

# max no steps per episode whilst training
TRAINING_MAX_STEPS = 500

# define ranges of values per level to be used in the arithmetic
RANGES = {
    'add':
        {'easy': {
            'first_term': range(0, 10),
            'second_term': range(0, 10)
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
            'second_term': range(0, 5)
        },
            'mod': {
                'first_term': range(20, 50),
                'second_term': range(0, 20)
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
            'first_term': range(0, 5),
            'second_term': range(0, 5)
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