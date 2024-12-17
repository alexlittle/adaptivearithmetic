
'''
Settings for levels and topics available
'''

ADAPTARITH_LEVELS = ['easy', 'mod', 'hard', 'vhard']

ADAPTARITH_TOPICS = ['add', 'subtract', 'multiply', 'divide']


'''
Settings for points, difficulty levels etc
'''
# knowledge level required for each topic to complete
ADAPTARITH_PASS_THRESHOLD = 100

# how many knowledge points for correct/incorrect answer
ADAPTARITH_POINTS_FOR_CORRECT = 5
ADAPTARITH_POINTS_FOR_INCORRECT = 0

# maximum increase in knowledge points per question
ADAPTARITH_MAX_GAIN = 5

# penalty for repeating same topic (used for training only)
ADAPTARITH_MAX_REPETITIONS = 2
ADAPTARITH_REPETITION_PENALTY = ADAPTARITH_POINTS_FOR_CORRECT * 2/5

# penalty for not selecting topic with the lowest knowledge level (used for training only)
ADAPTARITH_LOWEST_TOPIC_PENALTY = ADAPTARITH_POINTS_FOR_CORRECT * 1/5

# number of questions per topic per level in pre-test
ADAPTARITH_PRETEST_NUM = 2 # for 4 topics and 4 levels then total questions = 16 times this value

# for the pre-test, how many knowledge points per question (used for setting baseline)
ADAPTARITH_INITIAL_KNOWLEDGE_POINTS_PER_QUESTION = 10

# define ranges of values per level to be used in the arithmetic
ADAPTARITH_LEVEL_EASY_RANGE = {'lower': 0, 'upper': 25}
ADAPTARITH_LEVEL_MOD_RANGE = {'lower': 25, 'upper': 50}
ADAPTARITH_LEVEL_HARD_RANGE = {'lower': 50, 'upper': 75}
ADAPTARITH_LEVEL_VHARD_RANGE = {'lower': 75, 'upper': 100}


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
            'first_term': range(2, 5),
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
            'second_term': range(2, 5)
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