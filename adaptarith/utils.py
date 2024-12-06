import random
import torch
import os

from dqn_model.dqn import DQN
from django.conf import settings
from adaptarith.models import Question, KnowledgeLevel


def format_question(question):
    if question.topic == 'add':
        return f"{question.first_term} + {question.second_term}"
    if question.topic == 'subtract':
        return f"{question.first_term} - {question.second_term}"
    if question.topic == 'multiply':
        return f"{question.first_term} * {question.second_term}"
    if question.topic == 'divide':
        return f"{question.first_term} / {question.second_term}"

def generate_question(topic, level, pretest=False, training=False, user=None):
    question = Question()
    question.user = user
    question.pre_test=pretest
    question.topic = topic
    question.level = level

    ranges = settings.ADAPTARITH_RANGES[topic][level]
    question.first_term = random.choice(ranges['first_term'])
    question.second_term = random.choice(ranges['second_term'])

    # make sure division can always end up with whole number
    # inefficient but works for now
    if question.topic == 'divide':
        while question.first_term % question.second_term != 0:
            question.first_term += 1

    if not training:
        question.save()
    return question

def generate_pre_test(training=False, user=None):
    questions = []
    for topic in settings.ADAPTARITH_TOPICS:
        for level in settings.ADAPTARITH_LEVELS:
            q1 = generate_question(topic,level, pretest=True,training=training, user=user)
            questions.append(q1)
            #q2 = generate_question(topic,level, pretest=True,training=training, user=user)
            #questions.append(q2)
    # shuffle into random order
    random.shuffle(questions)
    return questions

def get_next_question(user):

    knowledge_level = KnowledgeLevel.get_latest_for_user_as_list(user)

    num_observations = len(settings.ADAPTARITH_TOPICS)
    num_actions = num_observations * len(settings.ADAPTARITH_LEVELS)

    state_dict_path = os.path.join(settings.BASE_DIR, 'dqn_model','model_dqn.pth')

    model = DQN(n_observations=num_observations, n_actions=num_actions)
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    model.eval()

    state_tensor = torch.tensor(knowledge_level, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():  # Disable gradient calculation since we are only inferring
        q_values = model(state_tensor)  # Get Q-values from the model

    # Select the action with the highest Q-value (for DQN)
    action = torch.argmax(q_values, dim=1).item()

    # 'translate' action into level & topic
    level = settings.ADAPTARITH_LEVELS[action // 4]
    topic = settings.ADAPTARITH_TOPICS[action % 4]

    question = generate_question(topic, level, user=user)
    return question