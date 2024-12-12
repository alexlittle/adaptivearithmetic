import random
import torch
import os

from rnn_dqn_model.rnn_dqn import LSTM_DQN
from rnn_dqn_model import rnn_dqn_config
from django.conf import settings
from adaptarith.models import Question, KnowledgeLevel


def format_question(question):
    if question.topic == 'add':
        return f"{question.first_term} + {question.second_term}"
    if question.topic == 'subtract':
        return f"{question.first_term} - {question.second_term}"
    if question.topic == 'multiply':
        return f"{question.first_term} × {question.second_term}"
    if question.topic == 'divide':
        return f"{question.first_term} ÷ {question.second_term}"

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
    for level in settings.ADAPTARITH_LEVELS:
        level_questions = []
        for topic in settings.ADAPTARITH_TOPICS:
            q1 = generate_question(topic,level, pretest=True, training=training, user=user)
            level_questions.append(q1)
            q2 = generate_question(topic,level, pretest=True,training=training, user=user)
            level_questions.append(q2)
        random.shuffle(level_questions)
        questions = questions + level_questions
    return questions

def get_next_question(user=None,
                      knowledge_level=None,
                      model_pth=settings.ADAPTARITH_MODEL_PATH):

    if not knowledge_level:
        knowledge_level = KnowledgeLevel.get_latest_for_user_as_list(user)

    num_observations = len(settings.ADAPTARITH_TOPICS)
    num_actions = num_observations * len(settings.ADAPTARITH_LEVELS)

    state_dict_path = os.path.join(settings.BASE_DIR, model_pth)

    model = LSTM_DQN(input_size=num_observations,
                     hidden_size=rnn_dqn_config.ADAPTARITH_TRAINING['hidden_dims'],
                     output_size=num_actions)
    model.load_state_dict(torch.load(state_dict_path, weights_only=False))
    model.eval()

    print(knowledge_level)
    input_kl = [x / 100.0 for x in knowledge_level]
    state_tensor = torch.tensor(input_kl, dtype=torch.float32).unsqueeze(0)
    print(state_tensor)
    with torch.no_grad():  # Disable gradient calculation since we are only inferring
        q_values, _ = model(state_tensor)  # Get Q-values from the model

    # Select the action with the highest Q-value (for DQN)
    if q_values.dim() == 1:
        q_values = q_values.unsqueeze(0)
    action = torch.argmax(q_values, dim=1).item()

    # 'translate' action into level & topic
    level = settings.ADAPTARITH_LEVELS[action // len(settings.ADAPTARITH_TOPICS)]
    topic = settings.ADAPTARITH_TOPICS[action % len(settings.ADAPTARITH_TOPICS)]
    print("")
    print("Q-values:", q_values.numpy())
    print(f"Action: {action},  Level: {level}, Topic: {topic}")

    return generate_question(topic, level, user=user)