import random
import torch
import os

from django.contrib.auth import login
from django.contrib.auth.models import User
from rnn_dqn_model.rnn_dqn import LSTM_DQN
from rnn_dqn_model import rnn_dqn_config
from django.conf import settings
from adaptarith.models import Question, KnowledgeLevel

def get_user(request):
    if request.user.is_authenticated:
        return request.user
    else:
        if not request.session.session_key:
            request.session.save()
        if not request.session.get('temp_user_id'):
            # Create a temporary user
            temp_user = User.objects.create_user(
                username=f"temp_user_{request.session.session_key}",
                first_name=request.GET.get('name', 'Guest'),
                is_active=False
            )
            request.session['temp_user_id'] = temp_user.id
            login(request, temp_user)
        else:
            temp_user = User.objects.get(id=request.session['temp_user_id'])
        return temp_user

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
            for x in range(settings.ADAPTARITH_PRETEST_NUM):
                q = generate_question(topic,level, pretest=True, training=training, user=user)
                level_questions.append(q)
        # only shuffle the level questions, so the easier questions come first in the pre-test, harder later
        random.shuffle(level_questions)
        questions = questions + level_questions
    return questions

def get_next_question(user=None,
                      knowledge_level=None,
                      model_pth=settings.ADAPTARITH_MODEL_PATH,
                      debug=False):

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

    input_kl = [x / 100.0 for x in knowledge_level]
    state_tensor = torch.tensor(input_kl, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():  # Disable gradient calculation since we are only inferring
        q_values, _ = model(state_tensor)  # Get Q-values from the model

    # Select the action with the highest Q-value (for DQN)
    if q_values.dim() == 1:
        q_values = q_values.unsqueeze(0)
    action = torch.argmax(q_values, dim=1).item()

    # 'translate' action into level & topic
    level = settings.ADAPTARITH_LEVELS[action // len(settings.ADAPTARITH_TOPICS)]
    topic = settings.ADAPTARITH_TOPICS[action % len(settings.ADAPTARITH_TOPICS)]

    if debug:
        print("")
        print("Q-values:", q_values.numpy())
        print(f"Action: {action},  Level: {level}, Topic: {topic}")
    return generate_question(topic, level, user=user)