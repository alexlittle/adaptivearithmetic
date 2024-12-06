import random
from django.conf import settings
from adaptarith.models import Question

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
        print("saving question")
        question.save()
    return question

def generate_pre_test(training=False):
    questions = []
    for topic in settings.ADAPTARITH_TOPICS:
        for level in settings.ADAPTARITH_LEVELS:
            q1 = generate_question(topic,level, pretest=True,training=training)
            questions.append(q1)
            q2 = generate_question(topic,level,pretest=True,training=training)
            questions.append(q2)
    # shuffle into random order
    random.shuffle(questions)
    return questions

def get_next_question():
    pass