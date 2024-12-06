from django.conf import settings
import random


class Quiz():

    def generate_test(self):
        self.questions = []
        for topic in settings.ADAPTARITH_TOPICS:
            for level in settings.ADAPTARITH_LEVELS:
                q1 = Question()
                q1.generate_question(topic, level)
                self.questions.append(q1)
                q2 = Question()
                q2.generate_question(topic, level)
                self.questions.append(q2)
        # shuffle into random order
        random.shuffle(self.questions)
        return self.questions

    def mark_test(self):
        self.score = 0
        for q in self.questions:
            if q.mark_question():
                self.score += 1

    def init_knowledge_levels(self):
        # for each topic init users knowledge level
        # basic "rules" for this - 12 increase in Knowledge level for each correct answer in each level (~100/8)
        knowledge = [0, 0, 0, 0]
        for q in self.questions:
            if q.mark_question():
                knowledge[settings.ADAPTARITH_TOPICS.index(q.topic)] += settings.ADAPTARITH_INITIAL_KNOWLEDGE_POINTS_PER_QUESTION
        return knowledge


class Question():

    def __str__(self):
        return f"{self.topic} {self.level}: {self.ft} {self.st}"

    def __init__(self):
        self.answer = None

    def generate_question(self, topic, level):
        self.topic = topic
        self.level = level

        ranges = settings.ADAPTARITH_RANGES[topic][level]
        self.ft = random.choice(ranges['first_term'])
        self.st = random.choice(ranges['second_term'])

        # make sure division can always end up with whole number
        # inefficient but works for now
        if self.topic == 'divide':
            while self.ft % self.st != 0:
                self.ft += 1

        return self.ft, self.st

    def get_correct_answer(self):
        if self.topic == 'add':
            return self.ft + self.st
        if self.topic == 'subtract':
            return self.ft - self.st
        if self.topic == 'multiply':
            return self.ft * self.st
        if self.topic == 'divide':
            return self.ft / self.st

    def set_answer(self, answer):
        self.answer = answer

    def mark_question(self):
        if self.topic == 'add' and self.ft + self.st == self.answer:
            return True
        if self.topic == 'subtract' and self.ft - self.st == self.answer:
            return True
        if self.topic == 'multiply' and self.ft * self.st == self.answer:
            return True
        if self.topic == 'divide' and self.ft / self.st == self.answer:
            return True
        return False