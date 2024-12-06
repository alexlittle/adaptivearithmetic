import random
from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone
from django.db.models import Max


class Question(models.Model):
    user = models.ForeignKey(User, null=True, on_delete=models.CASCADE)
    create_date = models.DateTimeField(default=timezone.now)
    pre_test = models.BooleanField(default=False)
    first_term = models.IntegerField(default=0)
    second_term = models.IntegerField(default=0)
    level = models.CharField(max_length=10, blank=False, null=False)
    topic = models.CharField(max_length=10, blank=False, null=False)
    response = models.IntegerField(default=None, blank=True, null=True)

    def get_correct_answer(self):
        if self.topic == 'add':
            return self.first_term + self.second_term
        if self.topic == 'subtract':
            return self.first_term - self.second_term
        if self.topic == 'multiply':
            return self.first_term * self.second_term
        if self.topic == 'divide':
            return self.first_term / self.second_term

    def mark_question(self, knowledge_level):
        correct = False
        if self.topic == 'add' and self.first_term + self.second_term == self.response:
            correct = True
        if self.topic == 'subtract' and self.first_term - self.second_term == self.response:
            correct = True
        if self.topic == 'multiply' and self.first_term * self.second_term == self.response:
            correct = True
        if self.topic == 'divide' and self.first_term / self.second_term == self.response:
            correct = True

        if not correct:
            return settings.ADAPTARITH_POINTS_FOR_INCORRECT

        kl_for_topic = knowledge_level[settings.ADAPTARITH_TOPICS.index(self.topic)]

        if kl_for_topic in settings.ADAPTARITH_LEVEL_EASY_RANGE and self.level == 'easy':
            return settings.ADAPTARITH_POINTS_FOR_CORRECT

        if kl_for_topic in settings.ADAPTARITH_LEVEL_MOD_RANGE and self.level == 'mod':
            return settings.ADAPTARITH_POINTS_FOR_CORRECT

        if kl_for_topic in settings.ADAPTARITH_LEVEL_HARD_RANGE and self.level == 'hard':
            return settings.ADAPTARITH_POINTS_FOR_CORRECT

        if kl_for_topic in settings.ADAPTARITH_LEVEL_VHARD_RANGE and self.level == 'vhard':
            return settings.ADAPTARITH_POINTS_FOR_CORRECT

        return settings.ADAPTARITH_POINTS_FOR_INCORRECT

class KnowledgeLevel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    create_date = models.DateTimeField(default=timezone.now)
    topic = models.CharField(max_length=10, blank=False, null=False)
    score = models.IntegerField(default=0)

    @staticmethod
    def get_latest_for_user(self, user):
        latest_dates = KnowledgeLevel.objects.filter(user=user).values('topic').annotate(
            latest_date=Max('create_date')
        )
        most_recent_rows = KnowledgeLevel.objects.filter(
            user=user,
            topic__in=[entry['topic'] for entry in latest_dates],
            create_date__in=[entry['latest_date'] for entry in latest_dates],
        )
        return most_recent_rows

    @staticmethod
    def get_training_init_knowledge_level(questions):
        knowledge = [0, 0, 0, 0]
        for q in questions:
            if q.mark_question(knowledge):
                knowledge[settings.ADAPTARITH_TOPICS.index(
                    q.topic)] += settings.ADAPTARITH_INITIAL_KNOWLEDGE_POINTS_PER_QUESTION
        return knowledge
