
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

    def mark_question(self):
        if self.topic == 'add' and self.first_term + self.second_term == self.response:
            return settings.ADAPTARITH_POINTS_FOR_CORRECT
        if self.topic == 'subtract' and self.first_term - self.second_term == self.response:
            return settings.ADAPTARITH_POINTS_FOR_CORRECT
        if self.topic == 'multiply' and self.first_term * self.second_term == self.response:
            return settings.ADAPTARITH_POINTS_FOR_CORRECT
        if self.topic == 'divide' and self.first_term / self.second_term == self.response:
            return settings.ADAPTARITH_POINTS_FOR_CORRECT

        return settings.ADAPTARITH_POINTS_FOR_INCORRECT

class KnowledgeLevel(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    create_date = models.DateTimeField(default=timezone.now)
    topic = models.CharField(max_length=10, blank=False, null=False)
    score = models.IntegerField(default=0)

    @staticmethod
    def has_passed(user):
        kls = KnowledgeLevel.get_latest_for_user(user)
        for k in kls:
            if k.score < settings.ADAPTARITH_PASS_THRESHOLD:
                return False
        return True

    @staticmethod
    def get_latest_for_topic(user, topic):
        try:
            latest_kl = KnowledgeLevel.objects.filter(user=user, topic=topic).latest('create_date')
            return latest_kl.score
        except KnowledgeLevel.DoesNotExist:
            return 0


    @staticmethod
    def get_latest_for_user(user):
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
    def get_latest_for_user_as_list(user):
        as_query = KnowledgeLevel.get_latest_for_user(user)
        knowledge = [0 for _ in settings.ADAPTARITH_TOPICS]
        for i in as_query:
            knowledge[settings.ADAPTARITH_TOPICS.index(i.topic)] = i.score
        return knowledge

    @staticmethod
    def pre_test_init_knowledge_level(questions):
        knowledge = [0 for _ in settings.ADAPTARITH_TOPICS]
        for q in questions:
            if q.response == q.get_correct_answer():
                knowledge[settings.ADAPTARITH_TOPICS.index(
                    q.topic)] += settings.ADAPTARITH_INITIAL_KNOWLEDGE_POINTS_PER_QUESTION
        return knowledge

