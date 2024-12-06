from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone
from django.db.models import Max


class Question(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    create_date = models.DateTimeField(default=timezone.now)
    pre_test = models.BooleanField(default=False)
    first_term = models.IntegerField(default=0)
    second_term = models.IntegerField(default=0)
    level = models.CharField(max_length=10, blank=False, null=False)
    topic = models.CharField(max_length=10, blank=False, null=False)
    response = models.IntegerField(default=None, blank=True, null=True)

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


