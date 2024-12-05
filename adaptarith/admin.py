from django.contrib import admin

from adaptarith.models import Question, KnowledgeLevel

@admin.register(Question)
class QuestionResourceAdmin(admin.ModelAdmin):
    list_display = ('user', 'pre_test', 'first_term', 'second_term', 'level', 'topic', 'response')


@admin.register(KnowledgeLevel)
class KnowledgeLevelResourceAdmin(admin.ModelAdmin):
    list_display = ('user', 'create_date', 'topic', 'score')