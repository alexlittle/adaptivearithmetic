from django.views.generic import TemplateView, FormView
from django.contrib.auth.views import LogoutView
from django.urls import reverse_lazy, reverse
from django.shortcuts import redirect
from django.conf import settings
from django.db.models import Max

from adaptarith.models import Question, KnowledgeLevel
from adaptarith.forms import AnswerForm
from adaptarith import utils

class HomeView(TemplateView):
    template_name = 'adaptarith/home.html'


class UserLogoutView(LogoutView):
    next_page = reverse_lazy('adaptarith:index')


def start_pretest(request):

    # remove any current questions from session
    request.session.pop('question_ids', None)
    request.session.pop('current_question_index', None)

    # Generate a pre test - saving questions to db
    questions = utils.generate_pre_test(user=request.user)
    question_ids = []

    for q in questions:
        question_ids.append(q.id)

    # Save the randomized question order in the session
    request.session['question_ids'] = question_ids
    request.session['current_question_index'] = 0
    return redirect(reverse('adaptarith:pretest_question'))


class PreTestQuestionView(FormView):
    template_name = 'adaptarith/pretest.html'
    form_class = AnswerForm

    def get_question(self):
        # Retrieve the current question based on the session index
        question_ids = self.request.session.get('question_ids', [])
        current_index = self.request.session.get('current_question_index', 0)

        if current_index < len(question_ids):
            question_id = question_ids[current_index]
            return Question.objects.get(pk=question_id)
        return None  # No more questions

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs.update({'initial': {'response': None}})
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['question'] = utils.format_question(self.get_question())
        context['progress'] = {
            'current': self.request.session.get('current_question_index', 0) + 1,
            'total': len(self.request.session.get('question_ids', [])),
        }
        return context

    def form_valid(self, form):
        question = self.get_question()
        if not question:
            return redirect('pretest_complete')

        # Process the user's answer
        question.response = form.cleaned_data['response']
        question.save()

        # Update session index
        self.request.session['current_question_index'] += 1
        self.request.session.modified = True

        # Redirect to the next question or finish
        if self.request.session['current_question_index'] >= len(self.request.session['question_ids']):
            # mark and save knowledge levels
            self.save_knowledge_levels(self.request.session['question_ids'])
            return redirect(reverse('adaptarith:pretest_complete'))
        return redirect(reverse('adaptarith:pretest_question'))

    def save_knowledge_levels(self, question_ids):
        questions = Question.objects.filter(pk__in=question_ids)

        kl = KnowledgeLevel()
        kls = kl.pre_test_init_knowledge_level(questions)
        for idx, kl in enumerate(kls):
            know_level = KnowledgeLevel()
            know_level.user = self.request.user
            know_level.topic = settings.ADAPTARITH_TOPICS[idx]
            know_level.score = kl
            know_level.save()



class PreTestCompleteView(TemplateView):
    template_name = 'adaptarith/pretest_complete.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['knowledge_levels'] = KnowledgeLevel.get_latest_for_user(self.request.user)
        return context


class RunView(FormView):
    template_name = 'adaptarith/run.html'
    form_class = AnswerForm

    def get_question(self, knowledge_levels):

        try:
            q_id = self.request.session['current_question']
        except KeyError:
            #generate
            q_id = utils.get_next_question(user=self.request.user).id

        self.request.session['current_question'] = q_id
        self.request.session.modified = True

        return q_id

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['knowledge_levels'] = KnowledgeLevel.get_latest_for_user(self.request.user)
        next_question_id = self.get_question(context['knowledge_levels'])
        question = Question.objects.get(pk=next_question_id)
        context['question'] = utils.format_question(question)
        return context

    def form_valid(self, form):
        # get from session
        q_id = self.request.session['current_question']
        question = Question.objects.get(pk=q_id)
        question.response = form.cleaned_data['response']
        question.save()

        score = 0
        if question.response == question.get_correct_answer():
            score = settings.ADAPTARITH_POINTS_FOR_CORRECT

        latest_score = KnowledgeLevel.get_latest_for_topic(self.request.user, question.topic)
        if score != 0:
            kl = KnowledgeLevel()
            kl.user = self.request.user
            kl.topic = question.topic
            kl.score = min(latest_score + score, 100)
            kl.save()

        # remove current question for session
        self.request.session.pop('current_question', None)
        # if is fully complete move to passed!
        passed = True
        kl = KnowledgeLevel.get_latest_for_user(self.request.user)
        for i in kl:
            if i.score < 90:
                passed = False
        if passed:
            return redirect(reverse('adaptarith:passed'))
        # else redirect to next question
        return redirect(reverse('adaptarith:run'))

class PassedView(TemplateView):
    template_name = 'adaptarith/passed.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['knowledge_levels'] = KnowledgeLevel.get_latest_for_user(self.request.user)
        return context