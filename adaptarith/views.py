from django.views.generic import TemplateView, FormView
from django.contrib.auth.views import LogoutView
from django.urls import reverse_lazy, reverse
from django.shortcuts import redirect
from django.conf import settings

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
    quiz = Quiz()
    questions = quiz.generate_pre_test()
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
        question = self.get_question()
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

        db_questions = Question.objects.filter(pk__in=question_ids)
        engine_questions = []
        for q in db_questions:
            qe = QuestionEngine()
            qe.ft = q.first_term
            qe.st = q.second_term
            qe.level = q.level
            qe.topic = q.topic
            qe.set_answer(q.response)
            engine_questions.append(qe)

        quiz = QuizEngine()
        quiz.questions = engine_questions
        kls = quiz.init_knowledge_levels()
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
        #generate
        # TODO
        next_question = utils.get_next_question(self.request.user, knowledge_levels)

        #save to DB
        next_q_db = Question()
        next_q_db.user = self.request.user
        next_q_db.first_term = next_question.ft
        next_q_db.second_term = next_question.st
        next_q_db.level = next_question.level
        next_q_db.topic = next_question.topic
        next_q_db.save()
        # put in session
        self.request.session['current_question'] = next_q_db.id
        self.request.session.modified = True

        return next_q_db

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['knowledge_levels'] = KnowledgeLevel.get_latest_for_user(self.request.user)
        next_question = self.get_question(context['knowledge_levels'])
        context['question'] = utils.format_question(next_question)
        return context

    def form_valid(self, form):
        # get from session
        question = self.request.session['current_question']
        question.response = form.cleaned_data['response']
        question.save()

        # mark and update knowledge level
        qe = QuestionEngine()
        qe.ft = question.first_term
        qe.st = question.second_term
        qe.level = question.level
        qe.topic = question.topic
        qe.set_answer(question.response)

        # TODO

        # if is fully complete move to passed!

        # else redirect to next question
        return redirect(reverse('adaptarith:run'))

