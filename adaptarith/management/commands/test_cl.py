from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from adaptarith import utils
from adaptarith.models import KnowledgeLevel


class Command(BaseCommand):
    help = _(u"For testing the model via command line")
    errors = []

    def add_arguments(self, parser):
        parser.add_argument(
            '--model_pth',
            type=str,
            default=settings.ADAPTARITH_MODEL_PATH,
            help=f'Model pth file to use, (default {settings.ADAPTARITH_MODEL_PATH})'
        )

    def handle(self, *args, **options):
        model_pth = options['model_pth']
        print(f"Using model: {model_pth}")
        print("Pre-test")
        print("-----------------")

        # training=True so questions etc not saved to DB
        pre_test = utils.generate_pre_test(training=True)

        for question in pre_test:
            default = question.get_correct_answer()
            question_string = utils.format_question(question)
            user_input = input(f"{question_string} = ({default}) ")
            question.response = user_input if user_input else default

        # mark and get initial knowledge
        knowledge_level = KnowledgeLevel.pre_test_init_knowledge_level(pre_test)
        print("-----------------")
        print("Pre-test completed: ")
        print(f"Knowledge level: {knowledge_level}")
        print("-----------------")


        while True:
            next_question = utils.get_next_question(user=None,
                                                    knowledge_level=knowledge_level,
                                                    model_pth=model_pth)

            question_string = utils.format_question(next_question)
            default = next_question.get_correct_answer()

            # get user input - pressing enter will just give the correct response
            user_input = input(f"{question_string} = ({default})")

            next_question.response = user_input if user_input else default
            score = next_question.mark_question()
            knowledge_level[settings.ADAPTARITH_TOPICS.index(next_question.topic)] += score
            print(f"New knowledge level: {knowledge_level}")
