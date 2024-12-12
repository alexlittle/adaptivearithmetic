import os
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from adaptarith import utils
from adaptarith.models import KnowledgeLevel


def find_max_subdir(directory):
    try:
        # Get the list of subdirectories
        subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

        # Filter subdirectories that are numbers
        number_subdirs = [int(d) for d in subdirs if d.isdigit()]

        # Return the maximum number, or None if no valid subdirectories are found
        return max(number_subdirs) if number_subdirs else None
    except Exception as e:
        print(f"Error: {e}")
        return None

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
        parser.add_argument(
            '--recent',
            type=str,
            default=None,
            help=f'Whether to use most recent from dir, (default false)'
        )

        parser.add_argument(
            '--no_pre_test',
            action='store_true',
            help=f'Whether to use a pretest, (default true)'
        )

    def handle(self, *args, **options):
        model_pth = options['model_pth']
        no_pre_test = options['no_pre_test']

        if options['recent']:
            directory = os.path.join(settings.BASE_DIR, options['recent'], 'results')
            max_number = find_max_subdir(directory)
            model_pth = f"{options['recent']}/results/{max_number}/model.pth"

        print(f"Using model: {model_pth}")

        if not no_pre_test:
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
        else:
            knowledge_level = [0 for _ in settings.ADAPTARITH_TOPICS]

        print(f"Knowledge level: {knowledge_level}")
        print("-----------------")


        while True:
            next_question = utils.get_next_question(user=None,
                                                    knowledge_level=knowledge_level,
                                                    model_pth=model_pth,
                                                    debug=True)

            question_string = utils.format_question(next_question)
            default = next_question.get_correct_answer()

            # get user input - pressing enter will just give the correct response
            user_input = input(f"{question_string} = ({default})")

            next_question.response = user_input if user_input else default

            score = next_question.mark_question()

            existing_score = knowledge_level[settings.ADAPTARITH_TOPICS.index(next_question.topic)]
            existing_score += score

            knowledge_level[settings.ADAPTARITH_TOPICS.index(next_question.topic)] = min(existing_score, 100)

            print(f"New knowledge level: {knowledge_level}")
