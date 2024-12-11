from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils.translation import gettext_lazy as _

from adaptarith.training_simulator import LearnerEnv

class Command(BaseCommand):
    help = _(u"For testing the simulator via command line")
    errors = []

    def add_arguments(self, parser):
        parser.add_argument(
            '--auto',
            type=bool,
            default=False,
            help=f'Auto test True/False)'
        )

    def handle(self, *args, **options):
        env = LearnerEnv()
        knowledge = env.reset()

        knowledge, reward, _, _ = env.step(0)
        print(knowledge, reward)

        knowledge, reward, _, _ = env.step(0)
        print(knowledge, reward)

        knowledge, reward, _, _ = env.step(0)
        print(knowledge, reward)

        return
        knowledge, reward, _, _ = env.step(0)
        print(knowledge, reward)
        knowledge, reward, _, _ = env.step(0)
        print(knowledge, reward)
        knowledge, reward, _, _ = env.step(0)
        print(knowledge, reward)



        """
        if not options['auto']:
            print(knowledge)
            while True:
                action = input("Action: ")
                knowledge, reward, _, _ = env.step(int(action))
                print(reward, knowledge)

        else:
            for i in range(0,85,5):
                knowledge = env.reset_to_value(i)
                print(knowledge)
                print("\t".join(str(x) for x in settings.ADAPTARITH_LEVELS))
                for l in range(0,len(settings.ADAPTARITH_LEVELS)):
                    reward =  env.calculate_reward(l,0, True)
                    print(reward, end="\t")
                print("")
        """


