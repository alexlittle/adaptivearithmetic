import gym
import random
import numpy as np

from gym import spaces

from django.conf import settings
from adaptarith import utils

class LearnerEnv(gym.Env):

    def __init__(self):
        super(LearnerEnv, self).__init__()

        # Observation space: 4 continuous values for knowledge levels of 4 topics
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)

        # Action space: num levels * num topics
        self.action_space = spaces.Discrete(len(settings.ADAPTARITH_LEVELS) * len(settings.ADAPTARITH_TOPICS))

        self.state = self._simulate_pre_test()
        self.current_step = 0

    def is_done(self):
        if self.current_step >= settings.ADAPTARITH_TRAINING['max_steps']:
            return True
        for i in range(0, len(self.state)):
            if self.state[i] < settings.ADAPTARITH_PASS_THRESHOLD:
                return False
        return True

    def _normalise_state(self):
        return [x / 100.0 for x in self.state]

    def _normalise_reward(self, reward):
        return reward / settings.ADAPTARITH_MAX_GAIN

    def step(self, action):
        # 'translate' action into level and topic
        level = action // len(settings.ADAPTARITH_LEVELS)
        topic = action % len(settings.ADAPTARITH_TOPICS)

        # Simulate knowledge gain or loss based on the action
        topic_str = settings.ADAPTARITH_TOPICS[topic]
        level_str = settings.ADAPTARITH_LEVELS[level]
        reward = self._simulate_learning(level_str, topic_str)

        # Update state (a change in knowledge levels)
        self.state[topic] += reward
        self.state[topic] = min(self.state[topic], 100)
        self.state[topic] = max(self.state[topic], 0)
        # Increment step
        self.current_step += 1

        # Check if the episode is done
        done = self.is_done()

        # Return updated state, reward, done
        return self._normalise_state(), self._normalise_reward(reward), done, {}

    def reset(self):
        # Reset state and step counter
        self.state = self._simulate_pre_test()
        self.current_step = 0
        return self._normalise_state()

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Knowledge Levels: {self.state}")

    def _simulate_pre_test(self):
        k_levels = [0,0,0,0]
        # randomly set from 0,5,10,15 ...95
        k_levels = [random.choice(range(0, 100, 5)) for _ in k_levels]
        return k_levels

    def _simulate_learning(self, level_str, topic_str, training=True):

        # get a question for given topic/level
        question = utils.generate_question(topic_str, level_str, pretest=False, training=training)

        # if the question is in the same level as the user state then mark correct 2/3 of the time
        if question.level == level_str and question.first_term % 3 != 0:
            question.response = question.get_correct_answer()

        # mark if correct & set the points
        return question.mark_question()