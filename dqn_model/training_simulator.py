import gym
import random
import numpy as np

from gym import spaces

from config import *
from quiz_engine import Quiz, Question


class LearnerEnv(gym.Env):

    def __init__(self):
        super(LearnerEnv, self).__init__()

        # Observation space: 4 continuous values for knowledge levels of 4 topics
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)

        # Action space: 4 levels × 4 topics = 16 discrete actions
        self.action_space = spaces.Discrete(16)

        self.state = self._simulate_pre_test()
        self.current_step = 0

    def is_done(self):
        if self.current_step >= TRAINING_MAX_STEPS:
            return True
        for i in range(0, len(self.state)):
            if self.state[i] < PASS_THRESHOLD:
                return False
        return True

    def _normalise_state(self):
        return [x / 100.0 for x in self.state]

    def _normalise_reward(self, reward):
        return reward / MAX_GAIN

    def step(self, action):
        # Decode action into level and topic
        level = action // 4  # 0, 1, 2
        topic = action % 4  # 0, 1, 2

        # Simulate knowledge gain or loss based on the action
        reward = self._simulate_learning(level, topic)

        # Update state (simulate a change in knowledge levels)
        self.state[topic] += reward
        self.state[topic] = min(self.state[topic], 100)
        self.state[topic] = max(self.state[topic], 0)
        # Increment step
        self.current_step += 1

        # Check if the episode is done
        done = self.is_done()

        # Return updated state, reward, done
        return self._normalise_state(), self._normalise_reward(reward), done

    def reset(self):
        # Reset state and step counter
        self.state = self._simulate_pre_test()
        self.current_step = 0
        return self._normalise_state()

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Knowledge Levels: {self.state}")

    def _simulate_pre_test(self):
        pre_test = Quiz()
        pre_test.generate_test()

        # add some randomness for the no questions correct
        q_idx_range = range(0, len(pre_test.questions))
        num_random_qs = random.randint(0, len(q_idx_range))
        random_correct_idxes = random.sample(q_idx_range, num_random_qs)

        for idx, q in enumerate(pre_test.questions):
            if idx in random_correct_idxes:
                q.set_answer(q.get_correct_answer())

        return pre_test.init_knowledge_levels()

    def _simulate_learning(self, level, topic):

        # get a question for given topic/level
        question = Question()
        ft, st = question.generate_question(TOPICS[topic], LEVELS[level])

        # set correct 2/3 of the time
        if ft % 3 != 0:
            question.set_answer(question.get_correct_answer())

        # mark if correct & set the points
        if not question.mark_question():
            return POINTS_FOR_INCORRECT

        if self.state[topic] in LEVEL_EASY_RANGE and level == LEVELS.index('easy'):
            return POINTS_FOR_CORRECT

        if self.state[topic] in LEVEL_MOD_RANGE and level == LEVELS.index('mod'):
            return POINTS_FOR_CORRECT

        if self.state[topic] in LEVEL_HARD_RANGE and level == LEVELS.index('hard'):
            return POINTS_FOR_CORRECT

        if self.state[topic] in LEVEL_VHARD_RANGE and level == LEVELS.index('vhard'):
            return POINTS_FOR_CORRECT

        return POINTS_FOR_INCORRECT