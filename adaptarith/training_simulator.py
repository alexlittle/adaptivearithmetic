import gym
import random
import numpy as np
import math
from gym import spaces

from django.conf import settings

class LearnerEnv(gym.Env):

    def __init__(self, max_steps):
        super(LearnerEnv, self).__init__()
        self.current_step = 0
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0, high=100, shape=(len(settings.ADAPTARITH_TOPICS),), dtype=np.float32)

        # Action space: num levels * num topics
        self.action_space = spaces.Discrete(len(settings.ADAPTARITH_LEVELS) * len(settings.ADAPTARITH_TOPICS))

        self.state = self._simulate_pre_test()
        self.last_actions = []

    def is_done(self):
        if self.current_step >= self.max_steps:
            return True
        for i in range(0, len(self.state)):
            if self.state[i] < settings.ADAPTARITH_PASS_THRESHOLD:
                return False
        return True

    def _normalise_state(self):
        return [x / 100.0 for x in self.state]

    def _normalise_reward(self, reward):
        normalized_reward =  reward / settings.ADAPTARITH_MAX_GAIN
        clipped_reward = np.clip(normalized_reward, -1, 1)
        return clipped_reward

    def _simulate_pre_test(self):
        return [random.choice(range(0,90)) for _ in settings.ADAPTARITH_TOPICS]

    def step(self, action):

        self.last_actions.append(action)
        if len(self.last_actions) > 7:
            self.last_actions.pop(0)

        # Apply a penalty for repetition
        if len(self.last_actions) == 7 and all(a == self.last_actions[0] for a in self.last_actions):
            repetition_penalty = 2
        else:
            repetition_penalty = 0

        # 'translate' action into level and topic
        level = action // len(settings.ADAPTARITH_TOPICS)
        topic = action % len(settings.ADAPTARITH_TOPICS)

        reward = self.simulate_learning(level, topic) - repetition_penalty
        self.state[topic] += reward
        self.state[topic] = min(self.state[topic], 100)
        self.state[topic] = max(self.state[topic], 0)

        self.current_step += 1
        done = self.is_done()

        return self._normalise_state(), self._normalise_reward(reward), done, {}


    def simulate_learning(self, level, topic):
        correct = self.is_correct(level, topic)
        reward = self.calculate_reward(level, topic, correct)

        return reward

    def is_correct(self, level, topic):
        alpha = 0.2  # Scaling factor for probability decay

        # Calculate the learner's knowledge band (0–3)
        knowledge_band = self.state[topic] // (100/len(settings.ADAPTARITH_LEVELS))

        # Calculate the difficulty mismatch
        difficulty_mismatch = abs(level - knowledge_band)

        # Calculate the probability of a correct answer
        p_correct = math.exp(-alpha * (difficulty_mismatch ** 2))

        # Check if the learner answers correctly (based on p_correct)
        return random.random() < p_correct

    def calculate_reward(self, level, topic, is_correct):
        # Base reward calculation
        base_reward = settings.ADAPTARITH_POINTS_FOR_CORRECT if is_correct else -settings.ADAPTARITH_POINTS_FOR_CORRECT
        knowledge_band = self.state[topic] // (100/len(settings.ADAPTARITH_LEVELS))
        difficulty_mismatch = abs(level - knowledge_band)
        # Apply reward penalty or bonus based on difficulty mismatch
        if difficulty_mismatch == 0:
            reward = base_reward
        elif difficulty_mismatch == 1:
            reward = 0
        elif difficulty_mismatch == 2:
            reward = -2.5
        else:
            reward = -settings.ADAPTARITH_POINTS_FOR_CORRECT

        # keep rewards within bounds
        return max(-settings.ADAPTARITH_POINTS_FOR_CORRECT, min(settings.ADAPTARITH_POINTS_FOR_CORRECT, reward))

    def reset(self):
        return self.reset_to_value(value=0)

    def reset_to_value(self, value=0):
        # Reset state and step counter
        self.state = [value for _ in settings.ADAPTARITH_TOPICS]
        self.current_step = 0
        self.last_actions = []
        return self._normalise_state()

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Knowledge Levels: {self.state}")