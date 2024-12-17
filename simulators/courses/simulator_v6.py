import gym
from gym import spaces
import numpy as np
import pickle
import pandas as pd
import random
import os

class LearningPredictorEnv(gym.Env):
    def __init__(self,  base_dir, max_sequence_length=11):
        super(LearningPredictorEnv, self).__init__()


        activity = pd.read_csv(os.path.join(base_dir, 'data','ou','studentassessment_demographic_coded.csv'))

        # filter for only one courses assessments
        self.all_activity = activity
        self.all_users = self.all_activity['id_student'].unique()
        self.max_sequence_length = max_sequence_length
        self.num_score_categories = 6  # Assuming 9 score categories (adjust as needed)
        self.observation_space = spaces.Box(low=0, high=self.num_score_categories+2, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_score_categories)  # Actions are predicted score categories
        self.score_category_sequence = []
        self.current_user_id = 0
        self.current_user_data = []
        self.current_user_data_index = 0
        self.hidden_state_sequence = []

    def reset(self):
        self.current_user_data_index = 0
        self.current_user_id = random.choice(self.all_users)
        self.current_user_data = self.all_activity.loc[self.all_activity['id_student'] == self.current_user_id].sort_values(by='date_submitted')

        first_score = self.categorize_score(self.current_user_data.iloc[self.current_user_data_index].score)
        self.score_category_sequence = [first_score]
        self.hidden_state_sequence = []
        return self._get_observation()

    def step(self, action):
        true_next_score_category = self._get_true_next_score()
        if true_next_score_category == -1:
            return self._get_observation(), 0, True, {}
        self.score_category_sequence.append(true_next_score_category)

        reward = 1 if action == true_next_score_category else 0
        # 5. Check for episode termination (adjust as needed)
        done = len(self.score_category_sequence) >= self.max_sequence_length

        next_observation = self._get_observation()

        return next_observation, reward, done, {}

    def _get_observation(self):
        padded_arr = self.score_category_sequence + [-1] * (self.max_sequence_length - len(self.score_category_sequence))

        return self.map_demographics() + [x / 11.0 if x != -1 else -1 for x in padded_arr]

    def _get_true_next_score(self):
        self.current_user_data_index += 1
        try:
            next_score = self.categorize_score(self.current_user_data.iloc[self.current_user_data_index].score)
            return next_score
        except IndexError:
            return -1

    def categorize_score(self, score):
        for idx, x in enumerate(range(50, 91, 10)):
            if score < x:
                return idx
        return idx + 1

    def map_demographics(self):
        demographics = []
        if self.current_user_data.iloc[0].gender_f:
            demographics.append(1)
        else:
            demographics.append(0)
        if self.current_user_data.iloc[0].gender_m:
            demographics.append(1)
        else:
            demographics.append(0)
        if self.current_user_data.iloc[0].highest_education_a_level_or_equivalent:
            demographics.append(1)
        else:
            demographics.append(0)

        return demographics


