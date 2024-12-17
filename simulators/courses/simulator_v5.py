import gym
from gym import spaces
import numpy as np
import pickle
import pandas as pd
import random
import os

class HMMPredictorEnv(gym.Env):
    def __init__(self,  base_dir, max_sequence_length=5):
        super(HMMPredictorEnv, self).__init__()

        with open(os.path.join(base_dir, 'data','ou_hmm_model_full.pkl'), "rb") as file:
            hmm_model = pickle.load(file)

        activity = pd.read_csv(os.path.join(base_dir, 'data','ou','studentassessment_filtered_full.csv'))

        # filter for only one courses assessments
        self.all_activity = activity
        self.all_users = self.all_activity['id_student'].unique()
        self.hmm_model = hmm_model
        self.max_sequence_length = max_sequence_length
        self.num_score_categories = 12  # Assuming 9 score categories (adjust as needed)
        self.observation_space = spaces.Box(low=0, high=self.num_score_categories - 1, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_score_categories)  # Actions are predicted score categories
        self.score_category_sequence = []
        self.hidden_state_sequence = []
        self.current_user_id = 0
        self.current_user_data = []
        self.current_user_data_index = 0

    def reset(self):
        self.current_user_data_index = 0
        self.current_user_id = random.choice(self.all_users)
        self.current_user_data = self.all_activity.loc[self.all_activity['id_student'] == self.current_user_id].sort_values(by='date_submitted')
        first_score = self.categorize_score(self.current_user_data.iloc[self.current_user_data_index].score)
        self.score_category_sequence = [first_score]
        self.hidden_state_sequence = []
        return self._get_observation()

    def step(self, action):
        # 1. Observe the actual next score category (replace with your actual data source)
        true_next_score_category = self._get_true_next_score()
        if true_next_score_category == -1:
            return self._get_observation(), 0, True, {}
        self.score_category_sequence.append(true_next_score_category)

        # 2. Update hidden state sequence using HMM
        #predicted_score_category = self.get_next_predicted(self.score_category_sequence)

        # 4. Calculate reward
        #reward_match_hmm = 1 if action == predicted_score_category else 0
        # Reward for matching the true next score
        reward_match_true = 1 if action == true_next_score_category else 0
        # Combine rewards (you can adjust weights as needed)
        reward = reward_match_true #+ 0.5 * reward_match_hmm

        #reward = 1 if predicted_score_category == true_next_score_category else 0

        # 5. Check for episode termination (adjust as needed)
        done = len(self.score_category_sequence) >= self.max_sequence_length

        # 6. Get next observation
        next_observation = self._get_observation()

        return next_observation, reward, done, {}

    def _get_observation(self):
        padded_arr = self.score_category_sequence + [-1] * (self.max_sequence_length - len(self.score_category_sequence))
        return [x / 12.0 for x in padded_arr]

    def _get_true_next_score(self):
        # Replace this with your logic to get the actual next score
        # (e.g., from a real-world data source, a simulation, etc.)
        # This is a placeholder for demonstration
        self.current_user_data_index += 1
        try:
            next_score = self.categorize_score(self.current_user_data.iloc[self.current_user_data_index].score)
            return next_score
        except IndexError:
            return -1

    def categorize_score(self, score):
        for idx, x in enumerate(range(49, 90, 4)):
            if score < x:
                return idx
        return idx + 1

    def get_next_predicted(self, sequence):
        prob, hidden_states = self.hmm_model.decode([sequence])  # Decode the first sequence in your data
        row = self.hmm_model.transmat_[hidden_states[-1]]
        max_idx = np.argmax(row)
        next_state_emissions = self.hmm_model.emissionprob_[max_idx]
        predicted_score_category = np.argmax(next_state_emissions)
        return predicted_score_category

