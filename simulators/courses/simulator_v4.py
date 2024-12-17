import gym
from gym import spaces
import numpy as np
import pandas as pd

class LearnerActivityEnv(gym.Env):
    """
    Gym environment for learner activity data.

    Args:
        data: Pandas DataFrame containing learner activity data.
            Columns: user_id, activity_id, score, timestamp
        num_activities: Total number of activities.
        max_activities_per_user: Maximum number of activities completed by a user.
    """

    def __init__(self, data, pre_score, num_activities, max_activities_per_user, max_steps, hmm_model):
        super(LearnerActivityEnv, self).__init__()

        self.data = data
        self.pre_score_data = pre_score
        self.num_activities = num_activities
        self.max_activities_per_user = max_activities_per_user
        self.hmm_model = hmm_model
        # Define action and observation spaces
        self.action_space = spaces.Discrete(num_activities)  # Action: next activity_id
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_activities + 4,), dtype=np.float32
        )  # Observation: one-hot encoding of completed activities

        self.max_steps = max_steps
        self.current_step = 0

        self.previous_action = None
        # Initialize episode
        self.reset()

    def step(self, action):
        """
        Take an action (choose next activity).

        Args:
            action: Index of the chosen activity.

        Returns:
            observation: One-hot encoding of completed activities.
            reward: Reward based on the chosen activity (e.g., score, completion rate).
            done: True if the episode is finished (all activities completed or maximum activities reached).
            info: Additional information (e.g., current activity, score).
        """
        self.current_step += 1
        # Check if the chosen activity is valid
        if action not in self.available_activities:
            done = self.check_episode_done()
            return self.observation, 0, done, {"info": "Invalid action"}

        # Calculate reward (e.g., based on score or completion rate)
        reward = self.calculate_reward(action)

        # Update user's activity history
        self.user_activities[action] = 1

        # Check if episode is finished
        done = self.check_episode_done()

        # Update observation (include HMM state and pre-score)
        current_state = np.array(self.current_state)

        self.observation = np.concatenate((self.user_activities, [current_state], [self.user_pre_score]))

        return self.observation, reward, done, {"info": {"current_activity": action}}

    def reset(self):
        """
        Reset the environment to a new user.

        Returns:
            observation: One-hot encoding of completed activities for the new user.
        """
        # Select a random user
        self.user_id = np.random.choice(self.data["user_id"].unique())
        self.user_data = self.data[self.data["user_id"] == self.user_id]

        self.user_pre_score = self.pre_score_data.loc[self.pre_score_data['user_id'] == self.user_id, 'pre_score'].iloc[0]

        # Initialize user's activity history
        self.user_activities = np.zeros(self.num_activities)

        # Get available activities for the user
        self.available_activities = set(self.user_data["activity_id"])

        self.predicted_state_sequence = []
        # Predict initial HMM state based on pre-score (assuming pre-score is informative)
        self.current_state = self.predict_hmm_state()
        current_state = np.array([self.current_state])
        # Update observation
        self.observation = np.concatenate((self.user_activities, current_state, [self.user_pre_score]))
        self.current_step = 0
        self.previous_action = None
        return self.observation

    def render(self, mode="human"):
        """
        Render the environment (optional).
        """
        print(f"User: {self.user_id}")
        print(f"Completed Activities: {np.where(self.user_activities == 1)[0]}")

    def calculate_score_reward(self, action):
        """
        Calculate reward based on the chosen action.

        Args:
            action: Index of the chosen activity.

        Returns:
            reward: Reward value.
        """
        # Example: Reward based on score
        if self.user_activities[action] == 1:
            # Activity already completed, give no reward or a small penalty
            return -0.1

        completed_activities = np.where(self.user_activities == 1)[0]
        if len(completed_activities) > 0:
            if action <= np.max(completed_activities):
                return 0  # Penalty for choosing an activity that should have been done earlier

        user_activity_data = self.user_data[self.user_data["activity_id"] == action]
        if not user_activity_data.empty:
            return user_activity_data["score"].values[0]
        else:
            return 0

    def calculate_reward(self, action):
        """
        Calculate reward based on chosen action and HMM alignment.

        Args:
            action: Index of the chosen activity.

        Returns:
            reward: Reward value.
        """
        base_reward = self.calculate_score_reward(action)

        # Get current HMM state
        self.current_state = self.predict_hmm_state()

        # Get HMM transition probabilities for the current state
        transition_probs = self.hmm_model.transmat_[self.current_state, :]
        most_probable_states = np.argsort(transition_probs)[::-1]
        if base_reward > 0.6:  # High success threshold
            # Prioritize states with high transition probabilities
            # and potentially move to a "higher" state
            expected_next_state = most_probable_states[0]
        elif base_reward < 0.3:  # Low performance threshold
            # Force transition to a lower/recovery state
            expected_next_state = 2  # or use a more dynamic selection
        else:
            # Medium reward: use probabilistic state transition
            expected_next_state = np.random.choice(
                len(transition_probs),
                p=transition_probs
            )
        print(expected_next_state)
        # Calculate HMM alignment reward
        hmm_alignment_reward = transition_probs[expected_next_state]

        # Combine base reward with HMM alignment reward
        total_reward = base_reward + 0.1 * hmm_alignment_reward  # Adjust the weight (0.1) as needed

        # Update previous action
        self.previous_action = action

        return total_reward

    def predict_hmm_state(self):
        """
        Predicts the current HMM state based on previous actions and scores.

        Returns:
            Predicted HMM state (index).
        """
        # Extract relevant observation sequence for HMM prediction
        observation_sequence = np.array(self.predicted_state_sequence)
        # Handle the case of no completed activities (e.g., at the beginning of an episode)
        if len(observation_sequence) == 0:
            #observation_sequence = np.pad(observation_sequence, (0, 3 - len(observation_sequence)), mode='constant', constant_values=0)
            observation_sequence = np.array([0])


        observation_sequence = observation_sequence.reshape(1, -1)
        observation_sequence = observation_sequence.astype(int)

        # Use HMM model to predict the most likely state sequence
        prob, state_sequence = self.hmm_model.decode(observation_sequence)

        self.predicted_state_sequence.append(state_sequence[-1])
        # Return the predicted current state
        return state_sequence[-1]

    def check_episode_done(self):
        """
        Check if the episode is finished.

        Returns:
            True if the episode is finished, False otherwise.
        """
        if self.current_step >= self.max_steps:
            return True
        return (
            len(np.where(self.user_activities == 1)[0])
            >= self.max_activities_per_user
            or len(self.available_activities) == 0
        )