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

    def __init__(self, data, num_activities, max_activities_per_user, max_steps):
        super(LearnerActivityEnv, self).__init__()

        self.data = data
        self.num_activities = num_activities
        self.max_activities_per_user = max_activities_per_user

        # Define action and observation spaces
        self.action_space = spaces.Discrete(num_activities)  # Action: next activity_id
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_activities,), dtype=np.float32
        )  # Observation: one-hot encoding of completed activities

        self.max_steps = max_steps
        self.current_step = 0
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

        # Update observation
        self.observation = self.user_activities

        return self.user_activities, reward, done, {"info": {"current_activity": action}}

    def reset(self):
        """
        Reset the environment to a new user.

        Returns:
            observation: One-hot encoding of completed activities for the new user.
        """
        # Select a random user
        self.user_id = np.random.choice(self.data["user_id"].unique())
        self.user_data = self.data[self.data["user_id"] == self.user_id]

        # Initialize user's activity history
        self.user_activities = np.zeros(self.num_activities)

        # Get available activities for the user
        self.available_activities = set(self.user_data["activity_id"])

        # Update observation
        self.observation = self.user_activities
        self.current_step = 0
        return self.user_activities

    def render(self, mode="human"):
        """
        Render the environment (optional).
        """
        print(f"User: {self.user_id}")
        print(f"Completed Activities: {np.where(self.user_activities == 1)[0]}")

    def calculate_reward(self, action):
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
            return 1 # user_activity_data["score"].values[0]
        else:
            return 0

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