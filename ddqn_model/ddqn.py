import torch.nn as nn
import torch.nn.functional as F

from django.conf import settings
"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, settings.ADAPTARITH_TRAINING['hidden_dims'])
        self.fc_2 = nn.Linear(settings.ADAPTARITH_TRAINING['hidden_dims'], settings.ADAPTARITH_TRAINING['hidden_dims'])
        self.fc_3 = nn.Linear(settings.ADAPTARITH_TRAINING['hidden_dims'], action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1