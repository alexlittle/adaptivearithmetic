import torch.nn as nn
import torch.nn.functional as F

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_dims, output_size):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(input_size, hidden_dims)
        self.fc_2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc_3 = nn.Linear(hidden_dims, output_size)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1