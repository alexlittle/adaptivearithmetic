import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNQNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim=128, rnn_layers=1):
        super(RNNQNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers

        # RNN Layer (LSTM)
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x, hidden=None):
        if len(x.shape) == 2:  # If single state, add sequence dimension
            x = x.unsqueeze(1)

        # Pass through RNN
        lstm_out, hidden = self.lstm(x, hidden)

        # Take only the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Pass through fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        return x, hidden