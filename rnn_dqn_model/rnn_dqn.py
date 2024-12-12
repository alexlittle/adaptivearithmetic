import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(LSTM_DQN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)

        # Fully connected layer to output Q-values
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # Pass input through LSTM
        out, hidden = self.lstm(x, hidden)

        # Get the last time-step output (batch_size, hidden_size)
        if out.dim() == 3:  # (batch_size, seq_len, hidden_size)
            out = out[:, -1, :]  # Take the last time step

        # If batch size is 1, the output will have shape [seq_len, hidden_size]
        elif out.dim() == 2:  # (seq_len, hidden_size)
            out = out[-1, :]

        # Pass the LSTM output through the fully connected layer
        q_values = self.fc(out)
        return q_values, hidden

    def init_hidden(self, batch_size, device):
        # If batch size is 1, hidden state is 2D, else it is 3D
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)

    def init_hidden(self, batch_size, device):
        # If batch size is 1, hidden state is 2D, else it is 3D
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)