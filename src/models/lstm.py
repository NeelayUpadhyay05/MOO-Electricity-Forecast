import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
        output_dim=24
    ):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """

        lstm_out, (hidden, cell) = self.lstm(x)

        # Take last hidden state from final layer
        last_hidden = hidden[-1]  # shape: (batch_size, hidden_dim)

        out = self.fc(last_hidden)  # shape: (batch_size, output_dim)

        return out