import torch
import torch.nn as nn

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # BiLSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Attention layer
        self.attention = nn.Linear(2 * hidden_dim, 1)

        # Fully connected layer
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # shape: [batch_size, seq_len, 2*hidden_dim]

        # Apply attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # shape: [batch_size, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # shape: [batch_size, 2*hidden_dim]

        # Output layer
        output = self.fc(context)  # shape: [batch_size, output_dim]
        return output
