import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        scores = self.attn_weights(lstm_output).squeeze(-1)  # [batch_size, seq_len]
        attn_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # [batch_size, hidden_dim]
        return context, attn_weights  # [batch_size, hidden_dim], [batch_size, seq_len]

class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = Attention(lstm_out_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        context, attn_weights = self.attention(lstm_out)  # [batch_size, hidden_dim*2], [batch_size, seq_len]
        return context, attn_weights, lstm_out  # [batch_size, hidden_dim*2], [batch_size, seq_len], [batch_size, seq_len, hidden_dim*2]