import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        attention_weights = F.softmax(self.attention(lstm_output).squeeze(-1), dim=1)
        # attention_weights: [batch_size, seq_len]

        # Apply attention weights to LSTM outputs
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_output
        ).squeeze(1)
        # context_vector: [batch_size, hidden_dim]

        return context_vector, attention_weights
    