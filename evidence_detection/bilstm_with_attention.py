import torch
import torch.nn as nn

from evidence_detection.attention_layer import AttentionLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        pretrained_embeddings=None
    ):
        super(BiLSTMAttention, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # BiLSTM layers
        self.lstm_claim = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 as bidirectional will double it
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.lstm_evidence = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 as bidirectional will double it
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention layers
        self.attention_claim = AttentionLayer(hidden_dim)
        self.attention_evidence = AttentionLayer(hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, claim_ids, claim_lengths, evidence_ids, evidence_lengths):
        # Embedding
        claim_embeds = self.embedding(claim_ids)  # [batch_size, claim_len, embedding_dim]
        evidence_embeds = self.embedding(evidence_ids)  # [batch_size, evidence_len, embedding_dim]

        # Pack padded sequences for efficient computation
        packed_claim = pack_padded_sequence(
            claim_embeds,
            claim_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_evidence = pack_padded_sequence(
            evidence_embeds,
            evidence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Pass through BiLSTM
        claim_outputs, _ = self.lstm_claim(packed_claim)
        evidence_outputs, _ = self.lstm_evidence(packed_evidence)

        # Unpack sequences
        claim_outputs, _ = pad_packed_sequence(claim_outputs, batch_first=True)
        evidence_outputs, _ = pad_packed_sequence(evidence_outputs, batch_first=True)

        # Apply attention
        claim_context, claim_attention = self.attention_claim(claim_outputs)
        evidence_context, evidence_attention = self.attention_evidence(evidence_outputs)

        # Combine claim and evidence representations
        # We concatenate claim vector, evidence vector, and their element-wise product
        combined = torch.cat([
            claim_context,
            evidence_context,
            claim_context * evidence_context
        ], dim=1)

        # Apply dropout
        combined = self.dropout(combined)

        # Classify
        logits = self.classifier(combined)

        return logits, (claim_attention, evidence_attention)
    