import torch
import torch.nn as nn
from transformers import AutoModel 

class CrossEncoder(nn.Module):
    """
    Cross-Encoder model for evidence detection.
    This model concatenates the claim and evidence into a single sequence.
    """
    def __init__(self):
        super(CrossEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, claim_ids, evidence_ids, claim_attention_mask, evidence_attention_mask):
        evidence_ids = evidence_ids[:, 1:]
        evidence_attention_mask = evidence_attention_mask[:, 1:]
        
        combined_input_ids = torch.cat((claim_ids, evidence_ids), dim=1)
        combined_attention_mask = torch.cat((claim_attention_mask, evidence_attention_mask), dim=1)
        
        outputs = self.bert(input_ids=combined_input_ids, attention_mask=combined_attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(self.dropout(pooled_output))
        return logits
