import torch
import torch.nn as nn
from transformers import AutoModel 

class SiameseBert(nn.Module):
    """
    Siamese Bert model for evidence detection
    """
    def __init__(self):
        super(SiameseBert, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size * 2, 2)
    
    def forward(self, claim_ids, evidence_ids, claim_attention_mask, evidence_attention_mask):
        claim_output = self.bert(input_ids = claim_ids, attention_mask = claim_attention_mask).last_hidden_state[:,0,:]
        evidence_output = self.bert(input_ids = evidence_ids, attention_mask = evidence_attention_mask).last_hidden_state[:,0,:]

        # Concatenate the output of the two branches
        combined = torch.cat((claim_output, evidence_output), dim=1)
        logits = self.fc(combined)
        return logits

   