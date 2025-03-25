import torch
import torch.nn as nn
from transformers import BertModel 

class EvidenceClassifier(nn.Module):
    def __init__(self, dropout_rate=0.2, num_labels=2):
        super(EvidenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # Get the BERT outputs; we use the pooled output ([CLS] token representation)
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits