import torch
import torch.nn as nn
from transformers import BertModel 

class EvidenceClassifier(nn.Module):
    def __init__(self, dropout_rate=0.1, num_labels=2):
        super(EvidenceClassifier, self).__init__()
        # Load the pretrained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Add a dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Linear layer for classification
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # Get the BERT outputs; we use the pooled output ([CLS] token representation)
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output  # shape: [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # If labels are provided, compute the loss (using cross entropy)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits}
   