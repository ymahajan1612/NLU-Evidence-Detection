import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizerFast

class ClaimEvidenceDataset(Dataset):
    def __init__(self, data_path, max_length=128, is_test=False):
        self.data = pd.read_csv(data_path)
        self.max_length = max_length
        self.is_test = is_test
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        claim = self.data.iloc[idx]['Claim']
        evidence = self.data.iloc[idx]['Evidence']
        claim_encoding = self.tokenizer(claim, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        evidence_encoding = self.tokenizer(evidence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        

        item = {
            'claim_input_ids': claim_encoding['input_ids'].flatten(),
            'claim_attention_mask': claim_encoding['attention_mask'].flatten(),
            'evidence_input_ids': evidence_encoding['input_ids'].flatten(),
            'evidence_attention_mask': evidence_encoding['attention_mask'].flatten()
        }

        if not self.is_test:
            label = self.data.iloc[idx]['label']
            item['labels'] = torch.tensor(label, dtype=torch.long)
    
        return item