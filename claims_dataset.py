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
        
        encoding = self.tokenizer(
            claim,
            evidence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove the batch dimension added by return_tensors='pt'
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        if not self.is_test:
            label = self.data.iloc[idx]['label']
            item['labels'] = torch.tensor(label, dtype=torch.long)
    
        return item
