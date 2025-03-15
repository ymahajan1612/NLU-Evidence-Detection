import torch
from torch.utils.data import Dataset

class EvidenceDetectionDataset(Dataset):
    def __init__(self, data_df, vocab, max_len=100, is_test=False):
        self.claims = data_df['Claim'].tolist()
        self.evidences = data_df['Evidence'].tolist()
        self.is_test = is_test
        self.vocab = vocab
        self.max_len = max_len

        # Only set labels if not in test mode
        if not self.is_test and 'label' in data_df.columns:
            self.labels = data_df['label'].tolist()
        else:
            self.labels = None           

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim = str(self.claims[idx])
        evidence = str(self.evidences[idx])

        # Convert text to numerical values
        claim_ids = self.vocab.numericalize(claim)
        evidence_ids = self.vocab.numericalize(evidence)

        # Truncate if necessary
        if len(claim_ids) > self.max_len:
            claim_ids = claim_ids[:self.max_len]

        if len(evidence_ids) > self.max_len:
            evidence_ids = evidence_ids[:self.max_len]

        item = {
            'claim_ids': torch.tensor(claim_ids),
            'claim_length': len(claim_ids),
            'evidence_ids': torch.tensor(evidence_ids),
            'evidence_length': len(evidence_ids),
        }

        # Add labels if available (not in test mode)
        if not self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item