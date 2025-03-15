import torch
from torch.utils.data import Dataset

class EvidenceDetectionDataset(Dataset):
    def __init__(self, data_df, vocab, max_len=100):
        self.claims = data_df['Claim'].tolist()
        self.evidences = data_df['Evidence'].tolist()
        self.labels = data_df['label'].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim = str(self.claims[idx])
        evidence = str(self.evidences[idx])
        label = self.labels[idx]

        # Convert text to numerical values
        claim_ids = self.vocab.numericalize(claim)
        evidence_ids = self.vocab.numericalize(evidence)

        # Truncate if necessary
        if len(claim_ids) > self.max_len:
            claim_ids = claim_ids[:self.max_len]

        if len(evidence_ids) > self.max_len:
            evidence_ids = evidence_ids[:self.max_len]

        return {
            'claim_ids': torch.tensor(claim_ids),
            'claim_length': len(claim_ids),
            'evidence_ids': torch.tensor(evidence_ids),
            'evidence_length': len(evidence_ids),
            'label': torch.tensor(label, dtype=torch.long)
        }