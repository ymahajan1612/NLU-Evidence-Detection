import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import time
from tqdm import tqdm

from early_stopping import EarlyStopping

class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the trainer with the model and datasets.
        
        Args:
            model: The BiLSTM with Attention model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to train on (cuda or cpu)
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=3, verbose=True)
        
    def collate_fn(self, batch):
        """
        Custom collate function to handle variable-length sequences.
        
        Args:
            batch: A batch of data from the dataset
            
        Returns:
            Dictionary with padded sequences and other batch information
        """
        # Separate batch elements
        claims = [item['claim_ids'] for item in batch]
        claim_lengths = torch.tensor([item['claim_length'] for item in batch])
        evidences = [item['evidence_ids'] for item in batch]
        evidence_lengths = torch.tensor([item['evidence_length'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        # Pad sequences
        padded_claims = pad_sequence(claims, batch_first=True, padding_value=0)
        padded_evidences = pad_sequence(evidences, batch_first=True, padding_value=0)
        
        return {
            'claim_ids': padded_claims,
            'claim_lengths': claim_lengths,
            'evidence_ids': padded_evidences,
            'evidence_lengths': evidence_lengths,
            'labels': labels
        }
    
    def train_epoch(self):
        """
        Train model for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            claim_ids = batch['claim_ids'].to(self.device)
            claim_lengths = batch['claim_lengths']
            evidence_ids = batch['evidence_ids'].to(self.device)
            evidence_lengths = batch['evidence_lengths']
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(claim_ids, claim_lengths, evidence_ids, evidence_lengths)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate(self):
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                claim_ids = batch['claim_ids'].to(self.device)
                claim_lengths = batch['claim_lengths']
                evidence_ids = batch['evidence_ids'].to(self.device)
                evidence_lengths = batch['evidence_lengths']
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits, _ = self.model(claim_ids, claim_lengths, evidence_ids, evidence_lengths)
                loss = self.criterion(logits, labels)
                
                # Accumulate metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, num_epochs=10):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Maximum number of epochs to train for
            
        Returns:
            Trained model
        """
        print(f"Starting training on device: {self.device}")
        print(f"Training set size: {len(self.train_loader.dataset)}")
        print(f"Validation set size: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate()
            
            elapsed_time = time.time() - start_time
            
            # Print metrics
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {elapsed_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['f1'])
            
            # Check early stopping
            self.early_stopping(val_metrics['f1'], self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load('best_bilstm_attention_model.pt'))
        print("Loaded best model from 'best_bilstm_attention_model.pt'")
        return self.model
    
    def get_attention_weights(self, claim, evidence, vocab):
        """
        Get attention weights for a claim-evidence pair.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            vocab: Vocabulary object
            
        Returns:
            Attention weights for claim and evidence
        """
        self.model.eval()
        
        # Prepare inputs
        claim_ids = self._prepare_input(claim, vocab)
        evidence_ids = self._prepare_input(evidence, vocab)
        
        with torch.no_grad():
            # Forward pass
            _, (claim_attention, evidence_attention) = self.model(
                claim_ids.unsqueeze(0).to(self.device),
                torch.tensor([claim_ids.size(0)]),
                evidence_ids.unsqueeze(0).to(self.device),
                torch.tensor([evidence_ids.size(0)])
            )
        
        return claim_attention.cpu().numpy(), evidence_attention.cpu().numpy()
    
    def _prepare_input(self, text, vocab):
        """
        Convert text to tensor of token IDs.
        
        Args:
            text: Input text
            vocab: Vocabulary object
            
        Returns:
            Tensor of token IDs
        """
        tokens = vocab.numericalize(text)
        return torch.tensor(tokens, dtype=torch.long)
    
    def predict(self, claim, evidence, vocab):
        """
        Make a prediction for a claim-evidence pair.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            vocab: Vocabulary object
            
        Returns:
            Prediction (0 or 1) and probabilities
        """
        self.model.eval()
        
        # Prepare inputs
        claim_ids = self._prepare_input(claim, vocab)
        evidence_ids = self._prepare_input(evidence, vocab)
        
        with torch.no_grad():
            # Forward pass
            logits, _ = self.model(
                claim_ids.unsqueeze(0).to(self.device),
                torch.tensor([claim_ids.size(0)]),
                evidence_ids.unsqueeze(0).to(self.device),
                torch.tensor([evidence_ids.size(0)])
            )
            
            probability = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
        
        return prediction, probability.cpu().numpy()[0]
    
    def save_model(self, path='trained_model.pt'):
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        