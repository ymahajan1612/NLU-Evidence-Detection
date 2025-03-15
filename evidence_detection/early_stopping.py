import torch

class EarlyStopping:
    """Early stops the training if validation F1 doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=True, delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            verbose (bool): If True, prints a message for each improvement
            delta (float): Minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = 0
        self.delta = delta

    def __call__(self, val_f1, model):
        score = val_f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1, model)
            self.counter = 0

    def save_checkpoint(self, val_f1, model):
        '''Saves model when validation F1 increases.'''
        if self.verbose:
            print(f'Validation F1 increased ({self.val_f1_max:.6f} --> {val_f1:.6f}). Saving model...')
        torch.save(model.state_dict(), 'best_bilstm_attention_model.pt')
        self.val_f1_max = val_f1
        