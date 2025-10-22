import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DICDataset(Dataset):
    def __init__(self, X, Y, F):
        """
        X: Input tensors (image identity, bc_disp)
        Y: (displacement fields ux, uy,)
        F: Force values
        """
        self.X = X
        self.Y = Y
        self.F = F
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.F[idx]


def create_dataloaders(X, Y, F, train_ratio=0.8, batch_size=32, shuffle=True):
    dataset_size = len(X)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = DICDataset(X[train_indices], Y[train_indices], F[train_indices])
    val_dataset = DICDataset(X[val_indices], Y[val_indices], F[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def collate_fn(batch):
    X_batch, Y_batch, F_batch = zip(*batch)
    return torch.stack(X_batch), torch.stack(Y_batch), torch.stack(F_batch)
