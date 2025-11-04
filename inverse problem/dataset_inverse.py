import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random


def create_inverse_dataloaders(X, Y_seg, Y_prop=None, batch_size=16, test_split=0.2, val_split=0.1):
    """
    Create data loaders for inverse problem
    
    Args:
        X: Input tensor [N, 3, H, W] - displacement fields + force/BC
        Y_seg: Segmentation target tensor [N, H, W] - material masks
        Y_prop: Property target tensor [N, 3, H, W] - material properties (optional)
        batch_size: Batch size for training
        test_split: Fraction of data for testing
        val_split: Fraction of remaining data for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Ensure tensors are on CPU for splitting
    X = X.cpu()
    Y_seg = Y_seg.cpu()
    if Y_prop is not None:
        Y_prop = Y_prop.cpu()
    
    # Get total number of samples
    total_samples = X.shape[0]
    indices = list(range(total_samples))
    random.shuffle(indices)
    
    # Calculate split sizes
    test_size = int(total_samples * test_split)
    remaining_size = total_samples - test_size
    val_size = int(remaining_size * val_split)
    train_size = remaining_size - val_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Create datasets
    if Y_prop is not None:
        # With property prediction
        train_dataset = TensorDataset(X[train_indices], Y_seg[train_indices], Y_prop[train_indices])
        val_dataset = TensorDataset(X[val_indices], Y_seg[val_indices], Y_prop[val_indices])
        test_dataset = TensorDataset(X[test_indices], Y_seg[test_indices], Y_prop[test_indices])
    else:
        # Segmentation only
        train_dataset = TensorDataset(X[train_indices], Y_seg[train_indices])
        val_dataset = TensorDataset(X[val_indices], Y_seg[val_indices])
        test_dataset = TensorDataset(X[test_indices], Y_seg[test_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader


def create_augmented_dataset(X, Y_seg, Y_prop=None, augment_factor=2):
    """
    Create augmented dataset for inverse problem
    
    Args:
        X: Input tensor [N, 3, H, W]
        Y_seg: Segmentation targets [N, H, W]
        Y_prop: Property targets [N, 3, H, W] (optional)
        augment_factor: How many times to augment the data
    
    Returns:
        Augmented X, Y_seg, Y_prop
    """
    
    def random_flip(x, y_seg, y_prop=None):
        """Random horizontal/vertical flip"""
        if random.random() > 0.5:
            x = torch.flip(x, [-1])  # Horizontal flip
            y_seg = torch.flip(y_seg, [-1])
            if y_prop is not None:
                y_prop = torch.flip(y_prop, [-1])
        
        if random.random() > 0.5:
            x = torch.flip(x, [-2])  # Vertical flip
            y_seg = torch.flip(y_seg, [-2])
            if y_prop is not None:
                y_prop = torch.flip(y_prop, [-2])
        
        return x, y_seg, y_prop
    
    def add_noise(x, noise_level=0.01):
        """Add Gaussian noise to input"""
        noise = torch.randn_like(x) * noise_level
        return x + noise
    
    augmented_X = [X]
    augmented_Y_seg = [Y_seg]
    if Y_prop is not None:
        augmented_Y_prop = [Y_prop]
    
    for _ in range(augment_factor - 1):
        aug_X = X.clone()
        aug_Y_seg = Y_seg.clone()
        aug_Y_prop = Y_prop.clone() if Y_prop is not None else None
        
        # Apply augmentations
        for i in range(len(aug_X)):
            # Flip augmentation
            aug_X[i], aug_Y_seg[i], aug_Y_prop_i = random_flip(
                aug_X[i], aug_Y_seg[i], 
                aug_Y_prop[i] if aug_Y_prop is not None else None
            )
            if aug_Y_prop is not None:
                aug_Y_prop[i] = aug_Y_prop_i
            
            # Noise augmentation
            aug_X[i] = add_noise(aug_X[i])
        
        augmented_X.append(aug_X)
        augmented_Y_seg.append(aug_Y_seg)
        if Y_prop is not None:
            augmented_Y_prop.append(aug_Y_prop)
    
    # Concatenate all augmented data
    final_X = torch.cat(augmented_X, dim=0)
    final_Y_seg = torch.cat(augmented_Y_seg, dim=0)
    final_Y_prop = torch.cat(augmented_Y_prop, dim=0) if Y_prop is not None else None
    
    print(f"Augmented dataset size: {final_X.shape[0]} samples (from {X.shape[0]} original)")
    
    return final_X, final_Y_seg, final_Y_prop


class InverseDataProcessor:
    """Process data specifically for inverse problem"""
    
    def __init__(self, normalize_displacement=True, normalize_force=True):
        self.normalize_displacement = normalize_displacement
        self.normalize_force = normalize_force
        self.displacement_stats = {}
        self.force_stats = {}
    
    def fit(self, X):
        """Calculate normalization statistics"""
        # X shape: [N, 3, H, W] where channels are [ux, uy, force]
        
        if self.normalize_displacement:
            # Calculate displacement statistics (channels 0 and 1)
            disp_data = X[:, :2, :, :].reshape(-1)  # Flatten ux and uy
            self.displacement_stats = {
                'mean': torch.mean(disp_data),
                'std': torch.std(disp_data),
                'min': torch.min(disp_data),
                'max': torch.max(disp_data)
            }
        
        if self.normalize_force:
            # Calculate force statistics (channel 2)
            force_data = X[:, 2, :, :].reshape(-1)  # Flatten force
            self.force_stats = {
                'mean': torch.mean(force_data),
                'std': torch.std(force_data),
                'min': torch.min(force_data),
                'max': torch.max(force_data)
            }
    
    def transform(self, X):
        """Apply normalization"""
        X_norm = X.clone()
        
        if self.normalize_displacement and self.displacement_stats:
            # Normalize displacement channels
            X_norm[:, :2, :, :] = (X_norm[:, :2, :, :] - self.displacement_stats['mean']) / (self.displacement_stats['std'] + 1e-8)
        
        if self.normalize_force and self.force_stats:
            # Normalize force channel
            X_norm[:, 2, :, :] = (X_norm[:, 2, :, :] - self.force_stats['mean']) / (self.force_stats['std'] + 1e-8)
        
        return X_norm
    
    def inverse_transform(self, X_norm):
        """Reverse normalization"""
        X = X_norm.clone()
        
        if self.normalize_displacement and self.displacement_stats:
            # Denormalize displacement channels
            X[:, :2, :, :] = X[:, :2, :, :] * self.displacement_stats['std'] + self.displacement_stats['mean']
        
        if self.normalize_force and self.force_stats:
            # Denormalize force channel
            X[:, 2, :, :] = X[:, 2, :, :] * self.force_stats['std'] + self.force_stats['mean']
        
        return X


def analyze_dataset_statistics(X, Y_seg, Y_prop=None):
    """Analyze dataset statistics for inverse problem"""
    print("Dataset Statistics:")
    print("=" * 50)
    
    # Input statistics
    print(f"Input shape: {X.shape}")
    print(f"Input range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Input mean: {X.mean():.4f}, std: {X.std():.4f}")
    
    # Channel-wise statistics
    channel_names = ['ux (displacement)', 'uy (displacement)', 'force/BC']
    for i, name in enumerate(channel_names):
        channel_data = X[:, i, :, :]
        print(f"{name}: mean={channel_data.mean():.4f}, std={channel_data.std():.4f}, "
              f"range=[{channel_data.min():.4f}, {channel_data.max():.4f}]")
    
    # Segmentation statistics
    print(f"\nSegmentation targets shape: {Y_seg.shape}")
    unique_labels = torch.unique(Y_seg)
    print(f"Unique material labels: {unique_labels.tolist()}")
    
    for label in unique_labels:
        count = (Y_seg == label).sum().item()
        percentage = count / Y_seg.numel() * 100
        print(f"  Label {label}: {count} pixels ({percentage:.2f}%)")
    
    # Property statistics (if available)
    if Y_prop is not None:
        print(f"\nProperty targets shape: {Y_prop.shape}")
        prop_names = ['Young\'s modulus', 'Poisson\'s ratio', 'Density']
        for i, name in enumerate(prop_names):
            prop_data = Y_prop[:, i, :, :]
            print(f"{name}: mean={prop_data.mean():.4f}, std={prop_data.std():.4f}, "
                  f"range=[{prop_data.min():.4f}, {prop_data.max():.4f}]")
    
    print("=" * 50)
