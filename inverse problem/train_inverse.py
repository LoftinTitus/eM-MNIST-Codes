import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm


class InverseTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu', learning_rate=1e-3, 
                 predict_properties=False, seg_weight=1.0, prop_weight=1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.predict_properties = predict_properties
        self.seg_weight = seg_weight
        self.prop_weight = prop_weight
        
        # Optimizers
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        # Loss functions
        self.seg_criterion = nn.CrossEntropyLoss()  # For material segmentation
        if predict_properties:
            self.prop_criterion = nn.MSELoss()  # For material properties
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_seg_losses = []
        self.val_seg_losses = []
        if predict_properties:
            self.train_prop_losses = []
            self.val_prop_losses = []
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        seg_loss_sum = 0.0
        prop_loss_sum = 0.0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            if self.predict_properties:
                inputs, seg_targets, prop_targets = batch
                inputs = inputs.to(self.device)
                seg_targets = seg_targets.to(self.device)
                prop_targets = prop_targets.to(self.device)
            else:
                inputs, seg_targets = batch
                inputs = inputs.to(self.device)
                seg_targets = seg_targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.predict_properties:
                seg_output, prop_output = self.model(inputs)
                
                # Segmentation loss
                seg_loss = self.seg_criterion(seg_output, seg_targets)
                
                # Property loss (only for non-background pixels)
                mask = (seg_targets > 0).float().unsqueeze(1)  # [B, 1, H, W]
                prop_loss = self.prop_criterion(prop_output * mask, prop_targets * mask)
                
                # Combined loss
                loss = self.seg_weight * seg_loss + self.prop_weight * prop_loss
                
                prop_loss_sum += prop_loss.item()
            else:
                seg_output = self.model(inputs)
                seg_loss = self.seg_criterion(seg_output, seg_targets)
                loss = seg_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            seg_loss_sum += seg_loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = seg_loss_sum / len(self.train_loader)
        
        self.train_losses.append(avg_loss)
        self.train_seg_losses.append(avg_seg_loss)
        
        if self.predict_properties:
            avg_prop_loss = prop_loss_sum / len(self.train_loader)
            self.train_prop_losses.append(avg_prop_loss)
            return avg_loss, avg_seg_loss, avg_prop_loss
        
        return avg_loss, avg_seg_loss
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        seg_loss_sum = 0.0
        prop_loss_sum = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                if self.predict_properties:
                    inputs, seg_targets, prop_targets = batch
                    inputs = inputs.to(self.device)
                    seg_targets = seg_targets.to(self.device)
                    prop_targets = prop_targets.to(self.device)
                else:
                    inputs, seg_targets = batch
                    inputs = inputs.to(self.device)
                    seg_targets = seg_targets.to(self.device)
                
                # Forward pass
                if self.predict_properties:
                    seg_output, prop_output = self.model(inputs)
                    
                    # Segmentation loss
                    seg_loss = self.seg_criterion(seg_output, seg_targets)
                    
                    # Property loss
                    mask = (seg_targets > 0).float().unsqueeze(1)
                    prop_loss = self.prop_criterion(prop_output * mask, prop_targets * mask)
                    
                    # Combined loss
                    loss = self.seg_weight * seg_loss + self.prop_weight * prop_loss
                    
                    prop_loss_sum += prop_loss.item()
                else:
                    seg_output = self.model(inputs)
                    seg_loss = self.seg_criterion(seg_output, seg_targets)
                    loss = seg_loss
                
                total_loss += loss.item()
                seg_loss_sum += seg_loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_seg_loss = seg_loss_sum / len(self.val_loader)
        
        self.val_losses.append(avg_loss)
        self.val_seg_losses.append(avg_seg_loss)
        
        if self.predict_properties:
            avg_prop_loss = prop_loss_sum / len(self.val_loader)
            self.val_prop_losses.append(avg_prop_loss)
            return avg_loss, avg_seg_loss, avg_prop_loss
        
        return avg_loss, avg_seg_loss
    
    def train(self, num_epochs):
        print(f"\nStarting inverse training:")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Predict properties: {self.predict_properties}")
        
        for epoch in range(num_epochs):
            # Training
            if self.predict_properties:
                train_loss, train_seg_loss, train_prop_loss = self.train_epoch()
                val_loss, val_seg_loss, val_prop_loss = self.validate_epoch()
            else:
                train_loss, train_seg_loss = self.train_epoch()
                val_loss, val_seg_loss = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print epoch results in forward problem style
            print(f"Epoch {epoch+1}/{num_epochs}")
            if self.predict_properties:
                print(f"Train Loss: {train_loss:.6f} (Seg: {train_seg_loss:.6f}, Prop: {train_prop_loss:.6f})")
                print(f"Val Loss: {val_loss:.6f} (Seg: {val_seg_loss:.6f}, Prop: {val_prop_loss:.6f})")
            else:
                print(f"Train Loss: {train_loss:.6f} (Seg: {train_seg_loss:.6f})")
                print(f"Val Loss: {val_loss:.6f} (Seg: {val_seg_loss:.6f})")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            print("-" * 50)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"../checkpoints/best_inverse_model.pt", epoch)
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"../checkpoints/inverse_checkpoint_epoch_{epoch+1}.pt", epoch)
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.6f}")
        self.plot_training_history()
    
    def save_checkpoint(self, path, epoch):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_seg_losses': self.train_seg_losses,
            'val_seg_losses': self.val_seg_losses,
            'best_val_loss': self.best_val_loss,
            'predict_properties': self.predict_properties,
            'seg_weight': self.seg_weight,
            'prop_weight': self.prop_weight
        }
        
        # Add property losses if applicable
        if self.predict_properties and hasattr(self, 'train_prop_losses'):
            checkpoint['train_prop_losses'] = self.train_prop_losses
            checkpoint['val_prop_losses'] = self.val_prop_losses
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {os.path.basename(path)}")
    
    def plot_training_history(self):
        """Plot training history"""
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # Total loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Total Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Val Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Segmentation loss
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_seg_losses, 'b-', label='Train Seg Loss')
        plt.plot(epochs, self.val_seg_losses, 'r-', label='Val Seg Loss')
        plt.title('Segmentation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Property loss (if applicable)
        if self.predict_properties:
            plt.subplot(1, 3, 3)
            plt.plot(epochs, self.train_prop_losses, 'b-', label='Train Prop Loss')
            plt.plot(epochs, self.val_prop_losses, 'r-', label='Val Prop Loss')
            plt.title('Property Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Create exports directory if it doesn't exist
        os.makedirs('../exports', exist_ok=True)
        plt.savefig('../exports/inverse_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Training history plot saved to '../exports/inverse_training_history.png'")


def calculate_segmentation_metrics(predictions, targets, num_classes):
    """Calculate segmentation metrics"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    metrics = {}
    
    # Overall accuracy
    correct = (predictions == targets).sum()
    total = targets.size
    metrics['accuracy'] = correct / total
    
    # Per-class metrics
    class_metrics = {}
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)
        
        tp = (pred_mask & target_mask).sum()
        fp = (pred_mask & ~target_mask).sum()
        fn = (~pred_mask & target_mask).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        class_metrics[f'class_{class_id}'] = {
            'precision': precision,
            'recall': recall,
            'iou': iou
        }
    
    metrics['class_metrics'] = class_metrics
    
    # Mean IoU
    mean_iou = np.mean([class_metrics[f'class_{i}']['iou'] for i in range(num_classes)])
    metrics['mean_iou'] = mean_iou
    
    return metrics
