import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', learning_rate=1e-3, 
                 force_weight=1.0, displacement_weight=1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.force_weight = force_weight
        self.displacement_weight = displacement_weight
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        self.criterion_displacement = nn.MSELoss()
        self.criterion_force = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_disp_losses = []
        self.train_force_losses = []
        self.val_disp_losses = []
        self.val_force_losses = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_disp_loss = 0
        total_force_loss = 0
        
        for batch_idx, (inputs, targets, forces) in enumerate(tqdm(self.train_loader, desc="Training")):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            forces = forces.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.model.predict_force:
                disp_outputs, force_outputs = self.model(inputs)
                disp_loss = self.criterion_displacement(disp_outputs, targets)
                force_loss = self.criterion_force(force_outputs, forces)
                loss = self.displacement_weight * disp_loss + self.force_weight * force_loss
                
                total_disp_loss += disp_loss.item()
                total_force_loss += force_loss.item()
            else:
                outputs = self.model(inputs)
                loss = self.criterion_displacement(outputs, targets)
                total_disp_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_total_loss = total_loss / len(self.train_loader)
        avg_disp_loss = total_disp_loss / len(self.train_loader)
        avg_force_loss = total_force_loss / len(self.train_loader) if self.model.predict_force else 0
        
        return avg_total_loss, avg_disp_loss, avg_force_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_disp_loss = 0
        total_force_loss = 0
        
        with torch.no_grad():
            for inputs, targets, forces in tqdm(self.val_loader, desc="Validating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                forces = forces.to(self.device)
                
                if self.model.predict_force:
                    disp_outputs, force_outputs = self.model(inputs)
                    disp_loss = self.criterion_displacement(disp_outputs, targets)
                    force_loss = self.criterion_force(force_outputs, forces)
                    loss = self.displacement_weight * disp_loss + self.force_weight * force_loss
                    
                    total_disp_loss += disp_loss.item()
                    total_force_loss += force_loss.item()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion_displacement(outputs, targets)
                    total_disp_loss += loss.item()
                
                total_loss += loss.item()
                
        avg_total_loss = total_loss / len(self.val_loader)
        avg_disp_loss = total_disp_loss / len(self.val_loader)
        avg_force_loss = total_force_loss / len(self.val_loader) if self.model.predict_force else 0
        
        return avg_total_loss, avg_disp_loss, avg_force_loss
    
    def train(self, num_epochs=100, save_dir="checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss, train_disp_loss, train_force_loss = self.train_epoch()
            val_loss, val_disp_loss, val_force_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_disp_losses.append(train_disp_loss)
            self.train_force_losses.append(train_force_loss)
            self.val_disp_losses.append(val_disp_loss)
            self.val_force_losses.append(val_force_loss)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.6f} (Disp: {train_disp_loss:.6f}, Force: {train_force_loss:.6f})")
            print(f"Val Loss: {val_loss:.6f} (Disp: {val_disp_loss:.6f}, Force: {val_force_loss:.6f})")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            print("-" * 50)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_disp_loss': train_disp_loss,
                    'train_force_loss': train_force_loss,
                    'val_disp_loss': val_disp_loss,
                    'val_force_loss': val_force_loss,
                }, os.path.join(save_dir, 'best_model.pt'))
                
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_disp_loss': train_disp_loss,
                    'train_force_loss': train_force_loss,
                    'val_disp_loss': val_disp_loss,
                    'val_force_loss': val_force_loss,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
                
    def plot_losses(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].plot(self.train_losses, label='Training Loss')
        axes[0].plot(self.val_losses, label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Training and Validation Loss')
        axes[0].legend()
        axes[0].set_yscale('log')
        axes[0].grid(True)
        
        axes[1].plot(self.train_disp_losses, label='Training Displacement Loss')
        axes[1].plot(self.val_disp_losses, label='Validation Displacement Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Displacement Loss')
        axes[1].set_title('Displacement Loss')
        axes[1].legend()
        axes[1].set_yscale('log')
        axes[1].grid(True)
        
        if hasattr(self, 'train_force_losses') and any(self.train_force_losses):
            axes[2].plot(self.train_force_losses, label='Training Force Loss')
            axes[2].plot(self.val_force_losses, label='Validation Force Loss')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Force Loss')
            axes[2].set_title('Force Loss')
            axes[2].legend()
            axes[2].set_yscale('log')
            axes[2].grid(True)
        else:
            axes[2].text(0.5, 0.5, 'Force prediction disabled', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Force Loss (N/A)')
        
        plt.tight_layout()
        plt.show()


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']
