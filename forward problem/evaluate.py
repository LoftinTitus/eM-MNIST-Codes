import torch
import numpy as np
import matplotlib.pyplot as plt
from fno_model import FNO2d
from train import load_checkpoint


def evaluate_model(model, test_loader, device='cpu'): # Change if you get a cuda
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for inputs, target, forces in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)
            
            output = model(inputs)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            predictions.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    return avg_loss, predictions, targets


def visualize(predictions, targets, inputs=None, num_samples=4):
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(min(num_samples, len(predictions))):
        # Real ux
        im1 = axes[i, 0].imshow(targets[i, 0], cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Target ux (Sample {i+1})')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Predicted ux
        im2 = axes[i, 1].imshow(predictions[i, 0], cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Predicted ux (Sample {i+1})')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # Real uy
        im3 = axes[i, 2].imshow(targets[i, 1], cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 2].set_title(f'Target uy (Sample {i+1})')
        plt.colorbar(im3, ax=axes[i, 2])
        
        # Predicted uy
        im4 = axes[i, 3].imshow(predictions[i, 1], cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 3].set_title(f'Predicted uy (Sample {i+1})')
        plt.colorbar(im4, ax=axes[i, 3])
        
        for ax in axes[i]:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def compute_metrics(predictions, targets):
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # Relative error
    rel_error = np.mean(np.abs(predictions - targets) / (np.abs(targets) + 1e-8))
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Relative Error': rel_error,
        'R²': r2
    }


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = FNO2d(modes1=12, modes2=12, width=64)
    epoch, train_loss, val_loss = load_checkpoint(model, 'checkpoints/best_model.pt', device)
    
    print(f"Loaded model from epoch {epoch}")
    print(f"Training loss: {train_loss:.6f}")
    print(f"Validation loss: {val_loss:.6f}")
    
    # Evaluate on test set (you would need to create a test_loader)
    # avg_loss, predictions, targets = evaluate_model(model, test_loader, device)
    # metrics = compute_metrics(predictions, targets)
    # print("Test Metrics:", metrics)
    # visualize(predictions, targets)
