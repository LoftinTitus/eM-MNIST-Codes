import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
import dataload
import dataprocess
import dataform
from fno_model import FNO2d
from cnn_model import BasicCNN, SimpleCNN
from dataset import create_dataloaders
import torch.nn as nn


def load_trained_models(device='cpu'):
    """Load both FNO and CNN models from checkpoints"""
    
    # Load FNO model
    fno_model = FNO2d(
        modes1=12,
        modes2=12,
        width=64,
        in_channels=2,
        out_channels=2,
        predict_force=True
    )
    
    fno_checkpoint_path = "../checkpoints/best_model.pt"
    if os.path.exists(fno_checkpoint_path):
        fno_model.load_state_dict(torch.load(fno_checkpoint_path, map_location=device))
        print("✓ FNO model loaded successfully")
    else:
        print(f"⚠️  FNO checkpoint not found at {fno_checkpoint_path}")
        return None, None
    
    # Try to load CNN model (check both architectures)
    cnn_model = None
    unet_path = "../checkpoints/best_unet_cnn_model.pt"
    simple_path = "../checkpoints/best_simple_cnn_model.pt"
    
    if os.path.exists(unet_path):
        cnn_model = BasicCNN(in_channels=2, out_channels=2, predict_force=True)
        cnn_model.load_state_dict(torch.load(unet_path, map_location=device))
        print("✓ U-Net CNN model loaded successfully")
    elif os.path.exists(simple_path):
        cnn_model = SimpleCNN(in_channels=2, out_channels=2, predict_force=True)
        cnn_model.load_state_dict(torch.load(simple_path, map_location=device))
        print("✓ Simple CNN model loaded successfully")
    else:
        print(f"⚠️  No CNN checkpoint found. Please train a CNN first using train_cnn.py")
        return fno_model, None
    
    return fno_model, cnn_model


def evaluate_model(model, val_loader, device, model_name):
    """Evaluate a model on validation data"""
    model.eval()
    model.to(device)
    
    criterion_displacement = nn.MSELoss()
    criterion_force = nn.MSELoss()
    
    total_disp_loss = 0
    total_force_loss = 0
    total_samples = 0
    
    all_disp_errors = []
    all_force_errors = []
    all_force_rel_errors = []
    
    with torch.no_grad():
        for inputs, targets, forces in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            forces = forces.to(device)
            
            if model.predict_force:
                disp_outputs, force_outputs = model(inputs)
                disp_loss = criterion_displacement(disp_outputs, targets)
                force_loss = criterion_force(force_outputs, forces)
                
                # Calculate per-sample force errors
                force_errors = torch.abs(force_outputs - forces).cpu().numpy()
                force_rel_errors = (force_errors / (torch.abs(forces).cpu().numpy() + 1e-8)) * 100
                
                all_force_errors.extend(force_errors.flatten())
                all_force_rel_errors.extend(force_rel_errors.flatten())
                
                total_force_loss += force_loss.item()
            else:
                disp_outputs = model(inputs)
                disp_loss = criterion_displacement(disp_outputs, targets)
                force_loss = 0
            
            # Calculate per-sample displacement errors
            disp_errors = torch.mean((disp_outputs - targets)**2, dim=[1,2,3]).cpu().numpy()
            all_disp_errors.extend(disp_errors)
            
            total_disp_loss += disp_loss.item()
            total_samples += inputs.size(0)
    
    avg_disp_loss = total_disp_loss / len(val_loader)
    avg_force_loss = total_force_loss / len(val_loader) if model.predict_force else 0
    
    results = {
        'model': model_name,
        'avg_displacement_mse': avg_disp_loss,
        'avg_force_mse': avg_force_loss,
        'std_displacement_mse': np.std(all_disp_errors),
        'std_force_error': np.std(all_force_errors) if all_force_errors else 0,
        'mean_force_rel_error': np.mean(all_force_rel_errors) if all_force_rel_errors else 0,
        'std_force_rel_error': np.std(all_force_rel_errors) if all_force_rel_errors else 0,
        'total_samples': total_samples
    }
    
    return results


def compare_predictions(fno_model, cnn_model, val_loader, device, num_samples=3):
    """Compare predictions from both models on the same samples"""
    
    fno_model.eval()
    cnn_model.eval()
    fno_model.to(device)
    cnn_model.to(device)
    
    with torch.no_grad():
        sample_input, sample_target, sample_force = next(iter(val_loader))
        num_samples = min(num_samples, sample_input.size(0))
        
        sample_input = sample_input[:num_samples].to(device)
        sample_target = sample_target[:num_samples]
        sample_force = sample_force[:num_samples]
        
        # Get predictions from both models
        fno_disp, fno_force = fno_model(sample_input)
        cnn_disp, cnn_force = cnn_model(sample_input)
        
        fno_disp = fno_disp.cpu()
        fno_force = fno_force.cpu()
        cnn_disp = cnn_disp.cpu()
        cnn_force = cnn_force.cpu()
        
        for sample_idx in range(num_samples):
            print(f"\n=== Sample {sample_idx + 1} Comparison ===")
            
            # Create comparison plot
            fig, axes = plt.subplots(3, 5, figsize=(25, 15))
            
            # Input data
            axes[0, 0].imshow(sample_input[sample_idx, 0].cpu(), cmap='gray')
            axes[0, 0].set_title('Material Mask')
            axes[1, 0].imshow(sample_input[sample_idx, 1].cpu(), cmap='viridis')
            axes[1, 0].set_title('BC Displacement')
            axes[2, 0].axis('off')
            
            # Target
            axes[0, 1].imshow(sample_target[sample_idx, 0], cmap='RdBu')
            axes[0, 1].set_title('Target ux')
            axes[1, 1].imshow(sample_target[sample_idx, 1], cmap='RdBu')
            axes[1, 1].set_title('Target uy')
            axes[2, 1].axis('off')
            
            # FNO predictions
            axes[0, 2].imshow(fno_disp[sample_idx, 0], cmap='RdBu')
            axes[0, 2].set_title('FNO Predicted ux')
            axes[1, 2].imshow(fno_disp[sample_idx, 1], cmap='RdBu')
            axes[1, 2].set_title('FNO Predicted uy')
            
            # CNN predictions
            axes[0, 3].imshow(cnn_disp[sample_idx, 0], cmap='RdBu')
            axes[0, 3].set_title('CNN Predicted ux')
            axes[1, 3].imshow(cnn_disp[sample_idx, 1], cmap='RdBu')
            axes[1, 3].set_title('CNN Predicted uy')
            
            # Error comparison
            fno_ux_error = torch.mean((fno_disp[sample_idx, 0] - sample_target[sample_idx, 0])**2).item()
            fno_uy_error = torch.mean((fno_disp[sample_idx, 1] - sample_target[sample_idx, 1])**2).item()
            fno_total_error = fno_ux_error + fno_uy_error
            
            cnn_ux_error = torch.mean((cnn_disp[sample_idx, 0] - sample_target[sample_idx, 0])**2).item()
            cnn_uy_error = torch.mean((cnn_disp[sample_idx, 1] - sample_target[sample_idx, 1])**2).item()
            cnn_total_error = cnn_ux_error + cnn_uy_error
            
            # Displacement error comparison
            error_data = ['FNO ux', 'CNN ux', 'FNO uy', 'CNN uy', 'FNO Total', 'CNN Total']
            error_values = [fno_ux_error, cnn_ux_error, fno_uy_error, cnn_uy_error, fno_total_error, cnn_total_error]
            colors = ['blue', 'red', 'blue', 'red', 'darkblue', 'darkred']
            
            axes[0, 4].bar(error_data, error_values, color=colors, alpha=0.7)
            axes[0, 4].set_title('Displacement MSE Comparison')
            axes[0, 4].set_ylabel('MSE')
            axes[0, 4].tick_params(axis='x', rotation=45)
            
            # Force comparison
            true_force = sample_force[sample_idx, 0].item()
            fno_pred_force = fno_force[sample_idx, 0].item()
            cnn_pred_force = cnn_force[sample_idx, 0].item()
            
            force_data = ['True', 'FNO', 'CNN']
            force_values = [true_force, fno_pred_force, cnn_pred_force]
            force_colors = ['green', 'blue', 'red']
            
            axes[1, 4].bar(force_data, force_values, color=force_colors, alpha=0.7)
            axes[1, 4].set_title('Force Comparison')
            axes[1, 4].set_ylabel('Force')
            
            # Error metrics text
            fno_force_error = abs(true_force - fno_pred_force)
            cnn_force_error = abs(true_force - cnn_pred_force)
            fno_force_rel_error = fno_force_error / (abs(true_force) + 1e-8) * 100
            cnn_force_rel_error = cnn_force_error / (abs(true_force) + 1e-8) * 100
            
            metrics_text = f"""Displacement MSE:
FNO: {fno_total_error:.6f}
CNN: {cnn_total_error:.6f}
Better: {'FNO' if fno_total_error < cnn_total_error else 'CNN'}

Force Error:
FNO: {fno_force_error:.6f} ({fno_force_rel_error:.2f}%)
CNN: {cnn_force_error:.6f} ({cnn_force_rel_error:.2f}%)
Better: {'FNO' if fno_force_error < cnn_force_error else 'CNN'}"""
            
            axes[2, 2].text(0.05, 0.95, metrics_text, transform=axes[2, 2].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[2, 2].set_title('Performance Comparison')
            axes[2, 2].axis('off')
            
            axes[2, 3].axis('off')
            axes[2, 4].axis('off')
            
            # Remove axes for all image plots
            for i in range(2):
                for j in range(4):
                    axes[i, j].axis('off')
            
            plt.tight_layout()
            plt.suptitle(f'FNO vs CNN - Sample {sample_idx + 1}', fontsize=16, y=0.98)
            plt.show()
            
            # Print numerical comparison
            print(f"Displacement MSE - FNO: {fno_total_error:.6f}, CNN: {cnn_total_error:.6f}")
            print(f"Force Error - FNO: {fno_force_error:.6f} ({fno_force_rel_error:.2f}%), CNN: {cnn_force_error:.6f} ({cnn_force_rel_error:.2f}%)")
            print(f"Winner - Displacement: {'FNO' if fno_total_error < cnn_total_error else 'CNN'}, Force: {'FNO' if fno_force_error < cnn_force_error else 'CNN'}")


def main():
    DATA_DIR = "/Users/tyloftin/Downloads/MNIST_comp_files"
    TARGET_SIZE = 56
    BATCH_SIZE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=== FNO vs CNN Model Comparison ===")
    print(f"Using device: {DEVICE}")
    
    # Load models
    print("\nLoading trained models...")
    fno_model, cnn_model = load_trained_models(DEVICE)
    
    if fno_model is None:
        print("Cannot proceed without FNO model. Please train FNO first.")
        return
    
    if cnn_model is None:
        print("Cannot proceed without CNN model. Please train CNN first using train_cnn.py")
        return
    
    # Load and prepare data
    print("\nLoading and preprocessing data...")
    raw_samples = dataload.load_dic_samples(DATA_DIR)
    processed_samples = []
    
    for i, raw_data in enumerate(raw_samples):
        try:
            raw_dict = {
                'DIC_disp': raw_data['ux_frames'][..., None],
                'label': raw_data['material_mask'],
                'instron_disp': raw_data['bc_disp'],
                'instron_force': raw_data['force']
            }
            
            ux = raw_data['ux_frames']
            uy = raw_data['uy_frames']
            raw_dict['DIC_disp'] = np.stack([ux, uy], axis=-1)
            
            processed = dataprocess.preprocess(raw_dict, target_size=TARGET_SIZE)
            processed = dataform.normalize(processed)
            processed_samples.append(processed)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    X, Y, F = dataform.build_dataset(processed_samples)
    train_loader, val_loader = create_dataloaders(X, Y, F, batch_size=BATCH_SIZE)
    
    print(f"Loaded {len(processed_samples)} samples for evaluation")
    
    # Evaluate both models
    print("\nEvaluating models...")
    fno_results = evaluate_model(fno_model, val_loader, DEVICE, "FNO")
    cnn_results = evaluate_model(cnn_model, val_loader, DEVICE, "CNN")
    
    # Print comparison results
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON RESULTS")
    print("="*60)
    
    results_df = pd.DataFrame([fno_results, cnn_results])
    print(results_df.to_string(index=False, float_format='%.6f'))
    
    print(f"\nWinner - Displacement MSE: {'FNO' if fno_results['avg_displacement_mse'] < cnn_results['avg_displacement_mse'] else 'CNN'}")
    print(f"Winner - Force MSE: {'FNO' if fno_results['avg_force_mse'] < cnn_results['avg_force_mse'] else 'CNN'}")
    print(f"Winner - Force Relative Error: {'FNO' if fno_results['mean_force_rel_error'] < cnn_results['mean_force_rel_error'] else 'CNN'}")
    
    # Model complexity comparison
    fno_params = sum(p.numel() for p in fno_model.parameters())
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    
    print(f"\nModel Complexity:")
    print(f"FNO Parameters: {fno_params:,}")
    print(f"CNN Parameters: {cnn_params:,}")
    print(f"Parameter Ratio (CNN/FNO): {cnn_params/fno_params:.2f}x")
    
    # Save detailed results
    results_df.to_csv("../exports/model_comparison_results.csv", index=False)
    print(f"\nDetailed results saved to ../exports/model_comparison_results.csv")
    
    # Visual comparison
    print("\nGenerating visual comparisons...")
    compare_predictions(fno_model, cnn_model, val_loader, DEVICE, num_samples=3)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
