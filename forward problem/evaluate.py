import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    
    rel_error = np.mean(np.abs(predictions - targets) / (np.abs(targets) + 1e-8))
    
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


def calculate_force_displacement_and_errors(model, val_loader, device, export_dir="exports"):
    os.makedirs(export_dir, exist_ok=True)
    
    model.eval()
    all_results = []
    
    print("Calculating force-displacement curves and errors:")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, forces) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            forces = forces.to(device)
            
            if hasattr(model, 'predict_force') and model.predict_force:
                disp_predictions, force_predictions = model(inputs)
            else:
                disp_predictions = model(inputs)
                force_predictions = None
            
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            disp_predictions_np = disp_predictions.cpu().numpy()
            forces_np = forces.cpu().numpy()
            
            if force_predictions is not None:
                force_predictions_np = force_predictions.cpu().numpy()
            else:
                force_predictions_np = None
            
            for i in range(inputs.size(0)):
                bc_map = inputs_np[i, 1] 
                
                bc_displacement = np.mean(bc_map[0, :]) 
                
                true_force = forces_np[i, 0]
                pred_force = force_predictions_np[i, 0] if force_predictions_np is not None else None
                
                target_ux = targets_np[i, 0]
                target_uy = targets_np[i, 1]
                pred_ux = disp_predictions_np[i, 0]
                pred_uy = disp_predictions_np[i, 1]
                
                target_ux_corrected, target_uy_corrected = remove_rigid_body_motion(
                    target_ux, target_uy, method='mean_subtraction'
                )
                pred_ux_corrected, pred_uy_corrected = remove_rigid_body_motion(
                    pred_ux, pred_uy, method='mean_subtraction'
                )
                
                mse_ux = mean_squared_error(target_ux_corrected.flatten(), pred_ux_corrected.flatten())
                mse_uy = mean_squared_error(target_uy_corrected.flatten(), pred_uy_corrected.flatten())
                mae_ux = mean_absolute_error(target_ux_corrected.flatten(), pred_ux_corrected.flatten())
                mae_uy = mean_absolute_error(target_uy_corrected.flatten(), pred_uy_corrected.flatten())
                
                mse_total = (mse_ux + mse_uy) / 2
                mae_total = (mae_ux + mae_uy) / 2
                
                target_magnitude = np.sqrt(target_ux_corrected**2 + target_uy_corrected**2)
                pred_magnitude = np.sqrt(pred_ux_corrected**2 + pred_uy_corrected**2)
                relative_error = np.mean(np.abs(target_magnitude - pred_magnitude) / (np.abs(target_magnitude) + 1e-8))
                
                force_error = None
                force_relative_error = None
                if pred_force is not None:
                    force_error = abs(true_force - pred_force)
                    force_relative_error = abs(true_force - pred_force) / (abs(true_force) + 1e-8)
                
                result_dict = {
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'displacement': bc_displacement,
                    'true_force': true_force,
                    'predicted_force': pred_force if pred_force is not None else np.nan,
                    'force_error': force_error if force_error is not None else np.nan,
                    'force_relative_error': force_relative_error if force_relative_error is not None else np.nan,
                    'mse_ux': mse_ux,
                    'mse_uy': mse_uy,
                    'mse_total': mse_total,
                    'mae_ux': mae_ux,
                    'mae_uy': mae_uy,
                    'mae_total': mae_total,
                    'relative_error': relative_error
                }
                
                all_results.append(result_dict)
    
    df = pd.DataFrame(all_results)
    
    df = df.sort_values('displacement')
    
    force_disp_file = os.path.join(export_dir, 'force_displacement_curve.csv')
    error_metrics_file = os.path.join(export_dir, 'error_metrics.csv')
    
    if 'predicted_force' in df.columns and not df['predicted_force'].isna().all():
        df[['displacement', 'true_force', 'predicted_force']].to_csv(force_disp_file, index=False)
    else:
        df[['displacement', 'true_force']].to_csv(force_disp_file, index=False)
    
    # All data including errors
    df.to_csv(error_metrics_file, index=False)
    
    print(f"Force-displacement curve exported to: {force_disp_file}")
    print(f"Error metrics exported to: {error_metrics_file}")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['displacement'], df['true_force'], 'b-', linewidth=2, label='True Force')
    if 'predicted_force' in df.columns and not df['predicted_force'].isna().all():
        plt.plot(df['displacement'], df['predicted_force'], 'r--', linewidth=2, label='Predicted Force')
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    plt.title('Force-Displacement Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if 'predicted_force' in df.columns and not df['predicted_force'].isna().all():
        plt.subplot(2, 2, 2)
        plt.plot(df['displacement'], df['force_error'], 'g-', linewidth=2)
        plt.xlabel('Displacement')
        plt.ylabel('Force Error')
        plt.title('Force Prediction Error vs Displacement')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.scatter(df['true_force'], df['predicted_force'], alpha=0.6, c='purple')
        min_force = min(df['true_force'].min(), df['predicted_force'].min())
        max_force = max(df['true_force'].max(), df['predicted_force'].max())
        plt.plot([min_force, max_force], [min_force, max_force], 'k--', alpha=0.5)
        plt.xlabel('True Force')
        plt.ylabel('Predicted Force')
        plt.title('Force Prediction Accuracy')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(df['displacement'], df['mse_total'], 'orange', linewidth=2)
    plt.xlabel('Displacement')
    plt.ylabel('MSE Total')
    plt.title('Displacement Prediction Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(export_dir, 'force_displacement_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Analysis plot saved to: {plot_file}")
    
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total samples analyzed: {len(df)}")
    print(f"Displacement range: {df['displacement'].min():.6f} to {df['displacement'].max():.6f}")
    print(f"True force range: {df['true_force'].min():.6f} to {df['true_force'].max():.6f}")
    
    if 'predicted_force' in df.columns and not df['predicted_force'].isna().all():
        print(f"Predicted force range: {df['predicted_force'].min():.6f} to {df['predicted_force'].max():.6f}")
        print(f"Average force error: {df['force_error'].mean():.6f} ± {df['force_error'].std():.6f}")
        print(f"Average force relative error: {df['force_relative_error'].mean():.6f} ± {df['force_relative_error'].std():.6f}")
    else:
        print("Force prediction was not enabled")
    
    print(f"Average displacement MSE: {df['mse_total'].mean():.6f} ± {df['mse_total'].std():.6f}")
    print(f"Average displacement MAE: {df['mae_total'].mean():.6f} ± {df['mae_total'].std():.6f}")
    print(f"Average displacement relative error: {df['relative_error'].mean():.6f} ± {df['relative_error'].std():.6f}")
    print("="*50)
    
    return df


def extract_boundary_displacement(bc_map, loading_type='uniaxial_tension'):
    height, width = bc_map.shape
    
    if loading_type == 'uniaxial_tension':

        top_edge_disp = np.mean(bc_map[0, :])
        return top_edge_disp
    
    else:
        # use maximum absolute displacement
        return bc_map[np.unravel_index(np.argmax(np.abs(bc_map)), bc_map.shape)]


def remove_rigid_body_motion(ux, uy, method='mean_subtraction'):
    if method == 'mean_subtraction':
        ux_corrected = ux - np.mean(ux)
        uy_corrected = uy - np.mean(uy)
        
    elif method == 'corner_reference':
        ux_corrected = ux - ux[0, 0]
        uy_corrected = uy - uy[0, 0]
        
    elif method == 'least_squares':
        h, w = ux.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        ux_flat = ux.flatten()
        uy_flat = uy.flatten()
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        
        A = np.column_stack([np.ones_like(x_flat), x_flat, y_flat])
    
        try:
            params_x = np.linalg.lstsq(A, ux_flat, rcond=None)[0]
            params_y = np.linalg.lstsq(A, uy_flat, rcond=None)[0]
            
            rigid_ux = (A @ params_x).reshape(h, w)
            rigid_uy = (A @ params_y).reshape(h, w)
            
            ux_corrected = ux - rigid_ux
            uy_corrected = uy - rigid_uy
        except:
            ux_corrected = ux - np.mean(ux)
            uy_corrected = uy - np.mean(uy)
    
    else:
        ux_corrected = ux.copy()
        uy_corrected = uy.copy()
    
    return ux_corrected, uy_corrected


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
