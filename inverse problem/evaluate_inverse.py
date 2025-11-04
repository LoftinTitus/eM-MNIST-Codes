import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

from inverse_models import InverseFNO2d, InverseUNet
from dataset_inverse import create_inverse_dataloaders
from train_inverse import calculate_segmentation_metrics


def load_trained_inverse_models(device='cpu'):
    """Load trained inverse models from checkpoints"""
    
    models = {}
    
    # Load FNO model
    fno_checkpoint_path = "../checkpoints/best_inverse_model.pt"
    if os.path.exists(fno_checkpoint_path):
        fno_model = InverseFNO2d(
            modes1=12,
            modes2=12,
            width=64,
            num_materials=3,
            predict_properties=False
        )
        checkpoint = torch.load(fno_checkpoint_path, map_location=device)
        fno_model.load_state_dict(checkpoint['model_state_dict'])
        fno_model.eval()
        models['FNO'] = fno_model
        print("✓ Inverse FNO model loaded successfully")
    else:
        print(f"⚠️  Inverse FNO checkpoint not found at {fno_checkpoint_path}")
    
    # Try to load UNet model
    unet_checkpoint_path = "../checkpoints/best_inverse_unet_model.pt"
    if os.path.exists(unet_checkpoint_path):
        unet_model = InverseUNet(
            n_channels=3,
            n_classes=3,
            predict_properties=False,
            bilinear=True
        )
        checkpoint = torch.load(unet_checkpoint_path, map_location=device)
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        unet_model.eval()
        models['UNet'] = unet_model
        print("✓ Inverse UNet model loaded successfully")
    else:
        print(f"⚠️  Inverse UNet checkpoint not found at {unet_checkpoint_path}")
    
    return models


def visualize_inverse_predictions(models, test_loader, device, num_samples=5, save_path="../exports/"):
    """Visualize inverse problem predictions"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Get a batch of test data
    test_batch = next(iter(test_loader))
    inputs, targets = test_batch
    inputs = inputs[:num_samples].to(device)
    targets = targets[:num_samples]
    
    # Define material colormap
    colors = ['black', 'red', 'blue']  # Background, Material 1, Material 2
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(len(models) + 2, num_samples, figsize=(4*num_samples, 4*(len(models) + 2)))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        # Show input displacement fields
        ux = inputs[i, 0].cpu().numpy()
        uy = inputs[i, 1].cpu().numpy()
        force = inputs[i, 2].cpu().numpy()
        target = targets[i].numpy()
        
        # Input displacement magnitude
        disp_mag = np.sqrt(ux**2 + uy**2)
        axes[0, i].imshow(disp_mag, cmap='viridis')
        axes[0, i].set_title(f'Sample {i+1}: Input Displacement')
        axes[0, i].axis('off')
        
        # Target material mask
        axes[1, i].imshow(target, cmap=cmap, vmin=0, vmax=len(colors)-1)
        axes[1, i].set_title(f'Target Material Mask')
        axes[1, i].axis('off')
        
        # Model predictions
        for j, (model_name, model) in enumerate(models.items()):
            with torch.no_grad():
                output = model(inputs[i:i+1])
                prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
            
            axes[j+2, i].imshow(prediction, cmap=cmap, vmin=0, vmax=len(colors)-1)
            axes[j+2, i].set_title(f'{model_name} Prediction')
            axes[j+2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'inverse_predictions_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to {save_path}")


def evaluate_inverse_models(models, test_loader, device, num_materials=3):
    """Evaluate all inverse models and compare performance"""
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} model...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets)
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = calculate_segmentation_metrics(all_predictions, all_targets, num_materials)
        results[model_name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        
        for class_id in range(num_materials):
            class_metrics = metrics['class_metrics'][f'class_{class_id}']
            print(f"  Class {class_id} - Precision: {class_metrics['precision']:.4f}, "
                  f"Recall: {class_metrics['recall']:.4f}, IoU: {class_metrics['iou']:.4f}")
    
    return results


def create_performance_comparison(results, save_path="../exports/"):
    """Create performance comparison plots and tables"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Extract metrics for comparison
    model_names = list(results.keys())
    metrics_data = []
    
    for model_name in model_names:
        metrics = results[model_name]
        row = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Mean_IoU': metrics['mean_iou']
        }
        
        # Add per-class IoU
        for class_id in range(len(metrics['class_metrics'])):
            class_metrics = metrics['class_metrics'][f'class_{class_id}']
            row[f'Class_{class_id}_IoU'] = class_metrics['iou']
            row[f'Class_{class_id}_Precision'] = class_metrics['precision']
            row[f'Class_{class_id}_Recall'] = class_metrics['recall']
        
        metrics_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    # Save to CSV
    df.to_csv(os.path.join(save_path, 'inverse_model_comparison.csv'), index=False)
    print(f"Performance comparison saved to {save_path}")
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Overall accuracy
    axes[0].bar(model_names, [results[name]['accuracy'] for name in model_names])
    axes[0].set_title('Overall Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    
    # Mean IoU
    axes[1].bar(model_names, [results[name]['mean_iou'] for name in model_names])
    axes[1].set_title('Mean IoU')
    axes[1].set_ylabel('IoU')
    axes[1].set_ylim(0, 1)
    
    # Per-class IoU
    num_classes = len(results[model_names[0]]['class_metrics'])
    x = np.arange(len(model_names))
    width = 0.25
    
    for class_id in range(num_classes):
        class_ious = [results[name]['class_metrics'][f'class_{class_id}']['iou'] 
                      for name in model_names]
        axes[2].bar(x + class_id * width, class_ious, width, 
                   label=f'Class {class_id}')
    
    axes[2].set_title('Per-Class IoU')
    axes[2].set_ylabel('IoU')
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels(model_names)
    axes[2].legend()
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'inverse_performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return df


def analyze_failure_cases(models, test_loader, device, save_path="../exports/"):
    """Analyze cases where models fail"""
    
    os.makedirs(save_path, exist_ok=True)
    
    print("Analyzing failure cases...")
    
    # Find samples where all models perform poorly
    poor_performance_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            
            batch_performance = []
            
            for model_name, model in models.items():
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                
                # Calculate per-sample accuracy
                correct = (predictions.cpu() == targets).float()
                sample_accuracies = correct.view(correct.shape[0], -1).mean(dim=1)
                
                batch_performance.append(sample_accuracies)
            
            # Find samples where all models perform poorly (< 50% accuracy)
            if batch_performance:
                min_performance = torch.stack(batch_performance).min(dim=0)[0]
                poor_indices = torch.where(min_performance < 0.5)[0]
                
                if len(poor_indices) > 0:
                    poor_performance_samples.extend([
                        (batch_idx, idx.item(), min_performance[idx].item()) 
                        for idx in poor_indices
                    ])
    
    print(f"Found {len(poor_performance_samples)} samples with poor performance")
    
    # Visualize worst performing samples
    if poor_performance_samples:
        worst_samples = sorted(poor_performance_samples, key=lambda x: x[2])[:5]
        
        fig, axes = plt.subplots(len(models) + 2, len(worst_samples), 
                                figsize=(4*len(worst_samples), 4*(len(models) + 2)))
        
        colors = ['black', 'red', 'blue']
        cmap = ListedColormap(colors)
        
        for i, (batch_idx, sample_idx, performance) in enumerate(worst_samples):
            # Re-get the specific sample (this is simplified - in practice you'd store the data)
            # For now, just show the structure
            axes[0, i].text(0.5, 0.5, f'Batch {batch_idx}\nSample {sample_idx}\nPerf: {performance:.3f}', 
                           ha='center', va='center')
            axes[0, i].set_title(f'Failure Case {i+1}')
            axes[0, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'failure_cases_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main evaluation function"""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading trained inverse models...")
    models = load_trained_inverse_models(device=DEVICE)
    
    if not models:
        print("No trained models found. Please train models first.")
        return
    
    print("Loading test data...")
    # You'll need to load your test data here
    # This is a placeholder - adapt based on your data loading
    print("Note: Please ensure test data is loaded properly in the main_inverse.py script")
    
    # For now, create dummy data for demonstration
    # In practice, you'd load your actual test data
    dummy_X = torch.randn(100, 3, 56, 56)  # 100 samples, 3 channels, 56x56
    dummy_Y = torch.randint(0, 3, (100, 56, 56))  # 100 samples, 56x56 material masks
    
    from torch.utils.data import TensorDataset, DataLoader
    test_dataset = TensorDataset(dummy_X, dummy_Y)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print("Evaluating models...")
    results = evaluate_inverse_models(models, test_loader, DEVICE)
    
    print("Creating performance comparison...")
    comparison_df = create_performance_comparison(results)
    
    print("Visualizing predictions...")
    visualize_inverse_predictions(models, test_loader, DEVICE)
    
    print("Analyzing failure cases...")
    analyze_failure_cases(models, test_loader, DEVICE)
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
