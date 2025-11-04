import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inverse_models import InverseFNO2d, InverseUNet
from dataform_inverse import normalize_inverse, build_inverse_dataset
import dataload
import dataprocess


def load_trained_model(model_path, model_type, device='cpu'):
    """Load a trained inverse model"""
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if model_type == 'FNO':
        model = InverseFNO2d(
            modes1=12, modes2=12, width=64,
            num_materials=3, predict_properties=False
        )
    elif model_type == 'UNet':
        model = InverseUNet(
            n_channels=3, n_classes=3,
            predict_properties=False, bilinear=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print(f"✓ {model_type} model loaded from {model_path}")
    return model


def create_detailed_predictions(model, model_name, test_data, device, save_path):
    """Create detailed prediction visualizations"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Material colormap
    colors = ['black', 'red', 'blue', 'green', 'yellow']  # Support up to 5 materials
    cmap = ListedColormap(colors[:3])  # Use first 3 for now
    
    inputs, targets = test_data
    num_samples = min(8, len(inputs))
    
    with torch.no_grad():
        model_inputs = inputs[:num_samples].to(device)
        outputs = model(model_inputs)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        inputs_cpu = inputs[:num_samples].cpu().numpy()
        targets_cpu = targets[:num_samples].numpy()
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(5, num_samples, figsize=(3*num_samples, 15))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            # Row 1: Input displacement X
            ux = inputs_cpu[i, 0]
            im1 = axes[0, i].imshow(ux, cmap='RdBu_r', vmin=-np.max(np.abs(ux)), vmax=np.max(np.abs(ux)))
            axes[0, i].set_title(f'Displacement X (ux)')
            axes[0, i].axis('off')
            if i == 0:
                plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # Row 2: Input displacement Y
            uy = inputs_cpu[i, 1]
            im2 = axes[1, i].imshow(uy, cmap='RdBu_r', vmin=-np.max(np.abs(uy)), vmax=np.max(np.abs(uy)))
            axes[1, i].set_title(f'Displacement Y (uy)')
            axes[1, i].axis('off')
            if i == 0:
                plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # Row 3: Input force field
            force = inputs_cpu[i, 2]
            im3 = axes[2, i].imshow(force, cmap='plasma')
            axes[2, i].set_title(f'Force Field')
            axes[2, i].axis('off')
            if i == 0:
                plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
            
            # Row 4: Ground truth material
            axes[3, i].imshow(targets_cpu[i], cmap=cmap, vmin=0, vmax=2)
            axes[3, i].set_title(f'True Geometry')
            axes[3, i].axis('off')
            
            # Row 5: Predicted material
            accuracy = np.mean(targets_cpu[i] == predictions[i])
            axes[4, i].imshow(predictions[i], cmap=cmap, vmin=0, vmax=2)
            axes[4, i].set_title(f'{model_name} Prediction\n(Acc: {accuracy:.3f})')
            axes[4, i].axis('off')
        
        plt.suptitle(f'{model_name} Material Identification: Input → Predicted Geometry', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{model_name.lower()}_detailed_predictions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create error analysis
        create_error_analysis(targets_cpu, predictions, model_name, save_path)
        
        # Create material class analysis
        create_material_analysis(targets_cpu, predictions, model_name, save_path)
        
        return predictions, targets_cpu


def create_error_analysis(targets, predictions, model_name, save_path):
    """Create error analysis visualizations"""
    
    num_samples = min(5, len(targets))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        # Error map (where predictions differ from truth)
        error_map = (targets[i] != predictions[i]).astype(int)
        axes[0, i].imshow(error_map, cmap='Reds', vmin=0, vmax=1)
        error_rate = np.mean(error_map)
        axes[0, i].set_title(f'Sample {i+1}: Error Map\n(Error Rate: {error_rate:.3f})')
        axes[0, i].axis('off')
        
        # Prediction confidence (distance from boundary)
        # This is a simplified confidence measure
        conf_map = np.ones_like(predictions[i], dtype=float)
        axes[1, i].imshow(conf_map, cmap='viridis')
        axes[1, i].set_title(f'Prediction Confidence')
        axes[1, i].axis('off')
    
    plt.suptitle(f'{model_name} Error Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name.lower()}_error_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def create_material_analysis(targets, predictions, model_name, save_path):
    """Analyze per-material performance"""
    
    # Calculate per-material metrics
    material_metrics = []
    
    for material_id in range(3):  # 3 materials
        # True positive, false positive, false negative
        true_mask = (targets == material_id)
        pred_mask = (predictions == material_id)
        
        tp = np.sum(true_mask & pred_mask)
        fp = np.sum(~true_mask & pred_mask)
        fn = np.sum(true_mask & ~pred_mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        material_metrics.append({
            'Material': f'Material {material_id}',
            'Precision': precision,
            'Recall': recall,
            'IoU': iou,
            'True_Pixels': np.sum(true_mask),
            'Pred_Pixels': np.sum(pred_mask)
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(material_metrics)
    df.to_csv(os.path.join(save_path, f'{model_name.lower()}_material_metrics.csv'), index=False)
    
    # Create bar plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    materials = df['Material'].values
    
    # Precision
    axes[0].bar(materials, df['Precision'], alpha=0.7, color='blue')
    axes[0].set_title('Precision by Material')
    axes[0].set_ylabel('Precision')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[1].bar(materials, df['Recall'], alpha=0.7, color='green')
    axes[1].set_title('Recall by Material')
    axes[1].set_ylabel('Recall')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    
    # IoU
    axes[2].bar(materials, df['IoU'], alpha=0.7, color='red')
    axes[2].set_title('IoU by Material')
    axes[2].set_ylabel('IoU')
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'{model_name} Per-Material Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name.lower()}_material_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n{model_name} Material Performance:")
    print(df.to_string(index=False, float_format='%.3f'))


def compare_models(fno_preds, unet_preds, targets, save_path="../exports/model_comparison/"):
    """Compare FNO vs U-Net predictions"""
    
    os.makedirs(save_path, exist_ok=True)
    
    num_samples = min(5, len(targets))
    colors = ['black', 'red', 'blue']
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        # Ground truth
        axes[0, i].imshow(targets[i], cmap=cmap, vmin=0, vmax=2)
        axes[0, i].set_title(f'Sample {i+1}: Ground Truth')
        axes[0, i].axis('off')
        
        # FNO prediction
        fno_acc = np.mean(targets[i] == fno_preds[i])
        axes[1, i].imshow(fno_preds[i], cmap=cmap, vmin=0, vmax=2)
        axes[1, i].set_title(f'FNO (Acc: {fno_acc:.3f})')
        axes[1, i].axis('off')
        
        # U-Net prediction
        unet_acc = np.mean(targets[i] == unet_preds[i])
        axes[2, i].imshow(unet_preds[i], cmap=cmap, vmin=0, vmax=2)
        axes[2, i].set_title(f'U-Net (Acc: {unet_acc:.3f})')
        axes[2, i].axis('off')
        
        # Difference map (where models disagree)
        diff_map = (fno_preds[i] != unet_preds[i]).astype(int)
        axes[3, i].imshow(diff_map, cmap='Reds', vmin=0, vmax=1)
        disagreement = np.mean(diff_map)
        axes[3, i].set_title(f'Model Disagreement ({disagreement:.3f})')
        axes[3, i].axis('off')
    
    plt.suptitle('FNO vs U-Net Model Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'fno_vs_unet_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main visualization function"""
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = "/Users/tyloftin/Downloads/MNIST_comp_files"
    TARGET_SIZE = 56
    
    print("="*60)
    print("GEOMETRY PREDICTION VISUALIZATION")
    print("="*60)
    
    # Load test data
    print("Loading test data...")
    raw_samples = dataload.load_dic_samples(DATA_DIR)
    processed_samples = []
    
    for i, raw_data in enumerate(raw_samples[:10]):  # Use first 10 samples
        try:
            raw_dict = {
                'ux_frames': raw_data['ux_frames'],
                'uy_frames': raw_data['uy_frames'],
                'material_mask': raw_data['material_mask'],
                'bc_disp': raw_data['bc_disp'],
                'force': raw_data['force'],
                'filename': raw_data.get('filename', f'sample_{i}')
            }
            
            processed = dataprocess.preprocess(raw_dict, target_size=TARGET_SIZE)
            processed = normalize_inverse(processed)
            processed_samples.append(processed)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    X, Y_seg, BC = build_inverse_dataset(processed_samples)
    test_data = (X, Y_seg)
    
    print(f"Loaded {len(X)} test samples")
    
    # Load models
    fno_model = load_trained_model("../checkpoints/best_inverse_fno_model.pt", "FNO", DEVICE)
    unet_model = load_trained_model("../checkpoints/best_inverse_unet_model.pt", "UNet", DEVICE)
    
    predictions = {}
    
    # Generate FNO predictions
    if fno_model is not None:
        print("\nGenerating FNO predictions...")
        fno_preds, targets = create_detailed_predictions(
            fno_model, "FNO", test_data, DEVICE, "../exports/fno_geometry_predictions/"
        )
        predictions['FNO'] = fno_preds
    
    # Generate U-Net predictions
    if unet_model is not None:
        print("\nGenerating U-Net predictions...")
        unet_preds, targets = create_detailed_predictions(
            unet_model, "UNet", test_data, DEVICE, "../exports/unet_geometry_predictions/"
        )
        predictions['UNet'] = unet_preds
    
    # Compare models if both are available
    if 'FNO' in predictions and 'UNet' in predictions:
        print("\nComparing models...")
        compare_models(predictions['FNO'], predictions['UNet'], targets)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("Check the following directories for results:")
    print("  - ../exports/fno_geometry_predictions/")
    print("  - ../exports/unet_geometry_predictions/")
    print("  - ../exports/model_comparison/")


if __name__ == "__main__":
    main()
