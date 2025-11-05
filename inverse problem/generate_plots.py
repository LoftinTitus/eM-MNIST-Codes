import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataform_inverse import normalize_inverse, build_inverse_dataset, build_material_property_dataset
from inverse_models import InverseFNO2d, InverseUNet
from dataset_inverse import create_inverse_dataloaders, analyze_dataset_statistics
from train_inverse import InverseTrainer
import dataload
import dataprocess


def load_model_and_generate_plots(model_type='fno'):
    """Generate plots and visualizations from trained model without retraining"""
    
    DATA_DIR = "/Users/tyloftin/Downloads/MNIST_comp_files"
    TARGET_SIZE = 56
    BATCH_SIZE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PREDICT_PROPERTIES = False
    
    print(f"="*60)
    print(f"GENERATING PLOTS FOR TRAINED {model_type.upper()} MODEL")
    print(f"="*60)
    print(f"Using device: {DEVICE}")
    
    print("Loading data")
    raw_samples = dataload.load_dic_samples(DATA_DIR)
    
    print("Preprocessing data:")
    processed_samples = []
    total_samples = len(raw_samples)
    
    for i, raw_data in enumerate(raw_samples):
        try:
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing sample {i+1}/{total_samples}...")
            
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
            print(f"  Error processing sample {i}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_samples)} samples")
    
    print("Building dataset:")
    if PREDICT_PROPERTIES:
        X, Y_seg, Y_prop = build_material_property_dataset(processed_samples)
    else:
        X, Y_seg, BC = build_inverse_dataset(processed_samples)
        Y_prop = None
    
    print("Remapping material labels to contiguous indices")
    unique_labels = torch.unique(Y_seg).tolist()
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    Y_seg_remapped = torch.zeros_like(Y_seg)
    for old_label, new_label in label_mapping.items():
        Y_seg_remapped[Y_seg == old_label] = new_label
    Y_seg = Y_seg_remapped
    NUM_MATERIALS = len(unique_labels)
    
    print("Creating data loaders:")
    train_loader, val_loader, test_loader = create_inverse_dataloaders(
        X, Y_seg, Y_prop, batch_size=BATCH_SIZE, test_split=0.2, val_split=0.1
    )
    
    if model_type.lower() == 'fno':
        checkpoint_path = "../checkpoints/best_model.pt"
        model = InverseFNO2d(
            modes1=12,
            modes2=12,
            width=64,
            num_materials=NUM_MATERIALS,
            predict_properties=PREDICT_PROPERTIES
        )
        save_path = "../exports/fno_predictions/"
        model_name = "FNO"
    else:  
        checkpoint_path = "../checkpoints/best_unet_cnn_model.pt"
        model = InverseUNet(
            n_channels=3,
            n_classes=NUM_MATERIALS,
            predict_properties=PREDICT_PROPERTIES,
            bilinear=True
        )
        save_path = "../exports/cnn_predictions/"
        model_name = "U-Net"
    
    print(f"Loading trained {model_name} model from: {checkpoint_path}")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        print(f"✓ Model loaded successfully!")
        
        if 'train_losses' in checkpoint:
            print("✓ Training history found in checkpoint")
            generate_training_history_plot(checkpoint)
        else:
            print("No training history in checkpoint")
    else:
        print(f"Checkpoint not found at: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = "../checkpoints/"
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt'):
                    print(f"  - {f}")
        return

    print(f"\nEvaluating {model_name} on test set:")
    test_results = evaluate_model(model, test_loader, DEVICE, NUM_MATERIALS, PREDICT_PROPERTIES)
    
    print(f"{model_name} Test Results:")
    print(f"  Segmentation Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Mean IoU: {test_results['mean_iou']:.4f}")
    
    if PREDICT_PROPERTIES and 'property_mse' in test_results:
        print(f"  Property MSE: {test_results['property_mse']:.6f}")
    
    print(f"\nGenerating {model_name} prediction visualizations:")
    visualize_predictions(model, test_loader, DEVICE, NUM_MATERIALS, 
                         num_samples=5, save_path=save_path, model_name=model_name)
    
    print(f" All plots and visualizations generated successfully!")
    print(f" Check the ../exports/ directory for all outputs")


def generate_training_history_plot(checkpoint):
    """Generate training history plot from checkpoint data"""
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    if not train_losses or not val_losses:
        print("  No loss data available in checkpoint")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    plt.plot(epochs, loss_diff, 'g-', label='|Train - Val|')
    plt.title('Training-Validation Loss Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    
    os.makedirs('../exports', exist_ok=True)
    plt.savefig('../exports/inverse_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Training history plot saved to '../exports/inverse_training_history.png'")


def evaluate_model(model, test_loader, device, num_materials, predict_properties=False):
    """Evaluate the model on test set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if predict_properties:
                inputs, seg_targets, prop_targets = batch
            else:
                inputs, seg_targets = batch
            
            inputs = inputs.to(device)
            seg_targets = seg_targets.to(device)
            
            seg_output = model(inputs)
            if predict_properties and isinstance(seg_output, tuple):
                seg_output = seg_output[0] 
            
            seg_predictions = torch.argmax(seg_output, dim=1)
            
            all_predictions.append(seg_predictions.cpu())
            all_targets.append(seg_targets.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    from train_inverse import calculate_segmentation_metrics
    seg_metrics = calculate_segmentation_metrics(all_predictions, all_targets, num_materials)
    
    return {
        'accuracy': seg_metrics['accuracy'],
        'mean_iou': seg_metrics['mean_iou'],
        'class_metrics': seg_metrics['class_metrics']
    }


def visualize_predictions(model, test_loader, device, num_materials, num_samples=5, save_path="../exports/predictions/", model_name="Model"):
    """Visualize model predictions vs ground truth"""
    from matplotlib.colors import ListedColormap
    
    os.makedirs(save_path, exist_ok=True)
    
    colors = ['black', 'red', 'blue', 'green', 'yellow', 'orange']
    cmap = ListedColormap(colors[:num_materials])
    
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        inputs, targets = test_batch
        inputs = inputs[:num_samples].to(device)
        targets = targets[:num_samples]
        
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0] 
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = targets.numpy()
        inputs_cpu = inputs.cpu().numpy()
        
        fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            ux, uy = inputs_cpu[i, 0], inputs_cpu[i, 1]
            disp_mag = np.sqrt(ux**2 + uy**2)
            
            im1 = axes[0, i].imshow(disp_mag, cmap='viridis')
            axes[0, i].set_title(f'Sample {i+1}: Input Displacement')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            force_field = inputs_cpu[i, 2]
            im2 = axes[1, i].imshow(force_field, cmap='plasma')
            axes[1, i].set_title(f'Input Force Field')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            axes[2, i].imshow(targets[i], cmap=cmap, vmin=0, vmax=num_materials-1)
            axes[2, i].set_title(f'True Material Distribution')
            axes[2, i].axis('off')
        
            axes[3, i].imshow(predictions[i], cmap=cmap, vmin=0, vmax=num_materials-1)
            accuracy = np.mean(targets[i] == predictions[i])
            axes[3, i].set_title(f'{model_name} Prediction (Acc: {accuracy:.3f})')
            axes[3, i].axis('off')
        
        plt.suptitle(f'{model_name} Inverse Problem: Material Identification Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{model_name.lower()}_material_predictions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ {model_name} prediction visualizations saved to: {save_path}")


if __name__ == "__main__":
    import sys
    
    model_type = 'fno'  
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
    
    if model_type not in ['fno', 'cnn', 'unet']:
        print("Usage: python generate_plots.py [fno|cnn|unet]")
        print("Defaulting to FNO...")
        model_type = 'fno'
    
    load_model_and_generate_plots(model_type)
