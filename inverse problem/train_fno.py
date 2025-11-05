import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataform_inverse import normalize_inverse, build_inverse_dataset, build_material_property_dataset
from inverse_models import InverseFNO2d
from dataset_inverse import create_inverse_dataloaders, analyze_dataset_statistics
from train_inverse import InverseTrainer
import dataload
import dataprocess


def main():

    DATA_DIR = "/Users/tyloftin/Downloads/MNIST_comp_files"
    TARGET_SIZE = 56
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 75
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    MODES1 = 12
    MODES2 = 12
    WIDTH = 64
    NUM_MATERIALS = 3
    PREDICT_PROPERTIES = False
    
    print("="*60)
    print("INVERSE FNO TRAINING")
    print("="*60)
    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"FNO Config: modes=({MODES1},{MODES2}), width={WIDTH}")
    print(f"Materials: {NUM_MATERIALS}, Properties: {PREDICT_PROPERTIES}")
    
    print("Loading data")
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} does not exist")
        return
    
    raw_samples = dataload.load_dic_samples(DATA_DIR)
    if len(raw_samples) == 0:
        print("No .npz files found in directory")
        return
    
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
            print(f"   Error processing sample {i}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_samples)} samples")
    
    print("Building dataset:")
    if PREDICT_PROPERTIES:
        X, Y_seg, Y_prop = build_material_property_dataset(processed_samples)
        print(f"  X (inputs): {X.shape} - displacement fields + force")
        print(f"  Y_seg (material masks): {Y_seg.shape}")
        print(f"  Y_prop (material properties): {Y_prop.shape}")
    else:
        X, Y_seg, BC = build_inverse_dataset(processed_samples)
        Y_prop = None
        print(f"  X (inputs): {X.shape} - displacement fields + force")
        print(f"  Y_seg (material masks): {Y_seg.shape}")
        print(f"  BC (boundary conditions): {BC.shape}")
    
    print("Analyzing dataset:")
    analyze_dataset_statistics(X, Y_seg, Y_prop)
    
    print("Remapping material labels to contiguous indices")
    unique_labels = torch.unique(Y_seg).tolist()
    print(f"  Original unique labels: {unique_labels}")
    
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    print(f"  Label mapping: {label_mapping}")
    
    Y_seg_remapped = torch.zeros_like(Y_seg)
    for old_label, new_label in label_mapping.items():
        Y_seg_remapped[Y_seg == old_label] = new_label
    
    Y_seg = Y_seg_remapped
    
    NUM_MATERIALS = len(unique_labels)
    print(f"  Updated NUM_MATERIALS to: {NUM_MATERIALS}")

    new_unique = torch.unique(Y_seg).tolist()
    assert new_unique == list(range(NUM_MATERIALS)), f"Remapping failed: {new_unique} != {list(range(NUM_MATERIALS))}"
    
    print("Creating data loaders:")
    train_loader, val_loader, test_loader = create_inverse_dataloaders(
        X, Y_seg, Y_prop, batch_size=BATCH_SIZE, test_split=0.2, val_split=0.1
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("Initializing FNO model:")
    model = InverseFNO2d(
        modes1=MODES1,
        modes2=MODES2,
        width=WIDTH,
        num_materials=NUM_MATERIALS,
        predict_properties=PREDICT_PROPERTIES
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Starting training:")
    trainer = InverseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        predict_properties=PREDICT_PROPERTIES,
        seg_weight=1.0,
        prop_weight=0.1 if PREDICT_PROPERTIES else 0.0
    )
    
    trainer.train(num_epochs=NUM_EPOCHS)
    
    print("\nSaving final FNO model...")
    final_model_path = "../checkpoints/best_inverse_fno_model.pt"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'InverseFNO2d',
        'config': {
            'modes1': MODES1,
            'modes2': MODES2,
            'width': WIDTH,
            'num_materials': NUM_MATERIALS,
            'predict_properties': PREDICT_PROPERTIES
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'target_size': TARGET_SIZE
        },
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'best_val_loss': trainer.best_val_loss
    }, final_model_path)
    
    print(f"✓ Final FNO model saved to: {final_model_path}")
    
    trainer.plot_training_history()
    
    print("\nCalculating material identification metrics and error analysis:")
    test_results = evaluate_fno_model(model, test_loader, DEVICE, NUM_MATERIALS, PREDICT_PROPERTIES)
    
    print("FNO Test Results:")
    print(f"  Segmentation Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Mean IoU: {test_results['mean_iou']:.4f}")
    
    if PREDICT_PROPERTIES and 'property_mse' in test_results:
        print(f"  Property MSE: {test_results['property_mse']:.6f}")
    
    print("\nTesting prediction on 3 samples:")
    visualize_predictions(model, test_loader, DEVICE, NUM_MATERIALS, num_samples=5, save_path="../exports/fno_predictions/")
    
    print("Training complete")
    print("\nGenerating prediction visualizations:")
    visualize_predictions(model, test_loader, DEVICE, NUM_MATERIALS, num_samples=5, save_path="../exports/fno_predictions/")


def evaluate_fno_model(model, test_loader, device, num_materials, predict_properties=False):
    """Evaluate the FNO inverse model on test set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_prop_predictions = []
    all_prop_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if predict_properties:
                inputs, seg_targets, prop_targets = batch
                inputs = inputs.to(device)
                seg_targets = seg_targets.to(device)
                prop_targets = prop_targets.to(device)
                
                seg_output, prop_output = model(inputs)
                
                all_prop_predictions.append(prop_output.cpu())
                all_prop_targets.append(prop_targets.cpu())
            else:
                inputs, seg_targets = batch
                inputs = inputs.to(device)
                seg_targets = seg_targets.to(device)
                
                seg_output = model(inputs)
            
            seg_predictions = torch.argmax(seg_output, dim=1)
            
            all_predictions.append(seg_predictions.cpu())
            all_targets.append(seg_targets.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    from train_inverse import calculate_segmentation_metrics
    seg_metrics = calculate_segmentation_metrics(all_predictions, all_targets, num_materials)
    
    results = {
        'accuracy': seg_metrics['accuracy'],
        'mean_iou': seg_metrics['mean_iou'],
        'class_metrics': seg_metrics['class_metrics']
    }
    
    if predict_properties and all_prop_predictions:
        all_prop_predictions = torch.cat(all_prop_predictions, dim=0)
        all_prop_targets = torch.cat(all_prop_targets, dim=0)

        prop_mse = torch.mean((all_prop_predictions - all_prop_targets) ** 2)
        results['property_mse'] = prop_mse.item()
    
    return results


def visualize_predictions(model, test_loader, device, num_materials, num_samples=5, save_path="../exports/fno_predictions/"):
    """Visualize FNO predictions vs ground truth"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    os.makedirs(save_path, exist_ok=True)
    
    colors = ['black', 'red', 'blue', 'green', 'yellow', 'orange'][:num_materials]
    cmap = ListedColormap(colors)
    
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        inputs, targets = test_batch
        inputs = inputs[:num_samples].to(device)
        targets = targets[:num_samples]
        
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = targets.numpy()
        inputs_cpu = inputs.cpu().numpy()
        
        fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            ux, uy = inputs_cpu[i, 0], inputs_cpu[i, 1]
            disp_mag = np.sqrt(ux**2 + uy**2)
            
            im1 = axes[0, i].imshow(disp_mag, cmap='RdBu')  
            axes[0, i].set_title(f'Sample {i+1}: Input Displacement')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            force_field = inputs_cpu[i, 2]
            im2 = axes[1, i].imshow(force_field, cmap='RdBu') 
            axes[1, i].set_title(f'Input Force Field')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            axes[2, i].imshow(targets[i], cmap=cmap, vmin=0, vmax=num_materials-1)
            axes[2, i].set_title(f'True Material Distribution')
            axes[2, i].axis('off')
            
            axes[3, i].imshow(predictions[i], cmap=cmap, vmin=0, vmax=num_materials-1)
            accuracy = np.mean(targets[i] == predictions[i])
            axes[3, i].set_title(f'FNO Prediction (Acc: {accuracy:.3f})')
            axes[3, i].axis('off')
        
        plt.suptitle('FNO Inverse Problem: Material Identification Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'fno_material_predictions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        for i in range(min(3, num_samples)):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            disp_mag = np.sqrt(inputs_cpu[i, 0]**2 + inputs_cpu[i, 1]**2)
            im1 = axes[0].imshow(disp_mag, cmap='RdBu') 
            axes[0].set_title('Input: Displacement Magnitude')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0])
            
            axes[1].imshow(targets[i], cmap=cmap, vmin=0, vmax=num_materials-1)
            axes[1].set_title('Ground Truth Material')
            axes[1].axis('off')
            
            axes[2].imshow(predictions[i], cmap=cmap, vmin=0, vmax=num_materials-1)
            accuracy = np.mean(targets[i] == predictions[i])
            axes[2].set_title(f'FNO Prediction (Acc: {accuracy:.3f})')
            axes[2].axis('off')
            
            plt.suptitle(f'FNO Sample {i+1} Detailed View', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'fno_sample_{i+1}_detail.png'), 
                        dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"FNO prediction visualizations saved to: {save_path}")
        
        prediction_data = {
            'inputs': inputs_cpu,
            'targets': targets,
            'predictions': predictions,
            'accuracies': [np.mean(targets[i] == predictions[i]) for i in range(num_samples)]
        }
        np.savez(os.path.join(save_path, 'fno_prediction_data.npz'), **prediction_data)
        print(f"✓ Prediction data saved to: {os.path.join(save_path, 'fno_prediction_data.npz')}")


if __name__ == "__main__":
    main()
