import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local inverse problem imports
from dataform_inverse import normalize_inverse, build_inverse_dataset, build_material_property_dataset
from inverse_models import InverseFNO2d, InverseUNet
from dataset_inverse import create_inverse_dataloaders, analyze_dataset_statistics, InverseDataProcessor
from train_inverse import InverseTrainer
import dataload
import dataprocess


def main():
    # Configuration
    DATA_DIR = "/Users/tyloftin/Downloads/MNIST_comp_files"
    TARGET_SIZE = 56
    BATCH_SIZE = 8  # Smaller batch size for inverse problem
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model configuration
    MODEL_TYPE = 'FNO'  # 'FNO' or 'UNet'
    PREDICT_PROPERTIES = False  # Set to True if you want to predict material properties
    NUM_MATERIALS = 3  # Number of material classes (including background)
    
    # FNO-specific parameters
    MODES1 = 12
    MODES2 = 12
    WIDTH = 64
    
    print(f"Using device: {DEVICE}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Predict properties: {PREDICT_PROPERTIES}")
    print(f"Data directory: {DATA_DIR}")
    
    # Load and preprocess data (same as forward problem)
    print("\nLoading data...")
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} does not exist")
        return
    
    raw_samples = dataload.load_dic_samples(DATA_DIR)
    if len(raw_samples) == 0:
        print("No .npz files found in directory")
        return
    
    print("Preprocessing data for inverse problem...")
    processed_samples = []
    for i, raw_data in enumerate(raw_samples):
        try:
            raw_dict = {
                'DIC_disp': raw_data['ux_frames'][..., None],
                'material_mask': raw_data['material_mask'],
                'instron_disp': raw_data['bc_disp'],
                'instron_force': raw_data['force']
            }
            
            # Stack displacement components
            ux = raw_data['ux_frames']
            uy = raw_data['uy_frames']
            raw_dict['ux_frames'] = ux
            raw_dict['uy_frames'] = uy
            raw_dict['bc_disp'] = raw_data['bc_disp']
            raw_dict['force'] = raw_data['force']
            
            # Preprocess using forward problem methods
            processed = dataprocess.preprocess(raw_dict, target_size=TARGET_SIZE)
            # Use inverse normalization
            processed = normalize_inverse(processed)
            processed_samples.append(processed)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_samples)} samples")
    
    # Build inverse dataset
    print("\nBuilding inverse dataset...")
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
    
    # Analyze dataset
    analyze_dataset_statistics(X, Y_seg, Y_prop)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_inverse_dataloaders(
        X, Y_seg, Y_prop, batch_size=BATCH_SIZE, test_split=0.2, val_split=0.1
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    print(f"\nInitializing {MODEL_TYPE} model...")
    if MODEL_TYPE == 'FNO':
        model = InverseFNO2d(
            modes1=MODES1,
            modes2=MODES2,
            width=WIDTH,
            num_materials=NUM_MATERIALS,
            predict_properties=PREDICT_PROPERTIES
        )
    elif MODEL_TYPE == 'UNet':
        model = InverseUNet(
            n_channels=3,  # ux, uy, force
            n_classes=NUM_MATERIALS,
            predict_properties=PREDICT_PROPERTIES,
            bilinear=True
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = InverseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        predict_properties=PREDICT_PROPERTIES,
        seg_weight=1.0,  # Weight for segmentation loss
        prop_weight=0.1 if PREDICT_PROPERTIES else 0.0  # Weight for property loss
    )
    
    # Start training
    print(f"\nStarting inverse training for {NUM_EPOCHS} epochs...")
    trainer.train(num_epochs=NUM_EPOCHS)
    
    print("\nInverse training completed!")
    
    # Test the model
    print("\nEvaluating on test set...")
    test_results = evaluate_inverse_model(model, test_loader, DEVICE, NUM_MATERIALS, PREDICT_PROPERTIES)
    
    print("Test Results:")
    print(f"  Segmentation Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Mean IoU: {test_results['mean_iou']:.4f}")
    
    if PREDICT_PROPERTIES and 'property_mse' in test_results:
        print(f"  Property MSE: {test_results['property_mse']:.6f}")


def evaluate_inverse_model(model, test_loader, device, num_materials, predict_properties=False):
    """Evaluate the inverse model on test set"""
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
            
            # Get predictions
            seg_predictions = torch.argmax(seg_output, dim=1)
            
            all_predictions.append(seg_predictions.cpu())
            all_targets.append(seg_targets.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate segmentation metrics
    from train_inverse import calculate_segmentation_metrics
    seg_metrics = calculate_segmentation_metrics(all_predictions, all_targets, num_materials)
    
    results = {
        'accuracy': seg_metrics['accuracy'],
        'mean_iou': seg_metrics['mean_iou'],
        'class_metrics': seg_metrics['class_metrics']
    }
    
    # Calculate property metrics if applicable
    if predict_properties and all_prop_predictions:
        all_prop_predictions = torch.cat(all_prop_predictions, dim=0)
        all_prop_targets = torch.cat(all_prop_targets, dim=0)
        
        # Calculate MSE for properties
        prop_mse = torch.mean((all_prop_predictions - all_prop_targets) ** 2)
        results['property_mse'] = prop_mse.item()
    
    return results


if __name__ == "__main__":
    main()
