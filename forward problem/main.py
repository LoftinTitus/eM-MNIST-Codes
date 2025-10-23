import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
import dataload
import dataprocess
import dataform
from fno_model import FNO2d
from dataset import create_dataloaders
from train import Trainer
from evaluate import calculate_force_displacement_and_errors


def main():
    DATA_DIR = "/Users/tyloftin/Downloads/MNIST_comp_files" 
    TARGET_SIZE = 56
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    MODES1 = 12  # Number of Fourier modes in first dimension
    MODES2 = 12  # Number of Fourier modes in second dimension
    WIDTH = 64   # Hidden dimension
    
    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    
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
    for i, raw_data in enumerate(raw_samples):
        try:
            raw_dict = {
                'DIC_disp': raw_data['ux_frames'][..., None],  # Add dummy channel dimension
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
    
    print(f"Successfully processed {len(processed_samples)} samples")
    
    print("Building dataset:")
    X, Y, F = dataform.build_dataset(processed_samples)
    
    print(f"  X (inputs): {X.shape}")
    print(f"  Y (outputs): {Y.shape}")
    print(f"  F (forces): {F.shape}")
    
    print("Creating data loaders:")
    train_loader, val_loader = create_dataloaders(X, Y, F, batch_size=BATCH_SIZE)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    print("Initializing FNO model:")
    model = FNO2d(
        modes1=MODES1,
        modes2=MODES2,
        width=WIDTH,
        in_channels=2,   # material_mask + bc_disp
        out_channels=2,   # ux + uy
        predict_force=True  # Enable force prediction
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Starting training:")
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE, learning_rate=LEARNING_RATE)
    trainer.train(num_epochs=NUM_EPOCHS)
    
    trainer.plot_losses()

    print("\nCalculating force-displacement curves and error metrics:")
    results_df = calculate_force_displacement_and_errors(model, val_loader, DEVICE)
    
    print("Testing prediction:")
    model.eval()
    with torch.no_grad():

        sample_input, sample_target, sample_force = next(iter(val_loader))
        sample_input = sample_input[:1].to(DEVICE) 
        sample_target = sample_target[:1]
        sample_force = sample_force[:1]
        
        # Get predictions
        if model.predict_force:
            disp_prediction, force_prediction = model(sample_input)
            disp_prediction = disp_prediction.cpu()
            force_prediction = force_prediction.cpu()
        else:
            disp_prediction = model(sample_input).cpu()
            force_prediction = None
        
        # Create visualization
        if force_prediction is not None:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Input visualizations
        axes[0, 0].imshow(sample_input[0, 0].cpu(), cmap='gray')
        axes[0, 0].set_title('Material Mask')
        axes[1, 0].imshow(sample_input[0, 1].cpu(), cmap='viridis')
        axes[1, 0].set_title('BC Displacement')
        
        # Target displacement fields
        axes[0, 1].imshow(sample_target[0, 0], cmap='RdBu')
        axes[0, 1].set_title('Target ux')
        axes[1, 1].imshow(sample_target[0, 1], cmap='RdBu')
        axes[1, 1].set_title('Target uy')
        
        # Predicted displacement fields
        axes[0, 2].imshow(disp_prediction[0, 0], cmap='RdBu')
        axes[0, 2].set_title('Predicted ux')
        axes[1, 2].imshow(disp_prediction[0, 1], cmap='RdBu')
        axes[1, 2].set_title('Predicted uy')
        
        # Force comparison (if available)
        if force_prediction is not None:
            true_force = sample_force[0, 0].item()
            pred_force = force_prediction[0, 0].item()
            
            axes[0, 3].bar(['True', 'Predicted'], [true_force, pred_force], 
                          color=['blue', 'red'], alpha=0.7)
            axes[0, 3].set_title('Force Comparison')
            axes[0, 3].set_ylabel('Force')
            
            # Force error
            force_error = abs(true_force - pred_force)
            force_rel_error = force_error / (abs(true_force) + 1e-8) * 100
            
            axes[1, 3].text(0.5, 0.7, f'True Force: {true_force:.6f}', 
                           ha='center', va='center', transform=axes[1, 3].transAxes, fontsize=12)
            axes[1, 3].text(0.5, 0.5, f'Pred Force: {pred_force:.6f}', 
                           ha='center', va='center', transform=axes[1, 3].transAxes, fontsize=12)
            axes[1, 3].text(0.5, 0.3, f'Error: {force_error:.6f}', 
                           ha='center', va='center', transform=axes[1, 3].transAxes, fontsize=12)
            axes[1, 3].text(0.5, 0.1, f'Rel Error: {force_rel_error:.2f}%', 
                           ha='center', va='center', transform=axes[1, 3].transAxes, fontsize=12)
            axes[1, 3].set_title('Force Metrics')
            axes[1, 3].axis('off')
        
        for ax in axes.flat:
            if ax != axes[1, 3] or force_prediction is None:  # Don't turn off axis for text display
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("Training complete")


if __name__ == "__main__":
    main()
