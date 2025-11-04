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
from cnn_model import BasicCNN, SimpleCNN
from dataset import create_dataloaders
from train import Trainer
from evaluate import calculate_force_displacement_and_errors


def main():
    DATA_DIR = "/Users/tyloftin/Downloads/MNIST_comp_files" 
    TARGET_SIZE = 56
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 75
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Choose CNN architecture
    USE_UNET_STYLE = True  # Set to False for SimpleCNN
    
    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"CNN Architecture: {'U-Net style' if USE_UNET_STYLE else 'Simple CNN'}")
    
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
    
    print("Initializing CNN model:")
    if USE_UNET_STYLE:
        model = BasicCNN(
            in_channels=2,   # material_mask + bc_disp
            out_channels=2,  # ux + uy
            predict_force=True
        )
    else:
        model = SimpleCNN(
            in_channels=2,   # material_mask + bc_disp
            out_channels=2,  # ux + uy
            predict_force=True
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Starting CNN training:")
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE, learning_rate=LEARNING_RATE)
    trainer.train(num_epochs=NUM_EPOCHS)
    
    trainer.plot_losses()

    print("\nCalculating force-displacement curves and error metrics:")
    results_df = calculate_force_displacement_and_errors(model, val_loader, DEVICE)
    
    print("Testing CNN prediction on 3 samples:")
    model.eval()
    with torch.no_grad():
        sample_input, sample_target, sample_force = next(iter(val_loader))
        num_samples = min(3, sample_input.size(0))  
        
        sample_input = sample_input[:num_samples].to(DEVICE)
        sample_target = sample_target[:num_samples]
        sample_force = sample_force[:num_samples]

        if model.predict_force:
            disp_prediction, force_prediction = model(sample_input)
            disp_prediction = disp_prediction.cpu()
            force_prediction = force_prediction.cpu()
        else:
            disp_prediction = model(sample_input).cpu()
            force_prediction = None
        
        for sample_idx in range(num_samples):
            print(f"\nSample {sample_idx + 1}:")
            
            if force_prediction is not None:
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            else:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            

            axes[0, 0].imshow(sample_input[sample_idx, 0].cpu(), cmap='gray')
            axes[0, 0].set_title(f'Sample {sample_idx + 1}: Material Mask')
            axes[1, 0].imshow(sample_input[sample_idx, 1].cpu(), cmap='viridis')
            axes[1, 0].set_title('BC Displacement')
            
            axes[0, 1].imshow(sample_target[sample_idx, 0], cmap='RdBu')
            axes[0, 1].set_title('Target ux')
            axes[1, 1].imshow(sample_target[sample_idx, 1], cmap='RdBu')
            axes[1, 1].set_title('Target uy')
            
            axes[0, 2].imshow(disp_prediction[sample_idx, 0], cmap='RdBu')
            axes[0, 2].set_title('CNN Predicted ux')
            axes[1, 2].imshow(disp_prediction[sample_idx, 1], cmap='RdBu')
            axes[1, 2].set_title('CNN Predicted uy')
            
            ux_error = torch.mean((disp_prediction[sample_idx, 0] - sample_target[sample_idx, 0])**2).item()
            uy_error = torch.mean((disp_prediction[sample_idx, 1] - sample_target[sample_idx, 1])**2).item()
            total_disp_error = ux_error + uy_error
            
            print(f"  Displacement MSE - ux: {ux_error:.6f}, uy: {uy_error:.6f}, total: {total_disp_error:.6f}")
            
            if force_prediction is not None:
                true_force = sample_force[sample_idx, 0].item()
                pred_force = force_prediction[sample_idx, 0].item()
                
                axes[0, 3].bar(['True', 'Predicted'], [true_force, pred_force], 
                              color=['blue', 'red'], alpha=0.7)
                axes[0, 3].set_title('Force Comparison')
                axes[0, 3].set_ylabel('Force')
                
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
                
                print(f"  Force - True: {true_force:.6f}, Predicted: {pred_force:.6f}")
                print(f"  Force Error: {force_error:.6f} ({force_rel_error:.2f}%)")
            
            for ax in axes.flat:
                if ax != axes[1, 3] or force_prediction is None:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.suptitle(f'CNN Sample {sample_idx + 1} Results', fontsize=16, y=0.98)
            plt.show()
    
    print("CNN Training complete")
    
    # Save the CNN model
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = "unet_cnn" if USE_UNET_STYLE else "simple_cnn"
    torch.save(model.state_dict(), f"{checkpoint_dir}/best_{model_name}_model.pt")
    print(f"CNN model saved to {checkpoint_dir}/best_{model_name}_model.pt")


if __name__ == "__main__":
    main()
