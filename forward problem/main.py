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
from train import Trainer, load_checkpoint


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
        out_channels=2   # ux + uy
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Starting training:")
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE, learning_rate=LEARNING_RATE)
    trainer.train(num_epochs=NUM_EPOCHS)
    
    trainer.plot_losses()
    
    
    print("Testing prediction:")
    model.eval()
    with torch.no_grad():

        sample_input, sample_target, sample_force = next(iter(val_loader))
        sample_input = sample_input[:1].to(DEVICE) 
        sample_target = sample_target[:1]
        
        prediction = model(sample_input).cpu()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        
        axes[0, 0].imshow(sample_input[0, 0].cpu(), cmap='gray')
        axes[0, 0].set_title('Material Mask')
        axes[1, 0].imshow(sample_input[0, 1].cpu(), cmap='viridis')
        axes[1, 0].set_title('BC Displacement')
        
        
        axes[0, 1].imshow(sample_target[0, 0], cmap='RdBu')
        axes[0, 1].set_title('Target ux')
        axes[1, 1].imshow(sample_target[0, 1], cmap='RdBu')
        axes[1, 1].set_title('Target uy')
        
        
        axes[0, 2].imshow(prediction[0, 0], cmap='RdBu')
        axes[0, 2].set_title('Predicted ux')
        axes[1, 2].imshow(prediction[0, 1], cmap='RdBu')
        axes[1, 2].set_title('Predicted uy')
        
        for ax in axes.flat:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("Training complete")


if __name__ == "__main__":
    main()
