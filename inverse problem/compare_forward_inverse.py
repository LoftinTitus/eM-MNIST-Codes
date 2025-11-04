import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Add paths for both forward and inverse problems
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'forward problem'))

# Import from forward problem
from fno_model import FNO2d
from cnn_model import BasicCNN

# Import from inverse problem
from inverse_models import InverseFNO2d, InverseUNet


class ForwardInverseComparison:
    """Compare forward and inverse models in a round-trip evaluation"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.forward_models = {}
        self.inverse_models = {}
        
    def load_forward_models(self):
        """Load trained forward models"""
        
        # Load FNO forward model
        fno_path = "../checkpoints/best_model.pt"
        if os.path.exists(fno_path):
            fno_model = FNO2d(
                modes1=12, modes2=12, width=64,
                in_channels=2, out_channels=2,
                predict_force=True
            )
            fno_model.load_state_dict(torch.load(fno_path, map_location=self.device))
            fno_model.eval()
            self.forward_models['FNO'] = fno_model
            print("✓ Forward FNO model loaded")
        
        # Load CNN forward model
        cnn_path = "../checkpoints/best_unet_cnn_model.pt"
        if os.path.exists(cnn_path):
            cnn_model = BasicCNN(in_channels=2, out_channels=2, predict_force=True)
            cnn_model.load_state_dict(torch.load(cnn_path, map_location=self.device))
            cnn_model.eval()
            self.forward_models['CNN'] = cnn_model
            print("✓ Forward CNN model loaded")
    
    def load_inverse_models(self):
        """Load trained inverse models"""
        
        # Load inverse FNO
        inv_fno_path = "../checkpoints/best_inverse_model.pt"
        if os.path.exists(inv_fno_path):
            inv_fno = InverseFNO2d(
                modes1=12, modes2=12, width=64,
                num_materials=3, predict_properties=False
            )
            checkpoint = torch.load(inv_fno_path, map_location=self.device)
            inv_fno.load_state_dict(checkpoint['model_state_dict'])
            inv_fno.eval()
            self.inverse_models['FNO'] = inv_fno
            print("✓ Inverse FNO model loaded")
        
        # Load inverse UNet
        inv_unet_path = "../checkpoints/best_inverse_unet_model.pt"
        if os.path.exists(inv_unet_path):
            inv_unet = InverseUNet(
                n_channels=3, n_classes=3,
                predict_properties=False, bilinear=True
            )
            checkpoint = torch.load(inv_unet_path, map_location=self.device)
            inv_unet.load_state_dict(checkpoint['model_state_dict'])
            inv_unet.eval()
            self.inverse_models['UNet'] = inv_unet
            print("✓ Inverse UNet model loaded")
    
    def round_trip_evaluation(self, material_masks, boundary_conditions, forces):
        """
        Perform round-trip evaluation:
        Material → Forward Model → Displacement → Inverse Model → Material
        """
        
        results = {}
        
        for forward_name, forward_model in self.forward_models.items():
            for inverse_name, inverse_model in self.inverse_models.items():
                
                print(f"\nEvaluating {forward_name} → {inverse_name} round trip...")
                
                round_trip_results = []
                
                with torch.no_grad():
                    for i in range(len(material_masks)):
                        # Original material mask
                        original_mask = material_masks[i]
                        bc = boundary_conditions[i]
                        
                        # Create input for forward model
                        bc_map = torch.ones_like(original_mask) * bc
                        forward_input = torch.stack([original_mask, bc_map], dim=0).unsqueeze(0)
                        forward_input = forward_input.to(self.device)
                        
                        # Forward pass: Material + BC → Displacement + Force
                        if forward_name == 'FNO':
                            disp_pred, force_pred = forward_model(forward_input)
                        else:  # CNN
                            outputs = forward_model(forward_input)
                            disp_pred = outputs[:, :2]  # First 2 channels are displacement
                            force_pred = outputs[:, 2:3]  # Third channel is force
                        
                        # Create input for inverse model
                        force_map = torch.ones_like(disp_pred[:, 0:1]) * force_pred
                        inverse_input = torch.cat([disp_pred, force_map], dim=1)
                        
                        # Inverse pass: Displacement + Force → Material
                        material_pred = inverse_model(inverse_input)
                        material_pred = torch.argmax(material_pred, dim=1)
                        
                        # Calculate accuracy
                        original_mask_cpu = original_mask.cpu().numpy()
                        material_pred_cpu = material_pred[0].cpu().numpy()
                        
                        accuracy = np.mean(original_mask_cpu == material_pred_cpu)
                        round_trip_results.append({
                            'sample_idx': i,
                            'accuracy': accuracy,
                            'original_mask': original_mask_cpu,
                            'predicted_mask': material_pred_cpu,
                            'displacement': disp_pred[0].cpu().numpy(),
                            'force': force_pred[0].cpu().numpy()
                        })
                
                combo_name = f"{forward_name}_{inverse_name}"
                results[combo_name] = round_trip_results
                
                avg_accuracy = np.mean([r['accuracy'] for r in round_trip_results])
                print(f"  Average round-trip accuracy: {avg_accuracy:.4f}")
        
        return results
    
    def visualize_round_trip(self, results, num_samples=3, save_path="../exports/"):
        """Visualize round-trip results"""
        
        os.makedirs(save_path, exist_ok=True)
        
        # Material colormap
        colors = ['black', 'red', 'blue']
        cmap = ListedColormap(colors)
        
        for combo_name, combo_results in results.items():
            
            fig, axes = plt.subplots(4, num_samples, figsize=(4*num_samples, 16))
            if num_samples == 1:
                axes = axes.reshape(-1, 1)
            
            # Show worst performing samples
            sorted_results = sorted(combo_results, key=lambda x: x['accuracy'])[:num_samples]
            
            for i, result in enumerate(sorted_results):
                # Original material mask
                axes[0, i].imshow(result['original_mask'], cmap=cmap, vmin=0, vmax=2)
                axes[0, i].set_title(f'Original Material\n(Sample {result["sample_idx"]})')
                axes[0, i].axis('off')
                
                # Predicted displacement (magnitude)
                ux, uy = result['displacement'][0], result['displacement'][1]
                disp_mag = np.sqrt(ux**2 + uy**2)
                im = axes[1, i].imshow(disp_mag, cmap='viridis')
                axes[1, i].set_title(f'Forward Model Output\n(Displacement Magnitude)')
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
                
                # Force prediction
                force_val = result['force'].item()
                axes[2, i].text(0.5, 0.5, f'Predicted Force:\n{force_val:.4f}', 
                               ha='center', va='center', fontsize=12,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[2, i].set_xlim(0, 1)
                axes[2, i].set_ylim(0, 1)
                axes[2, i].axis('off')
                
                # Predicted material mask
                axes[3, i].imshow(result['predicted_mask'], cmap=cmap, vmin=0, vmax=2)
                axes[3, i].set_title(f'Reconstructed Material\n(Accuracy: {result["accuracy"]:.3f})')
                axes[3, i].axis('off')
            
            plt.suptitle(f'Round-Trip Evaluation: {combo_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'round_trip_{combo_name}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_round_trip_summary(self, results, save_path="../exports/"):
        """Create summary statistics for round-trip evaluation"""
        
        os.makedirs(save_path, exist_ok=True)
        
        summary_data = []
        
        for combo_name, combo_results in results.items():
            accuracies = [r['accuracy'] for r in combo_results]
            
            summary_data.append({
                'Model_Combination': combo_name,
                'Mean_Accuracy': np.mean(accuracies),
                'Std_Accuracy': np.std(accuracies),
                'Min_Accuracy': np.min(accuracies),
                'Max_Accuracy': np.max(accuracies),
                'Median_Accuracy': np.median(accuracies),
                'Num_Samples': len(accuracies)
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(save_path, 'round_trip_summary.csv'), index=False)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        x_pos = range(len(summary_data))
        means = [d['Mean_Accuracy'] for d in summary_data]
        stds = [d['Std_Accuracy'] for d in summary_data]
        labels = [d['Model_Combination'] for d in summary_data]
        
        plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        plt.xlabel('Model Combination')
        plt.ylabel('Round-Trip Accuracy')
        plt.title('Round-Trip Evaluation: Forward → Inverse Model Combinations')
        plt.xticks(x_pos, labels, rotation=45)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'round_trip_comparison.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        return df


def main():
    """Main comparison function"""
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Initialize comparison
    comparison = ForwardInverseComparison(device=DEVICE)
    
    # Load models
    print("Loading forward models...")
    comparison.load_forward_models()
    
    print("Loading inverse models...")
    comparison.load_inverse_models()
    
    if not comparison.forward_models or not comparison.inverse_models:
        print("Error: Not all models could be loaded. Please train models first.")
        return
    
    # Create test data (in practice, use your actual test data)
    print("Creating test data...")
    num_samples = 10
    material_masks = []
    boundary_conditions = []
    forces = []
    
    for i in range(num_samples):
        # Create random material mask
        mask = torch.randint(0, 3, (56, 56)).float()
        material_masks.append(mask)
        
        # Random boundary condition
        bc = torch.rand(1) * 0.1  # Small displacement
        boundary_conditions.append(bc)
        
        # Random force (will be overwritten by forward model)
        force = torch.rand(1)
        forces.append(force)
    
    print("Performing round-trip evaluation...")
    results = comparison.round_trip_evaluation(material_masks, boundary_conditions, forces)
    
    print("Creating visualizations...")
    comparison.visualize_round_trip(results)
    
    print("Creating summary...")
    summary_df = comparison.create_round_trip_summary(results)
    
    print("\nRound-Trip Evaluation Summary:")
    print(summary_df.to_string(index=False))
    
    print("\nEvaluation completed! Check the exports folder for detailed results.")


if __name__ == "__main__":
    main()
