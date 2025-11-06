#!/usr/bin/env python3
"""
Evaluate model against ALL test samples and extract accuracy + label for each sample.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
import dataload
import dataprocess
from dataform_inverse import normalize_inverse, build_inverse_dataset, build_material_property_dataset
from dataset_inverse import create_inverse_dataloaders, analyze_dataset_statistics
from inverse_models import InverseUNet, InverseFNO2d

def load_model(model_path, model_type, input_channels=3, num_classes=6, device='cpu'):
    """Load a trained model from checkpoint."""
    
    if model_type.lower() == 'cnn' or model_type.lower() == 'unet':
        model = InverseUNet(n_channels=input_channels, n_classes=num_classes)
    elif model_type.lower() == 'fno':
        model = InverseFNO2d(modes1=16, modes2=16, width=64, num_materials=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded {model_type} model from {model_path}")
    return model

def get_mnist_label_from_sample(sample):
    """Extract MNIST digit label from sample filename or material mask."""
    # Try to extract from filename first
    filename = sample.get('filename', '')
    if filename:
        # Look for digit patterns in filename
        import re
        digit_match = re.search(r'(\d)', filename)
        if digit_match:
            return int(digit_match.group(1))
    
    # Fall back to dominant class in material mask
    material_mask = sample['material_mask']
    if material_mask is not None:
        unique_classes, counts = np.unique(material_mask, return_counts=True)
        if len(unique_classes) > 0:
            dominant_class = unique_classes[np.argmax(counts)]
            return int(dominant_class)
    
    return None

def evaluate_all_samples(model, test_loader, device, original_samples):
    """Evaluate model on all test samples and extract detailed metrics."""
    
    results = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[1]
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get predictions
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                predictions = outputs[0]  # Segmentation output
            else:
                predictions = outputs
            
            # Convert to class predictions
            if predictions.dim() > 3:  # [B, C, H, W]
                pred_classes = torch.argmax(predictions, dim=1)
            else:  # [B, H, W]
                pred_classes = predictions
            
            # Process each sample in the batch
            batch_size = inputs.shape[0]
            for i in range(batch_size):
                if sample_idx >= len(original_samples):
                    break
                
                # Get current sample data
                current_sample = original_samples[sample_idx]
                
                # Extract data for this sample
                target_np = targets[i].cpu().numpy()
                pred_np = pred_classes[i].cpu().numpy()
                
                # Calculate accuracy
                sample_accuracy = accuracy_score(target_np.flatten(), pred_np.flatten())
                
                # Get MNIST label
                mnist_label = get_mnist_label_from_sample(current_sample)
                
                # Get class distribution
                unique_classes, counts = np.unique(target_np, return_counts=True)
                dominant_class = unique_classes[np.argmax(counts)]
                dominant_percentage = (counts[np.argmax(counts)] / target_np.size) * 100
                
                class_distribution = dict(zip(unique_classes.astype(int), counts.astype(int)))
                
                # Store results
                result = {
                    'Sample_ID': sample_idx,
                    'Filename': current_sample.get('filename', f'sample_{sample_idx}'),
                    'MNIST_Label': mnist_label,
                    'Dominant_Class': int(dominant_class),
                    'Accuracy': sample_accuracy,
                    'Accuracy_Percentage': sample_accuracy * 100,
                    'Dominant_Class_Percentage': dominant_percentage,
                    'Total_Pixels': target_np.size,
                    'Correct_Pixels': int(np.sum(target_np == pred_np)),
                    'Incorrect_Pixels': int(np.sum(target_np != pred_np)),
                    'Num_Classes': len(unique_classes),
                    'Class_Distribution': str(class_distribution)
                }
                
                results.append(result)
                sample_idx += 1
                
                if sample_idx % 50 == 0:
                    print(f"Processed {sample_idx} samples...")
    
    return results

def create_comprehensive_analysis(results_df, output_dir):
    """Create comprehensive analysis and visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nCOMPREHENSIVE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total samples evaluated: {len(results_df)}")
    print(f"Average accuracy: {results_df['Accuracy'].mean():.6f} ({results_df['Accuracy'].mean()*100:.4f}%)")
    print(f"Accuracy std: {results_df['Accuracy'].std():.6f}")
    print(f"Min accuracy: {results_df['Accuracy'].min():.6f} ({results_df['Accuracy'].min()*100:.4f}%)")
    print(f"Max accuracy: {results_df['Accuracy'].max():.6f} ({results_df['Accuracy'].max()*100:.4f}%)")
    
    # MNIST label distribution
    if results_df['MNIST_Label'].notna().any():
        label_counts = results_df['MNIST_Label'].value_counts().sort_index()
        print(f"\nMNIST LABEL DISTRIBUTION:")
        for label, count in label_counts.items():
            subset = results_df[results_df['MNIST_Label'] == label]
            avg_acc = subset['Accuracy'].mean()
            print(f"  Digit {int(label)}: {count} samples ({count/len(results_df)*100:.1f}%) - Avg Acc: {avg_acc:.4f}")
    
    # Accuracy by MNIST digit
    if results_df['MNIST_Label'].notna().any():
        plt.figure(figsize=(12, 6))
        
        # Violin plot
        plt.subplot(1, 2, 1)
        sns.violinplot(data=results_df.dropna(subset=['MNIST_Label']), 
                      x='MNIST_Label', y='Accuracy')
        plt.title('Accuracy Distribution by MNIST Digit')
        plt.ylabel('Accuracy')
        plt.xlabel('MNIST Digit')
        
        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(data=results_df.dropna(subset=['MNIST_Label']), 
                   x='MNIST_Label', y='Accuracy')
        plt.title('Accuracy Statistics by MNIST Digit')
        plt.ylabel('Accuracy')
        plt.xlabel('MNIST Digit')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_by_mnist_digit_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Overall accuracy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['Accuracy'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(results_df['Accuracy'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["Accuracy"].mean():.4f}')
    plt.axvline(results_df['Accuracy'].median(), color='green', linestyle='--', 
                label=f'Median: {results_df["Accuracy"].median():.4f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Overall Accuracy Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'overall_accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Accuracy vs Sample ID (to check for patterns)
    plt.figure(figsize=(12, 6))
    plt.scatter(results_df['Sample_ID'], results_df['Accuracy'], alpha=0.6)
    plt.xlabel('Sample ID')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Sample ID')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'accuracy_vs_sample_id.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Analysis visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on ALL test samples')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'unet', 'fno'], 
                       default='fno', help='Type of model to evaluate')
    parser.add_argument('--data_dir', type=str, 
                       default="/Users/tyloftin/Downloads/MNIST_comp_files",
                       help='Directory containing the dataset')
    parser.add_argument('--output_file', type=str, 
                       default='all_samples_evaluation.csv',
                       help='Output CSV file for results')
    parser.add_argument('--output_dir', type=str, 
                       default='full_evaluation_analysis',
                       help='Directory for analysis outputs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("FULL DATASET EVALUATION")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {device}")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    # Load data
    print("\nLoading dataset...")
    try:
        raw_samples = dataload.load_dic_samples(args.data_dir)
        if len(raw_samples) == 0:
            print("No samples found in data directory!")
            return
        
        print(f"✓ Loaded {len(raw_samples)} samples")
        
        # Process data using the same pipeline as training
        TARGET_SIZE = 56
        processed_samples = []
        
        print("  Preprocessing samples...")
        for i, raw_data in enumerate(raw_samples):
            try:
                if (i + 1) % 10 == 0:
                    print(f"    Processing sample {i+1}/{len(raw_samples)}...")
                
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
                print(f"    Error processing sample {i}: {e}")
                continue
        
        print(f"  Successfully processed {len(processed_samples)} samples")
        
        # Build dataset
        X, Y_seg, BC = build_inverse_dataset(processed_samples)
        
        # Remap labels to contiguous indices (same as training)
        unique_labels = torch.unique(Y_seg).tolist()
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        Y_seg_remapped = torch.zeros_like(Y_seg)
        for old_label, new_label in label_mapping.items():
            Y_seg_remapped[Y_seg == old_label] = new_label
        Y_seg = Y_seg_remapped
        
        print(f"✓ Processed data: X shape = {X.shape}, Y_seg shape = {Y_seg.shape}")
        print(f"✓ Label mapping: {label_mapping}")
        
        # Create data loaders (we'll use the test loader)
        train_loader, val_loader, test_loader = create_inverse_dataloaders(
            X, Y_seg, batch_size=args.batch_size, test_split=0.2, val_split=0.1
        )
        
        # For evaluation, we want to use ALL samples, not just test split
        # So let's create a loader with all data
        from torch.utils.data import TensorDataset, DataLoader
        full_dataset = TensorDataset(X, Y_seg)
        full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"✓ Created data loader with {len(full_dataset)} samples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Load model
    print("\nLoading model...")
    try:
        model = load_model(args.model_path, args.model_type, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Evaluate all samples
    print(f"\nEvaluating model on all {len(raw_samples)} samples...")
    results = evaluate_all_samples(model, full_loader, device, raw_samples)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Export results
    results_df.to_csv(args.output_file, index=False)
    print(f"\n✓ Results exported to: {args.output_file}")
    
    # Create comprehensive analysis
    print("\nCreating comprehensive analysis...")
    create_comprehensive_analysis(results_df, args.output_dir)
    
    print(f"\n✓ Evaluation complete! Processed {len(results)} samples.")

if __name__ == "__main__":
    main()
