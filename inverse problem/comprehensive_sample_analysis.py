#!/usr/bin/env python3
"""
Comprehensive analysis of all samples with detailed accuracy and label extraction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path
import argparse

def detailed_sample_analysis(npz_file):
    """
    Comprehensive analysis of each sample including per-pixel accuracy, 
    class-wise metrics, and spatial error patterns.
    """
    
    # Load data
    data = np.load(npz_file)
    inputs = data['inputs']
    targets = data['targets']
    predictions = data['predictions']
    stored_accuracies = data.get('accuracies', None)
    
    print(f"COMPREHENSIVE SAMPLE ANALYSIS")
    print("="*60)
    print(f"Dataset: {npz_file}")
    print(f"Total samples: {len(inputs)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Prediction shape: {predictions.shape}")
    print("="*60)
    
    # Global analysis
    all_targets = targets.flatten()
    all_predictions = predictions.flatten()
    global_accuracy = accuracy_score(all_targets, all_predictions)
    unique_classes = np.unique(np.concatenate([all_targets, all_predictions]))
    
    print(f"\\nGLOBAL STATISTICS:")
    print(f"  Overall accuracy: {global_accuracy:.6f} ({global_accuracy*100:.4f}%)")
    print(f"  Classes present: {unique_classes}")
    
    # Per-sample detailed analysis
    sample_results = []
    
    for sample_idx in range(len(inputs)):
        print(f"\\n{'='*50}")
        print(f"SAMPLE {sample_idx} DETAILED ANALYSIS")
        print(f"{'='*50}")
        
        # Basic metrics
        target = targets[sample_idx]
        prediction = predictions[sample_idx]
        target_flat = target.flatten()
        pred_flat = prediction.flatten()
        
        sample_accuracy = accuracy_score(target_flat, pred_flat)
        
        # Class distribution in target
        unique_target_classes, target_counts = np.unique(target_flat, return_counts=True)
        dominant_class_idx = np.argmax(target_counts)
        mnist_digit = unique_target_classes[dominant_class_idx]
        dominant_percentage = (target_counts[dominant_class_idx] / len(target_flat)) * 100
        
        print(f"MNIST Digit (Dominant Class): {mnist_digit}")
        print(f"Overall Sample Accuracy: {sample_accuracy:.6f} ({sample_accuracy*100:.4f}%)")
        print(f"Dominant class coverage: {dominant_percentage:.2f}%")
        
        # Per-class metrics for this sample
        precision, recall, f1, support = precision_recall_fscore_support(
            target_flat, pred_flat, labels=unique_target_classes, average=None, zero_division=0
        )
        
        print(f"\\nPER-CLASS METRICS:")
        class_metrics = {}
        for i, class_id in enumerate(unique_target_classes):
            class_acc = accuracy_score(target_flat == class_id, pred_flat == class_id)
            
            # IoU for this class
            true_mask = (target_flat == class_id)
            pred_mask = (pred_flat == class_id)
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 1.0
            
            print(f"  Class {class_id}:")
            print(f"    Support: {support[i]} pixels ({support[i]/len(target_flat)*100:.2f}%)")
            print(f"    Accuracy: {class_acc:.6f}")
            print(f"    Precision: {precision[i]:.6f}")
            print(f"    Recall: {recall[i]:.6f}")
            print(f"    F1-Score: {f1[i]:.6f}")
            print(f"    IoU: {iou:.6f}")
            
            class_metrics[f'class_{class_id}'] = {
                'support': support[i],
                'accuracy': class_acc,
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'iou': iou
            }
        
        # Spatial error analysis
        error_map = (target != prediction).astype(int)
        total_errors = np.sum(error_map)
        error_percentage = (total_errors / error_map.size) * 100
        
        print(f"\\nSPATIAL ERROR ANALYSIS:")
        print(f"  Total error pixels: {total_errors}")
        print(f"  Error percentage: {error_percentage:.4f}%")
        print(f"  Error density: {total_errors / error_map.size:.6f}")
        
        # Confusion matrix for this sample
        conf_matrix = confusion_matrix(target_flat, pred_flat, labels=unique_target_classes)
        print(f"\\nCONFUSION MATRIX:")
        print("  True\\\\Pred", end="")
        for class_id in unique_target_classes:
            print(f"{class_id:>8}", end="")
        print()
        
        for i, true_class in enumerate(unique_target_classes):
            print(f"  {true_class:>8}", end="")
            for j, pred_class in enumerate(unique_target_classes):
                print(f"{conf_matrix[i,j]:>8}", end="")
            print()
        
        # Store results
        result = {
            'Sample_ID': sample_idx,
            'MNIST_Digit': int(mnist_digit),
            'Accuracy': sample_accuracy,
            'Accuracy_Percentage': sample_accuracy * 100,
            'Dominant_Class_Percentage': dominant_percentage,
            'Total_Pixels': len(target_flat),
            'Correct_Pixels': int(np.sum(target_flat == pred_flat)),
            'Error_Pixels': total_errors,
            'Error_Percentage': error_percentage,
            'Num_Classes': len(unique_target_classes),
            'Class_Distribution': str(dict(zip(unique_target_classes.astype(int), target_counts.astype(int)))),
            'Stored_Accuracy': stored_accuracies[sample_idx] if stored_accuracies is not None else None,
        }
        
        # Add per-class metrics to result
        for class_name, metrics in class_metrics.items():
            for metric_name, value in metrics.items():
                result[f'{class_name}_{metric_name}'] = value
        
        sample_results.append(result)
    
    return pd.DataFrame(sample_results), unique_classes

def create_comprehensive_visualizations(df, inputs, targets, predictions, unique_classes, output_dir):
    """Create detailed visualizations for comprehensive analysis."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Sample-by-sample detailed comparison
    n_samples = len(inputs)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Input (first channel)
        im1 = axes[i, 0].imshow(inputs[i, 0], cmap='RdBu', aspect='equal')
        axes[i, 0].set_title(f'Sample {i}: Input\\nMNIST Digit {df.iloc[i]["MNIST_Digit"]}')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Ground truth
        im2 = axes[i, 1].imshow(targets[i], cmap='viridis', aspect='equal', vmin=0, vmax=len(unique_classes)-1)
        axes[i, 1].set_title(f'Ground Truth\\nClasses: {df.iloc[i]["Num_Classes"]}')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Prediction
        im3 = axes[i, 2].imshow(predictions[i], cmap='viridis', aspect='equal', vmin=0, vmax=len(unique_classes)-1)
        axes[i, 2].set_title(f'Prediction\\nAcc: {df.iloc[i]["Accuracy"]:.4f}')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Error map
        error_map = (targets[i] != predictions[i]).astype(int)
        im4 = axes[i, 3].imshow(error_map, cmap='Reds', aspect='equal')
        axes[i, 3].set_title(f'Error Map\\nErrors: {df.iloc[i]["Error_Percentage"]:.2f}%')
        axes[i, 3].axis('off')
        plt.colorbar(im4, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_sample_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy metrics summary
    plt.figure(figsize=(15, 10))
    
    # Accuracy by MNIST digit
    plt.subplot(2, 3, 1)
    digit_acc = df.groupby('MNIST_Digit')['Accuracy'].agg(['mean', 'std']).reset_index()
    bars = plt.bar(digit_acc['MNIST_Digit'], digit_acc['mean'], yerr=digit_acc['std'], capsize=5)
    plt.xlabel('MNIST Digit')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by MNIST Digit')
    plt.grid(True, alpha=0.3)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, digit_acc['mean'])):
        plt.annotate(f'{acc:.4f}', (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                    ha='center', va='bottom', fontsize=9)
    
    # Error distribution
    plt.subplot(2, 3, 2)
    plt.bar(df['Sample_ID'], df['Error_Percentage'], color='coral', alpha=0.7)
    plt.xlabel('Sample ID')
    plt.ylabel('Error Percentage (%)')
    plt.title('Error Percentage by Sample')
    plt.grid(True, alpha=0.3)
    
    # Class distribution
    plt.subplot(2, 3, 3)
    plt.bar(df['Sample_ID'], df['Num_Classes'], color='lightgreen', alpha=0.7)
    plt.xlabel('Sample ID')
    plt.ylabel('Number of Classes')
    plt.title('Number of Classes per Sample')
    plt.grid(True, alpha=0.3)
    
    # Accuracy vs dominant class percentage
    plt.subplot(2, 3, 4)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, digit in enumerate(df['MNIST_Digit'].unique()):
        subset = df[df['MNIST_Digit'] == digit]
        plt.scatter(subset['Dominant_Class_Percentage'], subset['Accuracy'], 
                   label=f'Digit {digit}', color=colors[i % len(colors)], s=100, alpha=0.7)
    
    plt.xlabel('Dominant Class Percentage (%)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Dominant Class Coverage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Per-class IoU comparison (if available)
    plt.subplot(2, 3, 5)
    class_cols = [col for col in df.columns if col.startswith('class_') and col.endswith('_iou')]
    if class_cols:
        iou_data = []
        class_names = []
        for col in class_cols:
            class_id = col.split('_')[1]
            class_names.append(f'Class {class_id}')
            iou_values = df[col].dropna().values
            if len(iou_values) > 0:
                iou_data.append(iou_values)
        
        if iou_data:
            plt.boxplot(iou_data, labels=class_names)
            plt.ylabel('IoU')
            plt.title('IoU Distribution by Class')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
    
    # Sample accuracy histogram
    plt.subplot(2, 3, 6)
    plt.hist(df['Accuracy'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['Accuracy'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["Accuracy"].mean():.4f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Accuracy Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_metrics_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive analysis of prediction samples')
    parser.add_argument('--input_file', type=str, default='fno_prediction_data.npz',
                       help='Path to the NPZ prediction file')
    parser.add_argument('--output_dir', type=str, default='comprehensive_analysis',
                       help='Directory for analysis outputs')
    parser.add_argument('--output_csv', type=str, default='comprehensive_sample_analysis.csv',
                       help='CSV file for detailed results')
    
    args = parser.parse_args()
    
    print("COMPREHENSIVE SAMPLE ANALYSIS")
    print("="*60)
    
    # Load and analyze
    data = np.load(args.input_file)
    inputs = data['inputs']
    targets = data['targets'] 
    predictions = data['predictions']
    
    # Perform detailed analysis
    df, unique_classes = detailed_sample_analysis(args.input_file)
    
    # Export results
    df.to_csv(args.output_csv, index=False)
    print(f"\\n✓ Detailed results exported to: {args.output_csv}")
    
    # Create visualizations
    print(f"\\nCreating comprehensive visualizations...")
    create_comprehensive_visualizations(df, inputs, targets, predictions, unique_classes, args.output_dir)
    
    # Final summary
    print(f"\\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Samples analyzed: {len(df)}")
    print(f"Average accuracy: {df['Accuracy'].mean():.6f} ± {df['Accuracy'].std():.6f}")
    print(f"MNIST digits present: {sorted(df['MNIST_Digit'].unique())}")
    print(f"Best performing sample: {df.loc[df['Accuracy'].idxmax(), 'Sample_ID']} (Acc: {df['Accuracy'].max():.6f})")
    print(f"Worst performing sample: {df.loc[df['Accuracy'].idxmin(), 'Sample_ID']} (Acc: {df['Accuracy'].min():.6f})")
    print(f"Analysis complete!")

if __name__ == "__main__":
    main()
