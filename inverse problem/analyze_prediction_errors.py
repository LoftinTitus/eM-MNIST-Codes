#!/usr/bin/env python3
"""
Analyze prediction errors from NPZ file and export comprehensive metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from pathlib import Path
import argparse

def calculate_iou_per_class(y_true, y_pred, num_classes):
    """Calculate IoU for each class."""
    ious = []
    for class_id in range(num_classes):
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)
        
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        ious.append(iou)
    
    return np.array(ious)

def calculate_comprehensive_metrics(targets, predictions):
    """Calculate comprehensive error metrics."""
    # Flatten arrays for global metrics
    targets_flat = targets.flatten()
    predictions_flat = predictions.flatten()
    
    # Global metrics
    global_accuracy = accuracy_score(targets_flat, predictions_flat)
    
    # Get unique classes
    unique_classes = np.unique(np.concatenate([targets_flat, predictions_flat]))
    num_classes = len(unique_classes)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets_flat, predictions_flat, labels=unique_classes, average=None, zero_division=0
    )
    
    # IoU metrics
    global_iou_per_class = calculate_iou_per_class(targets_flat, predictions_flat, num_classes)
    
    # Per-sample metrics
    sample_accuracies = []
    sample_ious = []
    
    for i in range(len(targets)):
        sample_acc = accuracy_score(targets[i].flatten(), predictions[i].flatten())
        sample_iou_per_class = calculate_iou_per_class(targets[i], predictions[i], num_classes)
        sample_mean_iou = np.mean(sample_iou_per_class)
        
        sample_accuracies.append(sample_acc)
        sample_ious.append(sample_mean_iou)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(targets_flat, predictions_flat, labels=unique_classes)
    
    return {
        'global_accuracy': global_accuracy,
        'global_mean_iou': np.mean(global_iou_per_class),
        'global_iou_per_class': global_iou_per_class,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'sample_accuracies': np.array(sample_accuracies),
        'sample_ious': np.array(sample_ious),
        'confusion_matrix': conf_matrix,
        'unique_classes': unique_classes,
        'num_classes': num_classes
    }

def export_metrics_to_csv(metrics, output_dir):
    """Export metrics to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Summary metrics
    summary_data = {
        'Metric': [
            'Global Accuracy',
            'Global Mean IoU',
            'Mean Sample Accuracy',
            'Std Sample Accuracy',
            'Mean Sample IoU',
            'Std Sample IoU',
            'Min Sample Accuracy',
            'Max Sample Accuracy',
            'Min Sample IoU',
            'Max Sample IoU'
        ],
        'Value': [
            metrics['global_accuracy'],
            metrics['global_mean_iou'],
            np.mean(metrics['sample_accuracies']),
            np.std(metrics['sample_accuracies']),
            np.mean(metrics['sample_ious']),
            np.std(metrics['sample_ious']),
            np.min(metrics['sample_accuracies']),
            np.max(metrics['sample_accuracies']),
            np.min(metrics['sample_ious']),
            np.max(metrics['sample_ious'])
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'error_summary_metrics.csv', index=False)
    
    # Per-class metrics
    per_class_data = {
        'Class': metrics['unique_classes'],
        'IoU': metrics['global_iou_per_class'],
        'Precision': metrics['per_class_precision'],
        'Recall': metrics['per_class_recall'],
        'F1_Score': metrics['per_class_f1'],
        'Support': metrics['per_class_support']
    }
    
    per_class_df = pd.DataFrame(per_class_data)
    per_class_df.to_csv(output_dir / 'per_class_metrics.csv', index=False)
    
    # Per-sample metrics
    per_sample_data = {
        'Sample_ID': range(len(metrics['sample_accuracies'])),
        'Accuracy': metrics['sample_accuracies'],
        'Mean_IoU': metrics['sample_ious']
    }
    
    per_sample_df = pd.DataFrame(per_sample_data)
    per_sample_df.to_csv(output_dir / 'per_sample_metrics.csv', index=False)
    
    # Confusion matrix
    conf_matrix_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=metrics['unique_classes'],
        columns=metrics['unique_classes']
    )
    conf_matrix_df.to_csv(output_dir / 'confusion_matrix.csv')
    
    print(f"✓ Metrics exported to {output_dir}")
    return summary_df, per_class_df, per_sample_df

def create_error_visualizations(metrics, inputs, targets, predictions, output_dir):
    """Create comprehensive error visualization plots."""
    output_dir = Path(output_dir)
    
    # Set style
    plt.style.use('default')
    
    # 1. Sample accuracy and IoU distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(metrics['sample_accuracies'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(metrics['sample_accuracies']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(metrics["sample_accuracies"]):.4f}')
    ax1.set_xlabel('Sample Accuracy')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Sample Accuracies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(metrics['sample_ious'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(np.mean(metrics['sample_ious']), color='red', linestyle='--',
                label=f'Mean: {np.mean(metrics["sample_ious"]):.4f}')
    ax2.set_xlabel('Sample Mean IoU')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Sample Mean IoUs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_iou_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class metrics bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics['unique_classes']))
    width = 0.2
    
    ax.bar(x - width, metrics['global_iou_per_class'], width, label='IoU', alpha=0.8)
    ax.bar(x, metrics['per_class_precision'], width, label='Precision', alpha=0.8)
    ax.bar(x + width, metrics['per_class_recall'], width, label='Recall', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Metric Value')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics['unique_classes'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], 
                xticklabels=metrics['unique_classes'],
                yticklabels=metrics['unique_classes'],
                annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Sample-by-sample comparison (first 5 samples if available)
    n_samples_to_plot = min(5, len(inputs))
    if n_samples_to_plot > 0:
        fig, axes = plt.subplots(n_samples_to_plot, 3, figsize=(12, 3*n_samples_to_plot))
        if n_samples_to_plot == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples_to_plot):
            # Input (first channel if multi-channel)
            im1 = axes[i, 0].imshow(inputs[i, 0] if len(inputs[i].shape) == 3 else inputs[i], 
                                   cmap='RdBu', aspect='equal')
            axes[i, 0].set_title(f'Sample {i}: Input')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Ground truth
            im2 = axes[i, 1].imshow(targets[i], cmap='viridis', aspect='equal')
            axes[i, 1].set_title(f'Ground Truth\nAcc: {metrics["sample_accuracies"][i]:.4f}')
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Prediction
            im3 = axes[i, 2].imshow(predictions[i], cmap='viridis', aspect='equal')
            axes[i, 2].set_title(f'Prediction\nIoU: {metrics["sample_ious"][i]:.4f}')
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_comparisons.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Visualizations saved to {output_dir}")

def print_summary_statistics(metrics):
    """Print formatted summary statistics."""
    print("\n" + "="*60)
    print("PREDICTION ERROR ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nGLOBAL METRICS:")
    print(f"  Overall Accuracy: {metrics['global_accuracy']:.4f} ({metrics['global_accuracy']*100:.2f}%)")
    print(f"  Overall Mean IoU: {metrics['global_mean_iou']:.4f}")
    
    print(f"\nSAMPLE-WISE STATISTICS:")
    print(f"  Sample Accuracy  - Mean: {np.mean(metrics['sample_accuracies']):.4f} ± {np.std(metrics['sample_accuracies']):.4f}")
    print(f"                   - Range: [{np.min(metrics['sample_accuracies']):.4f}, {np.max(metrics['sample_accuracies']):.4f}]")
    print(f"  Sample Mean IoU  - Mean: {np.mean(metrics['sample_ious']):.4f} ± {np.std(metrics['sample_ious']):.4f}")
    print(f"                   - Range: [{np.min(metrics['sample_ious']):.4f}, {np.max(metrics['sample_ious']):.4f}]")
    
    print(f"\nPER-CLASS METRICS:")
    for i, class_id in enumerate(metrics['unique_classes']):
        print(f"  Class {class_id}:")
        print(f"    IoU: {metrics['global_iou_per_class'][i]:.4f}")
        print(f"    Precision: {metrics['per_class_precision'][i]:.4f}")
        print(f"    Recall: {metrics['per_class_recall'][i]:.4f}")
        print(f"    F1-Score: {metrics['per_class_f1'][i]:.4f}")
        print(f"    Support: {metrics['per_class_support'][i]}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Analyze prediction errors from NPZ file')
    parser.add_argument('--input_file', type=str, default='fno_prediction_data.npz',
                       help='Path to the NPZ prediction file')
    parser.add_argument('--output_dir', type=str, default='error_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    print("Loading prediction data...")
    try:
        data = np.load(args.input_file)
        inputs = data['inputs']
        targets = data['targets']
        predictions = data['predictions']
        
        print(f"✓ Loaded data: {len(inputs)} samples")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Target shape: {targets.shape}")
        print(f"  Prediction shape: {predictions.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("\nCalculating comprehensive metrics...")
    metrics = calculate_comprehensive_metrics(targets, predictions)
    
    print("\nExporting metrics to CSV...")
    summary_df, per_class_df, per_sample_df = export_metrics_to_csv(metrics, args.output_dir)
    
    print("\nCreating visualizations...")
    create_error_visualizations(metrics, inputs, targets, predictions, args.output_dir)
    
    print_summary_statistics(metrics)
    
    print(f"\n✓ Analysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
