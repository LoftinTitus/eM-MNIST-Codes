import torch
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def calculate_comprehensive_inverse_metrics(model, test_loader, device, num_materials, 
                                          predict_properties=False, export_dir="../exports"):
    """
    Calculate comprehensive metrics for inverse problem material identification,
    similar to the forward problem evaluation.
    """
    os.makedirs(export_dir, exist_ok=True)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_prop_predictions = []
    all_prop_targets = []
    all_inputs = []
    
    print("Calculating comprehensive inverse problem metrics...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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
            all_inputs.append(inputs.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)
    
    # Calculate segmentation metrics
    from train_inverse import calculate_segmentation_metrics
    seg_metrics = calculate_segmentation_metrics(all_predictions, all_targets, num_materials)
    
    # Prepare detailed results DataFrame
    results_data = []
    
    for i in range(len(all_predictions)):
        pred = all_predictions[i].numpy()
        target = all_targets[i].numpy()
        
        # Overall sample accuracy
        sample_accuracy = np.mean(pred == target)
        
        # Per-class accuracy for this sample
        class_accuracies = {}
        class_ious = {}
        
        for class_id in range(num_materials):
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            if target_mask.sum() > 0:  # Only calculate if class exists in target
                class_acc = np.mean(pred_mask[target_mask])
                
                tp = (pred_mask & target_mask).sum()
                fp = (pred_mask & ~target_mask).sum()
                fn = (~pred_mask & target_mask).sum()
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
                
                class_accuracies[f'class_{class_id}_accuracy'] = class_acc
                class_ious[f'class_{class_id}_iou'] = iou
            else:
                class_accuracies[f'class_{class_id}_accuracy'] = np.nan
                class_ious[f'class_{class_id}_iou'] = np.nan
        
        # Calculate input statistics for this sample
        if all_inputs[i].shape[0] >= 3:  # ux, uy, force
            ux = all_inputs[i, 0].numpy()
            uy = all_inputs[i, 1].numpy()
            force = all_inputs[i, 2].numpy()
            
            disp_magnitude = np.sqrt(ux**2 + uy**2)
            max_displacement = np.max(disp_magnitude)
            mean_displacement = np.mean(disp_magnitude)
            max_force = np.max(np.abs(force))
            mean_force = np.mean(np.abs(force))
        else:
            max_displacement = mean_displacement = max_force = mean_force = np.nan
        
        result_dict = {
            'sample_id': i,
            'overall_accuracy': sample_accuracy,
            'max_displacement': max_displacement,
            'mean_displacement': mean_displacement,
            'max_force': max_force,
            'mean_force': mean_force,
            **class_accuracies,
            **class_ious
        }
        
        results_data.append(result_dict)
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Add property prediction metrics if available
    if predict_properties and all_prop_predictions:
        all_prop_predictions = torch.cat(all_prop_predictions, dim=0)
        all_prop_targets = torch.cat(all_prop_targets, dim=0)
        
        prop_mse = torch.mean((all_prop_predictions - all_prop_targets) ** 2).item()
        prop_mae = torch.mean(torch.abs(all_prop_predictions - all_prop_targets)).item()
        prop_rel_error = torch.mean(torch.abs(all_prop_predictions - all_prop_targets) / 
                                   (torch.abs(all_prop_targets) + 1e-8)).item()
        
        df['property_mse'] = prop_mse
        df['property_mae'] = prop_mae
        df['property_rel_error'] = prop_rel_error
    
    # Export detailed metrics
    detailed_metrics_file = os.path.join(export_dir, 'inverse_detailed_metrics.csv')
    df.to_csv(detailed_metrics_file, index=False)
    
    # Create summary statistics
    summary_stats = {
        'total_samples': len(df),
        'overall_accuracy_mean': df['overall_accuracy'].mean(),
        'overall_accuracy_std': df['overall_accuracy'].std(),
        'mean_iou': seg_metrics['mean_iou'],
        'global_accuracy': seg_metrics['accuracy'],
    }
    
    # Add per-class summary statistics
    for class_id in range(num_materials):
        acc_col = f'class_{class_id}_accuracy'
        iou_col = f'class_{class_id}_iou'
        
        if acc_col in df.columns:
            summary_stats[f'class_{class_id}_accuracy_mean'] = df[acc_col].mean()
            summary_stats[f'class_{class_id}_accuracy_std'] = df[acc_col].std()
            summary_stats[f'class_{class_id}_iou_mean'] = df[iou_col].mean()
            summary_stats[f'class_{class_id}_iou_std'] = df[iou_col].std()
            
            # Global per-class metrics from seg_metrics
            if f'class_{class_id}' in seg_metrics['class_metrics']:
                class_metrics = seg_metrics['class_metrics'][f'class_{class_id}']
                summary_stats[f'class_{class_id}_precision'] = class_metrics['precision']
                summary_stats[f'class_{class_id}_recall'] = class_metrics['recall']
                summary_stats[f'class_{class_id}_iou_global'] = class_metrics['iou']
    
    # Add property summary stats if available
    if predict_properties and 'property_mse' in df.columns:
        summary_stats['property_mse'] = prop_mse
        summary_stats['property_mae'] = prop_mae
        summary_stats['property_rel_error'] = prop_rel_error
    
    # Export summary statistics
    summary_file = os.path.join(export_dir, 'inverse_summary_stats.csv')
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(summary_file, index=False)
    
    # Generate analysis plots
    plot_inverse_analysis(df, seg_metrics, num_materials, export_dir, predict_properties)
    
    # Print comprehensive summary (matching forward problem style)
    print_inverse_summary_statistics(summary_stats, seg_metrics, num_materials, predict_properties)
    
    print(f"Detailed metrics exported to: {detailed_metrics_file}")
    print(f"Summary statistics exported to: {summary_file}")
    
    return {
        'detailed_metrics': df,
        'summary_stats': summary_stats,
        'segmentation_metrics': seg_metrics
    }


def plot_inverse_analysis(df, seg_metrics, num_materials, export_dir, predict_properties=False):
    """Generate analysis plots for inverse problem results"""
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Accuracy distribution
    plt.subplot(3, 3, 1)
    plt.hist(df['overall_accuracy'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Sample Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sample Accuracies')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Per-class IoU
    plt.subplot(3, 3, 2)
    class_ious = [seg_metrics['class_metrics'][f'class_{i}']['iou'] for i in range(num_materials)]
    plt.bar(range(num_materials), class_ious, color=['red', 'blue', 'green', 'orange', 'purple'][:num_materials])
    plt.xlabel('Material Class')
    plt.ylabel('IoU')
    plt.title('Per-Class Intersection over Union')
    plt.xticks(range(num_materials))
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy vs Displacement
    plt.subplot(3, 3, 3)
    plt.scatter(df['max_displacement'], df['overall_accuracy'], alpha=0.6, color='purple')
    plt.xlabel('Max Displacement')
    plt.ylabel('Sample Accuracy')
    plt.title('Accuracy vs Max Displacement')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy vs Force
    plt.subplot(3, 3, 4)
    plt.scatter(df['max_force'], df['overall_accuracy'], alpha=0.6, color='orange')
    plt.xlabel('Max Force')
    plt.ylabel('Sample Accuracy')
    plt.title('Accuracy vs Max Force')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Per-class precision and recall
    plt.subplot(3, 3, 5)
    class_precisions = [seg_metrics['class_metrics'][f'class_{i}']['precision'] for i in range(num_materials)]
    class_recalls = [seg_metrics['class_metrics'][f'class_{i}']['recall'] for i in range(num_materials)]
    
    x = np.arange(num_materials)
    width = 0.35
    plt.bar(x - width/2, class_precisions, width, label='Precision', alpha=0.8)
    plt.bar(x + width/2, class_recalls, width, label='Recall', alpha=0.8)
    plt.xlabel('Material Class')
    plt.ylabel('Score')
    plt.title('Per-Class Precision and Recall')
    plt.xticks(range(num_materials))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Accuracy correlation matrix (if multiple classes)
    if num_materials > 1:
        plt.subplot(3, 3, 6)
        class_acc_cols = [f'class_{i}_accuracy' for i in range(num_materials) if f'class_{i}_accuracy' in df.columns]
        if len(class_acc_cols) > 1:
            corr_data = df[class_acc_cols].corr()
            im = plt.imshow(corr_data, cmap='RdBu', vmin=-1, vmax=1)
            plt.colorbar(im)
            plt.title('Class Accuracy Correlations')
            plt.xticks(range(len(class_acc_cols)), [f'C{i}' for i in range(len(class_acc_cols))])
            plt.yticks(range(len(class_acc_cols)), [f'C{i}' for i in range(len(class_acc_cols))])
    
    # Plot 7-9: Property prediction metrics if available
    if predict_properties and 'property_mse' in df.columns:
        plt.subplot(3, 3, 7)
        plt.hist(df['property_mse'], bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Property MSE')
        plt.ylabel('Frequency')
        plt.title('Property Prediction Error Distribution')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(export_dir, 'inverse_problem_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Analysis plot saved to: {plot_file}")


def print_inverse_summary_statistics(summary_stats, seg_metrics, num_materials, predict_properties=False):
    """Print comprehensive summary statistics matching forward problem style"""
    
    print("\n" + "="*60)
    print("INVERSE PROBLEM SUMMARY STATISTICS")
    print("="*60)
    print(f"Total samples analyzed: {summary_stats['total_samples']}")
    print(f"Model type: Material Identification (Segmentation)")
    print(f"Number of material classes: {num_materials}")
    
    print(f"\nGLOBAL METRICS:")
    print(f"  Overall accuracy: {summary_stats['global_accuracy']:.6f}")
    print(f"  Mean sample accuracy: {summary_stats['overall_accuracy_mean']:.6f} ± {summary_stats['overall_accuracy_std']:.6f}")
    print(f"  Mean Intersection over Union (IoU): {summary_stats['mean_iou']:.6f}")
    
    print(f"\nPER-CLASS METRICS:")
    for class_id in range(num_materials):
        if f'class_{class_id}_precision' in summary_stats:
            precision = summary_stats[f'class_{class_id}_precision']
            recall = summary_stats[f'class_{class_id}_recall']
            iou = summary_stats[f'class_{class_id}_iou_global']
            acc_mean = summary_stats.get(f'class_{class_id}_accuracy_mean', 0)
            acc_std = summary_stats.get(f'class_{class_id}_accuracy_std', 0)
            
            print(f"  Material Class {class_id}:")
            print(f"    Precision: {precision:.6f}")
            print(f"    Recall: {recall:.6f}")
            print(f"    IoU: {iou:.6f}")
            print(f"    Sample-wise accuracy: {acc_mean:.6f} ± {acc_std:.6f}")
    
    if predict_properties:
        print(f"\nMATERIAL PROPERTY PREDICTION:")
        if 'property_mse' in summary_stats:
            print(f"  Property MSE: {summary_stats['property_mse']:.6f}")
            print(f"  Property MAE: {summary_stats['property_mae']:.6f}")
            print(f"  Property relative error: {summary_stats['property_rel_error']:.6f}")
        else:
            print("  Property prediction was not enabled")
    
    print("="*60)


if __name__ == "__main__":
    # This can be used as a standalone script to generate metrics from saved models
    print("Inverse Problem Comprehensive Metrics Calculator")
    print("This module provides comprehensive evaluation metrics for inverse material identification problems.")
