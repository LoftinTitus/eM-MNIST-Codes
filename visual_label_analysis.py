#!/usr/bin/env python3
"""
Visual analysis of label spatial distribution to determine digit geometry vs surrounding material.
This will show you exactly which labels form the digit shape and which are the background.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

def visualize_label_patterns():
    """Create visual plots to show which labels form digit geometry vs background."""
    
    data_dir = "/Users/tyloftin/Downloads/MNIST_comp_files"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Find representative files for each label combination
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    examples = {
        '[0, 8]': None,
        '[3, 9]': None,
        '[4, 8]': None,
        '[7, 9]': None
    }
    
    # Find one example of each pattern
    for file_path in npz_files[:10]:  # Check first 10 files
        filename = os.path.basename(file_path)
        try:
            with np.load(file_path) as data:
                if 'label' in data:
                    labels = data['label']
                    unique_labels = sorted(np.unique(labels))
                    pattern = str(unique_labels)
                    
                    if pattern in examples and examples[pattern] is None:
                        examples[pattern] = file_path
        except:
            continue
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('eM-MNIST Label Distribution: Which Labels Form Digit Geometry?', fontsize=16, fontweight='bold')
    
    row = 0
    for pattern, file_path in examples.items():
        if file_path is None:
            continue
            
        filename = os.path.basename(file_path)
        
        try:
            with np.load(file_path) as data:
                labels = data['label']
                unique_labels = np.unique(labels)
                
                # Plot original label distribution
                ax1 = axes[row, 0] if row < 2 else axes[row-2, 2]
                im1 = ax1.imshow(labels, cmap='tab10', vmin=0, vmax=9)
                ax1.set_title(f'{filename}\nLabels {unique_labels}', fontweight='bold')
                ax1.set_xlabel('Original Labels')
                
                # Add colorbar to show which colors correspond to which labels
                cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
                cbar1.set_label('Label Value')
                
                # Create binary visualization showing spatial distribution
                ax2 = axes[row, 1] if row < 2 else axes[row-2, 3]
                
                # Count pixels for each label to determine which is minority (likely digit)
                unique, counts = np.unique(labels, return_counts=True)
                total_pixels = labels.size
                
                label_info = {}
                for label, count in zip(unique, counts):
                    percentage = (count / total_pixels) * 100
                    label_info[label] = {'count': count, 'percentage': percentage}
                
                # Create a binary mask: smaller percentage = likely digit geometry
                binary_labels = labels.copy()
                minority_label = min(label_info.keys(), key=lambda x: label_info[x]['percentage'])
                majority_label = max(label_info.keys(), key=lambda x: label_info[x]['percentage'])
                
                # Convert to binary: 1 for minority (likely digit), 0 for majority (likely background)
                binary_mask = (labels == minority_label).astype(int)
                
                im2 = ax2.imshow(binary_mask, cmap='RdYlBu', vmin=0, vmax=1)
                ax2.set_title(f'Interpretation:\nBlue=Label {majority_label} ({label_info[majority_label]["percentage"]:.1f}%)\nRed=Label {minority_label} ({label_info[minority_label]["percentage"]:.1f}%)', 
                             fontsize=10, fontweight='bold')
                ax2.set_xlabel('Binary: Red=Digit?, Blue=Background?')
                
                # Add text annotation
                ax2.text(0.02, 0.98, f'Red areas: {label_info[minority_label]["percentage"]:.1f}% of pixels', 
                        transform=ax2.transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                        fontweight='bold', color='white')
                ax2.text(0.02, 0.85, f'Blue areas: {label_info[majority_label]["percentage"]:.1f}% of pixels', 
                        transform=ax2.transAxes, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7),
                        fontweight='bold', color='white')
                
                print(f"\n📊 {filename} - Labels {unique_labels}:")
                print(f"   Label {minority_label}: {label_info[minority_label]['count']:,} pixels ({label_info[minority_label]['percentage']:.1f}%) - {'LIKELY DIGIT GEOMETRY' if label_info[minority_label]['percentage'] < 30 else 'LIKELY BACKGROUND'}")
                print(f"   Label {majority_label}: {label_info[majority_label]['count']:,} pixels ({label_info[majority_label]['percentage']:.1f}%) - {'LIKELY BACKGROUND' if label_info[majority_label]['percentage'] > 70 else 'LIKELY DIGIT GEOMETRY'}")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
        
        row += 1
        if row >= 2:  # Only show first 2 examples for clarity
            break
    
    plt.tight_layout()
    plt.savefig('emnist_label_spatial_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "="*80)
    print("🎯 VISUAL ANALYSIS INTERPRETATION")
    print("="*80)
    print("""
Based on the spatial distribution visualization:

🔍 LOOK FOR THESE PATTERNS:
   • RED areas should form digit-like shapes (0, 3, 4, 7)
   • BLUE areas should be the surrounding/background material
   • The minority label (smaller percentage) typically forms the digit geometry
   • The majority label (larger percentage) is usually the matrix/background

📊 PIXEL PERCENTAGE ANALYSIS:
   • Labels with ~15-25% of pixels → Likely digit geometry (0, 3, 4, 7)
   • Labels with ~75-85% of pixels → Likely background/matrix (8, 9)

💡 YOUR HYPOTHESIS CHECK:
   If your hypothesis is correct:
   • Labels 0, 3, 4, 7 should appear as digit-shaped regions (RED in plots)
   • Labels 8, 9 should appear as surrounding material (BLUE in plots)
   • The shapes should resemble actual MNIST digits (0, 3, 4, 7)
""")

def analyze_geometric_patterns():
    """Analyze the geometric patterns to confirm digit shapes."""
    
    data_dir = "/Users/tyloftin/Downloads/MNIST_comp_files"
    
    print(f"\n" + "="*80)
    print("🔢 DIGIT SHAPE ANALYSIS")
    print("="*80)
    
    digit_analysis = {
        0: {'files': [], 'label_pairs': []},
        3: {'files': [], 'label_pairs': []},
        4: {'files': [], 'label_pairs': []}, 
        7: {'files': [], 'label_pairs': []}
    }
    
    # Check if filenames correlate with expected digits
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    for file_path in npz_files[:20]:  # Check first 20 files
        filename = os.path.basename(file_path)
        file_number = int(filename.split('.')[0])
        
        try:
            with np.load(file_path) as data:
                if 'label' in data:
                    labels = data['label'] 
                    unique_labels = sorted(np.unique(labels))
                    
                    # Check if this could correspond to digits 0, 3, 4, 7
                    if len(unique_labels) == 2:
                        minority_label = min(unique_labels, key=lambda x: np.sum(labels == x))
                        majority_label = max(unique_labels, key=lambda x: np.sum(labels == x))
                        
                        # If minority label matches potential digit value
                        if minority_label in [0, 3, 4, 7]:
                            digit_analysis[minority_label]['files'].append(filename)
                            digit_analysis[minority_label]['label_pairs'].append(unique_labels)
                            
        except Exception as e:
            continue
    
    print("📋 POTENTIAL DIGIT-LABEL CORRESPONDENCE:")
    for digit, info in digit_analysis.items():
        if info['files']:
            print(f"   Digit {digit}: Found in files {info['files'][:5]} with label pairs {info['label_pairs'][:5]}")
        else:
            print(f"   Digit {digit}: No clear correspondence found")
    
    return digit_analysis

if __name__ == "__main__":
    print("🔍 Starting visual analysis of eM-MNIST label spatial distribution...")
    visualize_label_patterns()
    analyze_geometric_patterns()
    print(f"\n📸 Visualization saved as: emnist_label_spatial_analysis.png")
    print("👀 Check the generated plot to visually confirm which labels form digit geometry!")
