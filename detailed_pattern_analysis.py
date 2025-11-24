#!/usr/bin/env python3
"""
Examine specific examples from each label combination to understand the material mapping pattern.
"""

import numpy as np
import os

def examine_label_patterns():
    """Examine one example from each label combination pattern."""
    
    data_dir = "/Users/tyloftin/Downloads/MNIST_comp_files"
    
    # Representative files for each label combination
    examples = {
        '[0, 8]': '000.npz',  # Label 0 + Label 8
        '[3, 9]': '001.npz',  # Label 3 + Label 9  
        '[4, 8]': '002.npz',  # Label 4 + Label 8
        '[7, 9]': '003.npz'   # Label 7 + Label 9
    }
    
    print("="*80)
    print("DETAILED LABEL PATTERN ANALYSIS")
    print("="*80)
    
    for pattern, filename in examples.items():
        file_path = os.path.join(data_dir, filename)
        
        print(f"\n🔍 EXAMINING: {filename} (labels {pattern})")
        print("-" * 50)
        
        try:
            with np.load(file_path) as data:
                # Get all array info
                print(f"📦 Arrays in file: {list(data.keys())}")
                
                if 'label' in data:
                    labels = data['label']
                    print(f"📋 Label array shape: {labels.shape}")
                    print(f"🏷️  Unique labels: {np.unique(labels)}")
                    
                    # Count pixels for each label
                    unique, counts = np.unique(labels, return_counts=True)
                    total_pixels = labels.size
                    
                    print(f"📊 Label distribution:")
                    for label, count in zip(unique, counts):
                        percentage = (count / total_pixels) * 100
                        print(f"   Label {label}: {count:,} pixels ({percentage:.1f}%)")
                
                # Show some key info about the experiment
                if 'instron_force' in data and 'instron_disp' in data:
                    force = data['instron_force']
                    disp = data['instron_disp']
                    print(f"🔧 Experiment info:")
                    print(f"   Time steps: {len(force)}")
                    print(f"   Max force: {np.max(force):.3f}")
                    print(f"   Max displacement: {np.max(disp):.3f}")
                
                if 'DIC_X' in data:
                    coords = data['DIC_X']
                    print(f"📐 Spatial grid: {coords.shape}")
                    
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
    
    print(f"\n" + "="*80)
    print("📋 MATERIAL MAPPING INTERPRETATION")
    print("="*80)
    
    print("""
Based on the label patterns observed:

🏷️  LABEL COMBINATIONS PATTERN:
   • Files have exactly 2 labels each (binary material assignment)
   • Each file represents a digit with specific material properties
   
🎯 POSSIBLE DIGIT-MATERIAL MAPPING:
   • Labels [0, 8]: Could represent digit '0' (background=0, solid=8)
   • Labels [3, 9]: Could represent digit '3' (background=3, solid=9)  
   • Labels [4, 8]: Could represent digit '4' (background=4, solid=8)
   • Labels [7, 9]: Could represent digit '7' (background=7, solid=9)

🔬 ALTERNATIVE INTERPRETATION:
   • Label 0: True background/void regions
   • Labels 3,4,7: Different solid materials for the digit structure
   • Labels 8,9: Surrounding/matrix materials

📊 FREQUENCY ANALYSIS SUGGESTS:
   • Labels 8 & 9 appear in 50% of files each → likely matrix/background materials
   • Labels 0,3,4,7 appear in ~25% each → likely digit/solid materials
   
💡 RECOMMENDED FINAL MAPPING:
   Based on typical eM-MNIST structure and frequency analysis:
   • Label 0: Void/air (true background)
   • Labels 3,4,7: Different solid materials forming digit geometry
   • Labels 8,9: Matrix/surrounding materials (different properties)
""")

if __name__ == "__main__":
    examine_label_patterns()
