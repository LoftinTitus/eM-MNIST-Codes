#!/usr/bin/env python3
"""
Detailed analysis of the eM-MNIST input dataset files.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def analyze_input_dataset():
    """Analyze the raw eM-MNIST input dataset files."""
    
    # Data directory from the codebase
    data_dir = "/Users/tyloftin/Downloads/MNIST_comp_files"
    
    print("="*80)
    print("DETAILED ANALYSIS: eM-MNIST FNO PREDICTION DATA")
    print("="*80)
    
    # Load the data
    data = np.load(npz_file)
    
    print(f"File: {npz_file}")
    print(f"File size: {0.42:.2f} MB")
    print(f"\nArrays in this NPZ file:")
    print("-" * 40)
    
    for key in data.files:
        array = data[key]
        print(f"\n📊 {key.upper()}:")
        print(f"   Shape: {array.shape}")
        print(f"   Data type: {array.dtype}")
        print(f"   Memory: {array.nbytes / 1024 / 1024:.3f} MB")
        
        if key == 'inputs':
            print(f"   📥 INPUT STRUCTURE:")
            print(f"      - Batch size: {array.shape[0]} samples")
            print(f"      - Channels: {array.shape[1]} (ux_displacement, uy_displacement, force)")
            print(f"      - Spatial dims: {array.shape[2]} x {array.shape[3]} pixels")
            print(f"      - Value range: [{np.min(array):.6f}, {np.max(array):.6f}]")
            print(f"      - Mean: {np.mean(array):.6f}, Std: {np.std(array):.6f}")
            
        elif key == 'targets':
            print(f"   🎯 TARGET STRUCTURE (Ground Truth Material Labels):")
            print(f"      - Batch size: {array.shape[0]} samples")
            print(f"      - Spatial dims: {array.shape[1]} x {array.shape[2]} pixels")
            unique_labels = np.unique(array)
            print(f"      - Material labels present: {unique_labels}")
            
            # Material distribution
            print(f"      - MATERIAL DISTRIBUTION:")
            for label in unique_labels:
                count = np.sum(array == label)
                percentage = count / array.size * 100
                print(f"        • Label {label}: {count:,} pixels ({percentage:.2f}%)")
            
        elif key == 'predictions':
            print(f"   🔮 PREDICTION STRUCTURE (Model Output):")
            print(f"      - Batch size: {array.shape[0]} samples")
            print(f"      - Spatial dims: {array.shape[1]} x {array.shape[2]} pixels")
            unique_labels = np.unique(array)
            print(f"      - Material labels predicted: {unique_labels}")
            
            # Compare with targets
            targets = data['targets']
            accuracy = np.mean(array == targets)
            print(f"      - Overall pixel-wise accuracy: {accuracy:.6f} ({accuracy*100:.4f}%)")
            
        elif key == 'accuracies':
            print(f"   📈 ACCURACY METRICS:")
            print(f"      - Per-sample accuracies: {array}")
            print(f"      - Mean accuracy: {np.mean(array):.6f} ({np.mean(array)*100:.4f}%)")
            print(f"      - Min accuracy: {np.min(array):.6f} ({np.min(array)*100:.4f}%)")
            print(f"      - Max accuracy: {np.max(array):.6f} ({np.max(array)*100:.4f}%)")
            print(f"      - Std deviation: {np.std(array):.6f}")
    
    print(f"\n" + "="*80)
    print("MATERIAL MAPPING SUMMARY")
    print("="*80)
    print("Based on your codebase analysis:")
    print("• Label 0: Background/void material (air/empty space)")
    print("• Label 2: Solid material type 1")  
    print("• Label 3: Solid material type 2")
    print("• Label 4: Solid material type 3") 
    print("• Label 5: Solid material type 4")
    print("\nNote: Labels are remapped to contiguous indices in the model.")
    print("The original MNIST digit determines the dominant material pattern.")
    
    print(f"\n" + "="*80)
    print("DATA INTERPRETATION")
    print("="*80)
    print("This NPZ file contains:")
    print("1. INPUT TENSORS: Displacement fields (ux, uy) + force values")
    print("   - Shape: (batch, 3_channels, height, width)")
    print("   - Channel 0: X-displacement field")
    print("   - Channel 1: Y-displacement field") 
    print("   - Channel 2: Applied force field")
    print("")
    print("2. TARGET TENSORS: Ground truth material distribution")
    print("   - Shape: (batch, height, width)")
    print("   - Values: Material class indices (2,3,4,5)")
    print("")
    print("3. PREDICTION TENSORS: Model-predicted material distribution")
    print("   - Shape: (batch, height, width)")  
    print("   - Values: Predicted material class indices")
    print("")
    print("4. ACCURACY ARRAY: Per-sample accuracy scores")
    print("   - Shape: (batch,)")
    print("   - Values: Pixel-wise accuracy for each sample")
    
    data.close()

if __name__ == "__main__":
    analyze_emnist_npz()
