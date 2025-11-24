#!/usr/bin/env python3
"""
Script to examine the shape and contents of NPZ files in the eM-MNIST dataset.
"""

import numpy as np
import os
from pathlib import Path

def examine_npz_file(npz_path):
    """Examine the contents and shapes of an NPZ file."""
    print(f"\n{'='*60}")
    print(f"EXAMINING: {npz_path}")
    print(f"{'='*60}")
    
    try:
        # Load the NPZ file
        data = np.load(npz_path)
        
        print(f"File size: {os.path.getsize(npz_path) / 1024 / 1024:.2f} MB")
        print(f"Number of arrays: {len(data.files)}")
        print(f"Array names: {list(data.files)}")
        print()
        
        # Examine each array
        for key in data.files:
            array = data[key]
            print(f"Array '{key}':")
            print(f"  Shape: {array.shape}")
            print(f"  Dtype: {array.dtype}")
            print(f"  Size: {array.size:,} elements")
            print(f"  Memory: {array.nbytes / 1024 / 1024:.2f} MB")
            
            if array.size > 0:
                print(f"  Min: {np.min(array):.6f}")
                print(f"  Max: {np.max(array):.6f}")
                print(f"  Mean: {np.mean(array):.6f}")
                print(f"  Std: {np.std(array):.6f}")
                
                # Show unique values for small arrays or integer arrays
                if array.size < 1000 or np.issubdtype(array.dtype, np.integer):
                    unique_vals = np.unique(array)
                    if len(unique_vals) <= 20:
                        print(f"  Unique values: {unique_vals}")
                    else:
                        print(f"  Unique values: {len(unique_vals)} unique values")
                        print(f"  Sample values: {unique_vals[:10]}...")
                
                # For material masks, show class distribution
                if 'material' in key.lower() or 'mask' in key.lower() or 'target' in key.lower():
                    unique, counts = np.unique(array, return_counts=True)
                    print(f"  Class distribution:")
                    for val, count in zip(unique, counts):
                        percentage = count / array.size * 100
                        print(f"    Label {val}: {count:,} pixels ({percentage:.2f}%)")
            
            print()
        
        data.close()
        
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")

def find_and_examine_npz_files():
    """Find and examine all NPZ files in the workspace."""
    print("EXAMINING NPZ FILES IN eM-MNIST DATASET")
    print("="*60)
    
    # Current directory
    current_dir = Path("/Users/tyloftin/eM-MNIST-Codes")
    
    # Find all NPZ files
    npz_files = list(current_dir.glob("**/*.npz"))
    
    if not npz_files:
        print("No NPZ files found in the workspace.")
        return
    
    print(f"Found {len(npz_files)} NPZ file(s):")
    for npz_file in npz_files:
        print(f"  - {npz_file}")
    
    # Examine each NPZ file
    for npz_file in npz_files:
        examine_npz_file(npz_file)

if __name__ == "__main__":
    find_and_examine_npz_files()
