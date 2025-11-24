#!/usr/bin/env python3
"""
Analysis of the eM-MNIST input dataset files.
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
    print("ANALYZING eM-MNIST INPUT DATASET")
    print("="*80)
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Let me check for alternative locations...")
        
        # Look for common alternative paths
        alternative_paths = [
            "/Users/tyloftin/eM-MNIST-Codes/data",
            "/Users/tyloftin/eM-MNIST-Codes/dataset", 
            "/Users/tyloftin/eM-MNIST-Codes/MNIST_comp_files",
            "/Users/tyloftin/Desktop/MNIST_comp_files",
            "/Users/tyloftin/Documents/MNIST_comp_files"
        ]
        
        found_path = None
        for path in alternative_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path:
            print(f"✅ Found alternative data directory: {found_path}")
            data_dir = found_path
        else:
            print("No data directory found. Let me search for .npz files in the project...")
            search_for_dataset_files()
            return
    
    print(f"📁 Data directory: {data_dir}")
    
    # List all files in the directory
    try:
        files = os.listdir(data_dir)
        print(f"📄 Total files found: {len(files)}")
        
        # Categorize files
        npz_files = [f for f in files if f.endswith('.npz')]
        mat_files = [f for f in files if f.endswith('.mat')]
        dic_files = [f for f in files if f.endswith('.dic')]
        other_files = [f for f in files if not any(f.endswith(ext) for ext in ['.npz', '.mat', '.dic'])]
        
        print(f"\n📊 FILE BREAKDOWN:")
        print(f"   NPZ files: {len(npz_files)}")
        print(f"   MAT files: {len(mat_files)}")  
        print(f"   DIC files: {len(dic_files)}")
        print(f"   Other files: {len(other_files)}")
        
        # Analyze first few files in detail
        if npz_files:
            print(f"\n🔍 ANALYZING NPZ FILES:")
            analyze_npz_files(data_dir, npz_files[:5])  # First 5 files
            
        if dic_files:
            print(f"\n🔍 ANALYZING DIC FILES:")
            analyze_dic_files(data_dir, dic_files[:3])  # First 3 files
            
        if mat_files:
            print(f"\n🔍 ANALYZING MAT FILES:")
            print(f"   Found {len(mat_files)} MATLAB files")
            for i, f in enumerate(mat_files[:3]):
                print(f"   {i+1}. {f}")
        
    except Exception as e:
        print(f"❌ Error reading directory: {e}")

def search_for_dataset_files():
    """Search for dataset files in the project directory."""
    print("\n🔍 SEARCHING FOR DATASET FILES IN PROJECT...")
    
    project_root = "/Users/tyloftin/eM-MNIST-Codes"
    
    # Search for data files
    data_extensions = ['.npz', '.mat', '.dic', '.csv']
    found_files = []
    
    for root, dirs, files in os.walk(project_root):
        # Skip virtual environment and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if any(file.endswith(ext) for ext in data_extensions):
                full_path = os.path.join(root, file)
                found_files.append(full_path)
    
    print(f"📄 Found {len(found_files)} potential dataset files:")
    
    for file_path in found_files[:10]:  # Show first 10
        rel_path = os.path.relpath(file_path, project_root)
        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
        print(f"   📁 {rel_path} ({file_size:.2f} MB)")
    
    if len(found_files) > 10:
        print(f"   ... and {len(found_files) - 10} more files")
    
    # Try to analyze some of these files
    npz_files = [f for f in found_files if f.endswith('.npz')]
    if npz_files:
        print(f"\n🔍 ANALYZING FOUND NPZ FILES:")
        for npz_file in npz_files[:3]:
            analyze_single_npz(npz_file)

def analyze_npz_files(data_dir, npz_files):
    """Analyze NPZ files in detail."""
    
    for i, filename in enumerate(npz_files):
        print(f"\n   📦 FILE {i+1}: {filename}")
        filepath = os.path.join(data_dir, filename)
        analyze_single_npz(filepath)

def analyze_single_npz(filepath):
    """Analyze a single NPZ file."""
    try:
        file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
        print(f"      Size: {file_size:.3f} MB")
        
        data = np.load(filepath)
        print(f"      Arrays: {len(data.files)}")
        print(f"      Array names: {list(data.files)}")
        
        for key in data.files:
            array = data[key]
            print(f"         '{key}': shape={array.shape}, dtype={array.dtype}")
            
            # Show value ranges for smaller arrays
            if array.size < 100000:  # Less than 100k elements
                if np.issubdtype(array.dtype, np.number):
                    print(f"                 range=[{np.min(array):.3f}, {np.max(array):.3f}]")
                    
                    # Check for material labels
                    if 'material' in key.lower() or 'mask' in key.lower() or 'label' in key.lower():
                        unique_vals = np.unique(array)
                        if len(unique_vals) <= 10:
                            print(f"                 unique_values={unique_vals}")
        
        data.close()
        
    except Exception as e:
        print(f"      ❌ Error loading file: {e}")

def analyze_dic_files(data_dir, dic_files):
    """Analyze DIC files."""
    
    for i, filename in enumerate(dic_files):
        print(f"\n   📄 FILE {i+1}: {filename}")
        filepath = os.path.join(data_dir, filename)
        
        try:
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"      Size: {file_size:.1f} KB")
            
            # Try to read as text file
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:5]  # First 5 lines
                print(f"      Lines: {len(lines)} (showing first 5)")
                for j, line in enumerate(lines):
                    print(f"         {j+1}: {line.strip()[:100]}...")
                    
        except Exception as e:
            print(f"      ❌ Error reading file: {e}")

if __name__ == "__main__":
    analyze_input_dataset()
