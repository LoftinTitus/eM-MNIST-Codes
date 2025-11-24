#!/usr/bin/env python3
"""
Comprehensive analysis of all label values in the eM-MNIST input dataset.
This script examines all NPZ files to understand the complete label mapping.
"""

import numpy as np
import os
from collections import Counter, defaultdict
import glob

def analyze_all_labels():
    """Analyze label distributions across all input NPZ files."""
    
    data_dir = "/Users/tyloftin/Downloads/MNIST_comp_files"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Find all NPZ files
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    npz_files.sort()
    
    print("="*80)
    print("COMPREHENSIVE LABEL ANALYSIS - eM-MNIST INPUT DATASET")
    print("="*80)
    print(f"📁 Data directory: {data_dir}")
    print(f"📄 Total NPZ files: {len(npz_files)}")
    
    # Track all unique labels across dataset
    all_labels = set()
    label_combinations = defaultdict(int)
    file_label_info = {}
    
    print("\n🔍 ANALYZING LABELS IN ALL FILES:")
    print("-" * 50)
    
    for i, file_path in enumerate(npz_files):
        filename = os.path.basename(file_path)
        try:
            with np.load(file_path) as data:
                if 'label' in data:
                    labels = data['label']
                    unique_labels = np.unique(labels)
                    all_labels.update(unique_labels)
                    
                    # Track label combinations for each file
                    label_combo = tuple(sorted(unique_labels))
                    label_combinations[label_combo] += 1
                    file_label_info[filename] = unique_labels
                    
                    print(f"   📦 {filename}: labels {unique_labels} (shape: {labels.shape})")
                    
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")
    
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n🏷️  ALL UNIQUE LABELS FOUND: {sorted(all_labels)}")
    print(f"📈 TOTAL NUMBER OF UNIQUE LABELS: {len(all_labels)}")
    
    print(f"\n🔢 LABEL COMBINATIONS (how many files have each combination):")
    for combo, count in sorted(label_combinations.items(), key=lambda x: x[1], reverse=True):
        print(f"   Labels {list(combo)}: {count} files")
    
    # Analyze individual label frequencies
    label_file_count = Counter()
    for labels in file_label_info.values():
        for label in labels:
            label_file_count[label] += 1
    
    print(f"\n📊 INDIVIDUAL LABEL FREQUENCIES:")
    for label in sorted(label_file_count.keys()):
        count = label_file_count[label]
        percentage = (count / len(npz_files)) * 100
        print(f"   Label {label}: appears in {count}/{len(npz_files)} files ({percentage:.1f}%)")
    
    # Identify potential material mapping based on frequency
    print(f"\n🧠 MATERIAL MAPPING ANALYSIS:")
    print("   Based on the frequency and typical eM-MNIST structure:")
    
    # Sort labels by frequency to understand their role
    sorted_labels = sorted(label_file_count.items(), key=lambda x: x[1], reverse=True)
    
    for label, count in sorted_labels:
        percentage = (count / len(npz_files)) * 100
        if percentage > 80:
            role = "🔵 Background/Surrounding material (very common)"
        elif percentage > 50:
            role = "🟢 Primary solid material (common)"
        elif percentage > 20:
            role = "🟡 Secondary solid material (moderate)"
        else:
            role = "🟠 Rare material or special case (uncommon)"
        
        print(f"   Label {label}: {role}")
    
    print(f"\n📋 RECOMMENDED MATERIAL MAPPING:")
    print("   Based on analysis and typical eM-MNIST conventions:")
    print("   • Label 0: Background/void (surrounding material)")
    print("   • Labels 3,4,7,8,9: Different solid materials within digit geometry")
    print("   • The specific material properties would be defined in the training/evaluation code")
    
    return all_labels, label_combinations, file_label_info

if __name__ == "__main__":
    analyze_all_labels()
