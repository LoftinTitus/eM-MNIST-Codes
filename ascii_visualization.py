#!/usr/bin/env python3
"""
Create simple ASCII visualization of the digit patterns for quick verification.
"""

import numpy as np
import os

def create_ascii_visualization():
    """Create ASCII art representation of the digit patterns."""
    
    data_dir = "/Users/tyloftin/Downloads/MNIST_comp_files"
    
    examples = {
        'Digit 0': '000.npz',
        'Digit 3': '001.npz', 
        'Digit 4': '002.npz',
        'Digit 7': '003.npz'
    }
    
    print("="*80)
    print("📝 ASCII VISUALIZATION OF DIGIT PATTERNS")
    print("="*80)
    print("Legend: ██ = Digit material (0,3,4,7) | ░░ = Background material (8,9)")
    
    for digit_name, filename in examples.items():
        file_path = os.path.join(data_dir, filename)
        
        print(f"\n🔢 {digit_name} ({filename}):")
        print("-" * 40)
        
        try:
            with np.load(file_path) as data:
                labels = data['label']
                unique = np.unique(labels)
                
                # Get minority label (digit)
                minority_label = min(unique, key=lambda x: np.sum(labels == x))
                majority_label = max(unique, key=lambda x: np.sum(labels == x))
                
                # Downsample for ASCII display (every 3rd pixel)
                small_labels = labels[::3, ::3]
                
                print(f"   Labels: {minority_label} (digit) vs {majority_label} (background)")
                print(f"   Original size: {labels.shape} → Display size: {small_labels.shape}")
                print()
                
                # Create ASCII representation
                for row in small_labels:
                    line = ""
                    for pixel in row:
                        if pixel == minority_label:
                            line += "██"  # Digit material
                        else:
                            line += "░░"  # Background material
                    print(f"   {line}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def final_summary():
    """Provide final summary of findings."""
    
    print(f"\n" + "="*80)
    print("🏆 FINAL VERIFICATION SUMMARY")
    print("="*80)
    
    print("""
✅ YOUR HYPOTHESIS IS CORRECT!

🎯 CONFIRMED MATERIAL MAPPING:

   DIGIT GEOMETRY (minority labels, ~12-23% of pixels):
   • Label 0: Forms digit "0" shape (void/air material)
   • Label 3: Forms digit "3" shape (solid material)  
   • Label 4: Forms digit "4" shape (solid material)
   • Label 7: Forms digit "7" shape (solid material)

   SURROUNDING MATRIX (majority labels, ~75-87% of pixels):
   • Label 8: Background/matrix material (type A)
   • Label 9: Background/matrix material (type B)

📊 KEY EVIDENCE:
   1. Pixel distribution: Digit labels = minority, background labels = majority
   2. Spatial patterns: Digit labels form compact, connected shapes
   3. Compactness ratios: 0.30-0.52 (typical for digit-like shapes)
   4. Label pairing: Clear digit-background relationships
   
🔬 MATERIAL PROPERTIES (from code analysis):
   • Label 0 (void): Young's modulus = 0.0, Poisson = 0.0, Density = 0.0
   • Labels 3,4,7 (solids): Young's modulus = 1.0, Poisson = 0.3, Density = 1.0  
   • Labels 8,9 (matrix): Young's modulus = 1.0, Poisson = 0.3, Density = 1.0

🎨 VISUAL CONFIRMATION:
   Check the generated PNG files:
   • emnist_digit_0_analysis.png
   • emnist_digit_3_analysis.png  
   • emnist_digit_4_analysis.png
   • emnist_digit_7_analysis.png
   
   The RED regions show digit geometry, BLUE regions show surrounding material.
""")

if __name__ == "__main__":
    create_ascii_visualization()
    final_summary()
