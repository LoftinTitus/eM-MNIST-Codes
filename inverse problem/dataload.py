import os
import numpy as np

def load_dic_samples(folder_path):
    """Load DIC samples from .npz files"""
    samples = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(folder_path, fname)
        data = np.load(path)
        
        DIC = data['DIC_disp']  
        label = data['label'] 
        bc_disp = data['instron_disp']  
        force = data['instron_force']   

        sample = {
            "ux_frames": DIC[..., 0],  
            "uy_frames": DIC[..., 1],
            "material_mask": label,
            "bc_disp": bc_disp,
            "force": force,
            "filename": fname,
        }
        samples.append(sample)
    print(f"Loaded {len(samples)} samples from {folder_path}")
    return samples
