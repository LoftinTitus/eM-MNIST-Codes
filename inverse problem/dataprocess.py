import os
import numpy as np
import cv2
from scipy.ndimage import laplace

def check_dic_shapes(data_dir):
    """Check the shapes of DIC data"""
    shapes = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(data_dir, fname)
        data = np.load(path)
        shape = data["DIC_disp"].shape[1:3]  # (H, W)
        shapes[shape] = shapes.get(shape, 0) + 1
    return shapes


def harmonic_interpolation(u, mask, n_iter=300, alpha=0.2):
    """Physics constrained interpolation of edge effects"""
    u_filled = u.copy()
    missing = ~mask.astype(bool)
    for _ in range(n_iter):
        lap = laplace(u_filled)
        u_filled[missing] = u_filled[missing] - alpha * lap[missing]
    return u_filled


def interpolate_dic_edges(dic_disp, target_size=56):
    """Interpolate DIC displacement fields to target size"""
    T, H, W, _ = dic_disp.shape

    mask = np.ones((H, W), dtype=bool)

    pad_h = target_size - H
    pad_w = target_size - W
    dic_padded = np.pad(dic_disp, ((0,0),(0,pad_h),(0,pad_w),(0,0)), mode='edge')
    mask_padded = np.pad(mask, ((0,pad_h),(0,pad_w)), mode='constant', constant_values=True)

    dic_interp = np.zeros_like(dic_padded)
    for t in range(T):
        for c in range(2):  # u_x and u_y
            dic_interp[t, :, :, c] = harmonic_interpolation(dic_padded[t, :, :, c], mask_padded)
    return dic_interp


def resize_with_interpolation(array, target_size):
    """Resize array to target size using interpolation"""
    if len(array.shape) == 2:
        return cv2.resize(array, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    elif len(array.shape) == 3:
        resized = []
        for i in range(array.shape[0]):
            resized.append(cv2.resize(array[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR))
        return np.array(resized)
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")


def preprocess(sample, target_size=56):
    """Preprocess a single sample"""
    # Resize displacement fields
    ux_resized = resize_with_interpolation(sample["ux_frames"], target_size)
    uy_resized = resize_with_interpolation(sample["uy_frames"], target_size)
    
    # Resize material mask
    material_mask_resized = cv2.resize(
        sample["material_mask"].astype(np.float32), 
        (target_size, target_size), 
        interpolation=cv2.INTER_NEAREST
    ).astype(int)
    
    return {
        "ux_frames": ux_resized,
        "uy_frames": uy_resized,
        "material_mask": material_mask_resized,
        "bc_disp": sample["bc_disp"],
        "force": sample["force"],
        "filename": sample.get("filename", "unknown")
    }
