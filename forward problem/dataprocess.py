import os
import numpy as np
import cv2
from scipy.ndimage import laplace

def check_dic_shapes(data_dir):
    shapes = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(data_dir, fname)
        data = np.load(path)
        shape = data["DIC_disp"].shape[1:3]  # (H, W)
        shapes[shape] = shapes.get(shape, 0) + 1
    return shapes


## counts = check_dic_shapes("/Users/tyloftin/Downloads/MNIST_comp_files")
## print(counts)
## Data is not consistent in shape, meaning I need to pad them to a common size for the FNO harm

def harmonic_interpolation(u, mask, n_iter=300, alpha=0.2):
    ## Physics constrained interpolation of edge effects
    u_filled = u.copy()
    missing = ~mask.astype(bool)
    for _ in range(n_iter):
        lap = laplace(u_filled)
        u_filled[missing] = u_filled[missing] - alpha * lap[missing]
    return u_filled


def interpolate_dic_edges(dic_disp, target_size=56):
 
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

    ### I think this will be better, note to come back to it pls.
def resize_label(label, target_size=56):
    return cv2.resize(label, (target_size, target_size), interpolation=cv2.INTER_NEAREST)


def preprocess(data, target_size=56):

    label_resized = resize_label(data["label"], target_size)

    dic_interp = interpolate_dic_edges(data["DIC_disp"], target_size)

    ux = dic_interp[..., 0]
    uy = dic_interp[..., 1]

    sample = {
        "ux_frames": ux,            
        "uy_frames": uy,            
        "material_mask": label_resized,     
        "bc_disp": data["instron_disp"],    
        "force": data["instron_force"],   
    }
    return sample



'''Sanity check
s = processed[0]
print("ux_frames:", s["ux_frames"].shape)
print("uy_frames:", s["uy_frames"].shape)
print("material_mask:", s["material_mask"].shape)
print("mask:", s["mask"].shape)
print("bc_disp:", s["bc_disp"].shape)
print("force:", s["force"].shape)

print("ux range:", np.min(s["ux_frames"]), "to", np.max(s["ux_frames"]))
print("uy range:", np.min(s["uy_frames"]), "to", np.max(s["uy_frames"]))
print("force range:", np.min(s["force"]), "to", np.max(s["force"]))
print("bc_disp range:", np.min(s["bc_disp"]), "to", np.max(s["bc_disp"]))


import matplotlib.pyplot as plt

data_dir = "/Users/tyloftin/Downloads/MNIST_comp_files"
processed = []

for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith(".npz"):
        continue
    data = np.load(os.path.join(data_dir, fname))
    sample = preprocess(data, target_size=56)
    processed.append(sample)

s = processed[0]
frame = 50  # pick a mid-strain frame

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(s["ux_frames"][frame], cmap="RdBu")
axes[0].set_title("u_x frame 50")
axes[1].imshow(s["uy_frames"][frame], cmap="RdBu")
axes[1].set_title("u_y frame 50")
axes[2].imshow(s["material_mask"], cmap="gray")
axes[2].set_title("Material Mask")
plt.tight_layout()
plt.show() '''


