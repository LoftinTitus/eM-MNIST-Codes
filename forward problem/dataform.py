import numpy as np
import dataload
import torch

def normalize(sample):
    bc = sample["bc_disp"].reshape(-1, 1, 1)
    ux_norm = sample["ux_frames"] / (bc + 1e-8)
    uy_norm = sample["uy_frames"] / (bc + 1e-8)
    f_norm = sample["force"] / np.max(sample["force"])

    sample["ux_frames"] = ux_norm
    sample["uy_frames"] = uy_norm
    sample["force"] = f_norm
    return sample


def extract(sample):
    frames = []
    label = sample["material_mask"]
    for t in range(len(sample["bc_disp"])):
        bc_val = sample["bc_disp"][t]
        bc_map = np.ones_like(label) * bc_val
        input_t = np.stack([label, bc_map], axis=-1) 
        output_t = np.stack([sample["ux_frames"][t], sample["uy_frames"][t]], axis=-1)  
        frames.append((input_t, output_t, sample["force"][t]))
    return frames

def build_dataset(processed_samples):
    X_inputs, Y_outputs, F_forces = [], [], []
    for s in processed_samples:
        pairs = extract(s)
        for inp, out, f in pairs:
            X_inputs.append(inp)
            Y_outputs.append(out)
            F_forces.append(f)
    X = torch.tensor(np.array(X_inputs), dtype=torch.float32).permute(0, 3, 1, 2)
    Y = torch.tensor(np.array(Y_outputs), dtype=torch.float32).permute(0, 3, 1, 2)
    F = torch.tensor(np.array(F_forces), dtype=torch.float32).view(-1, 1)
    return X, Y, F


