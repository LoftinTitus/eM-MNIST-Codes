import numpy as np
import torch

def normalize_inverse(sample):
    """Normalize data for inverse problem"""
    # Normalize displacement fields by maximum displacement
    max_disp = max(np.max(np.abs(sample["ux_frames"])), np.max(np.abs(sample["uy_frames"])))
    if max_disp > 0:
        sample["ux_frames"] = sample["ux_frames"] / max_disp
        sample["uy_frames"] = sample["uy_frames"] / max_disp
    
    # Normalize force by maximum force
    max_force = np.max(sample["force"])
    if max_force > 0:
        sample["force"] = sample["force"] / max_force
    
    return sample


def extract_inverse(sample):
    """Extract frames for inverse problem:
    Input: displacement fields + force
    Output: material mask
    """
    frames = []
    material_mask = sample["material_mask"]
    
    for t in range(len(sample["bc_disp"])):
        # Input: displacement fields + force value
        force_val = sample["force"][t]
        force_map = np.ones_like(material_mask) * force_val
        
        # Stack displacement fields and force as input
        input_t = np.stack([
            sample["ux_frames"][t], 
            sample["uy_frames"][t], 
            force_map
        ], axis=-1)
        
        # Output: material mask (same for all time steps)
        output_t = material_mask
        
        frames.append((input_t, output_t, sample["bc_disp"][t]))
    
    return frames


def build_inverse_dataset(processed_samples):
    """Build dataset for inverse problem"""
    X_inputs, Y_outputs, BC_disps = [], [], []
    
    for s in processed_samples:
        pairs = extract_inverse(s)
        for inp, out, bc in pairs:
            X_inputs.append(inp)
            Y_outputs.append(out)
            BC_disps.append(bc)
    
    # Convert to tensors
    X = torch.tensor(np.array(X_inputs), dtype=torch.float32).permute(0, 3, 1, 2)  # [B, 3, H, W]
    Y = torch.tensor(np.array(Y_outputs), dtype=torch.long)  # [B, H, W] - long for classification
    BC = torch.tensor(np.array(BC_disps), dtype=torch.float32).view(-1, 1)
    
    return X, Y, BC


def extract_material_properties(sample):
    """Extract material properties from the sample"""
    
    properties = {}
    
    # Extract unique material labels
    unique_labels = np.unique(sample["material_mask"])
    
    # For each material, extract properties (you might need to adjust this)
    for label in unique_labels:
        if label == 0:  # Background/void
            properties[int(label)] = {
                'young_modulus': 0.0,
                'poisson_ratio': 0.0,
                'density': 0.0
            }
        else:
            properties[int(label)] = {
                'young_modulus': 1.0,  # Normalized values
                'poisson_ratio': 0.3,
                'density': 1.0
            }
    
    return properties


def build_material_property_dataset(processed_samples):
    """Build dataset that also predicts material properties"""
    X_inputs, Y_masks, Y_props = [], [], []
    
    for s in processed_samples:
        pairs = extract_inverse(s)
        material_props = extract_material_properties(s)
        
        for inp, mask, bc in pairs:
            X_inputs.append(inp)
            Y_masks.append(mask)
            
            # Convert material properties to tensor format
            # Create property maps for each material type
            prop_tensor = np.zeros((3, mask.shape[0], mask.shape[1]))  # [E, nu, rho, H, W]
            
            for label, props in material_props.items():
                mask_label = (mask == label)
                prop_tensor[0][mask_label] = props['young_modulus']
                prop_tensor[1][mask_label] = props['poisson_ratio'] 
                prop_tensor[2][mask_label] = props['density']
            
            Y_props.append(prop_tensor)
    
    X = torch.tensor(np.array(X_inputs), dtype=torch.float32).permute(0, 3, 1, 2)
    Y_masks = torch.tensor(np.array(Y_masks), dtype=torch.long)
    Y_props = torch.tensor(np.array(Y_props), dtype=torch.float32)
    
    return X, Y_masks, Y_props
