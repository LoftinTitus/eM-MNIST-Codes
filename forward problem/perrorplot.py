#!/usr/bin/env python3
"""
Plot predicted displacement vs real and force time-series for a single sample comparing FNO and CNN.

Usage:
    python perrorplot.py --sample 26 --timestep 2 --data-dir /path/to/MNIST_comp_files --output sample26_compare.png

This script assumes the repository's forward problem modules (dataload, dataprocess, dataform, compare_models)
are available in the same folder.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Ensure local package imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnn_model import BasicCNN
from fno_model import FNO2d
import dataload
import dataprocess
import dataform


def prepare_sample(raw_sample, target_size=56, device='cpu'):
    # Build raw_dict as used elsewhere
    raw_dict = {
        'DIC_disp': raw_sample.get('ux_frames')[..., None],
        'label': raw_sample.get('material_mask'),
        'instron_disp': raw_sample.get('bc_disp'),
        'instron_force': raw_sample.get('force')
    }

    ux = raw_sample.get('ux_frames')
    uy = raw_sample.get('uy_frames')
    raw_dict['DIC_disp'] = np.stack([ux, uy], axis=-1)

    processed = dataprocess.preprocess(raw_dict, target_size=target_size)
    processed = dataform.normalize(processed)
    return processed


def predict_for_sample(fno_model, cnn_model, sample_processed, device='cpu'):
    # extract frames -> list of (input_t, output_t, force)
    frames = dataform.extract(sample_processed)
    N = len(frames)
    inputs = np.stack([f[0] for f in frames], axis=0)  # (T, H, W, C)
    targets = np.stack([f[1] for f in frames], axis=0)  # (T, H, W, 2)
    forces = np.array([f[2] for f in frames])          # (T,)

    # convert to tensors in model expected shape (B, C, H, W)
    inputs_t = torch.tensor(inputs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

    # Run CNN
    cnn_disp = None
    cnn_force = None
    cnn_model.eval()
    cnn_model.to(device)
    with torch.no_grad():
        cnn_out = cnn_model(inputs_t)
        if isinstance(cnn_out, (tuple, list)):
            cnn_out_disp, cnn_out_force = cnn_out
        else:
            cnn_out_disp = cnn_out
            cnn_out_force = np.zeros((inputs_t.size(0),), dtype=float)
    cnn_disp = cnn_out_disp.cpu().numpy()
    try:
        cnn_force = cnn_out_force.cpu().numpy().reshape(-1)
    except Exception:
        cnn_force = np.array(cnn_out_force).reshape(-1)

    # Optionally run FNO if provided
    fno_disp = None
    fno_force = None
    if fno_model is not None:
        fno_model.eval()
        fno_model.to(device)
        with torch.no_grad():
            fno_out = fno_model(inputs_t)
            if isinstance(fno_out, (tuple, list)):
                fno_out_disp, fno_out_force = fno_out
            else:
                fno_out_disp = fno_out
                fno_out_force = np.zeros((inputs_t.size(0),), dtype=float)
        fno_disp = fno_out_disp.cpu().numpy()
        try:
            fno_force = fno_out_force.cpu().numpy().reshape(-1)
        except Exception:
            fno_force = np.array(fno_out_force).reshape(-1)

    # move targets to numpy and adjust axes to (T, H, W, 2)
    targets_np = targets.astype(float)

    # convert disp preds to (T, H, W, 2)
    cnn_disp = np.transpose(cnn_disp, (0, 2, 3, 1))
    if fno_disp is not None:
        fno_disp = np.transpose(fno_disp, (0, 2, 3, 1))

    return targets_np, forces, fno_disp, fno_force, cnn_disp, cnn_force


def plot_comparison_with_fno(targets, disp_values, forces_true, cnn_force, cnn_disp, fno_disp=None, fno_force=None, fno_df=None, timestep=None, output=None):
    """Create a single combined figure:
    - Row 1: ux maps [True | CNN]
    - Row 2: uy maps [True | CNN]
    - Row 3: force vs timestep spanning all columns
    """
    T = targets.shape[0]
    if timestep is None:
        timestep = T // 2
    if timestep < 0 or timestep >= T:
        raise ValueError('timestep out of range')

    # determine if fno model predictions available
    has_fno = fno_disp is not None

    if has_fno:
        vmin_ux = min(targets[..., 0].min(), fno_disp[..., 0].min(), cnn_disp[..., 0].min())
        vmax_ux = max(targets[..., 0].max(), fno_disp[..., 0].max(), cnn_disp[..., 0].max())
        vmin_uy = min(targets[..., 1].min(), fno_disp[..., 1].min(), cnn_disp[..., 1].min())
        vmax_uy = max(targets[..., 1].max(), fno_disp[..., 1].max(), cnn_disp[..., 1].max())

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.15)

        ax_ux_true = fig.add_subplot(gs[0, 0])
        ax_ux_fno = fig.add_subplot(gs[0, 1])
        ax_ux_cnn = fig.add_subplot(gs[0, 2])

        ax_uy_true = fig.add_subplot(gs[1, 0])
        ax_uy_fno = fig.add_subplot(gs[1, 1])
        ax_uy_cnn = fig.add_subplot(gs[1, 2])

        ax_force = fig.add_subplot(gs[2, :])

        im0 = ax_ux_true.imshow(targets[timestep, ..., 0], cmap='RdBu', vmin=vmin_ux, vmax=vmax_ux)
        ax_ux_true.set_title('True ux')
        ax_ux_true.axis('off')

        ax_ux_fno.imshow(fno_disp[timestep, ..., 0], cmap='RdBu', vmin=vmin_ux, vmax=vmax_ux)
        ax_ux_fno.set_title('FNO Pred ux')
        ax_ux_fno.axis('off')

        ax_ux_cnn.imshow(cnn_disp[timestep, ..., 0], cmap='RdBu', vmin=vmin_ux, vmax=vmax_ux)
        ax_ux_cnn.set_title('CNN Pred ux')
        ax_ux_cnn.axis('off')

        ax_uy_true.imshow(targets[timestep, ..., 1], cmap='RdBu', vmin=vmin_uy, vmax=vmax_uy)
        ax_uy_true.set_title('True uy')
        ax_uy_true.axis('off')

        ax_uy_fno.imshow(fno_disp[timestep, ..., 1], cmap='RdBu', vmin=vmin_uy, vmax=vmax_uy)
        ax_uy_fno.set_title('FNO Pred uy')
        ax_uy_fno.axis('off')

        ax_uy_cnn.imshow(cnn_disp[timestep, ..., 1], cmap='RdBu', vmin=vmin_uy, vmax=vmax_uy)
        ax_uy_cnn.set_title('CNN Pred uy')
        ax_uy_cnn.axis('off')
    else:
        vmin_ux = min(targets[..., 0].min(), cnn_disp[..., 0].min())
        vmax_ux = max(targets[..., 0].max(), cnn_disp[..., 0].max())
        vmin_uy = min(targets[..., 1].min(), cnn_disp[..., 1].min())
        vmax_uy = max(targets[..., 1].max(), cnn_disp[..., 1].max())

        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.15)

        ax_ux_true = fig.add_subplot(gs[0, 0])
        ax_ux_cnn = fig.add_subplot(gs[0, 1])

        ax_uy_true = fig.add_subplot(gs[1, 0])
        ax_uy_cnn = fig.add_subplot(gs[1, 1])

        ax_force = fig.add_subplot(gs[2, :])

        im0 = ax_ux_true.imshow(targets[timestep, ..., 0], cmap='RdBu', vmin=vmin_ux, vmax=vmax_ux)
        ax_ux_true.set_title('True ux')
        ax_ux_true.axis('off')

        ax_ux_cnn.imshow(cnn_disp[timestep, ..., 0], cmap='RdBu', vmin=vmin_ux, vmax=vmax_ux)
        ax_ux_cnn.set_title('CNN Pred ux')
        ax_ux_cnn.axis('off')

        ax_uy_true.imshow(targets[timestep, ..., 1], cmap='RdBu', vmin=vmin_uy, vmax=vmax_uy)
        ax_uy_true.set_title('True uy')
        ax_uy_true.axis('off')

        ax_uy_cnn.imshow(cnn_disp[timestep, ..., 1], cmap='RdBu', vmin=vmin_uy, vmax=vmax_uy)
        ax_uy_cnn.set_title('CNN Pred uy')
        ax_uy_cnn.axis('off')

    # Force time-series spanning full width
    time = np.arange(len(forces_true))
    # Plot forces. If fno_df is provided and contains displacement/predicted_force, prefer plotting vs displacement.
    if fno_df is not None and {'displacement', 'predicted_force'}.issubset(set(fno_df.columns)):
        ax_force.plot(fno_df['displacement'], fno_df['true_force'], label='True (CSV)', color='green', linewidth=1)
        ax_force.plot(fno_df['displacement'], fno_df['predicted_force'], label='FNO Pred (CSV)', color='blue', linewidth=1)
        if disp_values is not None and len(disp_values) == len(cnn_force):
            ax_force.plot(disp_values, forces_true, label='True (sample)', color='green', linewidth=2, alpha=0.6)
            ax_force.plot(disp_values, cnn_force, label='CNN Pred (sample)', color='red', linestyle=':')
        ax_force.set_xlabel('Displacement')
        ax_force.set_ylabel('Force')
        ax_force.legend()
        ax_force.set_title('Force vs Displacement')
        ax_force.grid(False)
    else:
        ax_force.plot(time, forces_true, label='True Force', color='green', linewidth=2)
        if fno_force is not None:
            ax_force.plot(time, fno_force, label='FNO Predicted Force', color='blue', linestyle='--')
        ax_force.plot(time, cnn_force, label='CNN Predicted Force', color='red', linestyle=':')
        ax_force.set_xlabel('Timestep')
        ax_force.set_ylabel('Force')
        ax_force.legend()
        ax_force.set_title('Force vs Timestep')
        ax_force.grid(False)

    fig.suptitle(f'CNN — Sample comparison at timestep {timestep}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output:
        outp = Path(output)
        combined_name = outp.with_name(outp.stem + '_combined' + outp.suffix)
        fig.savefig(combined_name, dpi=300, bbox_inches='tight')
        print(f'Saved combined plot to {combined_name}')

    plt.show()


def main():
    p = argparse.ArgumentParser(description='Per-sample prediction comparison FNO vs CNN')
    p.add_argument('--data-dir', default='/Users/tyloftin/Downloads/MNIST_comp_files', help='Directory with raw samples')
    p.add_argument('--sample', default='26', help='Sample index (0-based) or ID like 026')
    p.add_argument('--timestep', type=int, default=None, help='Timestep to show (default = middle)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    p.add_argument('--output', default=None, help='Output filename (PNG) base')
    p.add_argument('--output-csv', default=None, help='Output CSV filename to save numeric comparison (skip plotting)')
    p.add_argument('--cnn-checkpoint', default=None, help='Path to CNN checkpoint .pt')
    p.add_argument('--fno-force-csv', default=None, help='Path to FNO force-displacement CSV (optional)')
    p.add_argument('--fno-checkpoint', default=None, help='Path to FNO checkpoint .pt')
    args = p.parse_args()

    DATA_DIR = args.data_dir
    sample_idx = int(str(args.sample).lstrip('0') or '0')
    device = args.device

    # Default output CSV into repo-level exports folder when not provided
    if args.output_csv is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_csv = os.path.join(repo_root, 'exports', f'sample{sample_idx:03d}_metrics.csv')
        args.output_csv = default_csv
        print(f'No --output-csv provided; defaulting to: {args.output_csv}')

    print(f'Loading CNN model on device {device}...')
    # Determine CNN checkpoint: use provided or try common repo filenames
    cnn_checkpoint = args.cnn_checkpoint
    if cnn_checkpoint is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(repo_root, 'checkpoints', 'best_unet_cnn_model.pt'),
            os.path.join(repo_root, 'checkpoints', 'best_simple_cnn_model.pt'),
            os.path.join(repo_root, 'checkpoints', 'best_cnn_model.pt'),
        ]
        for c in candidates:
            if os.path.exists(c):
                cnn_checkpoint = c
                break
        if cnn_checkpoint is None:
            print('No CNN checkpoint provided and no default checkpoint found in repo checkpoints. Use --cnn-checkpoint to specify.')
            return

    if not os.path.exists(cnn_checkpoint):
        print(f'CNN checkpoint not found: {cnn_checkpoint}')
        return

    cnn_model = BasicCNN(in_channels=2, out_channels=2, predict_force=True)
    cnn_state = torch.load(cnn_checkpoint, map_location=device)
    cnn_model.load_state_dict(cnn_state)
    print(f'✓ CNN loaded from {cnn_checkpoint}')

    print('Loading raw samples...')
    raw_samples = dataload.load_dic_samples(DATA_DIR)
    if sample_idx < 0 or sample_idx >= len(raw_samples):
        print(f'Sample index {sample_idx} out of range (0..{len(raw_samples)-1})')
        return

    raw_sample = raw_samples[sample_idx]
    processed = prepare_sample(raw_sample, target_size=56, device=device)

    print('Running model predictions...')
    fno_model = None
    fno_force = None
    fno_disp = None
    # load FNO checkpoint if available
    fno_checkpoint = args.fno_checkpoint if hasattr(args, 'fno_checkpoint') else None
    if fno_checkpoint is None:
        # default to repo checkpoints
        fno_checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'best_model.pt')
    if os.path.exists(fno_checkpoint):
        try:
            fno_model = FNO2d(modes1=12, modes2=12, width=64, in_channels=2, out_channels=2, predict_force=True)
            raw = torch.load(fno_checkpoint, map_location=device)
            # Extract nested state dict if present
            if isinstance(raw, dict):
                if 'model_state_dict' in raw:
                    state = raw['model_state_dict']
                elif 'state_dict' in raw:
                    state = raw['state_dict']
                else:
                    # may be full checkpoint with other keys; try find likely key
                    # fallback: treat dict as state dict if it maps param names to tensors
                    sample_key = next(iter(raw.keys()))
                    if isinstance(raw[sample_key], torch.Tensor):
                        state = raw
                    else:
                        # try common nested key names
                        state = raw.get('model', raw)
            else:
                state = raw

            # strip possible 'module.' prefix from keys
            def _strip_module(sd):
                new = {}
                for k, v in sd.items():
                    nk = k
                    if k.startswith('module.'):
                        nk = k[len('module.'):]
                    new[nk] = v
                return new

            if isinstance(state, dict):
                state = _strip_module(state)

            # load with strict=False to avoid missing/unexpected key errors
            res = fno_model.load_state_dict(state, strict=False)
            # res is a NamedTuple with missing_keys and unexpected_keys
            if getattr(res, 'missing_keys', None):
                print('Warning: missing keys in FNO checkpoint:', res.missing_keys)
            if getattr(res, 'unexpected_keys', None):
                print('Warning: unexpected keys in FNO checkpoint:', res.unexpected_keys)
            print(f'✓ FNO loaded (strict=False) from {fno_checkpoint}')
        except Exception as e:
            print(f'Could not load FNO checkpoint {fno_checkpoint}: {e}')
            fno_model = None

    targets, forces_true, fno_disp, fno_force, cnn_disp, cnn_force = predict_for_sample(fno_model, cnn_model, processed, device=device)

    # If user provided FNO CSV, read it and plot its predicted_force vs displacement
    fno_csv = args.fno_force_csv
    fno_df = None
    if fno_csv:
        if os.path.exists(fno_csv):
            try:
                fno_df = pd.read_csv(fno_csv)
                print(f'Loaded FNO CSV: {fno_csv}')
            except Exception as e:
                print(f'Unable to read FNO CSV: {e}')
                fno_df = None
        else:
            print(f'FNO CSV not found: {fno_csv}')

    # For force plotting, prefer plotting vs displacement if available in raw_sample
    disp_values = raw_sample.get('bc_disp') if 'bc_disp' in raw_sample else None

    # If user requested CSV output, prepare numeric comparison and save
    if args.output_csv:
        out_csv = Path(args.output_csv)
        T = targets.shape[0]
        rows = []
        # compute CNN per-timestep MSEs
        for t in range(T):
            true_t = targets[t]  # H,W,2
            cnn_t = cnn_disp[t]
            mse_ux_cnn = np.mean((cnn_t[..., 0] - true_t[..., 0])**2)
            mse_uy_cnn = np.mean((cnn_t[..., 1] - true_t[..., 1])**2)
            mse_total_cnn = mse_ux_cnn + mse_uy_cnn

            # FNO MSEs if available
            if fno_disp is not None:
                fno_t = fno_disp[t]
                mse_ux_fno = np.mean((fno_t[..., 0] - true_t[..., 0])**2)
                mse_uy_fno = np.mean((fno_t[..., 1] - true_t[..., 1])**2)
                mse_total_fno = mse_ux_fno + mse_uy_fno
            else:
                mse_ux_fno = np.nan
                mse_uy_fno = np.nan
                mse_total_fno = np.nan

            # Forces: true and cnn
            f_true = float(forces_true[t])
            f_cnn = float(cnn_force[t]) if len(cnn_force) > t else np.nan

            # FNO force: prefer model prediction, otherwise try interpolate from CSV if disp_values provided
            f_fno = np.nan
            if fno_force is not None and len(fno_force) > t:
                f_fno = float(fno_force[t])
            elif fno_df is not None and disp_values is not None and 'displacement' in fno_df.columns and 'predicted_force' in fno_df.columns:
                # interpolate FNO predicted_force at sample displacement for this timestep
                try:
                    disp_t = float(disp_values[t])
                    f_fno = float(np.interp(disp_t, fno_df['displacement'].values, fno_df['predicted_force'].values))
                except Exception:
                    f_fno = np.nan

            rows.append({
                'timestep': t,
                'true_force': f_true,
                'cnn_force': f_cnn,
                'fno_force': f_fno,
                'mse_ux_cnn': mse_ux_cnn,
                'mse_uy_cnn': mse_uy_cnn,
                'mse_total_cnn': mse_total_cnn,
                'mse_ux_fno': mse_ux_fno,
                'mse_uy_fno': mse_uy_fno,
                'mse_total_fno': mse_total_fno,
            })

        df_out = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv, index=False)
        print(f'Wrote numeric comparison CSV to {out_csv}')
        return

    # Call plotting routine which will handle optional FNO CSV
    plot_comparison_with_fno(targets, disp_values, forces_true, cnn_force, cnn_disp, fno_disp=fno_disp, fno_force=fno_force, fno_df=fno_df, timestep=args.timestep, output=args.output)


if __name__ == '__main__':
    main()
