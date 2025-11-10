#!/usr/bin/env python3
"""
Force-displacement plotter: just set your CSV filename below and run the script.
Styled to match the MIG/Fit histogram plots (clean axes, modern look).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

csv_path = "/Users/tyloftin/Downloads/026forgraph.csv"  # Example: "mydata.csv" or "/Users/yourname/Desktop/mydata.csv"

# Column indices (0-based):
displacement_col = 0  # Displacement
true_force_col = 1    # True force
cnn_force_col = 2     # CNN force
fno_force_col = 5     # FNO force

# ================================================
csv_path = Path(csv_path)
if not csv_path.exists():
    print(f'CSV not found: {csv_path}')
    sys.exit(1)

df = pd.read_csv(csv_path)
headers = list(df.columns)
ncols = len(headers)

def get_col(idx):
    if idx < 0 or idx >= ncols:
        return None
    return headers[idx]

disp_col = get_col(displacement_col)
true_col = get_col(true_force_col)
cnn_col = get_col(cnn_force_col)
fno_col = get_col(fno_force_col)

if not disp_col:
    print(f"Error: Displacement column index {displacement_col} is out of range. Headers: {headers}")
    sys.exit(1)

x = df[disp_col].values
plt.figure(figsize=(8,4))
ax = plt.gca()

# Plot all as continuous lines with distinct colors
palette = sns.color_palette()
if true_col:
    ax.plot(x, df[true_col].values, label='True Force', color=palette[0], linewidth=2)
if cnn_col:
    ax.plot(x, df[cnn_col].values, label='CNN Prediction', color='#E56E94', linewidth=2)
if fno_col:
    ax.plot(x, df[fno_col].values, label='FNO Prediction', color=palette[1], linewidth=2)

ax.set_xlabel(f"{disp_col} (mm)")
ax.set_ylabel('Force (N)')
ax.legend()

# Remove top and right spines, emphasize left/bottom
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# Ticks only on left/bottom, outward
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.tick_params(which='both', direction='out')

plt.tight_layout()
out = csv_path.with_name(csv_path.stem + '_force_displacement.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved force-displacement plot to {out}')
plt.show()
