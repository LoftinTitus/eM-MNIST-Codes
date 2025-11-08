#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- File paths and column indices (1-based) ---
MIG_FILE = '/Users/tyloftin/Downloads/MNIST MIG - Sheet1.csv'
FIT_FILE = '/Users/tyloftin/Downloads/MNIST_Measuring - Sheet1.csv'
MIG_COL = 2
FIT_COL = 4

def read_col(file, col):
    df = pd.read_csv(file)
    col_name = df.columns[col - 1]
    s = df[col_name].astype(str).str.replace('%', '').str.replace(',', '').astype(float)
    return s.dropna(), col_name

def plot(series, label, color, save=None, xlabel=None):
    plt.figure(figsize=(8,4))
    ax = plt.gca()
    sns.kdeplot(series, color=color, lw=2, ax=ax)
    # get the KDE line data safely
    lines = ax.get_lines()
    if lines:
        x, y = lines[0].get_data()
        ax.fill_between(x, y, color=color, alpha=0.3)  # shaded area

    ax.set_title(label)
    ax.set_xlabel(xlabel or 'Value')
    ax.set_ylabel('Density')

    # Remove top and right spines so only x (bottom) and y (left) axes remain
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    # Ensure bottom and left spines are visible and slightly emphasized
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # Keep ticks on bottom and left only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(which='both', direction='out')

    plt.tight_layout()
    if save: plt.savefig(save, dpi=300)
    plt.show()

mig, mig_col = read_col(MIG_FILE, MIG_COL)
fit, fit_col = read_col(FIT_FILE, FIT_COL)

plot(mig, f"MIG — {mig_col}", '#1f77b4', 'mig_plot.png', xlabel='MIG Value')
plot(fit, f"Fit — {fit_col}", '#E56E94', 'fit_plot.png', xlabel='Fit Percentage')
