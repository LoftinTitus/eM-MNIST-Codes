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

def plot(series, label, color, save=None):
    plt.figure(figsize=(8,4))
    kde = sns.kdeplot(series, color=color, lw=2)
    x, y = kde.get_lines()[0].get_data()
    plt.fill_between(x, y, color=color, alpha=0.3)  # shaded area
    plt.title(label)
    plt.xlabel('Value'); plt.ylabel('Density'); plt.tight_layout()
    if save: plt.savefig(save, dpi=300)
    plt.show()

mig, mig_col = read_col(MIG_FILE, MIG_COL)
fit, fit_col = read_col(FIT_FILE, FIT_COL)

plot(mig, f"MIG — {mig_col}", '#1f77b4', 'mig_plot.png')
plot(fit, f"FIT — {fit_col}", '#E56E94', 'fit_plot.png')
