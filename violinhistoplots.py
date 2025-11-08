#!/usr/bin/env python3
"""
Violin plot visualization for speckle quality error metrics.
Creates a single violin plot with one violin for each error column.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from pathlib import Path


def create_error_violin_plot(csv_file='Raw Speckle Quality.errors.csv', output_file='error_metrics_violin_plot.png', 
                             custom_labels=None):
    """
    Create a violin plot showing distribution of all error metrics.
    Uses a blue to red gradient color scheme.
    
    Args:
        csv_file: Path to the CSV file containing error metrics
        output_file: Output filename for the plot
        custom_labels: Optional dictionary mapping original column names to display names
                      e.g., {'old_name': 'New Display Name'}
    """
    
    # Read the CSV file
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Get columns 5-10 (0-indexed, so columns 4-9)
    all_columns = df.columns.tolist()
    selected_columns = all_columns[4:10]  # Columns 5-10 (0-indexed: 4-9)
    
    print(f"Selected columns 5-10: {selected_columns}")
    
    # Filter to only numeric columns from the selection
    numeric_columns = [col for col in selected_columns if df[col].dtype in ['int64', 'float64']]
    
    print(f"Found {len(numeric_columns)} error metric columns")
    
    # Prepare data for violin plot - melt the dataframe to long format
    # Only include columns that have non-null values
    error_data = []
    error_labels = []
    
    for col in numeric_columns:
        # Get non-null values for this column
        values = df[col].dropna().values
        if len(values) > 0:  # Only include columns with data
            error_data.extend(values)
            # Use custom label if provided, otherwise use original column name
            display_name = custom_labels.get(col, col) if custom_labels else col
            error_labels.extend([display_name] * len(values))
    
    # Create a new dataframe for plotting
    plot_df = pd.DataFrame({
        'Error_Value': error_data,
        'Error_Metric': error_labels
    })
    
    # Create the figure
    # Ensure Arial font for all text elements (fallbacks provided)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.figure(figsize=(20, 10))
    
    # Custom blue to red gradient colors
    custom_colors = ['#1e3a8a', '#3b82f6', '#93c5fd', '#fca5a5', '#ef4444', '#7f1d1d']  # dark blue to dark red
    ax = sns.violinplot(data=plot_df, x='Error_Metric', y='Error_Value', 
                       palette=custom_colors, inner='quart', linewidth=0)
    
    # Remove violin edges for clean look
    for i, violin in enumerate(ax.collections[::2]):  # Every other collection is a violin body
        violin.set_edgecolor('none')
        violin.set_linewidth(0)
        
        # Special enhancement for Translation y (index 3 in your custom_labels order)
        if i == 3:  # Translation y position
            violin.set_edgecolor('black')
            violin.set_linewidth(3)  # Thicker border
            violin.set_alpha(1.0)    # Fully opaque
            # do NOT call set_linestyle on artists of this type (causes dash errors)
    
    # Add mean markers for better visual appeal
    means = plot_df.groupby('Error_Metric')['Error_Value'].mean()
    unique_metrics = plot_df['Error_Metric'].unique()
    
    
    # Style the quartile lines to be more visible (except for Translation y)
    unique_metrics = plot_df['Error_Metric'].unique()
    translation_y_index = list(unique_metrics).index('Translation y') if 'Translation y' in unique_metrics else -1
    
    for i, line in enumerate(ax.lines):
        # Skip styling the quartile lines for Translation y violin
        # Each violin typically has 3 lines (median + 2 quartile lines)
        violin_index = i // 3  # Approximate which violin this line belongs to
        
        if violin_index == translation_y_index:
            # Make Translation y lines invisible
            line.set_visible(False)
        else:
            line.set_color('black')
            line.set_linewidth(2.5)
            line.set_alpha(0.9)
    

    plt.ylabel('Error %', fontsize=22, fontweight='bold', labelpad=16)
    ax.tick_params(axis='x', labelsize=15)

    ax.tick_params(axis='y', labelsize=15)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # No grid lines
    plt.grid(False)

    # Show only left and bottom axes; remove top and right borders
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    # Thicken remaining spines (left and bottom)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Ensure ticks appear only on left and bottom
    ax.yaxis.set_ticks_position('left')

    # Add minor ticks and make ticks point outside
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Disable top/right ticks
    ax.tick_params(axis='x', which='both', top=False)
    ax.tick_params(axis='y', which='both', right=False)
    
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(left=0.1) 

    # index of Translation y
    unique_metrics = plot_df['Error_Metric'].unique()
    tx_i = list(unique_metrics).index('Translation y')
    tx_vals = plot_df.loc[plot_df['Error_Metric']=='Translation y','Error_Value'].values

    # overlay a thicker violin at same x position (matplotlib)
    parts = ax.violinplot(tx_vals, positions=[tx_i], widths=0.7,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor('#fca5a5')   # same blush red
        body.set_edgecolor('black')     # black border
        body.set_linewidth(3)
        body.set_alpha(1.0)
        body.set_zorder(10)

    
    plt.show()
    
    return plot_df


if __name__ == "__main__":
    print("Creating violin plots for speckle quality error metrics...")
    
    custom_labels = {
        'shear_u1_mean': 'Pure Shear x',
        'shear_u2_mean': 'Pure Shear y',
        'translation_u1_mean': 'Translation x',
        'translation_u2_mean': 'Translation y',
        'uniaxial_u1_mean': 'Uniaxial x',
        'uniaxial_u2_mean': 'Uniaxial y'
    }
    
    plot_df = create_error_violin_plot(custom_labels=custom_labels)

