# -*- coding: utf-8 -*-
"""
VANILLA+ PAPER FIGURES GENERATOR
Publication-Ready Figures for Vanilla+ Article

Figures:
1. Figure 1: Model Performance Comparison (Including Ablations)
2. Figure 2: Training Curves Comparison (All Models)
   - 2A: Training Accuracy
   - 2B: Validation Accuracy
   - 2C: Training Loss
   - 2D: Validation Loss
3. Figure 3: 10-Fold Cross-Validation Performance
4. Figure 4: Hyperparameter Grid Search Heatmap
5. Figure 5: Model Ranking with Error Bars
6. Figure 6: ROC Curves (One-vs-Rest)
7. Figure 7: Ablation Study Results

@author: H.A.R
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import warnings
warnings.filterwarnings("ignore")

# =============================================
# CONFIGURATION
# =============================================

RESULTS_DIR = r"D:\zebfish\new_class\Results\vanilla_plus_results_full3"
BEST_CSV_DIR = os.path.join(RESULTS_DIR, "best_configuration", "csv_files")
HISTORY_DIR = os.path.join(RESULTS_DIR, "best_configuration", "history_files")
BEST_TEST_DIR = os.path.join(RESULTS_DIR, "best_configuration", "test_predictions")

# Output directories
output_dir = r"D:\zebfish\new_class\Results\vanilla_plus_figures"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(tiff_dir, exist_ok=True)

# =============================================
# MODEL CONFIGURATION
# =============================================

# Main models
MAIN_MODELS = [
    "LogisticRegression",
    "VanillaMLP",
    "DNNBaseline",
    "VanillaPlus",
]

# Best ablation variants (top performers from results)
BEST_ABLATIONS = [
    "BatchNorm_instead_of_LayerNorm",  # Test MCC: 0.9165
    "Lower_Dropout_0.1",               # Test MCC: 0.9140
    "ReLU_instead_of_GELU",            # Test MCC: 0.9116
]

MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "VanillaMLP": "Vanilla MLP",
    "DNNBaseline": "DNN Baseline",
    "VanillaPlus": "Vanilla+ (Ours)",
    "BatchNorm_instead_of_LayerNorm": "BatchNorm Ablation",
    "Lower_Dropout_0.1": "Lower Dropout (0.1)",
    "ReLU_instead_of_GELU": "ReLU Ablation",
}

# Color palette
COLORS = {
    "LogisticRegression": "#808080",  # Gray
    "VanillaMLP": "#1f77b4",          # Blue
    "DNNBaseline": "#ff7f0e",         # Orange
    "VanillaPlus": "#2ca02c",         # Green
    "BatchNorm_instead_of_LayerNorm": "#d62728",  # Red
    "Lower_Dropout_0.1": "#9467bd",   # Purple
    "ReLU_instead_of_GELU": "#8c564b", # Brown
}

# For ROC curves - 8 classes
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 8))

# =============================================
# GLOBAL FONT SETTINGS
# =============================================

BASE_FONT_SIZE = 13
AXIS_LABEL_FONT_SIZE = 15
TITLE_FONT_SIZE = 15
SUPTITLE_FONT_SIZE = 17
LEGEND_FONT_SIZE = 12
TICK_FONT_SIZE = 12
ANNOTATION_FONT_SIZE = 12

plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelsize': AXIS_LABEL_FONT_SIZE,
    'axes.titlesize': TITLE_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.weight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# =============================================
# HELPER FUNCTIONS
# =============================================

def save_figure(fig, filename):
    """Save figure as both PNG and TIFF"""
    png_path = os.path.join(png_dir, f"{filename}.png")
    tiff_path = os.path.join(tiff_dir, f"{filename}.tiff")
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white', 
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"  ✅ Saved: {filename}")

def make_axis_text_bold(ax):
    """Make all axis labels and tick labels bold."""
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax.title.set_fontweight('bold')
    ax.title.set_fontsize(TITLE_FONT_SIZE)
    
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(TICK_FONT_SIZE)
    
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(TICK_FONT_SIZE)

def make_colorbar_text_bold(cbar):
    """Make colorbar label and tick labels bold."""
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(TICK_FONT_SIZE)

def load_history_from_npy(model_name):
    """Load training history from .npy file - try multiple naming conventions"""
    # Try with "ablation_" prefix if model is an ablation
    possible_names = []
    
    if model_name in BEST_ABLATIONS:
        possible_names.append(f"ablation_{model_name}_best_final_history.npy")
        possible_names.append(f"ablation_{model_name}_final_history.npy")
        possible_names.append(f"{model_name}_best_final_history.npy")
        possible_names.append(f"{model_name}_final_history.npy")
    else:
        possible_names.append(f"{model_name}_best_final_history.npy")
        possible_names.append(f"{model_name}_final_history.npy")
    
    for name in possible_names:
        history_path = os.path.join(HISTORY_DIR, name)
        if os.path.exists(history_path):
            try:
                history = np.load(history_path, allow_pickle=True).item()
                return history
            except Exception as e:
                print(f"    ⚠️ Error loading {name}: {e}")
                continue
    
    # If not found, try to find any file with the model name
    pattern = f"*{model_name}*history.npy"
    files = glob(os.path.join(HISTORY_DIR, pattern))
    if files:
        try:
            history = np.load(files[0], allow_pickle=True).item()
            return history
        except:
            pass
    
    return None

def load_test_array(model_name, array):
    """Load test array from test_predictions directory"""
    # Try with "ablation_" prefix if model is an ablation
    possible_names = []
    
    if model_name in BEST_ABLATIONS:
        possible_names.append(f"ablation_{model_name}_test_{array}.npy")
        possible_names.append(f"{model_name}_test_{array}.npy")
    else:
        possible_names.append(f"{model_name}_test_{array}.npy")
    
    for name in possible_names:
        path = os.path.join(BEST_TEST_DIR, name)
        if os.path.exists(path):
            try:
                return np.load(path, allow_pickle=True)
            except:
                continue
    
    return None

def get_model_color(model_name):
    """Get color for a model based on name"""
    if 'Logistic' in model_name:
        return COLORS['LogisticRegression']
    elif 'Vanilla MLP' in model_name or 'VanillaMLP' in model_name:
        return COLORS['VanillaMLP']
    elif 'DNN' in model_name or 'DNNBaseline' in model_name:
        return COLORS['DNNBaseline']
    elif 'Vanilla+' in model_name or 'VanillaPlus' in model_name:
        return COLORS['VanillaPlus']
    elif 'BatchNorm' in model_name:
        return COLORS['BatchNorm_instead_of_LayerNorm']
    elif 'Lower_Dropout' in model_name:
        return COLORS['Lower_Dropout_0.1']
    elif 'ReLU' in model_name:
        return COLORS['ReLU_instead_of_GELU']
    else:
        return '#808080'

# =============================================
# FIGURE 1: MODEL PERFORMANCE COMPARISON (Including Ablations)
# =============================================

print("\n" + "=" * 60)
print("VANILLA+: Generating Figure 1: Model Performance Comparison")
print("=" * 60)

final_summary_path = os.path.join(BEST_CSV_DIR, "final_comprehensive_summary.csv")

if os.path.exists(final_summary_path):
    df = pd.read_csv(final_summary_path)
    
    # Filter models: main models + best ablations
    all_models_to_plot = ['Logistic Regression', 'Vanilla MLP', 'DNN Baseline', 'Vanilla+ (Ours)']
    ablation_names_display = ['Ablation_BatchNorm_instead_of_LayerNorm', 
                              'Ablation_Lower_Dropout_0.1',
                              'Ablation_ReLU_instead_of_GELU']
    all_models_to_plot.extend(ablation_names_display)
    
    plot_df = df[df['Model'].isin(all_models_to_plot)]
    
    # Check available columns
    available_cols = plot_df.columns.tolist()
    
    # Find appropriate columns
    f1_col = 'Test_F1' if 'Test_F1' in available_cols else 'f1' if 'f1' in available_cols else None
    mcc_col = 'Test_MCC' if 'Test_MCC' in available_cols else 'mcc' if 'mcc' in available_cols else None
    acc_col = 'Test_Accuracy' if 'Test_Accuracy' in available_cols else 'accuracy' if 'accuracy' in available_cols else None
    auc_col = 'Test_AUC' if 'Test_AUC' in available_cols else 'auc_roc' if 'auc_roc' in available_cols else None
    
    metrics_config = [
        (mcc_col, 'Matthews Correlation Coefficient (MCC)'),
        (f1_col, 'F1 Score'),
        (acc_col, 'Accuracy'),
        (auc_col, 'AUC-ROC')
    ]
    
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Figure 1. Model Performance Comparison (Including Ablations)', 
                  fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')
    
    for i, (metric_col, title) in enumerate(metrics_config):
        ax = axes[i // 2, i % 2]
        
        if metric_col is None or metric_col not in available_cols:
            ax.text(0.5, 0.5, f'{title} (Not Available)', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
            continue
        
        data = plot_df[['Model', metric_col]].dropna()
        model_names = data['Model'].values
        values = data[metric_col].values
        
        # Get display names
        display_names = []
        for name in model_names:
            if 'Ablation_BatchNorm' in name:
                display_names.append('BatchNorm Ablation')
            elif 'Ablation_Lower_Dropout' in name:
                display_names.append('Lower Dropout (0.1)')
            elif 'Ablation_ReLU' in name:
                display_names.append('ReLU Ablation')
            else:
                display_names.append(name)
        
        bar_colors = [get_model_color(name) for name in display_names]
        bars = ax.bar(display_names, values, color=bar_colors, edgecolor='black', linewidth=1.5)
        
        # Highlight Vanilla+ and best ablation
        for j, name in enumerate(display_names):
            if 'Vanilla+' in name:
                bars[j].set_edgecolor('gold')
                bars[j].set_linewidth(3)
            elif 'BatchNorm' in name and bars[j].get_height() == max(values):
                bars[j].set_edgecolor('gold')
                bars[j].set_linewidth(3)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=ANNOTATION_FONT_SIZE,
                   fontweight='bold')
        
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.set_ylabel('Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        make_axis_text_bold(ax)
    
    plt.tight_layout()
    save_figure(fig1, "Figure1_Model_Performance_Comparison")
    plt.close(fig1)

# =============================================
# FIGURE 2: TRAINING CURVES COMPARISON (All Models) - FIXED
# =============================================

print("\n" + "=" * 60)
print("VANILLA+: Generating Figure 2: Training Curves Comparison")
print("=" * 60)

# Models to compare (main + best ablations)
compare_models = [
    ('VanillaPlus', 'Vanilla+ (Ours)'),
    ('VanillaMLP', 'Vanilla MLP'),
    ('DNNBaseline', 'DNN Baseline'),
    ('BatchNorm_instead_of_LayerNorm', 'BatchNorm Ablation'),
    ('Lower_Dropout_0.1', 'Lower Dropout (0.1)'),
    ('ReLU_instead_of_GELU', 'ReLU Ablation'),
]

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Figure 2. Training Curves Comparison Across Models', 
              fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')

curve_configs = [
    ('train_accuracy', 'Training Accuracy', axes[0, 0]),
    ('val_accuracy', 'Validation Accuracy', axes[0, 1]),
    ('train_loss', 'Training Loss', axes[1, 0]),
    ('val_loss', 'Validation Loss', axes[1, 1]),
]

for metric_name, title, ax in curve_configs:
    # Store lines for legend
    lines = []
    labels = []
    
    for model_name, display_name in compare_models:
        # Load history
        history = load_history_from_npy(model_name)
        
        if history is None:
            print(f"  ⚠️ No history found for {model_name}")
            continue
        
        # Check if metric exists
        if metric_name not in history:
            # Try alternative key names
            alt_key = metric_name.replace('train_', 'training_').replace('val_', 'validation_')
            if alt_key in history:
                metric_name_use = alt_key
            else:
                print(f"  ⚠️ Metric {metric_name} not found for {model_name}")
                continue
        else:
            metric_name_use = metric_name
        
        data = history[metric_name_use]
        if len(data) == 0:
            continue
            
        epochs = np.arange(1, len(data) + 1)
        color = get_model_color(display_name)
        
        # Determine line style
        if 'Ablation' in display_name or model_name in BEST_ABLATIONS:
            linestyle = '--'
            linewidth = 2.0
            alpha = 0.8
        else:
            linestyle = '-'
            linewidth = 2.5
            alpha = 0.9
        
        line, = ax.plot(epochs, data, linewidth=linewidth, color=color, 
                       linestyle=linestyle, alpha=alpha)
        lines.append(line)
        labels.append(display_name)
        
        print(f"  ✅ Loaded {display_name} - {metric_name}: {len(data)} epochs")
    
    ax.set_xlabel('Epoch', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel(title, fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    # Create legend with only the lines that exist
    if lines:
        ax.legend(lines, labels, loc='best', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    # Set y-limits based on metric
    if 'Accuracy' in title:
        ax.set_ylim([0.6, 1.0])
    elif 'Loss' in title:
        ax.set_ylim([0, 0.8])
    
    make_axis_text_bold(ax)

plt.tight_layout()
save_figure(fig2, "Figure2_Training_Curves_Comparison")
plt.close(fig2)

# =============================================
# FIGURE 3: 10-FOLD CV PERFORMANCE
# =============================================

print("\n" + "=" * 60)
print("VANILLA+: Generating Figure 3: 10-Fold CV Performance")
print("=" * 60)

vanilla_fold_path = os.path.join(BEST_CSV_DIR, "VanillaPlus_fold_summary.csv")

if os.path.exists(vanilla_fold_path):
    fold_df = pd.read_csv(vanilla_fold_path)
    
    fig3, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(1, 11)
    
    ax.plot(x, fold_df['mcc'], marker='o', linewidth=2.5, markersize=10, 
            color='#2E86AB', label='MCC')
    ax.plot(x, fold_df['f1'], marker='s', linewidth=2.5, markersize=10, 
            color='#A23B72', label='F1 Score')
    ax.plot(x, fold_df['accuracy'], marker='^', linewidth=2.5, markersize=10, 
            color='#F18F01', label='Accuracy')
    
    for metric, color, label in [('mcc', '#2E86AB', 'MCC'), ('f1', '#A23B72', 'F1'), 
                                  ('accuracy', '#F18F01', 'Accuracy')]:
        if metric in fold_df.columns:
            mean_val = fold_df[metric].mean()
            ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.5, 
                       label=f'{label} Mean: {mean_val:.4f}')
    
    ax.set_xlabel('Fold', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel('Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.set_title('Figure 3. Vanilla+ 10-Fold Cross-Validation Performance', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_ylim(0.6, 1.0)
    
    make_axis_text_bold(ax)
    
    plt.tight_layout()
    save_figure(fig3, "Figure3_10Fold_CV_Performance")
    plt.close(fig3)

# =============================================
# FIGURE 4: GRID SEARCH HEATMAP
# =============================================

print("\n" + "=" * 60)
print("VANILLA+: Generating Figure 4: Hyperparameter Grid Search Heatmap")
print("=" * 60)

grid_search_path = os.path.join(RESULTS_DIR, "csv_files", "full_grid_search_results.csv")

if os.path.exists(grid_search_path):
    grid_df = pd.read_csv(grid_search_path)
    
    lr_values = sorted(grid_df['learning_rate'].unique())
    bs_values = sorted(grid_df['batch_size'].unique())
    
    heatmap_data = np.zeros((len(bs_values), len(lr_values)))
    
    for i, bs in enumerate(bs_values):
        for j, lr in enumerate(lr_values):
            subset = grid_df[(grid_df['batch_size'] == bs) & (grid_df['learning_rate'] == lr)]
            if len(subset) > 0:
                heatmap_data[i, j] = subset['cv_mean_mcc'].values[0]
    
    fig4, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.81, vmax=0.85)
    
    ax.set_xticks(np.arange(len(lr_values)))
    ax.set_yticks(np.arange(len(bs_values)))
    ax.set_xticklabels([f'{lr:.4f}' for lr in lr_values], fontweight='bold')
    ax.set_yticklabels(bs_values, fontweight='bold')
    
    for i in range(len(bs_values)):
        for j in range(len(lr_values)):
            ax.text(j, i, f'{heatmap_data[i, j]:.4f}', ha='center', va='center',
                   color='black', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    
    ax.set_xlabel('Learning Rate', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel('Batch Size', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.set_title('Figure 4. Hyperparameter Grid Search Results (CV MCC)', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CV MCC', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    make_colorbar_text_bold(cbar)
    
    best_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1, 
                               fill=False, edgecolor='gold', linewidth=3))
    
    make_axis_text_bold(ax)
    
    plt.tight_layout()
    save_figure(fig4, "Figure4_Grid_Search_Heatmap")
    plt.close(fig4)

# =============================================
# FIGURE 5: MODEL RANKING WITH ERROR BARS
# =============================================

print("\n" + "=" * 60)
print("VANILLA+: Generating Figure 5: Model Ranking with Error Bars")
print("=" * 60)

if os.path.exists(final_summary_path):
    df = pd.read_csv(final_summary_path)
    
    # Include main models + best ablations
    all_models_to_plot = ['Logistic Regression', 'Vanilla MLP', 'DNN Baseline', 'Vanilla+ (Ours)']
    ablation_names_display = ['Ablation_BatchNorm_instead_of_LayerNorm', 
                              'Ablation_Lower_Dropout_0.1',
                              'Ablation_ReLU_instead_of_GELU']
    all_models_to_plot.extend(ablation_names_display)
    plot_df = df[df['Model'].isin(all_models_to_plot)]
    
    fig5, ax = plt.subplots(figsize=(14, 8))
    
    model_names = plot_df['Model'].values
    display_names = []
    for name in model_names:
        if 'Ablation_BatchNorm' in name:
            display_names.append('BatchNorm Ablation')
        elif 'Ablation_Lower_Dropout' in name:
            display_names.append('Lower Dropout (0.1)')
        elif 'Ablation_ReLU' in name:
            display_names.append('ReLU Ablation')
        else:
            display_names.append(name)
    
    mcc_col = 'Test_MCC' if 'Test_MCC' in plot_df.columns else 'mcc' if 'mcc' in plot_df.columns else None
    cv_mcc_col = 'CV_Mean_MCC' if 'CV_Mean_MCC' in plot_df.columns else 'cv_mean_mcc' if 'cv_mean_mcc' in plot_df.columns else None
    cv_std_col = 'CV_Std_MCC' if 'CV_Std_MCC' in plot_df.columns else 'cv_std_mcc' if 'cv_std_mcc' in plot_df.columns else None
    
    if mcc_col is None:
        print("  ⚠️ No MCC column found. Skipping Figure 5.")
    else:
        test_mcc = plot_df[mcc_col].values
        cv_mcc = plot_df[cv_mcc_col].values if cv_mcc_col else test_mcc
        cv_std = plot_df[cv_std_col].values if cv_std_col else np.zeros(len(test_mcc))
        
        bar_colors = [get_model_color(name) for name in display_names]
        
        x = np.arange(len(display_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, test_mcc, width, label='Test MCC', 
                       color=bar_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        bars2 = ax.bar(x + width/2, cv_mcc, width, yerr=cv_std, 
                       label='CV MCC', color=bar_colors, edgecolor='black', linewidth=1.5,
                       capsize=5, error_kw={'linewidth': 2, 'capsize': 5}, alpha=0.6)
        
        # Highlight best performers
        for j, name in enumerate(display_names):
            if 'Vanilla+' in name:
                bars1[j].set_edgecolor('gold')
                bars1[j].set_linewidth(3)
                bars2[j].set_edgecolor('gold')
                bars2[j].set_linewidth(3)
            elif 'BatchNorm' in name:
                bars1[j].set_edgecolor('gold')
                bars1[j].set_linewidth(3)
                bars2[j].set_edgecolor('gold')
                bars2[j].set_linewidth(3)
        
        ax.set_xlabel('Model', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylabel('MCC Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_title('Figure 5. Model Ranking with Error Bars', 
                    fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
        ax.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0.7, 0.95)
        
        for bar, val in zip(bars1, test_mcc):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.4f}', ha='center', va='bottom', 
                       fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
        
        make_axis_text_bold(ax)
        
        plt.tight_layout()
        save_figure(fig5, "Figure5_Model_Ranking_with_Errors")
        plt.close(fig5)

# =============================================
# FIGURE 6: ROC CURVES (One-vs-Rest)
# =============================================

print("\n" + "=" * 60)
print("VANILLA+: Generating Figure 6: ROC Curves (One-vs-Rest)")
print("=" * 60)

NUM_CLASSES = 8

# Select top models for ROC curves
roc_models = ['VanillaPlus', 'VanillaMLP', 'BatchNorm_instead_of_LayerNorm', 'LogisticRegression']

fig6, axes = plt.subplots(2, 2, figsize=(12, 10))
fig6.suptitle('Figure 6. ROC Curves (One-vs-Rest) - Test Set', 
              fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')

for idx, model_name in enumerate(roc_models):
    ax = axes[idx // 2, idx % 2]
    
    # Load test data
    probs = load_test_array(model_name, "probabilities")
    labels = load_test_array(model_name, "true_labels")
    display_name = MODEL_LABELS.get(model_name, model_name)
    
    if probs is None or labels is None:
        ax.set_title(f"{display_name}\n(no data)")
        continue
    
    # Compute ROC curve and AUC for each class
    aucs = []
    for c in range(NUM_CLASSES):
        y_bin = (labels == c).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=CLASS_COLORS[c], lw=2, alpha=0.8,
                label=f'Class {c} (AUC={roc_auc:.3f})')
    
    # Random classifier line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    
    # Calculate mean AUC
    mean_auc = np.mean(aucs) if aucs else 0.0
    
    # Add title and labels
    ax.set_title(f'{display_name}\nMean AUC = {mean_auc:.3f}', 
                 fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    if idx % 2 == 0:
        ax.set_ylabel('True Positive Rate', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    make_axis_text_bold(ax)

plt.tight_layout()
save_figure(fig6, "Figure6_ROC_Curves")
plt.close(fig6)

# =============================================
# FIGURE 7: ABLATION STUDY RESULTS
# =============================================

print("\n" + "=" * 60)
print("VANILLA+: Generating Figure 7: Ablation Study Results")
print("=" * 60)

ablation_path = os.path.join(BEST_CSV_DIR, "ablation_summary_best_config.csv")

if os.path.exists(ablation_path):
    abl_df = pd.read_csv(ablation_path)
    abl_df = abl_df.sort_values('Test_MCC', ascending=False)
    
    fig7, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig7.suptitle('Figure 7. Ablation Study Results', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')
    
    variants = abl_df['Variant'].values
    test_mcc = abl_df['Test_MCC'].values
    cv_mcc = abl_df['CV_Mean_MCC'].values
    test_f1 = abl_df['Test_F1'].values
    
    x = np.arange(len(variants))
    width = 0.35
    
    # Figure 7A: Test vs CV MCC
    bars1 = ax1.bar(x - width/2, test_mcc, width, label='Test MCC', 
                    color='#2E86AB', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, cv_mcc, width, label='CV MCC', 
                    color='#A23B72', edgecolor='black', linewidth=1.5)
    
    best_idx = np.argmax(test_mcc)
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(3)
    bars2[best_idx].set_edgecolor('gold')
    bars2[best_idx].set_linewidth(3)
    
    ax1.set_xlabel('Ablation Variant', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax1.set_ylabel('MCC Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax1.set_title('Figure 7A. Test and CV MCC Comparison', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
    ax1.legend(loc='lower right')
    ax1.grid(True, axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=ANNOTATION_FONT_SIZE,
                fontweight='bold')
    
    # Figure 7B: Performance Change
    full_model_mcc = abl_df[abl_df['Variant'] == 'Full_Model_VanillaPlus']['Test_MCC'].values[0]
    change = (test_mcc - full_model_mcc) / full_model_mcc * 100
    
    colors_bar = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in change]
    bars3 = ax2.bar(x, change, color=colors_bar, edgecolor='black', linewidth=1.5)
    
    bars3[best_idx].set_edgecolor('gold')
    bars3[best_idx].set_linewidth(3)
    worst_idx = np.argmin(change)
    bars3[worst_idx].set_edgecolor('red')
    bars3[worst_idx].set_linewidth(3)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Ablation Variant', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax2.set_ylabel('Change in Test MCC (%)', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax2.set_title('Figure 7B. Performance Change vs Full Model', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars3, change):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:+.2f}%', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    
    make_axis_text_bold(ax1)
    make_axis_text_bold(ax2)
    
    plt.tight_layout()
    save_figure(fig7, "Figure7_Ablation_Study")
    plt.close(fig7)

# =============================================
# SUMMARY
# =============================================

print("\n" + "=" * 60)
print("VANILLA+: FIGURE GENERATION COMPLETE!")
print("=" * 60)

print(f"\n📁 PNG files saved to: {png_dir}")
print(f"📁 TIFF files saved to: {tiff_dir}")

print("\nGenerated Figures:")
print("  - Figure 1: Model Performance Comparison (Including Ablations)")
print("  - Figure 2: Training Curves Comparison (4 panels)")
print("  - Figure 3: 10-Fold Cross-Validation Performance")
print("  - Figure 4: Hyperparameter Grid Search Heatmap")
print("  - Figure 5: Model Ranking with Error Bars")
print("  - Figure 6: ROC Curves (One-vs-Rest)")
print("  - Figure 7: Ablation Study Results")

print("\n" + "=" * 60)
print("✅ FIGURE GENERATION COMPLETE!")
print("=" * 60)