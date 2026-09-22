# -*- coding: utf-8 -*-
"""
VANILLA+ PAPER FIGURES GENERATOR
Publication-Ready Figures for Vanilla+ Article

Figures:
1. Model Performance Comparison
2. Training Curves Comparison
3. Hyperparameter Grid Search Heatmap
4. Model Ranking by Test and CV MCC
5. ROC Curves (One-vs-Rest)
6. Ablation Study Results

@author: H.A.R
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# =============================================
# CONFIGURATION
# =============================================

RESULTS_DIR = r"D:\zebfish\new_class\Results\vanilla_plus_results_full3"
BEST_CSV_DIR = os.path.join(RESULTS_DIR, "best_configuration", "csv_files")
HISTORY_DIR = os.path.join(RESULTS_DIR, "best_configuration", "history_files")
BEST_TEST_DIR = os.path.join(RESULTS_DIR, "best_configuration", "test_predictions")

output_dir = r"D:\zebfish\new_class\Results\vanilla_plus_figures1"
png_dir = os.path.join(output_dir, "PNG")
tiff_dir = os.path.join(output_dir, "TIFF")
os.makedirs(png_dir, exist_ok=True)
os.makedirs(tiff_dir, exist_ok=True)

# =============================================
# MODEL CONFIGURATION
# =============================================

MAIN_MODELS = ["LogisticRegression", "VanillaMLP", "DNNBaseline", "VanillaPlus"]

BEST_ABLATIONS = [
    "BatchNorm_instead_of_LayerNorm",
    "Lower_Dropout_0.1",
    "ReLU_instead_of_GELU",
]

MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "VanillaMLP": "Vanilla MLP",
    "DNNBaseline": "DNN Baseline",
    "VanillaPlus": "Vanilla+",
    "BatchNorm_instead_of_LayerNorm": "Vanilla+ (BatchNorm)",
    "Lower_Dropout_0.1": "Vanilla+ (Low Dropout)",
    "ReLU_instead_of_GELU": "Vanilla+ (ReLU)",
}

TEST_MCC_COLOR = "#2E86AB"
CV_MCC_COLOR   = "#F18F01"

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
    png_path = os.path.join(png_dir, f"{filename}.png")
    tiff_path = os.path.join(tiff_dir, f"{filename}.tiff")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(tiff_path, dpi=300, bbox_inches='tight', facecolor='white',
                format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    print(f"  [SAVED] {filename}")

def make_axis_text_bold(ax):
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
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(TICK_FONT_SIZE)

def load_test_array(model_name, array):
    possible_names = []
    if model_name in BEST_ABLATIONS:
        possible_names += [
            f"ablation_{model_name}_test_{array}.npy",
            f"{model_name}_test_{array}.npy",
        ]
    else:
        possible_names += [f"{model_name}_test_{array}.npy"]
    for name in possible_names:
        path = os.path.join(BEST_TEST_DIR, name)
        if os.path.exists(path):
            try:
                return np.load(path, allow_pickle=True)
            except Exception:
                continue
    return None

def load_history_exact(model_name, is_ablation=False):
    if is_ablation:
        candidates = [
            f"ablation_{model_name}_best_final_history.npy",
            f"ablation_{model_name}_final_history.npy",
        ]
    else:
        candidates = [
            f"{model_name}_best_final_history.npy",
            f"{model_name}_final_history.npy",
        ]
    for name in candidates:
        path = os.path.join(HISTORY_DIR, name)
        if os.path.exists(path):
            try:
                hist = np.load(path, allow_pickle=True).item()
                return hist, name
            except Exception:
                continue
    return None, None

def find_history_any(model_name):
    patterns = [
        f"*{model_name}*final_history.npy",
        f"*{model_name}*history.npy",
    ]
    for p in patterns:
        files = sorted(glob(os.path.join(HISTORY_DIR, p)))
        for f in files:
            try:
                hist = np.load(f, allow_pickle=True).item()
                return hist, os.path.basename(f)
            except Exception:
                continue
    return None, None

# =============================================
# FIGURE 1: MODEL PERFORMANCE COMPARISON
# =============================================

print("\n" + "=" * 60)
print("FIGURE 1: MODEL PERFORMANCE COMPARISON")
print("=" * 60)

final_summary_path = os.path.join(BEST_CSV_DIR, "final_comprehensive_summary.csv")

if os.path.exists(final_summary_path):
    df = pd.read_csv(final_summary_path)

    all_models_to_plot = ['Logistic Regression', 'Vanilla MLP', 'DNN Baseline', 'Vanilla+ (Ours)']
    ablation_names_display = ['Ablation_BatchNorm_instead_of_LayerNorm',
                              'Ablation_Lower_Dropout_0.1',
                              'Ablation_ReLU_instead_of_GELU']
    all_models_to_plot.extend(ablation_names_display)

    plot_df = df[df['Model'].isin(all_models_to_plot)]
    available_cols = plot_df.columns.tolist()

    f1_col = 'Test_F1' if 'Test_F1' in available_cols else 'f1' if 'f1' in available_cols else None
    mcc_col = 'Test_MCC' if 'Test_MCC' in available_cols else 'mcc' if 'mcc' in available_cols else None
    acc_col = 'Test_Accuracy' if 'Test_Accuracy' in available_cols else 'accuracy' if 'accuracy' in available_cols else None
    auc_col = 'Test_AUC' if 'Test_AUC' in available_cols else 'auc_roc' if 'auc_roc' in available_cols else None

    metrics_config = [
        (mcc_col, 'Matthews Correlation Coefficient (MCC)'),
        (f1_col, 'F1 Score'),
        (acc_col, 'Accuracy'),
        (auc_col, 'AUC-ROC'),
    ]

    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Model Performance Comparison', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')

    figure1_data = {}

    for i, (metric_col, title) in enumerate(metrics_config):
        ax = axes[i // 2, i % 2]
        if metric_col is None:
            ax.text(0.5, 0.5, f'{title} (Not Available)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
            continue

        data = plot_df[['Model', metric_col]].dropna()
        display_names = []
        for name in data['Model'].values:
            if 'Ablation_BatchNorm' in name:
                display_names.append('Vanilla+ (BatchNorm)')
            elif 'Ablation_Lower_Dropout' in name:
                display_names.append('Vanilla+ (Low Dropout)')
            elif 'Ablation_ReLU' in name:
                display_names.append('Vanilla+ (ReLU)')
            elif 'Vanilla+ (Ours)' in name:
                display_names.append('Vanilla+')
            else:
                display_names.append(name)

        values = data[metric_col].values
        bars = ax.bar(display_names, values, edgecolor='black', linewidth=1.5, alpha=0.85)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')

        ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.set_ylabel('Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        make_axis_text_bold(ax)

        figure1_data[title] = {n: v for n, v in zip(display_names, values)}

    plt.tight_layout()
    save_figure(fig1, "Figure1_Model_Performance_Comparison")
    plt.close(fig1)

    print("\n[Figure 1 Data - Test Set Performance]")
    for title, vals in figure1_data.items():
        print(f"\n  {title}:")
        for name, val in vals.items():
            print(f"    {name:<28} {val:.4f}")

# =============================================
# FIGURE 2: TRAINING CURVES COMPARISON
# =============================================

print("\n" + "=" * 60)
print("FIGURE 2: TRAINING CURVES COMPARISON")
print("=" * 60)

compare_models = [
    ('VanillaPlus', 'Vanilla+', False),
    ('VanillaMLP', 'Vanilla MLP', False),
    ('DNNBaseline', 'DNN Baseline', False),
    ('BatchNorm_instead_of_LayerNorm', 'Vanilla+ (BatchNorm)', True),
    ('Lower_Dropout_0.1', 'Vanilla+ (Low Dropout)', True),
    ('ReLU_instead_of_GELU', 'Vanilla+ (ReLU)', True),
]

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Training Curves Comparison Across Models',
              fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')

curve_configs = [
    ('train_accuracy', 'Training Accuracy', axes[0, 0]),
    ('val_accuracy', 'Validation Accuracy', axes[0, 1]),
    ('train_loss', 'Training Loss', axes[1, 0]),
    ('val_loss', 'Validation Loss', axes[1, 1]),
]

figure2_data = {}
loaded_histories = {}

for model_name, display_name, is_abl in compare_models:
    hist, src = load_history_exact(model_name, is_ablation=is_abl)
    if hist is None:
        hist, src = find_history_any(model_name)
    if hist is not None:
        loaded_histories[display_name] = {'hist': hist, 'file': src, 'is_ablation': is_abl}
        print(f"  [OK] {display_name:<28} loaded from: {src}")
    else:
        print(f"  [MISSING] {display_name:<28} no history file found")

colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(compare_models)))

for metric_name, title, ax in curve_configs:
    lines, labels = [], []
    metric_info = {}

    for (model_name, display_name, is_abl), color in zip(compare_models, colors_cycle):
        if display_name not in loaded_histories:
            continue
        hist = loaded_histories[display_name]['hist']
        if metric_name not in hist:
            continue
        data = hist[metric_name]
        if data is None or len(data) == 0:
            continue

        data = np.asarray(data)
        epochs = np.arange(1, len(data) + 1)
        linestyle = '--' if is_abl else '-'
        linewidth = 2.0 if is_abl else 2.5
        alpha = 0.85 if is_abl else 0.95

        line, = ax.plot(epochs, data, linewidth=linewidth, color=color,
                        linestyle=linestyle, alpha=alpha, label=display_name)
        lines.append(line)
        labels.append(display_name)

        if 'loss' in metric_name:
            metric_info[display_name] = {'epochs': len(data), 'final': float(data[-1]),
                                         'best': float(data.min())}
        else:
            metric_info[display_name] = {'epochs': len(data), 'final': float(data[-1]),
                                         'best': float(data.max())}

    ax.set_xlabel('Epoch', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel(title, fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
    if lines:
        ax.legend(lines, labels, loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    if 'Accuracy' in title:
        ax.set_ylim([0.6, 1.0])
    elif 'Loss' in title:
        ax.set_ylim([0, 0.8])

    make_axis_text_bold(ax)
    figure2_data[title] = metric_info

plt.tight_layout()
save_figure(fig2, "Figure2_Training_Curves_Comparison")
plt.close(fig2)

print("\n[Figure 2 Data - Training Curves]")
for title, info in figure2_data.items():
    print(f"\n  {title}:")
    if not info:
        print("    (no data)")
        continue
    for name, vals in info.items():
        print(f"    {name:<28} epochs={vals['epochs']:<4} "
              f"final={vals['final']:.4f} best={vals['best']:.4f}")

# =============================================
# FIGURE 3: HYPERPARAMETER GRID SEARCH HEATMAP
# =============================================

print("\n" + "=" * 60)
print("FIGURE 3: HYPERPARAMETER GRID SEARCH HEATMAP")
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

    fig3, ax = plt.subplots(figsize=(10, 8))
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
    ax.set_title('Hyperparameter Grid Search Results (CV MCC)',
                 fontsize=TITLE_FONT_SIZE, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CV MCC', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    make_colorbar_text_bold(cbar)

    best_idx = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                               fill=False, edgecolor='gold', linewidth=3))

    make_axis_text_bold(ax)
    plt.tight_layout()
    save_figure(fig3, "Figure3_Grid_Search_Heatmap")
    plt.close(fig3)

    print("\n[Figure 3 Data - Grid Search CV MCC]")
    print(f"  {'LR':<10}", end="")
    for bs in bs_values:
        print(f"BS={bs:<8}", end="")
    print()
    print("  " + "-" * (10 + 12 * len(bs_values)))
    for i, lr in enumerate(lr_values):
        print(f"  {lr:<10.4f}", end="")
        for j, bs in enumerate(bs_values):
            print(f"{heatmap_data[j, i]:<12.4f}", end="")
        print()
    print(f"\n  Best: LR={lr_values[best_idx[1]]}, BS={bs_values[best_idx[0]]}, "
          f"MCC={heatmap_data[best_idx]:.4f}")

# =============================================
# FIGURE 4: MODEL RANKING BY TEST AND CV MCC
# =============================================

print("\n" + "=" * 60)
print("FIGURE 4: MODEL RANKING BY TEST AND CV MCC")
print("=" * 60)

if os.path.exists(final_summary_path):
    df = pd.read_csv(final_summary_path)

    all_models_to_plot = ['Logistic Regression', 'Vanilla MLP', 'DNN Baseline', 'Vanilla+ (Ours)']
    ablation_names_display = ['Ablation_BatchNorm_instead_of_LayerNorm',
                              'Ablation_Lower_Dropout_0.1',
                              'Ablation_ReLU_instead_of_GELU']
    all_models_to_plot.extend(ablation_names_display)
    plot_df = df[df['Model'].isin(all_models_to_plot)]

    display_names = []
    for name in plot_df['Model'].values:
        if 'Ablation_BatchNorm' in name:
            display_names.append('Vanilla+ (BatchNorm)')
        elif 'Ablation_Lower_Dropout' in name:
            display_names.append('Vanilla+ (Low Dropout)')
        elif 'Ablation_ReLU' in name:
            display_names.append('Vanilla+ (ReLU)')
        elif 'Vanilla+ (Ours)' in name:
            display_names.append('Vanilla+')
        else:
            display_names.append(name)

    mcc_col = 'Test_MCC' if 'Test_MCC' in plot_df.columns else 'mcc' if 'mcc' in plot_df.columns else None
    cv_mcc_col = 'CV_Mean_MCC' if 'CV_Mean_MCC' in plot_df.columns else 'cv_mean_mcc' if 'cv_mean_mcc' in plot_df.columns else None

    if mcc_col is None:
        print("  [SKIPPED] No MCC column.")
    else:
        test_mcc = plot_df[mcc_col].values
        cv_mcc = plot_df[cv_mcc_col].values if cv_mcc_col else np.zeros_like(test_mcc)

        # Sort by test MCC descending
        order = np.argsort(test_mcc)[::-1]
        display_names_sorted = [display_names[i] for i in order]
        test_mcc_sorted = test_mcc[order]
        cv_mcc_sorted = cv_mcc[order]

        fig4, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(display_names_sorted))
        width = 0.38

        # UNIFORM COLOR BARS: Blue = Test MCC, Orange = CV MCC
        bars1 = ax.bar(x - width/2, test_mcc_sorted, width,
                       label='Test MCC', color=TEST_MCC_COLOR,
                       edgecolor='black', linewidth=1.5, alpha=0.9)
        bars2 = ax.bar(x + width/2, cv_mcc_sorted, width,
                       label='CV MCC', color=CV_MCC_COLOR,
                       edgecolor='black', linewidth=1.5, alpha=0.9)

        ax.set_xlabel('Model', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylabel('MCC Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_title('Model Ranking by Test and CV MCC',
                     fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(display_names_sorted, rotation=45, ha='right', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0.75, 0.95)

        for bar, val in zip(bars1, test_mcc_sorted):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')

        for bar, val in zip(bars2, cv_mcc_sorted):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')

        make_axis_text_bold(ax)
        plt.tight_layout()
        save_figure(fig4, "Figure4_Model_Ranking_Test_and_CV_MCC")
        plt.close(fig4)

        print("\n[Figure 4 Data - Test and CV MCC Ranking]")
        print(f"  {'Rank':<6} {'Model':<28} {'Test MCC':<12} {'CV MCC':<12}")
        print("  " + "-" * 60)
        for r, (name, tm, cm) in enumerate(zip(display_names_sorted, test_mcc_sorted, cv_mcc_sorted), 1):
            print(f"  {r:<6} {name:<28} {tm:<12.4f} {cm:<12.4f}")

# =============================================
# FIGURE 5: ROC CURVES (One-vs-Rest)
# =============================================

print("\n" + "=" * 60)
print("FIGURE 5: ROC CURVES (ONE-VS-REST)")
print("=" * 60)

NUM_CLASSES = 8
roc_models = ['VanillaPlus', 'VanillaMLP', 'BatchNorm_instead_of_LayerNorm', 'LogisticRegression']

fig5, axes = plt.subplots(2, 2, figsize=(12, 10))
fig5.suptitle('ROC Curves (One-vs-Rest) — Test Set',
              fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')

figure5_data = {}

for idx, model_name in enumerate(roc_models):
    ax = axes[idx // 2, idx % 2]
    probs = load_test_array(model_name, "probabilities")
    labels = load_test_array(model_name, "true_labels")
    display_name = MODEL_LABELS.get(model_name, model_name)

    if probs is None or labels is None:
        ax.set_title(f"{display_name}\n(no data)")
        continue

    aucs = []
    class_aucs = {}
    for c in range(NUM_CLASSES):
        y_bin = (labels == c).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        class_aucs[f'Class {c}'] = roc_auc
        ax.plot(fpr, tpr, color=CLASS_COLORS[c], lw=2, alpha=0.8,
                label=f'Class {c} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    mean_auc = float(np.mean(aucs)) if aucs else 0.0

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

    figure5_data[display_name] = {'mean_auc': mean_auc, 'per_class': class_aucs}

plt.tight_layout()
save_figure(fig5, "Figure5_ROC_Curves")
plt.close(fig5)

print("\n[Figure 5 Data - ROC AUC per Class]")
for model, info in figure5_data.items():
    print(f"\n  {model}: mean AUC = {info['mean_auc']:.4f}")
    for cls, val in info['per_class'].items():
        print(f"    {cls:<10} AUC = {val:.4f}")

# =============================================
# FIGURE 6: ABLATION STUDY RESULTS
# =============================================

print("\n" + "=" * 60)
print("FIGURE 6: ABLATION STUDY RESULTS")
print("=" * 60)

ablation_path = os.path.join(BEST_CSV_DIR, "ablation_summary_best_config.csv")

if os.path.exists(ablation_path):
    abl_df = pd.read_csv(ablation_path)
    abl_df = abl_df.sort_values('Test_MCC', ascending=False)

    fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig6.suptitle('Ablation Study Results', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')

    variants = abl_df['Variant'].values
    test_mcc = abl_df['Test_MCC'].values
    cv_mcc = abl_df['CV_Mean_MCC'].values
    test_f1 = abl_df['Test_F1'].values

    x = np.arange(len(variants))
    width = 0.35

    bars1 = ax1.bar(x - width/2, test_mcc, width, label='Test MCC',
                    color=TEST_MCC_COLOR, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, cv_mcc, width, label='CV MCC',
                    color=CV_MCC_COLOR, edgecolor='black', linewidth=1.5)

    best_idx = int(np.argmax(test_mcc))
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(3)
    bars2[best_idx].set_edgecolor('gold')
    bars2[best_idx].set_linewidth(3)

    ax1.set_xlabel('Ablation Variant', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax1.set_ylabel('MCC Score', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax1.set_title('Test and CV MCC Comparison', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=45, ha='right', fontsize=TICK_FONT_SIZE)
    ax1.legend(loc='lower right')
    ax1.grid(True, axis='y', alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                 f'{h:.4f}', ha='center', va='bottom',
                 fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')

    full_model_mcc = abl_df[abl_df['Variant'] == 'Full_Model_VanillaPlus']['Test_MCC'].values[0]
    change = (test_mcc - full_model_mcc) / full_model_mcc * 100

    colors_bar = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in change]
    bars3 = ax2.bar(x, change, color=colors_bar, edgecolor='black', linewidth=1.5)

    bars3[best_idx].set_edgecolor('gold')
    bars3[best_idx].set_linewidth(3)
    worst_idx = int(np.argmin(change))
    bars3[worst_idx].set_edgecolor('red')
    bars3[worst_idx].set_linewidth(3)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Ablation Variant', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax2.set_ylabel('Change in Test MCC (%)', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
    ax2.set_title('Performance Change vs Full Model', fontsize=TITLE_FONT_SIZE, fontweight='bold')
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
    save_figure(fig6, "Figure6_Ablation_Study")
    plt.close(fig6)

    print("\n[Figure 6 Data - Ablation Study]")
    print(f"  {'Variant':<35} {'Test MCC':<12} {'CV MCC':<12} {'Change':<10}")
    print("  " + "-" * 75)
    for v, tm, cm, ch in zip(variants, test_mcc, cv_mcc, change):
        print(f"  {v:<35} {tm:<12.4f} {cm:<12.4f} {ch:+.2f}%")
    print(f"\n  Full model Test MCC: {full_model_mcc:.4f}")
    print(f"  Best: {variants[best_idx]} (Test MCC = {test_mcc[best_idx]:.4f})")
    print(f"  Worst: {variants[worst_idx]} (Change = {change[worst_idx]:+.2f}%)")

# =============================================
# SUMMARY
# =============================================

print("\n" + "=" * 60)
print("VANILLA+: FIGURE GENERATION COMPLETE")
print("=" * 60)

print(f"\nPNG files: {png_dir}")
print(f"TIFF files: {tiff_dir}")

print("\nGenerated Figures:")
print("  - Figure 1: Model Performance Comparison")
print("  - Figure 2: Training Curves Comparison")
print("  - Figure 3: Hyperparameter Grid Search Heatmap")
print("  - Figure 4: Model Ranking by Test and CV MCC")
print("  - Figure 5: ROC Curves (One-vs-Rest)")
print("  - Figure 6: Ablation Study Results")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)