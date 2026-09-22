# -*- coding: utf-8 -*-
"""
Statistical Analysis for VANILLA+ Fish Enzyme Classification
Using FOLD-LEVEL history files for proper statistical tests
8-Class Classification (0 = Non-enzyme, 1-7 = EC Classes)

UPDATED: Analyzes ALL metrics (Accuracy, Precision, Recall, F1, MCC, AUC)
UPDATED: Using vanilla_plus_results_full3 results
UPDATED: Includes ALL 12 models (4 main + 8 ablations) uniformly in all tests
UPDATED: Wilcoxon pairwise comparisons across all 12 models with Bonferroni correction
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon
from itertools import combinations
from glob import glob
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("STATISTICAL ANALYSIS FOR VANILLA+ (WITH FOLD-LEVEL DATA)")
print("8-Class Classification | Species-Aware Split (370 Train, 19 Test Species)")
print("All 12 Models — Uniform Testing")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = r"D:\zebfish\new_class\Results\vanilla_plus_results_full3"
HISTORY_DIR = os.path.join(RESULTS_DIR, "best_configuration", "history_files")
CSV_DIR = os.path.join(RESULTS_DIR, "best_configuration", "csv_files")
RANDOM_STATE = 42

# Main models
MAIN_MODELS = ['VanillaPlus', 'VanillaMLP', 'DNNBaseline', 'LogisticRegression']

# Ablation models
ABLATION_MODELS = [
    'BatchNorm_instead_of_LayerNorm',
    'Lower_Dropout_0.1',
    'ReLU_instead_of_GELU',
    'Full_Model_VanillaPlus',
    'w_o_Residual',
    'w_o_LayerNorm',
    'w_o_Dropout',
    'Higher_Dropout_0.5'
]

# All 12 models
ALL_MODELS = MAIN_MODELS + ABLATION_MODELS

# Metrics to analyze
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']

# Display names
MODEL_DISPLAY_NAMES = {
    'VanillaPlus': 'Vanilla+ (Ours)',
    'VanillaMLP': 'Vanilla MLP',
    'DNNBaseline': 'DNN Baseline',
    'LogisticRegression': 'Logistic Regression',
    'BatchNorm_instead_of_LayerNorm': 'BatchNorm Ablation',
    'Lower_Dropout_0.1': 'Lower Dropout (0.1)',
    'ReLU_instead_of_GELU': 'ReLU Ablation',
    'Full_Model_VanillaPlus': 'Full Model (Ablation)',
    'w_o_Residual': 'w/o Residual',
    'w_o_LayerNorm': 'w/o LayerNorm',
    'w_o_Dropout': 'w/o Dropout',
    'Higher_Dropout_0.5': 'Higher Dropout (0.5)'
}

# ============================================================================
# LOAD FOLD-LEVEL DATA FOR ALL METRICS
# ============================================================================

print("\n" + "=" * 80)
print("LOADING FOLD-LEVEL DATA FOR ALL METRICS")
print("=" * 80)

fold_data = {}

def load_fold_history(model_name, fold, is_ablation=False):
    """Load fold history for a model."""
    possible_names = []
    if is_ablation:
        possible_names += [
            f"ablation_{model_name}_best_fold{fold}_history.npy",
            f"ablation_{model_name}_fold{fold}_history.npy",
            f"{model_name}_best_fold{fold}_history.npy",
            f"{model_name}_fold{fold}_history.npy",
        ]
    else:
        possible_names += [
            f"{model_name}_best_fold{fold}_history.npy",
            f"{model_name}_fold{fold}_history.npy",
        ]
    for name in possible_names:
        path = os.path.join(HISTORY_DIR, name)
        if os.path.exists(path):
            try:
                return np.load(path, allow_pickle=True).item()
            except Exception:
                continue
    return None

def load_model_history(model_name, is_ablation=False):
    """Load final history (fallback)."""
    possible_names = []
    if is_ablation:
        possible_names += [
            f"ablation_{model_name}_best_final_history.npy",
            f"ablation_{model_name}_final_history.npy",
        ]
    else:
        possible_names += [
            f"{model_name}_best_final_history.npy",
            f"{model_name}_final_history.npy",
        ]
    for name in possible_names:
        path = os.path.join(HISTORY_DIR, name)
        if os.path.exists(path):
            try:
                return np.load(path, allow_pickle=True).item()
            except Exception:
                continue
    return None

def extract_metrics_from_history(history):
    """Extract all metrics from history dict."""
    metrics = {}
    for metric in METRICS:
        val_key = f'val_{metric}'
        if val_key in history:
            val = history[val_key]
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                metrics[metric] = float(max(val))
            else:
                metrics[metric] = float(val)
        elif metric in history:
            val = history[metric]
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                metrics[metric] = float(max(val))
            else:
                metrics[metric] = float(val)
        else:
            metrics[metric] = 0.0
    return metrics

# Load data for all 12 models
for model_name in ALL_MODELS:
    is_ablation = model_name in ABLATION_MODELS
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    print(f"\n  Loading {display_name}...")

    model_folds = {metric: [] for metric in METRICS}
    found_folds = 0

    for fold in range(1, 11):
        history = load_fold_history(model_name, fold, is_ablation)
        if history:
            metrics = extract_metrics_from_history(history)
            for metric in METRICS:
                model_folds[metric].append(metrics[metric])
            found_folds += 1
            print(f"    Fold {found_folds}: MCC = {metrics['mcc']:.4f}")

    # Fallback: final history repeated
    if found_folds == 0:
        history = load_model_history(model_name, is_ablation)
        if history:
            metrics = extract_metrics_from_history(history)
            for metric in METRICS:
                model_folds[metric] = [metrics[metric]] * 10
            found_folds = 10
            print(f"    Using final history (10 folds with same value)")

    if found_folds > 0:
        for metric in METRICS:
            if len(model_folds[metric]) < 10:
                last_val = model_folds[metric][-1] if model_folds[metric] else 0
                model_folds[metric].extend([last_val] * (10 - len(model_folds[metric])))
            model_folds[metric] = np.array(model_folds[metric])

        fold_data[model_name] = {metric: model_folds[metric] for metric in METRICS}

        print(f"  ✅ Loaded {found_folds} folds for {display_name}")
        for metric in METRICS:
            mean_val = np.mean(model_folds[metric])
            std_val = np.std(model_folds[metric])
            print(f"     {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    else:
        print(f"  ❌ No data found for {display_name}")

# ============================================================================
# STATISTICAL TESTS FOR ALL METRICS — ALL 12 MODELS UNIFORMLY
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL TESTS FOR ALL METRICS (All 12 Models Uniformly)")
print("=" * 80)

all_friedman_results = {}
all_pairwise_results = {}
all_mannwhitney_results = {}

for metric in METRICS:
    print(f"\n{'='*80}")
    print(f"  METRIC: {metric.upper()}")
    print(f"{'='*80}")

    metric_data = {}
    for model_name in ALL_MODELS:
        if model_name in fold_data and metric in fold_data[model_name]:
            metric_data[MODEL_DISPLAY_NAMES.get(model_name, model_name)] = fold_data[model_name][metric]

    if len(metric_data) < 3:
        print(f"  ⚠️ Insufficient data ({len(metric_data)} models)")
        continue

    model_names = list(metric_data.keys())
    min_len = min(len(metric_data[m]) for m in model_names)

    # ---- Friedman test on all 12 models ----
    friedman_stat, friedman_p = friedmanchisquare(
        *[metric_data[m][:min_len] for m in model_names]
    )
    print(f"  Friedman (all {len(model_names)} models): "
          f"χ² = {friedman_stat:.4f}, p = {friedman_p:.6f}")
    print(f"  {'✅ Significant' if friedman_p < 0.05 else '❌ Not significant'}")

    all_friedman_results[metric] = {'stat': friedman_stat, 'p': friedman_p}

    # ---- Wilcoxon pairwise on all 12 models ----
    n_pairs = len(model_names) * (len(model_names) - 1) // 2
    alpha_corrected = 0.05 / n_pairs
    print(f"\n  Pairwise Wilcoxon: {n_pairs} comparisons, "
          f"Bonferroni α = {alpha_corrected:.6f}")

    pairwise_results = []
    for m1, m2 in combinations(model_names, 2):
        try:
            stat, p_val = wilcoxon(metric_data[m1][:min_len], metric_data[m2][:min_len])
            pairwise_results.append({
                'model_1':     m1,
                'model_2':     m2,
                'p_value':     p_val,
                'significant': p_val < alpha_corrected,
            })
        except Exception:
            pass

    sig_count = sum(1 for r in pairwise_results if r['significant'])
    print(f"  Significant pairs (Bonferroni): {sig_count}/{n_pairs}")

    for r in pairwise_results:
        if r['significant']:
            print(f"    ✅ {r['model_1']} vs {r['model_2']}: p = {r['p_value']:.6f}")

    all_pairwise_results[metric] = pairwise_results

    # ---- Mann-Whitney: Vanilla+ vs all 11 ----
    if 'Vanilla+ (Ours)' in metric_data:
        vp_scores = metric_data['Vanilla+ (Ours)']
        print(f"\n  Mann-Whitney U (Vanilla+ vs all 11 models):")
        all_mannwhitney_results[metric] = {}
        for m in model_names:
            if m == 'Vanilla+ (Ours)':
                continue
            try:
                stat, p_val = mannwhitneyu(vp_scores[:min_len], metric_data[m][:min_len])
                sig = '✅' if p_val < alpha_corrected else ''
                print(f"    Vanilla+ vs {m}: p = {p_val:.6f} {sig}")
                all_mannwhitney_results[metric][m] = p_val
            except Exception:
                pass

# ============================================================================
# SUMMARY TABLES FOR PAPER
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY TABLES FOR PAPER")
print("=" * 80)

# ----------------------------------------------------------------------------
# Table 1: Friedman Test Results
# ----------------------------------------------------------------------------
print("\n📊 Table 1: Friedman Test Results (All 12 Models):")
print("-" * 70)
print(f"{'Metric':<12} {'χ²':>10} {'p-value':>12} {'Significant':>12}")
print("-" * 70)
for metric in METRICS:
    if metric in all_friedman_results:
        chi2 = all_friedman_results[metric]['stat']
        p_val = all_friedman_results[metric]['p']
        sig = '✅' if p_val < 0.05 else '❌'
        print(f"{metric.upper():<12} {chi2:>10.4f} {p_val:>12.6f} {sig:>12}")
print("-" * 70)

# ----------------------------------------------------------------------------
# Table 2: Model Ranking by MCC
# ----------------------------------------------------------------------------
print("\n📊 Table 2: Model Ranking by MCC (All 12 Models):")
print("-" * 70)

model_ranking = []
for model_name in ALL_MODELS:
    if model_name in fold_data:
        scores = fold_data[model_name]['mcc']
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        model_ranking.append((display_name, mean_val, std_val))

model_ranking.sort(key=lambda x: x[1], reverse=True)

print(f"{'Rank':<6} {'Model':<30} {'Mean MCC':<12} {'Std MCC':<12}")
print("-" * 70)
for rank, (name, mean_val, std_val) in enumerate(model_ranking, 1):
    print(f"{rank:<6} {name:<30} {mean_val:.4f}     ±{std_val:.4f}")
print("-" * 70)

# ----------------------------------------------------------------------------
# Table 3: Significant Pairwise Wilcoxon Results (MCC)
# ----------------------------------------------------------------------------
print("\n📊 Table 3: Significant Pairwise Wilcoxon Results (MCC):")
print("-" * 80)

if 'mcc' in all_pairwise_results:
    sig_pairs = [r for r in all_pairwise_results['mcc'] if r['significant']]
    if sig_pairs:
        print(f"{'Model 1':<30} {'Model 2':<30} {'p-value':<12}")
        print("-" * 80)
        for r in sig_pairs:
            print(f"{r['model_1']:<30} {r['model_2']:<30} {r['p_value']:.6f}")
        print("-" * 80)
        print(f"Total significant pairs: {len(sig_pairs)}")
    else:
        print("  ⚠️ No significant pairs after Bonferroni correction.")

# ----------------------------------------------------------------------------
# Table 4: Mann-Whitney U (Vanilla+ vs All 11 Models, MCC)
# ----------------------------------------------------------------------------
print("\n📊 Table 4: Mann-Whitney U (Vanilla+ vs All Models, MCC):")
print("-" * 70)
if 'mcc' in all_mannwhitney_results:
    print(f"{'Comparison':<40} {'p-value':<12} {'Significant':<12}")
    print("-" * 70)
    for m, p_val in all_mannwhitney_results['mcc'].items():
        sig = '✅' if p_val < 0.05 else '❌'
        print(f"Vanilla+ vs {m:<30} {p_val:<12.6f} {sig:<12}")
    print("-" * 70)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Friedman results
if all_friedman_results:
    friedman_df = pd.DataFrame([
        {'Metric':       metric.upper(),
         'Chi_Square':   all_friedman_results[metric]['stat'],
         'P_Value':      all_friedman_results[metric]['p'],
         'Significant':  all_friedman_results[metric]['p'] < 0.05}
        for metric in all_friedman_results
    ])
    friedman_df.to_csv(os.path.join(RESULTS_DIR, "statistical_friedman_results.csv"), index=False)
    print("✅ Saved: statistical_friedman_results.csv")

# Model ranking
ranking_df = pd.DataFrame([
    {'Rank': rank, 'Model': name, 'Mean_MCC': mean_val, 'Std_MCC': std_val}
    for rank, (name, mean_val, std_val) in enumerate(model_ranking, 1)
])
ranking_df.to_csv(os.path.join(RESULTS_DIR, "statistical_model_ranking.csv"), index=False)
print("✅ Saved: statistical_model_ranking.csv")

# Pairwise Wilcoxon results
if all_pairwise_results:
    pairwise_rows = []
    for metric, results in all_pairwise_results.items():
        for r in results:
            pairwise_rows.append({
                'Metric':      metric.upper(),
                'Model_1':     r['model_1'],
                'Model_2':     r['model_2'],
                'P_Value':     r['p_value'],
                'Significant': r['significant'],
            })
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(os.path.join(RESULTS_DIR, "statistical_pairwise_wilcoxon.csv"), index=False)
    print("✅ Saved: statistical_pairwise_wilcoxon.csv")

# Mann-Whitney results
if all_mannwhitney_results:
    mw_rows = []
    for metric, comparisons in all_mannwhitney_results.items():
        for baseline, p_val in comparisons.items():
            mw_rows.append({
                'Metric':      metric.upper(),
                'Comparison':  f"Vanilla+ vs {baseline}",
                'P_Value':     p_val,
                'Significant': p_val < 0.05,
            })
    mw_df = pd.DataFrame(mw_rows)
    mw_df.to_csv(os.path.join(RESULTS_DIR, "statistical_mannwhitney_results.csv"), index=False)
    print("✅ Saved: statistical_mannwhitney_results.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS COMPLETE!")
print("=" * 80)

print("\n📊 Key Statistical Findings Summary:")

significant_metrics = [m for m in all_friedman_results if all_friedman_results[m]['p'] < 0.05]
non_significant_metrics = [m for m in all_friedman_results if all_friedman_results[m]['p'] >= 0.05]

if significant_metrics:
    print(f"\n  ✅ Significant differences found for: "
          f"{', '.join([m.upper() for m in significant_metrics])}")
if non_significant_metrics:
    print(f"  ❌ No significant differences for: "
          f"{', '.join([m.upper() for m in non_significant_metrics])}")

print("\n  📈 Top 5 Models by Mean MCC:")
for rank, (name, mean_val, std_val) in enumerate(model_ranking[:5], 1):
    print(f"    {rank}. {name}: {mean_val:.4f} ± {std_val:.4f}")

if 'mcc' in all_mannwhitney_results and 'BatchNorm Ablation' in all_mannwhitney_results['mcc']:
    p_val = all_mannwhitney_results['mcc']['BatchNorm Ablation']
    sig = '✅' if p_val < 0.05 else '❌'
    print(f"\n  Vanilla+ vs BatchNorm Ablation (MCC): p = {p_val:.6f} {sig}")

print(f"\n📁 Results saved in: {RESULTS_DIR}")
print("   - statistical_friedman_results.csv")
print("   - statistical_model_ranking.csv")
print("   - statistical_pairwise_wilcoxon.csv")
print("   - statistical_mannwhitney_results.csv")
print("=" * 80)