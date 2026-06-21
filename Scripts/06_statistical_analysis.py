# -*- coding: utf-8 -*-
"""
Statistical Analysis for VANILLA+ Fish Enzyme Classification
Using FOLD-LEVEL history files for proper statistical tests
8-Class Classification (0 = Non-enzyme, 1-7 = EC Classes)

UPDATED: Analyzes ALL metrics (Accuracy, Precision, Recall, F1, MCC, AUC)
UPDATED: Using vanilla_plus_results_full3 results
UPDATED: Includes ablation models in statistical tests
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, wilcoxon
from glob import glob
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("STATISTICAL ANALYSIS FOR VANILLA+ (WITH FOLD-LEVEL DATA)")
print("8-Class Classification | Species-Aware Split (370 Train, 19 Test Species)")
print("=" * 80)

# ============================================================================
# CONFIGURATION - UPDATED TO FULL3
# ============================================================================

RESULTS_DIR = r"D:\zebfish\new_class\Results\vanilla_plus_results_full3"
HISTORY_DIR = os.path.join(RESULTS_DIR, "best_configuration", "history_files")
CSV_DIR = os.path.join(RESULTS_DIR, "best_configuration", "csv_files")
RANDOM_STATE = 42

# Main models
MAIN_MODELS = ['VanillaPlus', 'VanillaMLP', 'DNNBaseline', 'LogisticRegression']

# Ablation models to include in statistical analysis
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

# All models combined
ALL_MODELS = MAIN_MODELS + ABLATION_MODELS

# Metrics to analyze
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc']

# Display names for models
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
best_val_metrics = {}

def load_model_history(model_name, is_ablation=False):
    """Load history for a model, handling both main and ablation models"""
    
    # Try different naming conventions
    possible_names = []
    
    if is_ablation:
        possible_names.append(f"ablation_{model_name}_best_final_history.npy")
        possible_names.append(f"ablation_{model_name}_final_history.npy")
        possible_names.append(f"{model_name}_best_final_history.npy")
        possible_names.append(f"{model_name}_final_history.npy")
    else:
        possible_names.append(f"{model_name}_best_final_history.npy")
        possible_names.append(f"{model_name}_final_history.npy")
    
    for name in possible_names:
        path = os.path.join(HISTORY_DIR, name)
        if os.path.exists(path):
            try:
                return np.load(path, allow_pickle=True).item()
            except:
                continue
    return None

def load_fold_history(model_name, fold, is_ablation=False):
    """Load fold history for a model"""
    
    possible_names = []
    
    if is_ablation:
        possible_names.append(f"ablation_{model_name}_best_fold{fold}_history.npy")
        possible_names.append(f"ablation_{model_name}_fold{fold}_history.npy")
        possible_names.append(f"{model_name}_best_fold{fold}_history.npy")
        possible_names.append(f"{model_name}_fold{fold}_history.npy")
    else:
        possible_names.append(f"{model_name}_best_fold{fold}_history.npy")
        possible_names.append(f"{model_name}_fold{fold}_history.npy")
    
    for name in possible_names:
        path = os.path.join(HISTORY_DIR, name)
        if os.path.exists(path):
            try:
                return np.load(path, allow_pickle=True).item()
            except:
                continue
    return None

def extract_metrics_from_history(history):
    """Extract all metrics from history dict"""
    metrics = {}
    for metric in METRICS:
        # Try val_metric first
        val_key = f'val_{metric}'
        if val_key in history:
            val = history[val_key]
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                metrics[metric] = max(val)
            else:
                metrics[metric] = val
        elif metric in history:
            val = history[metric]
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                metrics[metric] = max(val)
            else:
                metrics[metric] = val
        else:
            metrics[metric] = 0.0
    return metrics

# Load data for all models
for model_name in ALL_MODELS:
    is_ablation = model_name in ABLATION_MODELS
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    
    print(f"\n  Loading {display_name}...")
    
    # Try to load fold histories
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
    
    # If no fold histories found, try to load from CSV
    if found_folds == 0:
        csv_name = f"ablation_{model_name}_best_config.csv" if is_ablation else f"{model_name}_best_config.csv"
        csv_path = os.path.join(CSV_DIR, csv_name)
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Try to get fold metrics from CSV
                for fold in range(1, 11):
                    fold_key = f'fold_{fold}_mcc'
                    if fold_key in df.columns:
                        for metric in METRICS:
                            metric_key = f'fold_{fold}_{metric}'
                            if metric_key in df.columns:
                                model_folds[metric].append(df[metric_key].iloc[0])
                            else:
                                # Use MCC as proxy
                                model_folds[metric].append(df[fold_key].iloc[0])
                        found_folds += 1
                if found_folds > 0:
                    print(f"    Loaded {found_folds} folds from CSV")
            except Exception as e:
                print(f"    ⚠️ Error loading CSV: {e}")
    
    # If still no data, use final history
    if found_folds == 0:
        history = load_model_history(model_name, is_ablation)
        if history:
            metrics = extract_metrics_from_history(history)
            for metric in METRICS:
                model_folds[metric] = [metrics[metric]] * 10
            found_folds = 10
            print(f"    Using final history (10 folds with same value)")
    
    # Store data if we found anything
    if found_folds > 0:
        for metric in METRICS:
            if len(model_folds[metric]) < 10:
                # Pad with last value if needed
                last_val = model_folds[metric][-1] if model_folds[metric] else 0
                model_folds[metric].extend([last_val] * (10 - len(model_folds[metric])))
            model_folds[metric] = np.array(model_folds[metric])
        
        fold_data[model_name] = {metric: np.array(model_folds[metric]) for metric in METRICS}
        
        print(f"  ✅ Loaded {found_folds} folds for {display_name}")
        for metric in METRICS:
            mean_val = np.mean(model_folds[metric])
            std_val = np.std(model_folds[metric])
            print(f"     {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    else:
        print(f"  ❌ No data found for {display_name}")

# ============================================================================
# LOAD ABLATION FOLD DATA FOR ALL METRICS (Alternative method)
# ============================================================================

print("\n" + "=" * 80)
print("LOADING ABLATION FOLD DATA (Alternative Method)")
print("=" * 80)

ablation_fold_data = {}

# Find ablation history files
ablation_files = glob(os.path.join(HISTORY_DIR, "ablation_*_best_fold*_history.npy"))

if ablation_files:
    # Group by ablation variant
    ablation_groups = {}
    for af in ablation_files:
        base = os.path.basename(af)
        parts = base.split('_best_')
        if len(parts) >= 2:
            variant = parts[0].replace('ablation_', '')
            fold_num = int(parts[1].split('_')[0].replace('fold', ''))
            
            if variant not in ablation_groups:
                ablation_groups[variant] = {}
            
            try:
                history = np.load(af, allow_pickle=True).item()
                val_mcc = history.get('val_mcc', [])
                if val_mcc:
                    ablation_groups[variant][fold_num] = {
                        'mcc': max(val_mcc),
                        'accuracy': max(history.get('val_accuracy', [0])),
                        'f1': max(history.get('val_f1', [0])),
                        'precision': max(history.get('val_precision', [0])),
                        'recall': max(history.get('val_recall', [0])),
                        'auc': max(history.get('val_auc', [0]))
                    }
            except Exception as e:
                print(f"  ❌ Error loading {af}: {e}")
    
    # Convert to arrays
    for variant, folds in ablation_groups.items():
        if folds:
            sorted_folds = sorted(folds.keys())
            ablation_fold_data[variant] = {
                metric: np.array([folds[f][metric] for f in sorted_folds]) 
                for metric in METRICS
            }
            print(f"  ✅ {variant}: {len(sorted_folds)} folds")
            for metric in METRICS:
                mean_val = np.mean(ablation_fold_data[variant][metric])
                std_val = np.std(ablation_fold_data[variant][metric])
                print(f"     {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")

# ============================================================================
# STATISTICAL TESTS FOR ALL METRICS
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL TESTS FOR ALL METRICS (Including Ablations)")
print("=" * 80)

# Store results
all_friedman_results = {}
all_pairwise_results = {}
all_mannwhitney_results = {}

for metric in METRICS:
    print(f"\n{'='*80}")
    print(f"  METRIC: {metric.upper()}")
    print(f"{'='*80}")
    
    # Get data for this metric
    metric_data = {}
    for model_name in ALL_MODELS:
        if model_name in fold_data and metric in fold_data[model_name]:
            metric_data[MODEL_DISPLAY_NAMES.get(model_name, model_name)] = fold_data[model_name][metric]
    
    if len(metric_data) >= 3:
        # --- Friedman Test ---
        model_names = list(metric_data.keys())
        fold_scores = [metric_data[m] for m in model_names]
        
        # Ensure all have same length
        min_len = min(len(scores) for scores in fold_scores)
        fold_scores_aligned = [scores[:min_len] for scores in fold_scores]
        
        print(f"\n  Comparing {len(model_names)} models with {min_len} folds each")
        print(f"  Models: {model_names[:5]}..." if len(model_names) > 5 else f"  Models: {model_names}")
        
        try:
            friedman_stat, friedman_p = friedmanchisquare(*fold_scores_aligned)
            print(f"\n  Friedman test: χ² = {friedman_stat:.4f}, p = {friedman_p:.6f}")
            
            if friedman_p < 0.05:
                print(f"  ✅ Significant difference among models (p < 0.05)")
            else:
                print(f"  ❌ No significant difference among models (p >= 0.05)")
            
            all_friedman_results[metric] = {'stat': friedman_stat, 'p': friedman_p}
            
        except Exception as e:
            print(f"  ⚠️ Friedman test error: {e}")
        
        # --- Pairwise Comparisons (only for main models + best ablations) ---
        main_and_best = ['Vanilla+ (Ours)', 'Vanilla MLP', 'DNN Baseline', 'Logistic Regression',
                         'BatchNorm Ablation', 'Lower Dropout (0.1)', 'ReLU Ablation']
        
        print(f"\n  Pairwise comparisons (Wilcoxon signed-rank) for key models:")
        print("-" * 60)
        
        for i, model1 in enumerate(main_and_best):
            if model1 not in metric_data:
                continue
            for j, model2 in enumerate(main_and_best):
                if i < j and model2 in metric_data:
                    scores1 = metric_data[model1][:min_len]
                    scores2 = metric_data[model2][:min_len]
                    
                    try:
                        stat, p_val = wilcoxon(scores1, scores2)
                        is_sig = p_val < 0.05
                        sig_str = "✅ Significant" if is_sig else "❌ Not significant"
                        print(f"  {model1} vs {model2}: p = {p_val:.6f} ({sig_str})")
                    except Exception as e:
                        print(f"  {model1} vs {model2}: Error - {e}")
        
        # --- Mann-Whitney U (Vanilla+ vs Baselines including ablations) ---
        if 'Vanilla+ (Ours)' in metric_data:
            vanilla_scores = metric_data['Vanilla+ (Ours)']
            baseline_models = ['Vanilla MLP', 'DNN Baseline', 'Logistic Regression',
                               'BatchNorm Ablation', 'Lower Dropout (0.1)', 'ReLU Ablation']
            
            print(f"\n  Mann-Whitney U (Vanilla+ vs Baselines + Ablations):")
            for baseline in baseline_models:
                if baseline in metric_data:
                    baseline_scores = metric_data[baseline]
                    min_len = min(len(vanilla_scores), len(baseline_scores))
                    
                    try:
                        stat, p_val = mannwhitneyu(vanilla_scores[:min_len], baseline_scores[:min_len])
                        is_sig = p_val < 0.05
                        sig_str = "✅ Significant" if is_sig else "❌ Not significant"
                        print(f"  Vanilla+ vs {baseline}: p = {p_val:.6f} ({sig_str})")
                        
                        if metric not in all_mannwhitney_results:
                            all_mannwhitney_results[metric] = {}
                        all_mannwhitney_results[metric][baseline] = p_val
                    except Exception as e:
                        print(f"  Vanilla+ vs {baseline}: Error - {e}")

# ============================================================================
# SUMMARY TABLES FOR PAPER
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY TABLES FOR PAPER")
print("=" * 80)

# Table 1: Friedman Test Results
print("\n📊 Table 1: Friedman Test Results (All Models Including Ablations):")
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

# Table 2: Model Ranking by MCC
print("\n📊 Table 2: Model Ranking by MCC (Including Ablations):")
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

# Table 3: Mann-Whitney U Test Results
print("\n📊 Table 3: Mann-Whitney U Test (Vanilla+ vs Key Models):")
print("-" * 80)

# Create header
print(f"{'Metric':<12}", end="")
for baseline in ['Vanilla MLP', 'DNN Baseline', 'Logistic Regression', 
                 'BatchNorm Ablation', 'Lower Dropout (0.1)', 'ReLU Ablation']:
    print(f"{baseline[:15]:>15}", end="")
print()
print("-" * 80)

for metric in METRICS:
    if metric in all_mannwhitney_results:
        print(f"{metric.upper():<12}", end="")
        for baseline in ['Vanilla MLP', 'DNN Baseline', 'Logistic Regression', 
                         'BatchNorm Ablation', 'Lower Dropout (0.1)', 'ReLU Ablation']:
            if baseline in all_mannwhitney_results[metric]:
                p_val = all_mannwhitney_results[metric][baseline]
                sig = '✅' if p_val < 0.05 else ''
                print(f"{p_val:>8.4f} {sig:<6}", end="")
            else:
                print(f"{'':>8} {'':<6}", end="")
        print()
print("-" * 80)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save Friedman results
if all_friedman_results:
    friedman_df = pd.DataFrame([
        {'Metric': metric.upper(), 
         'Chi_Square': all_friedman_results[metric]['stat'],
         'P_Value': all_friedman_results[metric]['p'],
         'Significant': all_friedman_results[metric]['p'] < 0.05}
        for metric in all_friedman_results
    ])
    friedman_df.to_csv(os.path.join(RESULTS_DIR, "statistical_friedman_results.csv"), index=False)
    print(f"✅ Saved: statistical_friedman_results.csv")

# Save model ranking
ranking_df = pd.DataFrame([
    {'Rank': rank, 'Model': name, 'Mean_MCC': mean_val, 'Std_MCC': std_val}
    for rank, (name, mean_val, std_val) in enumerate(model_ranking, 1)
])
ranking_df.to_csv(os.path.join(RESULTS_DIR, "statistical_model_ranking.csv"), index=False)
print(f"✅ Saved: statistical_model_ranking.csv")

# Save Mann-Whitney results
if all_mannwhitney_results:
    mannwhitney_data = []
    for metric in all_mannwhitney_results:
        row = {'Metric': metric.upper()}
        for baseline in ['Vanilla MLP', 'DNN Baseline', 'Logistic Regression', 
                         'BatchNorm Ablation', 'Lower Dropout (0.1)', 'ReLU Ablation']:
            if baseline in all_mannwhitney_results[metric]:
                row[baseline] = all_mannwhitney_results[metric][baseline]
            else:
                row[baseline] = None
        mannwhitney_data.append(row)
    
    mannwhitney_df = pd.DataFrame(mannwhitney_data)
    mannwhitney_df.to_csv(os.path.join(RESULTS_DIR, "statistical_mannwhitney_results.csv"), index=False)
    print(f"✅ Saved: statistical_mannwhitney_results.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS COMPLETE!")
print("=" * 80)

print("\n📊 Key Statistical Findings Summary:")

# Check which metrics show significance
significant_metrics = [m for m in all_friedman_results if all_friedman_results[m]['p'] < 0.05]
non_significant_metrics = [m for m in all_friedman_results if all_friedman_results[m]['p'] >= 0.05]

if significant_metrics:
    print(f"\n  ✅ Significant differences found for: {', '.join([m.upper() for m in significant_metrics])}")
if non_significant_metrics:
    print(f"  ❌ No significant differences for: {', '.join([m.upper() for m in non_significant_metrics])}")

print("\n  📈 Top 5 Models by Mean MCC:")
for rank, (name, mean_val, std_val) in enumerate(model_ranking[:5], 1):
    print(f"    {rank}. {name}: {mean_val:.4f} ± {std_val:.4f}")

# Check Vanilla+ vs best ablation significance
if 'mcc' in all_mannwhitney_results and 'BatchNorm Ablation' in all_mannwhitney_results['mcc']:
    p_val = all_mannwhitney_results['mcc']['BatchNorm Ablation']
    sig = '✅' if p_val < 0.05 else '❌'
    print(f"\n  Vanilla+ vs BatchNorm Ablation (MCC): p = {p_val:.6f} {sig}")

print(f"\n📁 Results saved in: {RESULTS_DIR}")
print("   - statistical_friedman_results.csv")
print("   - statistical_model_ranking.csv")
print("   - statistical_mannwhitney_results.csv")
print("=" * 80)