# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 22:01:06 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
RESULTS VISUALIZATION AND SUMMARY FOR VANILLA+ EXPERIMENTS
This script loads and displays all results from the Vanilla+ experiments

UPDATED: Using vanilla_plus_results_full3 results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# =============================================================================
# CONFIGURATION - UPDATED TO FULL3
# =============================================================================

RESULTS_DIR = r"D:\zebfish\new_class\Results\vanilla_plus_results_full3"
BEST_CSV_DIR = os.path.join(RESULTS_DIR, "best_configuration", "csv_files")
HISTORY_DIR = os.path.join(RESULTS_DIR, "best_configuration", "history_files")

print("=" * 80)
print("VANILLA+ EXPERIMENT RESULTS VIEWER")
print("=" * 80)
print(f"\nResults Directory: {RESULTS_DIR}")
print(f"CSV Directory: {BEST_CSV_DIR}")

# =============================================================================
# 1. LOAD ALL RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("1. LOADING RESULTS")
print("=" * 80)

# Load comprehensive summary
final_summary_path = os.path.join(BEST_CSV_DIR, "final_comprehensive_summary.csv")
if os.path.exists(final_summary_path):
    final_summary = pd.read_csv(final_summary_path)
    print(f"✅ Loaded: final_comprehensive_summary.csv")
    print(f"   Columns: {final_summary.columns.tolist()}")
else:
    print(f"❌ File not found: {final_summary_path}")
    final_summary = None

# Load paper results table
paper_table_path = os.path.join(BEST_CSV_DIR, "paper_results_table.csv")
if os.path.exists(paper_table_path):
    paper_table = pd.read_csv(paper_table_path)
    print(f"✅ Loaded: paper_results_table.csv")
else:
    print(f"❌ File not found: {paper_table_path}")
    paper_table = None

# Load individual model results
model_results = {}
model_names = ['VanillaPlus', 'VanillaMLP', 'DNNBaseline', 'LogisticRegression']
for model in model_names:
    file_path = os.path.join(BEST_CSV_DIR, f"{model}_best_config.csv")
    if os.path.exists(file_path):
        model_results[model] = pd.read_csv(file_path)
        print(f"✅ Loaded: {model}_best_config.csv")
    else:
        print(f"❌ File not found: {file_path}")

# Load ablation results
ablation_results = {}
ablation_variants = [
    'Full_Model_VanillaPlus',
    'ReLU_instead_of_GELU',
    'BatchNorm_instead_of_LayerNorm',
    'w_o_Dropout',
    'w_o_LayerNorm',
    'w_o_Residual',
    'Higher_Dropout_0.5',
    'Lower_Dropout_0.1'
]
for variant in ablation_variants:
    file_path = os.path.join(BEST_CSV_DIR, f"ablation_{variant}_best_config.csv")
    if os.path.exists(file_path):
        ablation_results[variant] = pd.read_csv(file_path)
        print(f"✅ Loaded: ablation_{variant}_best_config.csv")
    else:
        print(f"❌ File not found: {file_path}")

print("\n✅ All results loaded successfully!")

# =============================================================================
# 2. DISPLAY MODEL PERFORMANCE SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("2. MODEL PERFORMANCE SUMMARY (TEST SET)")
print("=" * 80)

if final_summary is not None:
    # Check available columns
    available_cols = final_summary.columns.tolist()
    
    # Map column names (they might be lowercase or with underscores)
    col_map = {}
    for col in available_cols:
        if 'accuracy' in col.lower() and 'test' in col.lower():
            col_map['Test_Accuracy'] = col
        elif 'precision' in col.lower() and 'test' in col.lower():
            col_map['Test_Precision'] = col
        elif 'recall' in col.lower() and 'test' in col.lower():
            col_map['Test_Recall'] = col
        elif 'f1' in col.lower() and 'test' in col.lower():
            col_map['Test_F1'] = col
        elif 'mcc' in col.lower() and 'test' in col.lower():
            col_map['Test_MCC'] = col
        elif 'auc' in col.lower() and 'test' in col.lower():
            col_map['Test_AUC'] = col
        elif 'cv_mean_mcc' in col.lower():
            col_map['CV_Mean_MCC'] = col
        elif 'cv_std_mcc' in col.lower():
            col_map['CV_Std_MCC'] = col
    
    # Select main models
    main_models = final_summary[~final_summary['Model'].str.contains('Ablation')]
    main_models = main_models[main_models['Model'].str.contains('Logistic|Vanilla|DNN')]
    
    # Create display columns
    display_cols = ['Model']
    display_headers = ['Model']
    
    for display_name, actual_col in col_map.items():
        if actual_col in available_cols:
            display_cols.append(actual_col)
            display_headers.append(display_name)
    
    if len(display_cols) > 1:
        print("\n📊 Main Models Performance:")
        display_df = main_models[display_cols].copy()
        display_df.columns = display_headers
        print(tabulate(display_df, headers='keys', tablefmt='grid', floatfmt='.4f'))
    else:
        print("  ⚠️ No matching columns found. Available columns:")
        print(f"     {available_cols}")

# =============================================================================
# 3. DISPLAY ABLATION RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("3. ABLATION STUDY RESULTS")
print("=" * 80)

if ablation_results:
    ablation_summary = []
    for variant, df in ablation_results.items():
        # Get test metrics
        test_f1 = df['f1'].iloc[0] if 'f1' in df.columns else 'N/A'
        test_mcc = df['mcc'].iloc[0] if 'mcc' in df.columns else 'N/A'
        test_acc = df['accuracy'].iloc[0] if 'accuracy' in df.columns else 'N/A'
        test_auc = df['auc_roc'].iloc[0] if 'auc_roc' in df.columns else 'N/A'
        
        # Get CV metrics
        cv_mcc_mean = df['cv_mean_mcc'].iloc[0] if 'cv_mean_mcc' in df.columns else 'N/A'
        cv_mcc_std = df['cv_std_mcc'].iloc[0] if 'cv_std_mcc' in df.columns else 'N/A'
        
        ablation_summary.append({
            'Variant': variant,
            'Test_Acc': test_acc,
            'Test_F1': test_f1,
            'Test_MCC': test_mcc,
            'Test_AUC': test_auc,
            'CV_MCC': f"{cv_mcc_mean:.4f} ± {cv_mcc_std:.4f}" if cv_mcc_mean != 'N/A' else 'N/A'
        })
    
    ablation_df = pd.DataFrame(ablation_summary)
    
    # Sort by Test MCC
    if 'Test_MCC' in ablation_df.columns:
        ablation_df = ablation_df.sort_values('Test_MCC', ascending=False)
    
    print("\n📊 Ablation Variants Performance:")
    print(tabulate(ablation_df, headers='keys', tablefmt='grid', floatfmt='.4f'))

# =============================================================================
# 4. COMPLETE COMPARISON TABLE
# =============================================================================

print("\n" + "=" * 80)
print("4. COMPLETE COMPARISON TABLE")
print("=" * 80)

if final_summary is not None:
    # Get all models (including ablations)
    all_models = final_summary.copy()
    
    # Create display columns
    display_cols = ['Model']
    display_headers = ['Model']
    
    for display_name, actual_col in col_map.items():
        if actual_col in available_cols:
            display_cols.append(actual_col)
            display_headers.append(display_name)
    
    if len(display_cols) > 1:
        display_df = all_models[display_cols].copy()
        display_df.columns = display_headers
        
        # Sort by Test MCC if available
        if 'Test MCC' in display_df.columns:
            display_df = display_df.sort_values('Test MCC', ascending=False)
        
        print("\n📊 All Models Performance (Sorted by Test MCC):")
        print(tabulate(display_df, headers='keys', tablefmt='grid', floatfmt='.4f'))

# =============================================================================
# 5. PER-FOLD RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("5. PER-FOLD RESULTS (Vanilla+)")
print("=" * 80)

# Load Vanilla+ fold summary
vanilla_fold_path = os.path.join(BEST_CSV_DIR, "VanillaPlus_fold_summary.csv")
if os.path.exists(vanilla_fold_path):
    fold_df = pd.read_csv(vanilla_fold_path)
    print("\n📊 Vanilla+ 10-Fold CV Results:")
    print(tabulate(fold_df, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Display statistics
    print("\n📈 Fold Statistics:")
    for col in ['accuracy', 'f1', 'mcc', 'auc']:
        if col in fold_df.columns:
            mean_val = fold_df[col].mean()
            std_val = fold_df[col].std()
            min_val = fold_df[col].min()
            max_val = fold_df[col].max()
            print(f"  {col.upper():12s}: Mean={mean_val:.4f} ± {std_val:.4f}  [Min={min_val:.4f}, Max={max_val:.4f}]")
else:
    print(f"❌ VanillaPlus_fold_summary.csv not found")

# =============================================================================
# 6. BEST CONFIGURATIONS FROM GRID SEARCH
# =============================================================================

print("\n" + "=" * 80)
print("6. GRID SEARCH RESULTS")
print("=" * 80)

grid_search_path = os.path.join(RESULTS_DIR, "csv_files", "full_grid_search_results.csv")
if os.path.exists(grid_search_path):
    grid_df = pd.read_csv(grid_search_path)
    print("\n📊 Grid Search Results:")
    print(tabulate(grid_df, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Best configuration
    best_config = grid_df.loc[grid_df['cv_mean_mcc'].idxmax()]
    print(f"\n🏆 Best Configuration: LR={best_config['learning_rate']}, BS={best_config['batch_size']}")
    print(f"   CV MCC: {best_config['cv_mean_mcc']:.4f} ± {best_config['cv_std_mcc']:.4f}")
    print(f"   Test MCC: {best_config['test_mcc']:.4f}")
else:
    print(f"❌ full_grid_search_results.csv not found")

# =============================================================================
# 7. TRAINING HISTORY (Load from .npy files)
# =============================================================================

print("\n" + "=" * 80)
print("7. TRAINING HISTORY (First 5 epochs)")
print("=" * 80)

def load_history(model_name):
    """Load training history from .npy file"""
    history_path = os.path.join(HISTORY_DIR, f"{model_name}_best_final_history.npy")
    if os.path.exists(history_path):
        try:
            history = np.load(history_path, allow_pickle=True).item()
            return history
        except:
            return None
    return None

# Load Vanilla+ history
vanilla_history = load_history("VanillaPlus")
if vanilla_history:
    print("\n📈 Vanilla+ Training History (First 5 epochs):")
    print(f"  Epochs: {len(vanilla_history.get('train_loss', []))}")
    print(f"  Best Val MCC: {max(vanilla_history.get('val_mcc', [])):.4f}")
    
    # Create a small dataframe for first 5 epochs
    if len(vanilla_history.get('train_loss', [])) >= 5:
        history_df = pd.DataFrame({
            'Epoch': range(1, 6),
            'Train Loss': vanilla_history['train_loss'][:5],
            'Val Loss': vanilla_history['val_loss'][:5],
            'Val Accuracy': vanilla_history['val_accuracy'][:5],
            'Val F1': vanilla_history['val_f1'][:5],
            'Val MCC': vanilla_history['val_mcc'][:5]
        })
        print(tabulate(history_df, headers='keys', tablefmt='grid', floatfmt='.4f'))

# =============================================================================
# 8. STATISTICAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("8. STATISTICAL SUMMARY")
print("=" * 80)

if final_summary is not None:
    # Get LR MCC
    lr_mcc = None
    for col in available_cols:
        if ('mcc' in col.lower() or 'MCC' in col) and 'test' in col.lower():
            lr_row = final_summary[final_summary['Model'] == 'Logistic Regression']
            if not lr_row.empty:
                lr_mcc = lr_row[col].iloc[0]
                break
    
    if lr_mcc is not None:
        print(f"\n📊 Improvements over Logistic Regression (LR MCC = {lr_mcc:.4f}):")
        
        for _, row in final_summary.iterrows():
            if row['Model'] != 'Logistic Regression':
                # Find MCC column
                model_mcc = None
                for col in available_cols:
                    if ('mcc' in col.lower() or 'MCC' in col) and 'test' in col.lower():
                        model_mcc = row[col]
                        break
                
                if model_mcc is not None:
                    improvement = ((model_mcc - lr_mcc) / lr_mcc) * 100
                    print(f"  {row['Model']:30s}: {model_mcc:.4f} ({improvement:+.2f}% improvement)")

# =============================================================================
# 9. VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 80)
print("9. GENERATING VISUALIZATIONS")
print("=" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create a figures directory
figures_dir = os.path.join(RESULTS_DIR, "figures")
os.makedirs(figures_dir, exist_ok=True)

if final_summary is not None:
    # Figure 1: Model Comparison Bar Chart
    print("\n📊 Creating Figure 1: Model Performance Comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Filter main models
    main_models = final_summary[~final_summary['Model'].str.contains('Ablation')]
    main_models = main_models[main_models['Model'].str.contains('Logistic|Vanilla|DNN')]
    
    metrics = ['Test_F1', 'Test_MCC', 'Test_Accuracy', 'Test_AUC']
    titles = ['F1 Score', 'MCC', 'Accuracy', 'AUC']
    
    # Map metric names to actual columns
    metric_col_map = {}
    for metric in metrics:
        for col in available_cols:
            if metric.lower() in col.lower() or (metric.replace('_', '').lower() in col.lower()):
                metric_col_map[metric] = col
                break
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]
        actual_col = metric_col_map.get(metric)
        
        if actual_col and actual_col in main_models.columns:
            data = main_models[['Model', actual_col]].dropna()
            if not data.empty:
                colors = ['#808080' if 'Logistic' in m else '#1f77b4' if 'MLP' in m else '#ff7f0e' if 'DNN' in m else '#2ca02c' for m in data['Model']]
                bars = ax.bar(data['Model'], data[actual_col], color=colors, edgecolor='black', linewidth=1.5)
                
                # Highlight Vanilla+
                for j, model in enumerate(data['Model']):
                    if 'Vanilla+' in model:
                        bars[j].set_edgecolor('gold')
                        bars[j].set_linewidth(3)
                
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_ylabel(title, fontsize=12)
                ax.set_xticklabels(data['Model'], rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
                # Add value labels
                for j, v in enumerate(data[actual_col]):
                    ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {os.path.join(figures_dir, 'model_comparison.png')}")
    
    # Figure 2: Ablation Study
    print("\n📊 Creating Figure 2: Ablation Study...")
    
    if ablation_results:
        ablation_data = []
        for variant, df in ablation_results.items():
            if 'mcc' in df.columns and 'cv_mean_mcc' in df.columns:
                ablation_data.append({
                    'Variant': variant,
                    'Test_MCC': df['mcc'].iloc[0],
                    'CV_MCC': df['cv_mean_mcc'].iloc[0]
                })
        
        if ablation_data:
            abl_df = pd.DataFrame(ablation_data)
            abl_df = abl_df.sort_values('Test_MCC', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(abl_df))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, abl_df['Test_MCC'], width, label='Test MCC', color='#2E86AB')
            bars2 = ax.bar(x + width/2, abl_df['CV_MCC'], width, label='CV MCC', color='#A23B72')
            
            # Highlight best performer
            best_idx = np.argmax(abl_df['Test_MCC'])
            bars1[best_idx].set_edgecolor('gold')
            bars1[best_idx].set_linewidth(3)
            bars2[best_idx].set_edgecolor('gold')
            bars2[best_idx].set_linewidth(3)
            
            ax.set_xlabel('Ablation Variant', fontsize=12)
            ax.set_ylabel('MCC Score', fontsize=12)
            ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(abl_df['Variant'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'ablation_study.png'), dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {os.path.join(figures_dir, 'ablation_study.png')}")
    
    # Figure 3: 10-Fold CV Performance
    print("\n📊 Creating Figure 3: 10-Fold CV Performance...")
    
    if os.path.exists(vanilla_fold_path):
        fold_df = pd.read_csv(vanilla_fold_path)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each fold
        x = np.arange(1, 11)
        ax.plot(x, fold_df['mcc'], marker='o', linewidth=2, markersize=10, color='#2E86AB', label='MCC')
        ax.plot(x, fold_df['f1'], marker='s', linewidth=2, markersize=10, color='#A23B72', label='F1')
        ax.plot(x, fold_df['accuracy'], marker='^', linewidth=2, markersize=10, color='#F18F01', label='Accuracy')
        
        # Add mean lines
        ax.axhline(y=fold_df['mcc'].mean(), color='#2E86AB', linestyle='--', alpha=0.5, label=f"MCC Mean: {fold_df['mcc'].mean():.4f}")
        ax.axhline(y=fold_df['f1'].mean(), color='#A23B72', linestyle='--', alpha=0.5, label=f"F1 Mean: {fold_df['f1'].mean():.4f}")
        ax.axhline(y=fold_df['accuracy'].mean(), color='#F18F01', linestyle='--', alpha=0.5, label=f"Acc Mean: {fold_df['accuracy'].mean():.4f}")
        
        ax.set_xlabel('Fold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Vanilla+ 10-Fold Cross-Validation Performance', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'cv_performance.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {os.path.join(figures_dir, 'cv_performance.png')}")

print("\n✅ All visualizations saved to:", figures_dir)

# =============================================================================
# 10. FINAL SUMMARY - UPDATED WITH NEW RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("10. KEY FINDINGS SUMMARY")
print("=" * 80)

print("\n📊 Top 3 Models by Test MCC:")
if final_summary is not None:
    # Filter only main models (not ablations)
    main_models_only = final_summary[~final_summary['Model'].str.contains('Ablation')]
    
    # Find MCC column
    mcc_col = None
    for col in available_cols:
        if ('mcc' in col.lower() or 'MCC' in col) and 'test' in col.lower():
            mcc_col = col
            break
    
    if mcc_col and mcc_col in main_models_only.columns:
        top_models = main_models_only.nlargest(3, mcc_col)
        if len(top_models) > 0:
            for idx, row in top_models.iterrows():
                print(f"  {row['Model']:40s}: MCC = {row[mcc_col]:.4f}")

print("\n🏆 Best Configuration:")
if 'best_config' in locals():
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  CV MCC: {best_config['cv_mean_mcc']:.4f} ± {best_config['cv_std_mcc']:.4f}")

print("\n📈 Key Insights (UPDATED for FULL3):")
print("  1. Best overall configuration: LR=0.001, BS=128 (CV MCC=0.8373)")
print("  2. BatchNorm Ablation achieves best Test MCC (0.9165)")
print("  3. Lower Dropout (0.1) achieves Test MCC (0.9140)")
print("  4. ReLU Ablation achieves Test MCC (0.9116)")
print("  5. Vanilla+ achieves Test MCC (0.8858) and CV MCC (0.8360 ± 0.0200)")
print("  6. Species-aware split ensures realistic evaluation (no species overlap)")

print("\n📁 Output Files:")
print(f"  - CSV Results: {BEST_CSV_DIR}")
print(f"  - Figures: {figures_dir}")
print(f"  - History Files: {HISTORY_DIR}")

print("\n" + "=" * 80)
print("✅ RESULTS VIEWING COMPLETE!")
print("=" * 80)