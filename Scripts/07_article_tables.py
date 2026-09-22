import pandas as pd
import os

BEST_CSV = r"D:\zebfish\new_class\Results\vanilla_plus_results_full3\best_configuration\csv_files"

# ============================================================
# MAIN MODELS
# ============================================================
print("=" * 90)
print("MAIN MODELS — TEST METRICS")
print("=" * 90)
print(f"{'Model':<22} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'MCC':>8} {'AUC':>8}")
print("-" * 90)

for model in ['LogisticRegression', 'VanillaMLP', 'DNNBaseline', 'VanillaPlus']:
    path = os.path.join(BEST_CSV, f"{model}_best_config.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{model:<22} {df['accuracy'].iloc[0]:>8.4f} {df['precision'].iloc[0]:>8.4f} "
              f"{df['recall'].iloc[0]:>8.4f} {df['f1'].iloc[0]:>8.4f} "
              f"{df['mcc'].iloc[0]:>8.4f} {df['auc_roc'].iloc[0]:>8.4f}")

print("\n" + "=" * 90)
print("MAIN MODELS — CV METRICS (mean ± std)")
print("=" * 90)
print(f"{'Model':<22} {'Acc':>14} {'Prec':>14} {'Rec':>14} {'F1':>14} {'MCC':>14} {'AUC':>14}")
print("-" * 90)

for model in ['LogisticRegression', 'VanillaMLP', 'DNNBaseline', 'VanillaPlus']:
    path = os.path.join(BEST_CSV, f"{model}_best_config.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{model:<22} "
              f"{df['cv_mean_accuracy'].iloc[0]:>6.4f}±{df['cv_std_accuracy'].iloc[0]:.4f} "
              f"{df['cv_mean_precision'].iloc[0]:>6.4f}±{df['cv_std_precision'].iloc[0]:.4f} "
              f"{df['cv_mean_recall'].iloc[0]:>6.4f}±{df['cv_std_recall'].iloc[0]:.4f} "
              f"{df['cv_mean_f1'].iloc[0]:>6.4f}±{df['cv_std_f1'].iloc[0]:.4f} "
              f"{df['cv_mean_mcc'].iloc[0]:>6.4f}±{df['cv_std_mcc'].iloc[0]:.4f} "
              f"{df['cv_mean_auc_roc'].iloc[0]:>6.4f}±{df['cv_std_auc_roc'].iloc[0]:.4f}")

# ============================================================
# ABLATION VARIANTS
# ============================================================
ablation_variants = [
    'Full_Model_VanillaPlus', 'w_o_Residual', 'w_o_LayerNorm',
    'ReLU_instead_of_GELU', 'w_o_Dropout', 'Higher_Dropout_0.5',
    'Lower_Dropout_0.1', 'BatchNorm_instead_of_LayerNorm'
]

print("\n" + "=" * 90)
print("ABLATION VARIANTS — TEST METRICS")
print("=" * 90)
print(f"{'Variant':<35} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'MCC':>8} {'AUC':>8}")
print("-" * 90)

for label in ablation_variants:
    path = os.path.join(BEST_CSV, f"ablation_{label}_best_config.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{label:<35} {df['accuracy'].iloc[0]:>8.4f} {df['precision'].iloc[0]:>8.4f} "
              f"{df['recall'].iloc[0]:>8.4f} {df['f1'].iloc[0]:>8.4f} "
              f"{df['mcc'].iloc[0]:>8.4f} {df['auc_roc'].iloc[0]:>8.4f}")

print("\n" + "=" * 90)
print("ABLATION VARIANTS — CV METRICS (mean ± std)")
print("=" * 90)
print(f"{'Variant':<35} {'Acc':>14} {'Prec':>14} {'Rec':>14} {'F1':>14} {'MCC':>14} {'AUC':>14}")
print("-" * 90)

for label in ablation_variants:
    path = os.path.join(BEST_CSV, f"ablation_{label}_best_config.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{label:<35} "
              f"{df['cv_mean_accuracy'].iloc[0]:>6.4f}±{df['cv_std_accuracy'].iloc[0]:.4f} "
              f"{df['cv_mean_precision'].iloc[0]:>6.4f}±{df['cv_std_precision'].iloc[0]:.4f} "
              f"{df['cv_mean_recall'].iloc[0]:>6.4f}±{df['cv_std_recall'].iloc[0]:.4f} "
              f"{df['cv_mean_f1'].iloc[0]:>6.4f}±{df['cv_std_f1'].iloc[0]:.4f} "
              f"{df['cv_mean_mcc'].iloc[0]:>6.4f}±{df['cv_std_mcc'].iloc[0]:.4f} "
              f"{df['cv_mean_auc_roc'].iloc[0]:>6.4f}±{df['cv_std_auc_roc'].iloc[0]:.4f}")