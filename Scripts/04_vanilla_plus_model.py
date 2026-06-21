# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:27:15 2026

@author: H.A.R
"""

# -*- coding: utf-8 -*-
"""
VANILLA+: ENHANCED MLP FOR FISH ENZYME CLASSIFICATION (8 Classes)
Using UniProt Protein Embeddings (1024-dimensional)

FULL REPRODUCIBILITY:
- Grid search over ALL hyperparameters (9 combinations)
- Saves ALL configurations results (.npy, .csv, histories)
- Separately saves BEST configuration results
- Saves test predictions for ALL models and ALL configurations
- Saves COMPLETE CV metrics for ALL models (accuracy, precision, recall, f1, mcc, auc)

Models Evaluated:
  1. Logistic Regression   (Baseline)
  2. Vanilla MLP           (Baseline)
  3. DNN Baseline          (Baseline)
  4. Vanilla+              (Proposed Novel Model)
  5. Vanilla+ Ablations    (8 variants — WITH 10-fold CV)

BUG FIXES OVER ORIGINAL:
  FIX-1: save_test_predictions now receives the TRAINED model, not a fresh untrained instance.
  FIX-2: train_model_with_config now accepts and forwards lr to AdamW (was hardcoded 0.001).
  FIX-3: Final summary walrus-operator bug fixed — each model's metrics stored in named variables
          (m_vanilla, m_dnn, m_vp) BEFORE the summary loop; no more identical rows.
  FIX-4: train_model_with_config returns the trained model as a third return value.
  FIX-5: Complete .npy bundle per fold: confusion_matrix, per_class_precision/recall/f1,
          model weights (.pt) for every fold, and full per-epoch history for every metric.
  FIX-6: History dict is now symmetric — train_mcc and train_auc stored every epoch.

Output Structure:
  OUTPUT_DIR/
    all_configurations/LR_*/   — all 9 grid-search combinations
    best_configuration/        — best combination, all models, all artefacts
      csv_files/               — summary tables ready for paper
      npy_files/               — val predictions + confusion matrices per fold
      history_files/           — per-epoch training curves for every model × fold
      models/                  — .pt weights for every model × fold
      test_predictions/        — test-set predictions + confusion matrices
      figures/                 — all 14 figures (run vanilla_plus_figures.py)
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

LEARNING_RATES          = [0.001, 0.0005, 0.0001]
BATCH_SIZES             = [64, 128, 256]
EPOCHS                  = 100
EARLY_STOPPING_PATIENCE = 10
N_FOLDS                 = 10
RANDOM_STATE            = 42
SELECTION_METRIC        = "mcc"
NUM_CLASSES             = 8

DATA_DIR   = r"D:\zebfish\new_class\ml_ready_data_20percent"
OUTPUT_DIR = r"D:\zebfish\new_class\Results\vanilla_plus_results_full3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Flat top-level directories (legacy compatibility) ────────────────────────
NPY_DIR     = os.path.join(OUTPUT_DIR, "npy_files")
CSV_DIR     = os.path.join(OUTPUT_DIR, "csv_files")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history_files")
MODELS_DIR  = os.path.join(OUTPUT_DIR, "models")
TEST_DIR    = os.path.join(OUTPUT_DIR, "test_predictions")
for d in [NPY_DIR, CSV_DIR, HISTORY_DIR, MODELS_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Per-configuration directory trees ───────────────────────────────────────
ALL_CONFIGS_DIR = os.path.join(OUTPUT_DIR, "all_configurations")
BEST_CONFIG_DIR = os.path.join(OUTPUT_DIR, "best_configuration")
os.makedirs(ALL_CONFIGS_DIR, exist_ok=True)
os.makedirs(BEST_CONFIG_DIR, exist_ok=True)

BEST_NPY_DIR     = os.path.join(BEST_CONFIG_DIR, "npy_files")
BEST_CSV_DIR     = os.path.join(BEST_CONFIG_DIR, "csv_files")
BEST_HISTORY_DIR = os.path.join(BEST_CONFIG_DIR, "history_files")
BEST_MODELS_DIR  = os.path.join(BEST_CONFIG_DIR, "models")
BEST_TEST_DIR    = os.path.join(BEST_CONFIG_DIR, "test_predictions")
for d in [BEST_NPY_DIR, BEST_CSV_DIR, BEST_HISTORY_DIR, BEST_MODELS_DIR, BEST_TEST_DIR]:
    os.makedirs(d, exist_ok=True)

for lr in LEARNING_RATES:
    for bs in BATCH_SIZES:
        cfg = os.path.join(ALL_CONFIGS_DIR, f"LR_{lr}_BS_{bs}")
        for sub in ["npy_files", "csv_files", "history_files", "models", "test_predictions"]:
            os.makedirs(os.path.join(cfg, sub), exist_ok=True)

print(f"✅ All directories created under: {OUTPUT_DIR}")


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(RANDOM_STATE)
DEVICE = torch.device("cpu")


# =============================================================================
# 3. DATASET
# =============================================================================

class FishEmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        assert embeddings.shape[1] == 1024, "Expected 1024-dim embeddings"
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels,     dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# 4. MODEL DEFINITIONS
# =============================================================================

# ── 4a. Residual block (used by VanillaPlus and VanillaPlusAblation) ─────────

class ResidualBlock(nn.Module):
    """Two-layer residual block with optional GELU/ReLU activation."""
    def __init__(self, in_dim, out_dim, dropout=0.3, use_gelu=True):
        super().__init__()
        act = nn.GELU() if use_gelu else nn.ReLU()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            act,
            nn.Dropout(dropout),
        )
        self.skip = (nn.Linear(in_dim, out_dim, bias=False)
                     if in_dim != out_dim else nn.Identity())

    def forward(self, x):
        return self.block(x) + self.skip(x)


# ── 4b. Baseline 1: Vanilla MLP ──────────────────────────────────────────────

class VanillaMLP(nn.Module):
    """Simple MLP — no LayerNorm, no residual, no GELU."""
    def __init__(self, input_dim=1024, hidden_dims=None,
                 num_classes=NUM_CLASSES, dropout=0.3, **_):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 4c. Baseline 2: DNN Baseline ─────────────────────────────────────────────

class DNNBaseline(nn.Module):
    """Deeper DNN with LayerNorm — no residual connections."""
    def __init__(self, input_dim=1024, hidden_dims=None,
                 num_classes=NUM_CLASSES, dropout=0.3, **_):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 4d. Proposed: Vanilla+ ───────────────────────────────────────────────────

class VanillaPlus(nn.Module):
    """
    Vanilla+ — proposed model.
    Input projection → stack of residual blocks → classifier head.
    """
    def __init__(self, input_dim=1024, hidden_dims=None,
                 num_classes=NUM_CLASSES, dropout=0.3, use_gelu=True, **_):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        act = nn.GELU() if use_gelu else nn.ReLU()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            act,
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dims[i], hidden_dims[i + 1],
                          dropout=dropout, use_gelu=use_gelu)
            for i in range(len(hidden_dims) - 1)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1] // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.blocks(self.input_proj(x)))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 4e. Ablation variant ─────────────────────────────────────────────────────

class VanillaPlusAblation(nn.Module):
    """
    Ablation variant — each component of Vanilla+ can be toggled off.
    Toggles: residual connections, LayerNorm, BatchNorm, GELU/ReLU, dropout.
    """
    def __init__(self, input_dim=1024, hidden_dims=None,
                 num_classes=NUM_CLASSES, dropout=0.3,
                 use_residual=True, use_layernorm=True,
                 use_batchnorm=False, use_gelu=True, **_):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        def _norm(d):
            if use_layernorm:  return nn.LayerNorm(d)
            if use_batchnorm:  return nn.BatchNorm1d(d)
            return nn.Identity()

        act = nn.GELU() if use_gelu else nn.ReLU()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            _norm(hidden_dims[0]),
            act,
            nn.Dropout(dropout),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            if use_residual:
                blocks.append(ResidualBlock(hidden_dims[i], hidden_dims[i + 1],
                                            dropout=dropout, use_gelu=use_gelu))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    _norm(hidden_dims[i + 1]),
                    act,
                    nn.Dropout(dropout),
                ))
        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1] // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.blocks(self.input_proj(x)))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Ablation catalogue
ABLATION_VARIANTS = [
    ("Full_Model_VanillaPlus",
     dict(use_residual=True,  use_layernorm=True,  use_gelu=True,  dropout=0.3)),
    ("w_o_Residual",
     dict(use_residual=False, use_layernorm=True,  use_gelu=True,  dropout=0.3)),
    ("w_o_LayerNorm",
     dict(use_residual=True,  use_layernorm=False, use_gelu=True,  dropout=0.3)),
    ("ReLU_instead_of_GELU",
     dict(use_residual=True,  use_layernorm=True,  use_gelu=False, dropout=0.3)),
    ("w_o_Dropout",
     dict(use_residual=True,  use_layernorm=True,  use_gelu=True,  dropout=0.0)),
    ("Higher_Dropout_0.5",
     dict(use_residual=True,  use_layernorm=True,  use_gelu=True,  dropout=0.5)),
    ("Lower_Dropout_0.1",
     dict(use_residual=True,  use_layernorm=True,  use_gelu=True,  dropout=0.1)),
    ("BatchNorm_instead_of_LayerNorm",
     dict(use_residual=True,  use_layernorm=False, use_batchnorm=True,
          use_gelu=True, dropout=0.3)),
]


# =============================================================================
# 5. TRAINER
# =============================================================================

class Trainer:
    def __init__(self, model, optimizer, criterion,
                 scheduler=None, patience=EARLY_STOPPING_PATIENCE):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.patience  = patience

        # FIX-6: symmetric history — train_mcc and train_auc now included
        self.history = {
            "train_loss":      [], "val_loss":      [],
            "train_accuracy":  [], "val_accuracy":  [],
            "train_precision": [], "val_precision": [],
            "train_recall":    [], "val_recall":    [],
            "train_f1":        [], "val_f1":        [],
            "train_mcc":       [], "val_mcc":       [],   # FIX-6
            "train_auc":       [], "val_auc":       [],   # FIX-6
        }
        self.best_val_mcc = -1.0
        self.best_weights = None
        self._no_improve  = 0

    # ── single epoch ─────────────────────────────────────────────────────────
    def _epoch(self, loader, training):
        self.model.train(training)
        total_loss, all_labels, all_probs, all_preds = 0.0, [], [], []

        with torch.set_grad_enabled(training):
            for X_b, y_b in loader:
                logits = self.model(X_b)
                loss   = self.criterion(logits, y_b)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total_loss += loss.item() * len(y_b)
                probs = torch.softmax(logits, 1).detach().numpy()
                preds = np.argmax(probs, axis=1)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y_b.numpy())

        labels = np.array(all_labels)
        probs  = np.array(all_probs)
        preds  = np.array(all_preds)

        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average="macro", zero_division=0)
        rec  = recall_score(labels, preds,    average="macro", zero_division=0)
        f1   = f1_score(labels, preds,        average="macro", zero_division=0)
        mcc  = matthews_corrcoef(labels, preds)
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0

        return (total_loss / len(loader.dataset),
                acc, prec, rec, f1, mcc, auc,
                labels, probs, preds)

    # ── training loop ─────────────────────────────────────────────────────────
    def fit(self, train_loader, val_loader, epochs=EPOCHS, verbose=True):
        if verbose:
            print(f"\n  Params: {self.model.count_parameters():,} | Max epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            tr = self._epoch(train_loader, True)
            vl = self._epoch(val_loader,   False)

            if self.scheduler:
                self.scheduler.step(vl[0])

            # store all metrics symmetrically — FIX-6
            keys = ["loss", "accuracy", "precision", "recall", "f1", "mcc", "auc"]
            for i, k in enumerate(keys):
                self.history[f"train_{k}"].append(tr[i])
                self.history[f"val_{k}"].append(vl[i])

            if vl[5] > self.best_val_mcc:   # vl[5] = mcc
                self.best_val_mcc = vl[5]
                self.best_weights = {k: v.clone()
                                     for k, v in self.model.state_dict().items()}
                self._no_improve  = 0
                flag = " ✓"
            else:
                self._no_improve += 1
                flag = ""

            if verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"    Ep {epoch:03d} | TrLoss {tr[0]:.4f} | VlLoss {vl[0]:.4f} | "
                      f"VlAcc {vl[1]:.4f} | VlF1 {vl[4]:.4f} | VlMCC {vl[5]:.4f}{flag}")

            if self._no_improve >= self.patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}.")
                break

        if self.best_weights:
            self.model.load_state_dict(self.best_weights)

    # ── evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, loader):
        self.model.eval()
        loss, acc, prec, rec, f1, mcc, auc, labels, probs, preds = \
            self._epoch(loader, False)

        # FIX-5: per-class metrics and confusion matrix computed here
        cm       = confusion_matrix(labels, preds,
                                    labels=list(range(NUM_CLASSES)))
        cls_prec = precision_score(labels, preds, average=None, zero_division=0,
                                   labels=list(range(NUM_CLASSES)))
        cls_rec  = recall_score(labels, preds, average=None, zero_division=0,
                                labels=list(range(NUM_CLASSES)))
        cls_f1   = f1_score(labels, preds, average=None, zero_division=0,
                            labels=list(range(NUM_CLASSES)))
        return {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc_roc": auc, "mcc": mcc, "loss": loss,
            "labels": labels, "probs": probs, "preds": preds,
            # arrays needed for figure reproduction
            "confusion_matrix":    cm,
            "per_class_precision": cls_prec,
            "per_class_recall":    cls_rec,
            "per_class_f1":        cls_f1,
        }

    def get_history(self):
        return self.history


# =============================================================================
# 6. HELPERS
# =============================================================================

def make_loaders(X_tr, y_tr, X_te, y_te, batch_size=128):
    counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)
    sample_weights = (1.0 / counts)[y_tr]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)
    return (DataLoader(FishEmbeddingDataset(X_tr, y_tr),
                       batch_size=batch_size, sampler=sampler),
            DataLoader(FishEmbeddingDataset(X_te, y_te),
                       batch_size=batch_size, shuffle=False))


def _save_npy_bundle(out_dir, prefix, preds, probs, labels,
                     cm=None, cls_prec=None, cls_rec=None, cls_f1=None,
                     fold=None, data_type="val"):
    """
    FIX-5: central helper — saves every array needed to regenerate any figure.
    Naming: {prefix}_{data_type}[_fold{N}]_{array}.npy
    """
    tag = f"{prefix}_{data_type}" + (f"_fold{fold}" if fold is not None else "")
    np.save(os.path.join(out_dir, f"{tag}_predictions.npy"),   preds)
    np.save(os.path.join(out_dir, f"{tag}_probabilities.npy"), probs)
    np.save(os.path.join(out_dir, f"{tag}_true_labels.npy"),   labels)
    if cm       is not None:
        np.save(os.path.join(out_dir, f"{tag}_confusion_matrix.npy"),    cm)
    if cls_prec is not None:
        np.save(os.path.join(out_dir, f"{tag}_per_class_precision.npy"), cls_prec)
    if cls_rec  is not None:
        np.save(os.path.join(out_dir, f"{tag}_per_class_recall.npy"),    cls_rec)
    if cls_f1   is not None:
        np.save(os.path.join(out_dir, f"{tag}_per_class_f1.npy"),        cls_f1)


def _save_history(hist_dir, prefix, history, fold=None):
    """Save complete per-epoch history dict as .npy."""
    tag = f"{prefix}" + (f"_fold{fold}" if fold is not None else "_final")
    np.save(os.path.join(hist_dir, f"{tag}_history.npy"), history)


def _save_model(models_dir, prefix, model, fold=None):
    """Save model state_dict for every fold."""
    tag = f"{prefix}" + (f"_fold{fold}" if fold is not None else "_final")
    torch.save(model.state_dict(),
               os.path.join(models_dir, f"{tag}.pt"))


# FIX-1 + FIX-2 + FIX-4 + FIX-5:
#   - accepts lr parameter (FIX-2)
#   - returns trained model as third value (FIX-4)
#   - saves complete npy bundle (FIX-5)
def train_model_with_config(model_class, X_tr, y_tr, X_te, y_te,
                             epochs=EPOCHS, batch_size=128, lr=0.001,
                             verbose=True, model_name="model",
                             fold=None, output_dirs=None, **model_kwargs):
    """
    Train one model, evaluate on held-out set, save all artefacts.
    Returns (metrics_dict, history_dict, trained_model).
    """
    tr_loader, te_loader = make_loaders(X_tr, y_tr, X_te, y_te, batch_size)

    model     = model_class(**model_kwargs).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # FIX-2
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=7)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion, scheduler)
    trainer.fit(tr_loader, te_loader, epochs=epochs, verbose=verbose)
    metrics = trainer.evaluate(te_loader)
    history = trainer.get_history()

    if output_dirs is not None:
        _save_model(output_dirs["models"], model_name, model, fold)
        _save_history(output_dirs["history"], model_name, history, fold)
        _save_npy_bundle(
            output_dirs["npy"], model_name,
            metrics["preds"], metrics["probs"], metrics["labels"],
            cm=metrics["confusion_matrix"],
            cls_prec=metrics["per_class_precision"],
            cls_rec=metrics["per_class_recall"],
            cls_f1=metrics["per_class_f1"],
            fold=fold, data_type="val",
        )

    return metrics, history, model   # FIX-4: return trained model


# =============================================================================
# 7. GRID SEARCH   (FIX-1 + FIX-2)
# =============================================================================

def run_grid_search(X_train, y_train, X_test, y_test):
    print(f"\n{'='*60}")
    print("  HYPERPARAMETER GRID SEARCH — ALL CONFIGURATIONS")
    print(f"{'='*60}")

    all_results  = []
    best_score   = -1.0
    best_params  = {}
    best_cfg_dir = None

    for lr in LEARNING_RATES:
        for bs in BATCH_SIZES:
            print(f"\n  ▶ Testing  LR={lr}  BS={bs}")

            cfg_root    = os.path.join(ALL_CONFIGS_DIR, f"LR_{lr}_BS_{bs}")
            output_dirs = {
                "npy":     os.path.join(cfg_root, "npy_files"),
                "csv":     os.path.join(cfg_root, "csv_files"),
                "history": os.path.join(cfg_root, "history_files"),
                "models":  os.path.join(cfg_root, "models"),
                "test":    os.path.join(cfg_root, "test_predictions"),
            }

            skf       = StratifiedKFold(n_splits=10, shuffle=True,
                                        random_state=RANDOM_STATE)
            cv_scores = []
            fold_rows = []

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                m, _, _ = train_model_with_config(
                    VanillaPlus, X_tr, y_tr, X_val, y_val,
                    epochs=EPOCHS // 2, batch_size=bs, lr=lr,  # FIX-2
                    verbose=False,
                    model_name=f"VanillaPlus_LR{lr}_BS{bs}",
                    fold=fold, output_dirs=output_dirs,
                    num_classes=NUM_CLASSES,
                )
                cv_scores.append(m["mcc"])
                fold_rows.append({**m, "fold": fold,
                                  "learning_rate": lr, "batch_size": bs})

            mean_mcc = float(np.mean(cv_scores))
            std_mcc  = float(np.std(cv_scores))
            print(f"    CV MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")

            pd.DataFrame(fold_rows).to_csv(
                os.path.join(output_dirs["csv"], "grid_cv_fold_metrics.csv"),
                index=False)

            # ── FIX-1 + FIX-4: train final model, keep reference ──────────
            print("    Testing on unseen species…")
            m_final, _, trained_model = train_model_with_config(
                VanillaPlus, X_train, y_train, X_test, y_test,
                epochs=EPOCHS, batch_size=bs, lr=lr,           # FIX-2
                verbose=False,
                model_name=f"VanillaPlus_LR{lr}_BS{bs}_final",
                fold=None, output_dirs=output_dirs,
                num_classes=NUM_CLASSES,
            )

            # FIX-1: use TRAINED model for test bundle
            _save_npy_bundle(
                output_dirs["test"],
                f"VanillaPlus_LR{lr}_BS{bs}",
                m_final["preds"], m_final["probs"], m_final["labels"],
                cm=m_final["confusion_matrix"],
                cls_prec=m_final["per_class_precision"],
                cls_rec=m_final["per_class_recall"],
                cls_f1=m_final["per_class_f1"],
                data_type="test",
            )

            cfg_summary = {
                "learning_rate":  lr,
                "batch_size":     bs,
                "cv_mean_mcc":    mean_mcc,
                "cv_std_mcc":     std_mcc,
                "test_mcc":       m_final["mcc"],
                "test_f1":        m_final["f1"],
                "test_accuracy":  m_final["accuracy"],
                "test_precision": m_final["precision"],
                "test_recall":    m_final["recall"],
                "test_auc":       m_final["auc_roc"],
            }
            all_results.append(cfg_summary)
            pd.DataFrame([cfg_summary]).to_csv(
                os.path.join(output_dirs["csv"], "config_summary.csv"), index=False)

            if mean_mcc > best_score:
                best_score   = mean_mcc
                best_params  = {"lr": lr, "bs": bs}
                best_cfg_dir = cfg_root

    pd.DataFrame(all_results).to_csv(
        os.path.join(CSV_DIR, "full_grid_search_results.csv"), index=False)

    print(f"\n  Best: LR={best_params['lr']}  BS={best_params['bs']}  "
          f"(CV MCC={best_score:.4f})")
    return best_params, best_cfg_dir


# =============================================================================
# 8. BEST CONFIGURATION — ALL MODELS + ABLATIONS  (all FIX applied)
# =============================================================================

def _run_cv_for_model(model_class, model_name, X_train, y_train, skf,
                      batch_size, lr, output_dirs, **model_kwargs):
    """
    10-fold CV for any model; returns (aggregate cv_metrics, per-fold list).
    FIX-5: saves history + full npy bundle for every fold.
    """
    metric_keys = ["mcc", "accuracy", "f1", "precision", "recall", "auc_roc"]
    accumulator = {k: [] for k in metric_keys}
    cv_rows     = []

    print(f"\n    Running {N_FOLDS}-fold CV for {model_name}…")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        m, _, _ = train_model_with_config(
            model_class, X_tr, y_tr, X_val, y_val,
            epochs=EPOCHS, batch_size=batch_size, lr=lr,
            verbose=False,
            model_name=f"{model_name}_best",
            fold=fold, output_dirs=output_dirs,
            **model_kwargs,
        )
        for k in metric_keys:
            accumulator[k].append(m[k])

        cv_rows.append({"fold": fold, **{k: m[k] for k in metric_keys}})
        print(f"      Fold {fold:2d}: Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
              f"MCC={m['mcc']:.4f}  AUC={m['auc_roc']:.4f}")

    cv_metrics = {}
    for k in metric_keys:
        cv_metrics[f"cv_mean_{k}"] = float(np.mean(accumulator[k]))
        cv_metrics[f"cv_std_{k}"]  = float(np.std(accumulator[k]))

    return cv_metrics, cv_rows


def run_best_configuration(X_train, y_train, X_test, y_test,
                           best_params, best_cfg_dir):
    print(f"\n{'='*60}")
    print(f"  BEST CONFIG: LR={best_params['lr']}  BS={best_params['bs']}")
    print(f"  Running ALL models with best hyperparameters")
    print(f"{'='*60}")

    lr = best_params["lr"]
    bs = best_params["bs"]

    output_dirs = {
        "npy":     BEST_NPY_DIR,
        "csv":     BEST_CSV_DIR,
        "history": BEST_HISTORY_DIR,
        "models":  BEST_MODELS_DIR,
        "test":    BEST_TEST_DIR,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_summary_rows = []

    # ── helper: train full model on all train data → test set, save bundle ──
    def _train_final(model_class, model_name, verbose=True, **model_kwargs):
        print(f"\n  ▶ Final training (full train→test): {model_name}")
        m, _, trained = train_model_with_config(
            model_class, X_train, y_train, X_test, y_test,
            epochs=EPOCHS, batch_size=bs, lr=lr,
            verbose=verbose,
            model_name=f"{model_name}_final",
            fold=None, output_dirs=output_dirs,
            **model_kwargs,
        )
        # FIX-1: use TRAINED model weights for test bundle
        _save_npy_bundle(
            output_dirs["test"], model_name,
            m["preds"], m["probs"], m["labels"],
            cm=m["confusion_matrix"],
            cls_prec=m["per_class_precision"],
            cls_rec=m["per_class_recall"],
            cls_f1=m["per_class_f1"],
            data_type="test",
        )
        return m

    # =========================================================================
    # A. Logistic Regression
    # =========================================================================
    print("\n  ── Logistic Regression ──────────────────────────────────────")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                              multi_class="multinomial", class_weight="balanced",
                              random_state=RANDOM_STATE)
    clf.fit(X_train_sc, y_train)
    test_probs = clf.predict_proba(X_test_sc)
    test_preds = np.argmax(test_probs, axis=1)

    lr_test = {
        "accuracy":  accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds, average="macro", zero_division=0),
        "recall":    recall_score(y_test, test_preds,    average="macro", zero_division=0),
        "f1":        f1_score(y_test, test_preds,        average="macro", zero_division=0),
        "mcc":       matthews_corrcoef(y_test, test_preds),
        "auc_roc":   roc_auc_score(y_test, test_probs,   multi_class="ovr", average="macro"),
    }
    _save_npy_bundle(
        output_dirs["test"], "LogisticRegression",
        test_preds, test_probs, y_test,
        cm=confusion_matrix(y_test, test_preds, labels=list(range(NUM_CLASSES))),
        cls_prec=precision_score(y_test, test_preds, average=None, zero_division=0,
                                 labels=list(range(NUM_CLASSES))),
        cls_rec=recall_score(y_test, test_preds, average=None, zero_division=0,
                             labels=list(range(NUM_CLASSES))),
        cls_f1=f1_score(y_test, test_preds, average=None, zero_division=0,
                        labels=list(range(NUM_CLASSES))),
        data_type="test",
    )

    # 10-fold CV for LR
    print(f"    Running {N_FOLDS}-fold CV for Logistic Regression…")
    lr_cv_acc    = {k: [] for k in
                    ["mcc", "accuracy", "f1", "precision", "recall", "auc_roc"]}
    lr_fold_rows = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        sc_f     = StandardScaler()
        X_tr_sc  = sc_f.fit_transform(X_train[tr_idx])
        X_val_sc = sc_f.transform(X_train[val_idx])
        y_tr_f, y_val_f = y_train[tr_idx], y_train[val_idx]

        clf_f = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                   multi_class="multinomial", class_weight="balanced",
                                   random_state=RANDOM_STATE)
        clf_f.fit(X_tr_sc, y_tr_f)
        vp = clf_f.predict_proba(X_val_sc)
        vd = np.argmax(vp, axis=1)

        fm = {
            "accuracy":  accuracy_score(y_val_f, vd),
            "precision": precision_score(y_val_f, vd, average="macro", zero_division=0),
            "recall":    recall_score(y_val_f, vd,    average="macro", zero_division=0),
            "f1":        f1_score(y_val_f, vd,        average="macro", zero_division=0),
            "mcc":       matthews_corrcoef(y_val_f, vd),
        }
        try:
            fm["auc_roc"] = roc_auc_score(y_val_f, vp, multi_class="ovr", average="macro")
        except Exception:
            fm["auc_roc"] = 0.0

        for k in lr_cv_acc:
            lr_cv_acc[k].append(fm[k])

        # FIX-5: save full npy bundle for LR folds
        fold_cm = confusion_matrix(y_val_f, vd, labels=list(range(NUM_CLASSES)))
        _save_npy_bundle(
            output_dirs["npy"], "LogisticRegression",
            vd, vp, y_val_f,
            cm=fold_cm,
            cls_prec=precision_score(y_val_f, vd, average=None, zero_division=0,
                                     labels=list(range(NUM_CLASSES))),
            cls_rec=recall_score(y_val_f, vd, average=None, zero_division=0,
                                 labels=list(range(NUM_CLASSES))),
            cls_f1=f1_score(y_val_f, vd, average=None, zero_division=0,
                            labels=list(range(NUM_CLASSES))),
            fold=fold, data_type="val",
        )
        # FIX-6: history dict for LR (single-entry, uniform format)
        lr_hist = {f"val_{k}": [v] for k, v in fm.items()}
        lr_hist.update({f"train_{k}": [v] for k, v in fm.items()})
        np.save(os.path.join(output_dirs["history"],
                             f"LogisticRegression_best_fold{fold}_history.npy"), lr_hist)

        lr_fold_rows.append({"fold": fold, **fm})
        print(f"      Fold {fold:2d}: Acc={fm['accuracy']:.4f}  F1={fm['f1']:.4f}  "
              f"MCC={fm['mcc']:.4f}  AUC={fm['auc_roc']:.4f}")

    lr_cv = {f"cv_mean_{k}": float(np.mean(v)) for k, v in lr_cv_acc.items()}
    lr_cv.update({f"cv_std_{k}":  float(np.std(v))  for k, v in lr_cv_acc.items()})

    # FIX-3: store as named variable, not reused 'm'
    lr_metrics = {**lr_test, **lr_cv}
    pd.DataFrame([lr_metrics]).to_csv(
        os.path.join(output_dirs["csv"], "LogisticRegression_best_config.csv"),
        index=False)
    pd.DataFrame(lr_fold_rows).to_csv(
        os.path.join(output_dirs["csv"], "LogisticRegression_fold_summary.csv"),
        index=False)
    print(f"    Test F1={lr_test['f1']:.4f}  MCC={lr_test['mcc']:.4f} | "
          f"CV MCC={lr_cv['cv_mean_mcc']:.4f}±{lr_cv['cv_std_mcc']:.4f}")
    all_summary_rows.append({"Model": "Logistic Regression", **lr_metrics})

    # =========================================================================
    # B. Vanilla MLP  — FIX-3: result stored in named variable m_vmlp
    # =========================================================================
    print("\n  ── Vanilla MLP ──────────────────────────────────────────────")
    m_vmlp = _train_final(VanillaMLP, "VanillaMLP", num_classes=NUM_CLASSES)
    cv_vmlp, vmlp_folds = _run_cv_for_model(
        VanillaMLP, "VanillaMLP", X_train, y_train, skf,
        bs, lr, output_dirs, num_classes=NUM_CLASSES)
    vmlp_metrics = {**m_vmlp, **cv_vmlp}
    pd.DataFrame([vmlp_metrics]).to_csv(
        os.path.join(output_dirs["csv"], "VanillaMLP_best_config.csv"), index=False)
    pd.DataFrame(vmlp_folds).to_csv(
        os.path.join(output_dirs["csv"], "VanillaMLP_fold_summary.csv"), index=False)
    print(f"    Test F1={m_vmlp['f1']:.4f}  MCC={m_vmlp['mcc']:.4f} | "
          f"CV MCC={cv_vmlp['cv_mean_mcc']:.4f}±{cv_vmlp['cv_std_mcc']:.4f}")
    all_summary_rows.append({"Model": "Vanilla MLP", **vmlp_metrics})

    # =========================================================================
    # C. DNN Baseline  — FIX-3: result stored in named variable m_dnn
    # =========================================================================
    print("\n  ── DNN Baseline ─────────────────────────────────────────────")
    m_dnn = _train_final(DNNBaseline, "DNNBaseline", num_classes=NUM_CLASSES)
    cv_dnn, dnn_folds = _run_cv_for_model(
        DNNBaseline, "DNNBaseline", X_train, y_train, skf,
        bs, lr, output_dirs, num_classes=NUM_CLASSES)
    dnn_metrics = {**m_dnn, **cv_dnn}
    pd.DataFrame([dnn_metrics]).to_csv(
        os.path.join(output_dirs["csv"], "DNNBaseline_best_config.csv"), index=False)
    pd.DataFrame(dnn_folds).to_csv(
        os.path.join(output_dirs["csv"], "DNNBaseline_fold_summary.csv"), index=False)
    print(f"    Test F1={m_dnn['f1']:.4f}  MCC={m_dnn['mcc']:.4f} | "
          f"CV MCC={cv_dnn['cv_mean_mcc']:.4f}±{cv_dnn['cv_std_mcc']:.4f}")
    all_summary_rows.append({"Model": "DNN Baseline", **dnn_metrics})

    # =========================================================================
    # D. Vanilla+  — FIX-3: result stored in named variable m_vp
    # =========================================================================
    print("\n  ── Vanilla+ (Proposed) ──────────────────────────────────────")
    m_vp = _train_final(VanillaPlus, "VanillaPlus",
                         num_classes=NUM_CLASSES, use_gelu=True)
    cv_vp, vp_folds = _run_cv_for_model(
        VanillaPlus, "VanillaPlus", X_train, y_train, skf,
        bs, lr, output_dirs, num_classes=NUM_CLASSES, use_gelu=True)
    vp_metrics = {**m_vp, **cv_vp}
    pd.DataFrame([vp_metrics]).to_csv(
        os.path.join(output_dirs["csv"], "VanillaPlus_best_config.csv"), index=False)
    pd.DataFrame(vp_folds).to_csv(
        os.path.join(output_dirs["csv"], "VanillaPlus_fold_summary.csv"), index=False)
    print(f"    Test F1={m_vp['f1']:.4f}  MCC={m_vp['mcc']:.4f} | "
          f"CV MCC={cv_vp['cv_mean_mcc']:.4f}±{cv_vp['cv_std_mcc']:.4f}")
    all_summary_rows.append({"Model": "Vanilla+ (Ours)", **vp_metrics})

    # =========================================================================
    # E. Ablation Study
    # =========================================================================
    print("\n  ── Ablation Study ───────────────────────────────────────────")
    abl_all_folds = []

    for label, kwargs in ABLATION_VARIANTS:
        print(f"\n    Variant: {label}")
        m_abl = _train_final(VanillaPlusAblation, f"ablation_{label}",
                              verbose=False,
                              num_classes=NUM_CLASSES, **kwargs)
        cv_abl, abl_folds = _run_cv_for_model(
            VanillaPlusAblation, f"ablation_{label}",
            X_train, y_train, skf,
            bs, lr, output_dirs, num_classes=NUM_CLASSES, **kwargs)
        abl_metrics = {**m_abl, **cv_abl}

        pd.DataFrame([abl_metrics]).to_csv(
            os.path.join(output_dirs["csv"], f"ablation_{label}_best_config.csv"),
            index=False)
        for r in abl_folds:
            r["variant"] = label
        abl_all_folds.extend(abl_folds)
        pd.DataFrame(abl_folds).to_csv(
            os.path.join(output_dirs["csv"], f"ablation_{label}_fold_summary.csv"),
            index=False)
        print(f"      Test F1={m_abl['f1']:.4f}  MCC={m_abl['mcc']:.4f} | "
              f"CV MCC={cv_abl['cv_mean_mcc']:.4f}±{cv_abl['cv_std_mcc']:.4f}")
        all_summary_rows.append({"Model": f"Ablation_{label}", **abl_metrics})

    pd.DataFrame(abl_all_folds).to_csv(
        os.path.join(output_dirs["csv"], "ablation_all_folds_combined.csv"), index=False)

    # =========================================================================
    # F. Final comprehensive CSV outputs
    # =========================================================================
    final_df = pd.DataFrame(all_summary_rows)

    # drop un-serialisable array columns
    drop_cols = ["labels", "probs", "preds", "confusion_matrix",
                 "per_class_precision", "per_class_recall", "per_class_f1"]
    final_df  = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])

    final_df.to_csv(
        os.path.join(output_dirs["csv"], "final_comprehensive_summary.csv"), index=False)

    paper_cols = [
        "Model",
        "accuracy", "f1", "mcc", "auc_roc",
        "cv_mean_accuracy", "cv_std_accuracy",
        "cv_mean_f1",       "cv_std_f1",
        "cv_mean_mcc",      "cv_std_mcc",
        "cv_mean_auc_roc",  "cv_std_auc_roc",
    ]
    paper_df = final_df[[c for c in paper_cols if c in final_df.columns]]
    paper_df.to_csv(
        os.path.join(output_dirs["csv"], "paper_results_table.csv"), index=False)

    print(f"\n  ✅ All results saved to: {output_dirs['csv']}")
    return {
        "logistic":    lr_metrics,
        "vanillamlp":  vmlp_metrics,
        "dnnbaseline": dnn_metrics,
        "vanillaplus": vp_metrics,
    }


# =============================================================================
# 9. LOAD DATA
# =============================================================================

def load_data():
    print(f"\n📁 Loading data from: {DATA_DIR}")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}  classes: {np.bincount(y_train, minlength=NUM_CLASSES)}")
    print(f"  y_test  : {y_test.shape}   classes: {np.bincount(y_test,  minlength=NUM_CLASSES)}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  VANILLA+: ENHANCED MLP FOR FISH ENZYME CLASSIFICATION")
    print("  FULL REPRODUCIBILITY — ALL CONFIGURATIONS SAVED")
    print("=" * 70)
    print(f"\n  CV Folds        : {N_FOLDS}")
    print(f"  Selection metric: {SELECTION_METRIC}")
    print(f"  All configs     : {ALL_CONFIGS_DIR}")
    print(f"  Best config     : {BEST_CONFIG_DIR}")

    X_train, X_test, y_train, y_test = load_data()

    train_df      = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_df       = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    train_species = set(train_df["Organism"].unique())
    test_species  = set(test_df["Organism"].unique())
    overlap       = train_species & test_species
    print(f"\n  Training species : {len(train_species)}")
    print(f"  Test species     : {len(test_species)}")
    print(f"  Overlap          : {len(overlap)} "
          f"({'✅ No overlap!' if not overlap else '⚠️  Overlap found!'})")

    best_params, best_cfg_dir = run_grid_search(X_train, y_train, X_test, y_test)

    best_results = run_best_configuration(
        X_train, y_train, X_test, y_test, best_params, best_cfg_dir)

    print("\n" + "=" * 70)
    print("  ✅ ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\n  Best params: LR={best_params['lr']}  BS={best_params['bs']}")
    print(f"\n  📁 best_configuration/")
    print(f"     csv_files/")
    print(f"       ├─ final_comprehensive_summary.csv")
    print(f"       ├─ paper_results_table.csv")
    print(f"       ├─ *_fold_summary.csv")
    print(f"       └─ *_best_config.csv")
    print(f"     npy_files/          ← val predictions + CM + per-class arrays")
    print(f"     history_files/      ← full epoch curves (all 14 metrics)")
    print(f"     models/             ← .pt weights per fold + final")
    print(f"     test_predictions/   ← test arrays + CM + per-class arrays")
    print(f"\n  Run vanilla_plus_figures.py to regenerate all 14 figures.")
    print("=" * 70)
    return {"best_params": best_params, "best_results": best_results}


if __name__ == "__main__":
    results = main()