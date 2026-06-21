# 🧬 Fish Enzyme Classification using Enhanced MLP Framework (Vanilla+)

## 📌 Overview

This repository contains a **complete bioinformatics deep learning pipeline** for fish enzyme classification using **UniProt protein embeddings (1024-dimensional)**.

The study proposes an **Enhanced MLP-based framework (Vanilla+)** and evaluates it using species-aware biological validation, cross-validation, statistical testing, and publication-ready figures.

---

# 📄 Research Article

## 🧠 Title:
**Vanilla+: A Reproducible Enhanced MLP Framework for Fish Enzyme Classification**

---

# 🚀 Key Contribution

- Enhanced Multi-Layer Perceptron (MLP) architecture
- Residual learning improvements
- Advanced normalization strategies (BatchNorm / LayerNorm)
- Activation function optimization (ReLU / GELU)
- Dropout regularization study
- Strong baseline benchmarking for enzyme classification

---

# 🧬 Dataset

- Source: UniProt fish proteome dataset
- Input: 1024-dimensional protein embeddings (HDF5 format)
- Task: 8-class enzyme classification

### Classes:
- 0 → Non-enzyme  
- 1–7 → EC enzyme classes  

---

# ⚙️ Pipeline Overview (UNCHANGED FROM ORIGINAL FRAMEWORK)

```
01 → UniProt dataset analysis
02 → Species distribution analysis
03 → Species-aware train/test split (NO leakage)
04 → Vanilla+ model training (THIS REPO CHANGE ONLY HERE)
05 → Results aggregation
06 → Statistical significance testing
07 → Visualization dashboard
08 → Publication figure generation
``` id="m4xv1a"

---

# 📂 Repository Structure

```
Enzyme-MLP-Enhanced/
│
├── 01_fish_uniprot_analysis.py
│   → EC classification + dataset exploration
│
├── 02_fish_species_distribution_analysis.py
│   → Biological species profiling
│
├── 03_species_aware_split.py
│   → Train/test split with species separation
│
├── 04_vanilla_plus_model.py   ⭐ (ONLY CHANGED FILE)
│   → Enhanced MLP architecture (Vanilla+)
│
├── 05_results_aggregation.py
│   → Combines CV + grid search results
│
├── 06_statistical_analysis.py
│   → Friedman, Wilcoxon, Mann–Whitney tests
│
├── 07_results_visualization_viewer.py
│   → Full results dashboard
│
├── 08_generate_figures.py
│   → Publication-ready figures generator
│
├── data/
│   ├── UniProt TSV files
│   ├── EC classification CSV
│   ├── protein_embeddings.h5
│
├── results/
│   ├── cv_results/
│   ├── grid_search/
│   ├── best_model/
│   ├── ablation/
│
└── figures/
    ├── PNG/
    └── TIFF/
``` id="z3q7tc"

---

# 🔄 Experimental Setup

### ✔ Cross-validation
- 10-fold stratified CV

### ✔ Evaluation metrics
- Accuracy
- Precision
- Recall
- F1-score
- MCC (primary metric)
- ROC-AUC

---

# 🧪 Statistical Validation

- ✔ Friedman Test → overall model comparison
- ✔ Wilcoxon Signed-Rank Test → pairwise comparison
- ✔ Mann–Whitney U Test → distribution comparison

---

# 📊 Outputs

### Figures:
1. Model performance comparison  
2. Training curves  
3. CV stability plots  
4. Grid search heatmaps  
5. Model ranking  
6. ROC curves  
7. Analysis plots  

---

# 🧠 Scientific Purpose

This study focuses on:

- Strong deep learning baseline (MLP enhancement)
- Biological enzyme classification
- Robust evaluation using unseen species
- Statistical validation of performance improvements

---

# 🏆 Highlights

✔ Species-aware dataset split  
✔ No data leakage evaluation  
✔ Full reproducibility pipeline  
✔ Strong baseline benchmarking study  
✔ Publication-ready figures (300 DPI TIFF/PNG)  

- Author: [Your Name]
- Email: [Add email]
```
