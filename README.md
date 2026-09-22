# Vanilla+: A Species-Aware Deep Learning Framework for Protein Embedding-Based Enzyme Classification Across Fish Diversity

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Bioinformatics](https://img.shields.io/badge/Field-Bioinformatics-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

Vanilla+ is a novel deep learning framework developed for **species-aware multi-class enzyme classification in fish** using protein language model-derived embeddings.

The framework integrates:

- UniProt-reviewed fish protein sequences
- ProtT5 protein language model embeddings
- Species-aware evaluation strategy
- Residual deep neural network architecture
- Extensive baseline comparison
- Ablation experiments
- Statistical validation

The objective of this study is to improve enzyme function classification by incorporating evolutionary diversity among fish species while avoiding information leakage between closely related species.

---

# Key Features

## 1. Biological Dataset Construction

Protein sequences were collected from the UniProt Knowledgebase using reviewed Swiss-Prot entries.

The dataset contains:

- Fish proteins from Actinopterygii (ray-finned fishes)
- Multiple fish species
- Enzyme and non-enzyme protein classes
- 1024-dimensional ProtT5 protein embeddings

Protein sequences were classified into eight functional categories:

| Class | Function |
|---|---|
| 0 | Non-enzyme |
| 1 | Oxidoreductases |
| 2 | Transferases |
| 3 | Hydrolases |
| 4 | Lyases |
| 5 | Isomerases |
| 6 | Ligases |
| 7 | Translocases |

---

# Methodological Workflow


UniProt Fish Proteins
|
|
v
Protein Sequence Collection
|
|
v
ProtT5 Protein Embeddings
(1024 dimensions)
|
|
v
Species-Aware Data Split
|
|
v
Vanilla+ Deep Learning Model
|
|
v
Performance Evaluation
|
|
v
Statistical Validation


---

# Model Architecture

## Vanilla+

Vanilla+ is an enhanced multilayer perceptron architecture designed for protein embedding classification.

The architecture incorporates:

- Residual learning blocks
- Layer normalization
- GELU activation
- Dropout regularization
- Optimized training strategy


The proposed model was compared with:

### Baseline Models

1. Logistic Regression
2. Vanilla MLP
3. DNN Baseline

### Vanilla+ Ablation Models

- Without residual connection
- Without LayerNorm
- Without dropout
- Different dropout rates
- ReLU instead of GELU
- BatchNorm instead of LayerNorm

---

# Repository Structure


VanillaPlus/
│
├── data/
│ └── Fish protein datasets
│
├── scripts/
│
│ ├── 01_fish_uniprot_analysis.py
│ │ Dataset preparation and enzyme classification
│ │
│ ├── 02_fish_species_distribution_analysis.py
│ │ Species distribution analysis
│ │
│ ├── 03_species_aware_split.py
│ │ Evolutionary-aware train/test separation
│ │
│ ├── 04_vanilla_plus_model.py
│ │ Deep learning model training
│ │
│ ├── 05_results_aggregation.py
│ │ Combine model results
│ │
│ ├── 06_statistical_analysis.py
│ │ Statistical comparison of models
│ │
│ ├── 07_article_tables.py
│ │ Generate manuscript tables
│ │
│ └── 08_generate_figures.py
│ Generate publication-quality figures
│
├── results/
│
├── figures/
│
├── requirements.txt
│
└── README.md


---

# Installation

Clone this repository:

```bash
git clone https://github.com/username/VanillaPlus.git

cd VanillaPlus

Install dependencies:

pip install -r requirements.txt
Requirements

Main packages:

python >= 3.9

numpy
pandas
scikit-learn
pytorch
torchvision
h5py
matplotlib
seaborn
scipy
Data Preparation
Step 1: Analyze UniProt Fish Proteins
python 01_fish_uniprot_analysis.py

This step:

Reads UniProt fish protein files
Assigns enzyme classes
Generates class distribution summaries
Step 2: Species Distribution Analysis
python 02_fish_species_distribution_analysis.py

This evaluates:

Number of fish species
Protein distribution among species
Major represented fish groups
Step 3: Species-Aware Data Splitting
python 03_species_aware_split.py

The dataset is divided while maintaining species independence.

Purpose:

Prevent evolutionary information leakage
Evaluate model generalization across fish diversity
Model Training

Run Vanilla+ training:

python 04_vanilla_plus_model.py

The training pipeline includes:

10-fold cross validation
Hyperparameter optimization
Multiple baseline comparisons
Ablation experiments
Model checkpoint saving
Evaluation

Generate combined results:

python 05_results_aggregation.py

The following metrics are calculated:

Accuracy
Precision
Recall
F1-score
Matthews Correlation Coefficient (MCC)
Area Under ROC Curve (AUC)
Statistical Analysis

Run:

python 06_statistical_analysis.py

Statistical evaluation includes:

Friedman test
Pairwise comparisons
Fold-level performance analysis
Comparison among all models
Figure Generation

Generate manuscript figures:

python 08_generate_figures.py

Generated figures include:

Model performance comparison
Training curves
Hyperparameter analysis
MCC ranking
ROC curves
Ablation study visualization
Reproducibility

Random seeds are fixed to ensure reproducible experiments.

The repository saves:

Model weights
Training history
Cross-validation results
Predictions
Confusion matrices
Statistical outputs
Citation

If you use this framework, please cite:

Author(s).

Vanilla+: A Species-Aware Deep Learning Framework 
for Protein Embedding-Based Enzyme Classification 
Across Fish Diversity.

Submitted manuscript.
Contact

For questions regarding the implementation or collaboration:

Author:
Anjum Shahzad

Email:
your.email@example.com

License

This project is released under the MIT License.


---

One recommendation before uploading to GitHub: **rename the repository** from `VanillaPlus` to something more discoverable, for example:

**`VanillaPlus-Fish-Enzyme-Classification`**

or

**`SpeciesAware-VanillaPlus-Bioinformatics`**

because reviewers and future researchers searching GitHub will find it more easily.
