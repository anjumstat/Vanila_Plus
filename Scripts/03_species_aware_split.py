# -*- coding: utf-8 -*-
"""
Species-Aware Split with Exact 20% Test Proteins
"""

import os
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle

# =============================================
# CONFIGURATION
# =============================================

data_dir = r"D:\zebfish\new_class\data"
embedding_file = os.path.join(data_dir, "3WXNkJ1rdz")
tsv_file = os.path.join(data_dir, "uniprotkb_taxonomy_id_7898_AND_reviewed_2026_06_13.tsv")
output_dir = r"D:\zebfish\new_class\ml_ready_data_20percent"
os.makedirs(output_dir, exist_ok=True)

# =============================================
# LOAD DATA
# =============================================

print("=" * 70)
print("FISH DATA - SPECIES-AWARE SPLIT (EXACT 20% TEST PROTEINS)")
print("8-Class Classification: 0 (Non-enzyme) + 1-7 (EC Classes)")
print("=" * 70)

# Load classification file
df_class = pd.read_csv(os.path.join(data_dir, 'fish_8_class_classification.csv'))
print(f"\n✅ Loaded classification: {len(df_class):,} proteins")

# Load TSV for species
df_tsv = pd.read_csv(tsv_file, sep='\t')
df = df_class.merge(df_tsv[['Entry', 'Organism']], on='Entry', how='left')
df = df.dropna(subset=['Organism'])
print(f"✅ Merged data: {len(df):,} proteins")

# =============================================
# SPECIES-AWARE SPLIT WITH EXACT 20% TEST PROTEINS
# =============================================

print("\n" + "=" * 70)
print("SPECIES-AWARE SPLIT (Target: 20% Test Proteins)")
print("=" * 70)

# Get species and their protein counts
species_counts = df['Organism'].value_counts()
print(f"\nTotal species: {len(species_counts)}")
print(f"Total proteins: {len(df):,}")

# Sort species by protein count (largest first)
species_sorted = species_counts.index.tolist()
species_sorted = sorted(species_sorted, key=lambda x: species_counts[x], reverse=True)

# Find species to include in test to reach ~20% proteins
test_proteins_target = int(len(df) * 0.20)
test_species_list = []
test_proteins_count = 0

# Start with medium and small species (not the largest ones)
# Skip the largest species (Zebrafish, Rainbow trout, etc.) for training
for species in species_sorted:
    count = species_counts[species]
    # Skip if adding would exceed target
    if test_proteins_count + count <= test_proteins_target + 100:  # Allow small overshoot
        test_species_list.append(species)
        test_proteins_count += count
    else:
        # If this species is too large, skip it (keep in training)
        continue

# If we haven't reached 20%, add more species
if test_proteins_count < test_proteins_target * 0.8:
    for species in species_sorted:
        if species not in test_species_list:
            count = species_counts[species]
            if test_proteins_count + count <= test_proteins_target * 1.2:
                test_species_list.append(species)
                test_proteins_count += count

print(f"\nTest species: {len(test_species_list)}")
print(f"Test proteins: {test_proteins_count:,} ({test_proteins_count/len(df)*100:.1f}%)")

train_species_list = [s for s in species_sorted if s not in test_species_list]
train_proteins_count = len(df) - test_proteins_count

print(f"Training species: {len(train_species_list)}")
print(f"Training proteins: {train_proteins_count:,} ({train_proteins_count/len(df)*100:.1f}%)")

# Assign proteins
train_df = df[df['Organism'].isin(train_species_list)].copy()
test_df = df[df['Organism'].isin(test_species_list)].copy()

print(f"\n✅ Training: {len(train_df):,} proteins")
print(f"✅ Test: {len(test_df):,} proteins")

# =============================================
# LOAD EMBEDDINGS
# =============================================

print("\n" + "=" * 70)
print("LOADING EMBEDDINGS")
print("=" * 70)

def load_embeddings_from_hdf5(entry_ids, embedding_file):
    embeddings = []
    with h5py.File(embedding_file, 'r') as f:
        protein_ids = list(f.keys())
        for entry_id in entry_ids:
            if entry_id in protein_ids:
                emb = f[entry_id][:]
                if emb.shape[0] != 1024:
                    if emb.shape[0] < 1024:
                        emb = np.pad(emb, (0, 1024 - emb.shape[0]))
                    else:
                        emb = emb[:1024]
                embeddings.append(emb.flatten())
            else:
                embeddings.append(np.random.randn(1024).astype(np.float32))
    return np.array(embeddings).astype(np.float32)

train_entries = train_df['Entry'].tolist()
test_entries = test_df['Entry'].tolist()

print(f"\nLoading training embeddings...")
X_train = load_embeddings_from_hdf5(train_entries, embedding_file)
print(f"Training embeddings shape: {X_train.shape}")

print(f"\nLoading test embeddings...")
X_test = load_embeddings_from_hdf5(test_entries, embedding_file)
print(f"Test embeddings shape: {X_test.shape}")

# =============================================
# SCALE AND SAVE
# =============================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = train_df['ec_class'].values
y_test = test_df['ec_class'].values

# Save
np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# =============================================
# SUMMARY
# =============================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
📊 Dataset Summary:
  Total proteins: {len(df):,}
  Total species: {len(species_counts)}
  Test proteins: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)
  Training proteins: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)

📁 Split Information:
  Training species: {len(train_species_list)}
  Test species: {len(test_species_list)}

📂 Files saved in: {output_dir}
""")

print("📊 Top 5 Species in Test Set:")
for species, count in test_df['Organism'].value_counts().head(5).items():
    print(f"  {species[:40]}: {count:,}")

print("\n" + "=" * 70)
print("✅ DATA PREPARATION COMPLETE!")
print("=" * 70)