# -*- coding: utf-8 -*-
"""
Analyze Species Distribution in Fish UniProt Data
Shows which fish species are included and how many proteins each has
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# =============================================
# CONFIGURATION
# =============================================

data_dir = r"D:\zebfish\new_class\data"

# Find the TSV file
tsv_files = glob(os.path.join(data_dir, "*.tsv"))

print("=" * 70)
print("FISH UNIPROT DATA - SPECIES DISTRIBUTION")
print("=" * 70)

# =============================================
# LOAD DATA
# =============================================

if not tsv_files:
    print("\n❌ No TSV files found!")
    exit()

# Load the first TSV file (your downloaded data)
file_path = tsv_files[0]
print(f"\n📁 Loading: {os.path.basename(file_path)}")

df = pd.read_csv(file_path, sep='\t')
print(f"✅ Loaded: {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")

# =============================================
# FIND ORGANISM COLUMN
# =============================================

organism_col = None
for col in ['Organism', 'Organism Name', 'organism', 'organism_name']:
    if col in df.columns:
        organism_col = col
        break

if organism_col is None:
    print("\n❌ No organism column found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

print(f"\n🔬 Using organism column: '{organism_col}'")

# =============================================
# SPECIES DISTRIBUTION
# =============================================

print("\n" + "=" * 70)
print("SPECIES DISTRIBUTION")
print("=" * 70)

# Count proteins per species
species_counts = df[organism_col].value_counts()
total_species = len(species_counts)

print(f"\nTotal number of fish species: {total_species}")
print(f"Total proteins: {len(df):,}")

# Display all species with counts (limit to first 50 for readability)
print("\n📊 Species Distribution (Top 50):")
print("-" * 70)
print(f"{'#':<5} {'Species':<50} {'Count':<12} {'Percentage':<10}")
print("-" * 70)

for i, (species, count) in enumerate(species_counts.items(), 1):
    if i > 50:
        print(f"... and {total_species - 50} more species")
        break
    percentage = count / len(df) * 100
    print(f"{i:<5} {species[:50]:<50} {count:>10,}   {percentage:>7.2f}%")

print("-" * 70)
print(f"{'Total':<5} {'All Species':<50} {len(df):>10,}   {100:>7.2f}%")

# =============================================
# TOP SPECIES SUMMARY
# =============================================

print("\n" + "=" * 70)
print("TOP 20 FISH SPECIES (by protein count)")
print("=" * 70)

top_20 = species_counts.head(20)
print(f"\n{'Rank':<5} {'Species':<50} {'Count':<12} {'Percentage':<10}")
print("-" * 70)

for i, (species, count) in enumerate(top_20.items(), 1):
    percentage = count / len(df) * 100
    print(f"{i:<5} {species[:50]:<50} {count:>10,}   {percentage:>7.2f}%")

print("-" * 70)
top_20_total = top_20.sum()
print(f"{'Top 20':<5} {'Total':<50} {top_20_total:>10,}   {top_20_total/len(df)*100:>7.2f}%")

# =============================================
# FISH GROUPS SUMMARY
# =============================================

print("\n" + "=" * 70)
print("FISH GROUPS SUMMARY")
print("=" * 70)

# Define major fish groups for categorization
fish_groups = {
    'Zebrafish': ['Danio rerio'],
    'Salmonids': ['Salmo salar', 'Oncorhynchus mykiss', 'Oncorhynchus keta', 'Oncorhynchus tshawytscha'],
    'Pufferfish': ['Takifugu rubripes', 'Tetraodon nigroviridis'],
    'Catfish': ['Ictalurus punctatus'],
    'Carps': ['Cyprinus carpio', 'Carassius auratus'],
    'Tilapia': ['Oreochromis niloticus', 'Oreochromis mossambicus'],
    'Medaka': ['Oryzias latipes'],
    'Eels': ['Anguilla japonica', 'Anguilla anguilla'],
    'Tuna': ['Thunnus obesus', 'Thunnus albacares', 'Thunnus thynnus'],
    'Seabass/Seabream': ['Dicentrarchus labrax', 'Sparus aurata', 'Pagrus major'],
    'Cod': ['Gadus morhua'],
    'Flatfish': ['Paralichthys olivaceus'],
}

print("\n📊 Fish Groups Protein Counts:")
print("-" * 50)
total_group_proteins = 0
for group, species_list in fish_groups.items():
    group_count = 0
    found_species = []
    for species in species_list:
        if species in species_counts.index:
            count = species_counts[species]
            group_count += count
            found_species.append(f"{species} ({count:,})")
    if group_count > 0:
        total_group_proteins += group_count
        print(f"  {group}: {group_count:,} proteins")
        print(f"    Species: {', '.join(found_species[:3])}")
        if len(found_species) > 3:
            print(f"    ... and {len(found_species)-3} more")

print(f"\n  Total from major groups: {total_group_proteins:,} proteins")
print(f"  Percentage of dataset: {total_group_proteins/len(df)*100:.2f}%")

# =============================================
# CHECK FOR KEY FISH SPECIES
# =============================================

print("\n" + "=" * 70)
print("CHECK FOR KEY FISH SPECIES")
print("=" * 70)

key_fish = [
    'Danio rerio',           # Zebrafish
    'Salmo salar',           # Atlantic salmon
    'Oncorhynchus mykiss',   # Rainbow trout
    'Takifugu rubripes',     # Japanese pufferfish
    'Ictalurus punctatus',   # Channel catfish
    'Cyprinus carpio',       # Common carp
    'Carassius auratus',     # Goldfish
    'Oryzias latipes',       # Japanese medaka
    'Tetraodon nigroviridis', # Spotted green pufferfish
    'Oreochromis niloticus', # Nile tilapia
]

found_key_fish = []

for fish in key_fish:
    matches = [s for s in species_counts.index if fish.lower() in s.lower()]
    if matches:
        for match in matches:
            count = species_counts[match]
            found_key_fish.append((match, count))
            print(f"✅ {match}: {count:,} proteins")
    else:
        print(f"❌ {fish}: NOT FOUND")

if found_key_fish:
    total_key = sum([count for _, count in found_key_fish])
    print(f"\n📊 Total key fish proteins: {total_key:,}")
    print(f"   Percentage of dataset: {total_key/len(df)*100:.2f}%")

# =============================================
# VISUALIZATION
# =============================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Figure 1: Top 20 Fish Species Bar Chart
fig1, ax1 = plt.subplots(figsize=(14, 10))

top_20_sorted = top_20.sort_values(ascending=True)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_20_sorted)))[::-1]

bars = ax1.barh(range(len(top_20_sorted)), top_20_sorted.values, color=colors, edgecolor='black', alpha=0.8)

ax1.set_yticks(range(len(top_20_sorted)))
ax1.set_yticklabels(top_20_sorted.index, fontsize=10)
ax1.set_xlabel('Number of Proteins', fontsize=14, fontweight='bold')
ax1.set_title('Top 20 Fish Species in UniProt Dataset', fontsize=16, fontweight='bold')
ax1.grid(True, axis='x', alpha=0.3)

# Add value labels
for bar, count in zip(bars, top_20_sorted.values):
    ax1.text(bar.get_width() + max(top_20_sorted.values)*0.01, 
             bar.get_y() + bar.get_height()/2, 
             f'{count:,}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'Fish_Species_Distribution_Top20.png'), dpi=300, bbox_inches='tight')
print("  ✅ Saved: Fish_Species_Distribution_Top20.png")

# Figure 2: Pie Chart - Top 10 Fish Species vs Others
fig2, ax2 = plt.subplots(figsize=(10, 10))

top_10 = species_counts.head(10)
others_count = species_counts.iloc[10:].sum()

pie_data = pd.concat([top_10, pd.Series({'Others': others_count})])
pie_labels = [f"{label[:30]} ({count:,})" if count > 100 else label[:30] for label, count in pie_data.items()]

colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))

ax2.pie(pie_data.values, labels=pie_labels, autopct='%1.1f%%', 
        colors=colors_pie, startangle=90, explode=[0.03] * len(pie_data))
ax2.set_title('Fish Species Distribution in UniProt Dataset\n(Top 10 Species + Others)', 
              fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'Fish_Species_Distribution_Pie.png'), dpi=300, bbox_inches='tight')
print("  ✅ Saved: Fish_Species_Distribution_Pie.png")

# Figure 3: Zebrafish vs Other Species Bar Chart
fig3, ax3 = plt.subplots(figsize=(10, 6))

zebrafish_count = species_counts.get('Danio rerio (Zebrafish) (Brachydanio rerio)', 0)
other_count = len(df) - zebrafish_count

category_labels = ['Zebrafish', 'Other Species']
category_counts = [zebrafish_count, other_count]
category_colors = ['#1f77b4', '#808080']

bars = ax3.bar(category_labels, category_counts, color=category_colors, edgecolor='black', alpha=0.8)

ax3.set_ylabel('Number of Proteins', fontsize=14, fontweight='bold')
ax3.set_title('Zebrafish vs Other Fish Species', fontsize=16, fontweight='bold')
ax3.grid(True, axis='y', alpha=0.3)

# Add value labels
for bar, count in zip(bars, category_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(category_counts)*0.01, 
             f'{count:,} ({count/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'Fish_Zebrafish_vs_Others.png'), dpi=300, bbox_inches='tight')
print("  ✅ Saved: Fish_Zebrafish_vs_Others.png")

plt.close('all')

# =============================================
# SAVE RESULTS
# =============================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save species distribution to CSV
species_df = pd.DataFrame({
    'Species': species_counts.index,
    'Count': species_counts.values,
    'Percentage': (species_counts.values / len(df) * 100).round(2)
})
species_df.to_csv(os.path.join(data_dir, 'fish_species_distribution.csv'), index=False)
print(f"✅ Saved: fish_species_distribution.csv")

# Save top 20 to CSV
top_20_df = pd.DataFrame({
    'Rank': range(1, 21),
    'Species': top_20.index,
    'Count': top_20.values,
    'Percentage': (top_20.values / len(df) * 100).round(2)
})
top_20_df.to_csv(os.path.join(data_dir, 'fish_top_20_species.csv'), index=False)
print(f"✅ Saved: fish_top_20_species.csv")

# =============================================
# SUMMARY FOR PAPER
# =============================================

print("\n" + "=" * 70)
print("SUMMARY FOR PAPER - FISH DATA")
print("=" * 70)

print(f"\n📊 Fish Dataset Summary:")
print(f"  Total proteins: {len(df):,}")
print(f"  Total species: {total_species}")
print(f"  Top species: {top_20.index[0]} ({top_20.iloc[0]:,} proteins)")
print(f"  Zebrafish: {zebrafish_count:,} proteins ({zebrafish_count/len(df)*100:.2f}%)")

if found_key_fish:
    print(f"\n🐟 Key Fish Species:")
    for species, count in found_key_fish[:10]:
        print(f"  {species}: {count:,} proteins")
    print(f"  Total key fish: {total_key:,} proteins ({total_key/len(df)*100:.2f}%)")

print("\n" + "=" * 70)
print("✅ FISH DATA ANALYSIS COMPLETE!")
print("=" * 70)