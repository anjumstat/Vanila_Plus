# -*- coding: utf-8 -*-
"""
Analyze Fish UniProt Data for 8-Class Classification
Classes: 0 = Non-enzyme, 1-7 = EC Classes
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

# Path to your fish data
data_dir = r"D:\zebfish\new_class\data"

# Find the TSV file (you may have multiple species or combined)
tsv_files = glob(os.path.join(data_dir, "*.tsv"))

print("=" * 70)
print("FISH UNIPROT DATA ANALYSIS")
print("8-Class Classification: 0 (Non-enzyme) + 1-7 (EC Classes)")
print("=" * 70)

# =============================================
# LOAD DATA
# =============================================

all_dfs = []

for tsv_file in tsv_files:
    file_path = os.path.join(data_dir, tsv_file)
    print(f"\n📁 Loading: {os.path.basename(tsv_file)}")
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"   ✅ Loaded: {len(df):,} rows")
        print(f"   Columns: {df.columns.tolist()}")
        all_dfs.append(df)
    except Exception as e:
        print(f"   ❌ Error: {e}")

if not all_dfs:
    print("\n❌ No TSV files found!")
    exit()

# Combine all species
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ Combined dataset: {len(combined_df):,} proteins")

# =============================================
# FUNCTION TO CLASSIFY PROTEINS (8 Classes)
# =============================================

def get_ec_class(ec_text):
    """
    Extract EC class (1-7) from EC number
    Returns 0 for non-enzyme
    """
    if pd.isna(ec_text):
        return 0
    
    ec_str = str(ec_text).strip()
    if ec_str == '' or ec_str.lower() in ['nan', 'none', 'null', '-']:
        return 0
    
    # Check if it starts with a digit (EC number)
    if ec_str and ec_str[0].isdigit():
        try:
            ec_class = int(ec_str[0])
            if 1 <= ec_class <= 7:
                return ec_class
        except:
            pass
    
    return 0

def get_class_name(ec_class):
    """Get class name from EC class number"""
    names = {
        0: 'Non-enzyme',
        1: 'Oxidoreductases',
        2: 'Transferases',
        3: 'Hydrolases',
        4: 'Lyases',
        5: 'Isomerases',
        6: 'Ligases',
        7: 'Translocases'
    }
    return names.get(ec_class, 'Unknown')

# =============================================
# CLASSIFY ALL PROTEINS
# =============================================

# Find EC column
ec_column = None
for col in ['EC number', 'EC Number', 'EC_number', 'ec']:
    if col in combined_df.columns:
        ec_column = col
        break

if ec_column is None:
    print("\n❌ No EC number column found!")
    print(f"Available columns: {combined_df.columns.tolist()}")
    exit()

print(f"\n🔬 Using EC column: '{ec_column}'")

# Apply classification
combined_df['ec_class'] = combined_df[ec_column].apply(get_ec_class)
combined_df['class_name'] = combined_df['ec_class'].apply(get_class_name)

# =============================================
# CLASS DISTRIBUTION
# =============================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

# Count each class
class_counts = combined_df['ec_class'].value_counts().sort_index()
class_percentages = (class_counts / len(combined_df) * 100).round(2)

# Create summary DataFrame
summary_df = pd.DataFrame({
    'Class': class_counts.index,
    'Class_Name': [get_class_name(c) for c in class_counts.index],
    'Count': class_counts.values,
    'Percentage': class_percentages.values
})

print("\n📊 8-Class Distribution:")
print("-" * 70)
print(f"{'Class':<6} {'Name':<20} {'Count':<12} {'Percentage':<10}")
print("-" * 70)

for _, row in summary_df.iterrows():
    print(f"{row['Class']:<6} {row['Class_Name']:<20} {row['Count']:>10,}   {row['Percentage']:>7.2f}%")

print("-" * 70)
total = summary_df['Count'].sum()
print(f"{'Total':<6} {'All Proteins':<20} {total:>10,}   {100:>7.2f}%")

# =============================================
# SPECIES DISTRIBUTION (FISH SPECIES)
# =============================================

print("\n" + "=" * 70)
print("FISH SPECIES DISTRIBUTION")
print("=" * 70)

# Find organism column
organism_col = None
for col in ['Organism', 'Organism Name', 'organism', 'organism_name']:
    if col in combined_df.columns:
        organism_col = col
        break

if organism_col:
    # Count proteins per species
    species_counts = combined_df[organism_col].value_counts()
    total_species = len(species_counts)
    
    print(f"\nTotal number of fish species: {total_species}")
    print(f"Total proteins: {len(combined_df):,}")
    
    print("\n📊 Top 20 Fish Species:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Species':<45} {'Count':<12} {'Percentage':<10}")
    print("-" * 70)
    
    top_20 = species_counts.head(20)
    for i, (species, count) in enumerate(top_20.items(), 1):
        percentage = count / len(combined_df) * 100
        print(f"{i:<5} {species[:45]:<45} {count:>10,}   {percentage:>7.2f}%")
    
    print("-" * 70)
    top_20_total = top_20.sum()
    print(f"{'Top 20':<5} {'Total':<45} {top_20_total:>10,}   {top_20_total/len(combined_df)*100:>7.2f}%")
    
    # Check for specific fish species
    target_fish = ['Danio rerio', 'Zebrafish', 'Salmo salar', 'Oncorhynchus mykiss', 
                   'Takifugu rubripes', 'Ictalurus punctatus', 'Carassius auratus',
                   'Cyprinus carpio', 'Tetraodon nigroviridis', 'Oryzias latipes']
    
    print("\n" + "=" * 70)
    print("CHECK FOR TARGET FISH SPECIES")
    print("=" * 70)
    
    found_fish = []
    for fish in target_fish:
        matches = [s for s in species_counts.index if fish.lower() in s.lower()]
        if matches:
            for match in matches:
                count = species_counts[match]
                found_fish.append((match, count))
                print(f"✅ {match}: {count:,} proteins")
        else:
            print(f"❌ {fish}: NOT FOUND")
    
    if found_fish:
        total_fish = sum([count for _, count in found_fish])
        print(f"\n📊 Total target fish proteins: {total_fish:,}")
        print(f"   Percentage of dataset: {total_fish/len(combined_df)*100:.2f}%")
    
    # Save species distribution
    species_df = pd.DataFrame({
        'Species': species_counts.index,
        'Count': species_counts.values,
        'Percentage': (species_counts.values / len(combined_df) * 100).round(2)
    })
    species_df.to_csv(os.path.join(data_dir, 'fish_species_distribution.csv'), index=False)
    print(f"\n✅ Saved: fish_species_distribution.csv")
    
    # Save top 20 species
    top_20_df = pd.DataFrame({
        'Rank': range(1, 21),
        'Species': top_20.index,
        'Count': top_20.values,
        'Percentage': (top_20.values / len(combined_df) * 100).round(2)
    })
    top_20_df.to_csv(os.path.join(data_dir, 'fish_top_20_species.csv'), index=False)
    print(f"✅ Saved: fish_top_20_species.csv")

# =============================================
# VISUALIZATION
# =============================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure 1: Bar Chart - 8-Class Distribution
fig1, ax1 = plt.subplots(figsize=(12, 6))

colors = ['#808080', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
bars = ax1.bar(summary_df['Class_Name'], summary_df['Count'], color=colors, edgecolor='black', alpha=0.8)

ax1.set_xlabel('Class', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Proteins', fontsize=14, fontweight='bold')
ax1.set_title('8-Class Distribution: Fish UniProt Data\n(0 = Non-enzyme, 1-7 = EC Classes)', 
              fontsize=16, fontweight='bold')
ax1.tick_params(axis='x', rotation=45, labelsize=11)
ax1.tick_params(axis='y', labelsize=11)
ax1.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, count in zip(bars, summary_df['Count']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(summary_df['Count'])*0.01, 
             f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'Fish_Class_Distribution_Bar_Chart.png'), dpi=300, bbox_inches='tight')
print("  ✅ Saved: Fish_Class_Distribution_Bar_Chart.png")

# Figure 2: Pie Chart - 8-Class Distribution
fig2, ax2 = plt.subplots(figsize=(10, 10))

# Combine small classes for better visualization
threshold = 1000
small_classes = summary_df[summary_df['Count'] < threshold]
if len(small_classes) > 0:
    other_count = small_classes['Count'].sum()
    
    # Create new DataFrame for pie chart
    pie_data = summary_df[summary_df['Count'] >= threshold].copy()
    pie_data = pd.concat([pie_data, pd.DataFrame({
        'Class': ['Other'],
        'Class_Name': ['Other (Small Classes)'],
        'Count': [other_count],
        'Percentage': [other_count / total * 100]
    })], ignore_index=True)
else:
    pie_data = summary_df

colors_pie = ['#808080', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

ax2.pie(pie_data['Count'], labels=pie_data['Class_Name'], autopct='%1.1f%%', 
        colors=colors_pie[:len(pie_data)], startangle=90, explode=[0.05] * len(pie_data))
ax2.set_title('8-Class Distribution: Fish UniProt Data', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'Fish_Class_Distribution_Pie_Chart.png'), dpi=300, bbox_inches='tight')
print("  ✅ Saved: Fish_Class_Distribution_Pie_Chart.png")

# Figure 3: Top 20 Fish Species Bar Chart
if organism_col:
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    
    top_20_sorted = top_20.sort_values(ascending=True)
    colors_fish = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_20_sorted)))[::-1]
    
    bars = ax3.barh(range(len(top_20_sorted)), top_20_sorted.values, color=colors_fish, edgecolor='black', alpha=0.8)
    
    ax3.set_yticks(range(len(top_20_sorted)))
    ax3.set_yticklabels(top_20_sorted.index, fontsize=10)
    ax3.set_xlabel('Number of Proteins', fontsize=14, fontweight='bold')
    ax3.set_title('Top 20 Fish Species in UniProt Dataset', fontsize=16, fontweight='bold')
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, top_20_sorted.values):
        ax3.text(bar.get_width() + max(top_20_sorted.values)*0.01, 
                 bar.get_y() + bar.get_height()/2, 
                 f'{count:,}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'Fish_Top_20_Species.png'), dpi=300, bbox_inches='tight')
    print("  ✅ Saved: Fish_Top_20_Species.png")

plt.close('all')

# =============================================
# DETAILED STATISTICS
# =============================================

print("\n" + "=" * 70)
print("DETAILED STATISTICS")
print("=" * 70)

# Total proteins
print(f"\nTotal proteins: {len(combined_df):,}")

# Enzymes vs Non-enzymes
enzymes = (combined_df['ec_class'] != 0).sum()
non_enzymes = (combined_df['ec_class'] == 0).sum()

print(f"\n🔬 Enzyme Classification:")
print(f"  Enzymes: {enzymes:,} ({enzymes/len(combined_df)*100:.2f}%)")
print(f"  Non-enzymes: {non_enzymes:,} ({non_enzymes/len(combined_df)*100:.2f}%)")

# EC Class distribution (enzymes only)
print(f"\n📊 EC Class Distribution (Enzymes only):")
enzyme_df = combined_df[combined_df['ec_class'] != 0]
ec_dist = enzyme_df['ec_class'].value_counts().sort_index()
for ec_class, count in ec_dist.items():
    name = get_class_name(ec_class)
    print(f"  Class {ec_class} ({name}): {count:,} ({count/enzymes*100:.2f}% of enzymes)")

# =============================================
# SAVE RESULTS
# =============================================

# Save classification file
output_file = os.path.join(data_dir, 'fish_8_class_classification.csv')
combined_df[['Entry', 'ec_class', 'class_name']].to_csv(output_file, index=False)
print(f"\n✅ Saved classification: {output_file}")

# Save summary
summary_output = os.path.join(data_dir, 'fish_class_distribution_summary.csv')
summary_df.to_csv(summary_output, index=False)
print(f"✅ Saved summary: {summary_output}")

# =============================================
# PRINT SUMMARY TABLE FOR PAPER (LaTeX)
# =============================================

print("\n" + "=" * 70)
print("SUMMARY TABLE FOR PAPER (LaTeX Format)")
print("=" * 70)

print("\n\\begin{table}[htbp]")
print("\\centering")
print("\\caption{8-Class Distribution of Fish Proteins from UniProt}")
print("\\label{tab:fish_class_distribution}")
print("\\begin{tabular}{lrrr}")
print("\\hline")
print("\\textbf{Class} & \\textbf{Class Name} & \\textbf{Count} & \\textbf{Percentage} \\\\")
print("\\hline")

for _, row in summary_df.iterrows():
    print(f"{row['Class']} & {row['Class_Name']} & {row['Count']:,} & {row['Percentage']:.2f}\\% \\\\")

print("\\hline")
print(f"\\textbf{{Total}} & \\textbf{{All Proteins}} & \\textbf{{{total:,}}} & \\textbf{{100.00\\%}} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

print("\n" + "=" * 70)
print("✅ FISH DATA ANALYSIS COMPLETE!")
print("=" * 70)