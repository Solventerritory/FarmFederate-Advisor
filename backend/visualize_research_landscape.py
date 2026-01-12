#!/usr/bin/env python3
"""
Quick visualization of research paper landscape
"""

import matplotlib.pyplot as plt
import numpy as np
from research_paper_comparison import RESEARCH_PAPERS

# Group papers by year and category
timeline = {}
for name, info in RESEARCH_PAPERS.items():
    year = info['year']
    cat = info['category']
    if year not in timeline:
        timeline[year] = {}
    if cat not in timeline[year]:
        timeline[year][cat] = []
    timeline[year][cat].append((name, info['f1']))

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Papers per year
years = sorted(timeline.keys())
counts = [sum(len(cats) for cats in timeline[y].values()) for y in years]
ax1.bar(years, counts, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
ax1.set_title('Research Papers Timeline (2016-2024)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for year, count in zip(years, counts):
    ax1.text(year, count + 0.1, str(count), ha='center', fontweight='bold')

# Plot 2: F1 scores over time by category
categories = set()
for year_data in timeline.values():
    categories.update(year_data.keys())

category_colors = {
    'Federated Learning': '#FF6B6B',
    'Plant Disease Detection': '#4ECDC4',
    'Federated Agriculture': '#45B7D1',
    'Vision Transformer': '#FFA07A',
    'Multimodal': '#98D8C8',
    'LLM': '#FFD93D',
    'Federated Multimodal': '#6BCF7F',
}

for cat in categories:
    cat_years = []
    cat_f1s = []
    for year in years:
        if year in timeline and cat in timeline[year]:
            for name, f1 in timeline[year][cat]:
                cat_years.append(year)
                cat_f1s.append(f1 * 100)
    
    if cat_years:
        ax2.scatter(cat_years, cat_f1s, s=150, alpha=0.7, 
                   label=cat, color=category_colors.get(cat, 'gray'),
                   edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Research Progress: F1 Score Evolution by Category', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(alpha=0.3)
ax2.set_ylim(70, 100)

plt.tight_layout()
plt.savefig('results/research_landscape.png', dpi=300, bbox_inches='tight')
print("\n[✓] Research landscape visualization saved to: results/research_landscape.png")
print(f"[📊] Total Papers: {len(RESEARCH_PAPERS)}")
print(f"[📅] Timeline: {min(years)} - {max(years)}")
print(f"[📂] Categories: {len(categories)}")
