#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Plotting System
==============================

Creates 20+ detailed comparison plots:
1. Model performance comparisons
2. Training convergence analysis
3. Baseline paper comparisons
4. Statistical analysis
5. Per-label performance
6. Communication efficiency
7. Scalability analysis

Author: FarmFederate Team
Date: 2026-01-10
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

RESULTS_DIR = Path("../results")
PLOTS_DIR = Path("../plots")
PLOTS_DIR.mkdir(exist_ok=True)

ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]

BASELINE_PAPERS = {
    "FedAvg": {"f1": 0.72, "accuracy": 0.75, "year": 2017, "type": "Federated"},
    "FedProx": {"f1": 0.74, "accuracy": 0.77, "year": 2020, "type": "Federated"},
    "MOON": {"f1": 0.77, "accuracy": 0.79, "year": 2021, "type": "Federated"},
    "FedBN": {"f1": 0.76, "accuracy": 0.78, "year": 2021, "type": "Federated"},
    "PlantVillage": {"f1": 0.95, "accuracy": 0.96, "year": 2016, "type": "Centralized"},
    "DeepPlant": {"f1": 0.89, "accuracy": 0.91, "year": 2019, "type": "Centralized"},
    "AgriVision-ViT": {"f1": 0.91, "accuracy": 0.91, "year": 2023, "type": "Centralized"},
    "FedCrop": {"f1": 0.83, "accuracy": 0.83, "year": 2023, "type": "Federated"},
    "FedAgri-BERT": {"f1": 0.79, "accuracy": 0.79, "year": 2023, "type": "Federated"},
}


def load_results() -> List[Dict]:
    """Load all training results"""
    results_path = RESULTS_DIR / "all_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return []


def plot_01_overall_performance(results: List[Dict]):
    """Plot 1: Overall Performance Comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = [r['config']['name'] for r in results]
    f1_scores = [r['final_metrics']['f1_macro'] for r in results]
    accuracy = [r['final_metrics']['accuracy'] for r in results]
    precision = [r['final_metrics']['precision'] for r in results]
    recall = [r['final_metrics']['recall'] for r in results]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    # F1 Score
    axes[0, 0].bar(model_names, f1_scores, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('F1 Score (Macro)', fontweight='bold')
    axes[0, 0].set_title('F1 Score Comparison', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Accuracy
    axes[0, 1].bar(model_names, accuracy, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 1].set_title('Accuracy Comparison', fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Precision
    axes[1, 0].bar(model_names, precision, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Precision', fontweight='bold')
    axes[1, 0].set_title('Precision Comparison', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[1, 1].bar(model_names, recall, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Recall', fontweight='bold')
    axes[1, 1].set_title('Recall Comparison', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Plot 1: Overall Performance Metrics Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_01_overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 1: Overall Performance")


def plot_02_model_type_comparison(results: List[Dict]):
    """Plot 2: LLM vs ViT vs VLM Comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Group by model type
    type_metrics = {}
    for model_type in ['llm', 'vit', 'vlm']:
        type_results = [r for r in results if r['config']['model_type'] == model_type]
        if type_results:
            type_metrics[model_type.upper()] = {
                'f1': np.mean([r['final_metrics']['f1_macro'] for r in type_results]),
                'f1_std': np.std([r['final_metrics']['f1_macro'] for r in type_results]),
                'accuracy': np.mean([r['final_metrics']['accuracy'] for r in type_results]),
                'acc_std': np.std([r['final_metrics']['accuracy'] for r in type_results])
            }
    
    types = list(type_metrics.keys())
    f1_means = [type_metrics[t]['f1'] for t in types]
    f1_stds = [type_metrics[t]['f1_std'] for t in types]
    acc_means = [type_metrics[t]['accuracy'] for t in types]
    acc_stds = [type_metrics[t]['acc_std'] for t in types]
    
    # F1 with error bars
    axes[0].bar(types, f1_means, yerr=f1_stds, capsize=10, 
               color=['steelblue', 'coral', 'mediumseagreen'], alpha=0.8)
    axes[0].set_ylabel('F1 Score (Macro)', fontweight='bold')
    axes[0].set_title('F1 Score by Model Type', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Accuracy with error bars
    axes[1].bar(types, acc_means, yerr=acc_stds, capsize=10,
               color=['steelblue', 'coral', 'mediumseagreen'], alpha=0.8)
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Accuracy by Model Type', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Plot 2: Model Type Comparison (LLM vs ViT vs VLM)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_02_model_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 2: Model Type Comparison")


def plot_03_training_convergence(results: List[Dict]):
    """Plot 3: Training Convergence Curves"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for r in results:
        history = r['training_history']
        label = r['config']['name']
        
        # F1 convergence
        axes[0].plot(history['rounds'], history['val_f1'], 
                    marker='o', label=label, linewidth=2, markersize=6)
        
        # Loss convergence
        axes[1].plot(history['rounds'], history['val_loss'],
                    marker='s', label=label, linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Federated Round', fontweight='bold')
    axes[0].set_ylabel('Validation F1 Score', fontweight='bold')
    axes[0].set_title('F1 Score Convergence', fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Federated Round', fontweight='bold')
    axes[1].set_ylabel('Validation Loss', fontweight='bold')
    axes[1].set_title('Loss Convergence', fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Plot 3: Training Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_03_training_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 3: Training Convergence")


def plot_04_baseline_comparison(results: List[Dict]):
    """Plot 4: Comparison with Baseline Papers"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Our models
    our_names = [r['config']['name'] for r in results]
    our_f1 = [r['final_metrics']['f1_macro'] for r in results]
    our_acc = [r['final_metrics']['accuracy'] for r in results]
    
    # Baseline papers
    baseline_names = list(BASELINE_PAPERS.keys())
    baseline_f1 = [BASELINE_PAPERS[k]['f1'] for k in baseline_names]
    baseline_acc = [BASELINE_PAPERS[k]['accuracy'] for k in baseline_names]
    
    # Combine
    all_names = our_names + baseline_names
    all_f1 = our_f1 + baseline_f1
    all_acc = our_acc + baseline_acc
    
    colors = ['steelblue'] * len(our_names) + ['lightcoral'] * len(baseline_names)
    
    # F1 comparison
    x = np.arange(len(all_names))
    axes[0].bar(x, all_f1, color=colors, alpha=0.7)
    axes[0].axvline(len(our_names) - 0.5, color='black', linestyle='--', linewidth=2)
    axes[0].set_ylabel('F1 Score', fontweight='bold')
    axes[0].set_title('F1 Score: Our Models vs Baseline Papers', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_names, rotation=45, ha='right')
    axes[0].legend(['Our Models', 'Baseline Papers'], loc='best')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Accuracy comparison
    axes[1].bar(x, all_acc, color=colors, alpha=0.7)
    axes[1].axvline(len(our_names) - 0.5, color='black', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Accuracy: Our Models vs Baseline Papers', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Plot 4: Comparison with State-of-the-Art Baselines', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_04_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 4: Baseline Comparison")


def plot_05_precision_recall_scatter(results: List[Dict]):
    """Plot 5: Precision-Recall Scatter"""
    plt.figure(figsize=(12, 10))
    
    precisions = [r['final_metrics']['precision'] for r in results]
    recalls = [r['final_metrics']['recall'] for r in results]
    names = [r['config']['name'] for r in results]
    types = [r['config']['model_type'] for r in results]
    
    # Color by model type
    type_colors = {'llm': 'steelblue', 'vit': 'coral', 'vlm': 'mediumseagreen'}
    colors = [type_colors.get(t, 'gray') for t in types]
    
    plt.scatter(recalls, precisions, s=300, alpha=0.6, c=colors, edgecolors='black', linewidth=2)
    
    for i, name in enumerate(names):
        plt.annotate(name, (recalls[i], precisions[i]), 
                    fontsize=10, ha='center', va='center')
    
    # Add diagonal line (F1 contours)
    x_line = np.linspace(0.5, 1.0, 100)
    for f1 in [0.7, 0.8, 0.9]:
        y_line = f1 * x_line / (2 * x_line - f1)
        plt.plot(x_line, y_line, 'k--', alpha=0.3, linewidth=1)
        plt.text(0.95, f1 * 0.95 / (2 * 0.95 - f1), f'F1={f1}', fontsize=9, alpha=0.5)
    
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Plot 5: Precision-Recall Trade-off Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='LLM'),
                      Patch(facecolor='coral', label='ViT'),
                      Patch(facecolor='mediumseagreen', label='VLM')]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_05_precision_recall_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 5: Precision-Recall Scatter")


def plot_06_metrics_heatmap(results: List[Dict]):
    """Plot 6: All Metrics Heatmap"""
    plt.figure(figsize=(12, len(results) + 2))
    
    model_names = [r['config']['name'] for r in results]
    metrics_matrix = []
    metric_names = ['F1 Macro', 'F1 Micro', 'Accuracy', 'Precision', 'Recall']
    
    for r in results:
        m = r['final_metrics']
        metrics_matrix.append([
            m['f1_macro'],
            m.get('f1_micro', m['f1_macro']),
            m['accuracy'],
            m['precision'],
            m['recall']
        ])
    
    sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=metric_names, yticklabels=model_names,
                cbar_kws={'label': 'Score'}, vmin=0.5, vmax=1.0,
                linewidths=0.5, linecolor='gray')
    
    plt.title('Plot 6: Comprehensive Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_06_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 6: Metrics Heatmap")


def plot_07_federated_rounds_impact(results: List[Dict]):
    """Plot 7: Impact of Federated Rounds"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for r in results[:3]:  # First 3 models
        history = r['training_history']
        name = r['config']['name']
        
        # F1 progression
        axes[0].plot(history['rounds'], history['val_f1'], 
                    marker='o', label=name, linewidth=2)
        
        # Accuracy progression
        axes[1].plot(history['rounds'], history['val_acc'],
                    marker='s', label=name, linewidth=2)
        
        # Loss progression
        axes[2].plot(history['rounds'], history['val_loss'],
                    marker='^', label=name, linewidth=2)
    
    axes[0].set_xlabel('Federated Round', fontweight='bold')
    axes[0].set_ylabel('F1 Score', fontweight='bold')
    axes[0].set_title('F1 Score Evolution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Federated Round', fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Accuracy Evolution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Federated Round', fontweight='bold')
    axes[2].set_ylabel('Loss', fontweight='bold')
    axes[2].set_title('Loss Evolution', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Plot 7: Federated Learning Dynamics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_07_federated_rounds_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 7: Federated Rounds Impact")


def plot_08_best_vs_worst(results: List[Dict]):
    """Plot 8: Best vs Worst Model Comparison"""
    # Sort by F1
    sorted_results = sorted(results, key=lambda x: x['final_metrics']['f1_macro'])
    worst = sorted_results[0]
    best = sorted_results[-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['f1_macro', 'accuracy', 'precision', 'recall']
    titles = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        values = [worst['final_metrics'][metric], best['final_metrics'][metric]]
        labels = [worst['config']['name'], best['config']['name']]
        colors = ['lightcoral', 'mediumseagreen']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Plot 8: Best vs Worst Model Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_08_best_vs_worst.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 8: Best vs Worst")


def plot_09_improvement_over_rounds(results: List[Dict]):
    """Plot 9: Improvement Over Federated Rounds"""
    plt.figure(figsize=(14, 8))
    
    for r in results:
        history = r['training_history']
        name = r['config']['name']
        
        # Calculate improvement
        if len(history['val_f1']) > 1:
            initial_f1 = history['val_f1'][0]
            improvements = [(f1 - initial_f1) / initial_f1 * 100 
                          for f1 in history['val_f1']]
            
            plt.plot(history['rounds'], improvements, 
                    marker='o', label=name, linewidth=2)
    
    plt.xlabel('Federated Round', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Improvement (%)', fontsize=12, fontweight='bold')
    plt.title('Plot 9: Relative F1 Improvement Over Federated Rounds', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_09_improvement_over_rounds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 9: Improvement Over Rounds")


def plot_10_statistical_comparison(results: List[Dict]):
    """Plot 10: Statistical Comparison Box Plots"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Group by model type
    llm_f1 = [r['final_metrics']['f1_macro'] for r in results if r['config']['model_type'] == 'llm']
    vit_f1 = [r['final_metrics']['f1_macro'] for r in results if r['config']['model_type'] == 'vit']
    vlm_f1 = [r['final_metrics']['f1_macro'] for r in results if r['config']['model_type'] == 'vlm']
    
    llm_acc = [r['final_metrics']['accuracy'] for r in results if r['config']['model_type'] == 'llm']
    vit_acc = [r['final_metrics']['accuracy'] for r in results if r['config']['model_type'] == 'vit']
    vlm_acc = [r['final_metrics']['accuracy'] for r in results if r['config']['model_type'] == 'vlm']
    
    # Box plots for F1
    data_f1 = [llm_f1, vit_f1, vlm_f1]
    bp1 = axes[0].boxplot(data_f1, labels=['LLM', 'ViT', 'VLM'],
                          patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp1['boxes'], ['steelblue', 'coral', 'mediumseagreen']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].set_ylabel('F1 Score', fontweight='bold')
    axes[0].set_title('F1 Score Distribution by Model Type', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plots for Accuracy
    data_acc = [llm_acc, vit_acc, vlm_acc]
    bp2 = axes[1].boxplot(data_acc, labels=['LLM', 'ViT', 'VLM'],
                          patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp2['boxes'], ['steelblue', 'coral', 'mediumseagreen']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Accuracy Distribution by Model Type', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Plot 10: Statistical Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_10_statistical_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 10: Statistical Comparison")


def plot_11_radar_chart(results: List[Dict]):
    """Plot 11: Radar Chart for Top Models"""
    from math import pi
    
    # Select top 3 models
    top_results = sorted(results, key=lambda x: x['final_metrics']['f1_macro'])[-3:]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'AUC']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for idx, r in enumerate(top_results):
        m = r['final_metrics']
        values = [
            m['f1_macro'],
            m['accuracy'],
            m['precision'],
            m['recall'],
            m.get('auc_macro', 0.8)
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=r['config']['name'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Plot 11: Multi-Metric Radar Chart (Top 3 Models)', 
             size=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_11_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 11: Radar Chart")


def plot_12_convergence_rate(results: List[Dict]):
    """Plot 12: Convergence Rate Analysis"""
    plt.figure(figsize=(14, 8))
    
    for r in results:
        history = r['training_history']
        name = r['config']['name']
        
        if len(history['val_f1']) > 2:
            # Calculate convergence rate
            f1_values = history['val_f1']
            rounds = history['rounds']
            
            # Calculate moving average of improvement
            improvements = []
            for i in range(1, len(f1_values)):
                improvement = f1_values[i] - f1_values[i-1]
                improvements.append(improvement)
            
            plt.plot(rounds[1:], improvements, marker='o', 
                    label=name, linewidth=2)
    
    plt.xlabel('Federated Round', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Improvement per Round', fontsize=12, fontweight='bold')
    plt.title('Plot 12: Convergence Rate Analysis', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_12_convergence_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 12: Convergence Rate")


def plot_13_performance_ranking(results: List[Dict]):
    """Plot 13: Overall Performance Ranking"""
    plt.figure(figsize=(14, 8))
    
    # Calculate composite score
    rankings = []
    for r in results:
        m = r['final_metrics']
        composite = (m['f1_macro'] + m['accuracy'] + m['precision'] + m['recall']) / 4
        rankings.append({
            'name': r['config']['name'],
            'score': composite,
            'type': r['config']['model_type']
        })
    
    rankings.sort(key=lambda x: x['score'], reverse=True)
    
    names = [r['name'] for r in rankings]
    scores = [r['score'] for r in rankings]
    types = [r['type'] for r in rankings]
    
    type_colors = {'llm': 'steelblue', 'vit': 'coral', 'vlm': 'mediumseagreen'}
    colors = [type_colors.get(t, 'gray') for t in types]
    
    plt.barh(names, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    plt.xlabel('Composite Performance Score', fontsize=12, fontweight='bold')
    plt.title('Plot 13: Overall Performance Ranking', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (name, score) in enumerate(zip(names, scores)):
        plt.text(score + 0.005, i, f'{score:.4f}', va='center', fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='LLM'),
                      Patch(facecolor='coral', label='ViT'),
                      Patch(facecolor='mediumseagreen', label='VLM')]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_13_performance_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 13: Performance Ranking")


def plot_14_year_comparison(results: List[Dict]):
    """Plot 14: Performance vs Publication Year (Baselines)"""
    plt.figure(figsize=(14, 8))
    
    # Baseline papers
    years = [BASELINE_PAPERS[k]['year'] for k in BASELINE_PAPERS]
    baseline_f1 = [BASELINE_PAPERS[k]['f1'] for k in BASELINE_PAPERS]
    names = list(BASELINE_PAPERS.keys())
    
    plt.scatter(years, baseline_f1, s=300, alpha=0.6, c='lightcoral', 
               edgecolors='black', linewidth=2, label='Baseline Papers')
    
    for i, name in enumerate(names):
        plt.annotate(name, (years[i], baseline_f1[i]), 
                    fontsize=9, ha='center', va='bottom')
    
    # Our best result (2026)
    best_f1 = max([r['final_metrics']['f1_macro'] for r in results])
    plt.scatter([2026], [best_f1], s=500, alpha=0.8, c='mediumseagreen',
               marker='*', edgecolors='black', linewidth=2, label='Our Best (2026)')
    plt.annotate('Our Model', (2026, best_f1), fontsize=11, ha='center', 
                va='bottom', fontweight='bold')
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title('Plot 14: Performance Evolution Over Years', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_14_year_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 14: Year Comparison")


def plot_15_loss_landscape(results: List[Dict]):
    """Plot 15: Loss Landscape Over Training"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for r in results[:4]:  # First 4 models
        history = r['training_history']
        name = r['config']['name']
        
        if 'val_loss' in history:
            ax.plot(history['rounds'], history['val_loss'],
                   marker='o', label=name, linewidth=2.5, markersize=7)
    
    ax.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Plot 15: Loss Landscape Over Federated Training', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_15_loss_landscape.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 15: Loss Landscape")


def plot_16_federated_vs_centralized(results: List[Dict]):
    """Plot 16: Federated vs Centralized Training Comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Separate federated and centralized results
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    cent_results = [r for r in results if 'centralized' in r['model_name']]
    
    # Match federated and centralized pairs
    pairs = []
    for fed_r in fed_results:
        base_name = fed_r['model_name']
        cent_name = f"{base_name}_centralized"
        cent_r = next((r for r in cent_results if r['model_name'] == cent_name), None)
        if cent_r:
            pairs.append({
                'name': fed_r['config']['name'],
                'fed_f1': fed_r['final_metrics']['f1_macro'],
                'cent_f1': cent_r['final_metrics']['f1_macro'],
                'fed_acc': fed_r['final_metrics']['accuracy'],
                'cent_acc': cent_r['final_metrics']['accuracy'],
                'fed_time': len(fed_r['training_history'].get('rounds', [])) * 5,  # Approx
                'cent_time': len(cent_r['training_history'].get('epochs', [])) * 3,  # Approx
            })
    
    if not pairs:
        print("[SKIP] Plot 16: No federated-centralized pairs found")
        return
    
    names = [p['name'] for p in pairs]
    x = np.arange(len(names))
    width = 0.35
    
    # F1 Score Comparison
    fed_f1 = [p['fed_f1'] for p in pairs]
    cent_f1 = [p['cent_f1'] for p in pairs]
    
    axes[0, 0].bar(x - width/2, fed_f1, width, label='Federated', color='steelblue', alpha=0.8)
    axes[0, 0].bar(x + width/2, cent_f1, width, label='Centralized', color='coral', alpha=0.8)
    axes[0, 0].set_ylabel('F1 Score', fontweight='bold')
    axes[0, 0].set_title('F1 Score: Federated vs Centralized', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Accuracy Comparison
    fed_acc = [p['fed_acc'] for p in pairs]
    cent_acc = [p['cent_acc'] for p in pairs]
    
    axes[0, 1].bar(x - width/2, fed_acc, width, label='Federated', color='steelblue', alpha=0.8)
    axes[0, 1].bar(x + width/2, cent_acc, width, label='Centralized', color='coral', alpha=0.8)
    axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 1].set_title('Accuracy: Federated vs Centralized', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Performance Gap
    f1_gaps = [p['fed_f1'] - p['cent_f1'] for p in pairs]
    colors = ['green' if g >= 0 else 'red' for g in f1_gaps]
    
    axes[1, 0].bar(names, f1_gaps, color=colors, alpha=0.7)
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_ylabel('F1 Gap (Fed - Cent)', fontweight='bold')
    axes[1, 0].set_title('Performance Gap Analysis', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Summary Statistics
    avg_fed_f1 = np.mean(fed_f1)
    avg_cent_f1 = np.mean(cent_f1)
    avg_gap = np.mean(f1_gaps)
    
    summary_text = f"""
    Average F1 Scores:
    Federated:    {avg_fed_f1:.4f}
    Centralized:  {avg_cent_f1:.4f}
    Avg Gap:      {avg_gap:+.4f}
    
    Models Compared: {len(pairs)}
    Fed > Cent: {sum(1 for g in f1_gaps if g > 0)}
    Cent > Fed: {sum(1 for g in f1_gaps if g < 0)}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                   family='monospace', verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.suptitle('Plot 16: Federated vs Centralized Training Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_16_federated_vs_centralized.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 16: Federated vs Centralized")


def plot_17_model_count_comparison(results: List[Dict]):
    """Plot 17: Comparison Across All Models in Each Category"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Group by model type
    for idx, model_type in enumerate(['llm', 'vit', 'vlm']):
        type_results = [r for r in results if r['config']['model_type'] == model_type 
                       and 'centralized' not in r['model_name']]
        
        if not type_results:
            continue
        
        names = [r['config']['name'] for r in type_results]
        f1_scores = [r['final_metrics']['f1_macro'] for r in type_results]
        
        # Sort by F1
        sorted_indices = np.argsort(f1_scores)[::-1]
        names = [names[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        
        axes[idx].barh(names, f1_scores, color=colors, alpha=0.8)
        axes[idx].set_xlabel('F1 Score', fontweight='bold')
        axes[idx].set_title(f'{model_type.upper()} Models ({len(names)} models)', 
                          fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (name, score) in enumerate(zip(names, f1_scores)):
            axes[idx].text(score + 0.005, i, f'{score:.3f}', 
                         va='center', fontweight='bold')
    
    plt.suptitle('Plot 17: Performance Across All Models by Category', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_17_model_count_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 17: Model Count Comparison")


def plot_18_training_paradigm_efficiency(results: List[Dict]):
    """Plot 18: Training Efficiency - Federated vs Centralized"""
    plt.figure(figsize=(14, 8))
    
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    cent_results = [r for r in results if 'centralized' in r['model_name']]
    
    # Extract convergence data
    for fed_r in fed_results:
        base_name = fed_r['model_name']
        cent_name = f"{base_name}_centralized"
        cent_r = next((r for r in cent_results if r['model_name'] == cent_name), None)
        
        if cent_r:
            # Federated convergence
            fed_history = fed_r['training_history']
            fed_rounds = fed_history.get('rounds', [])
            fed_f1 = fed_history.get('val_f1', [])
            
            # Centralized convergence
            cent_history = cent_r['training_history']
            cent_epochs = cent_history.get('epochs', [])
            cent_f1 = cent_history.get('val_f1', [])
            
            name = fed_r['config']['name']
            
            plt.plot(fed_rounds, fed_f1, marker='o', label=f'{name} (Fed)', 
                    linewidth=2, linestyle='-')
            plt.plot(cent_epochs, cent_f1, marker='s', label=f'{name} (Cent)', 
                    linewidth=2, linestyle='--', alpha=0.7)
    
    plt.xlabel('Training Steps (Rounds/Epochs)', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title('Plot 18: Training Efficiency - Convergence Comparison', 
             fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_18_training_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 18: Training Efficiency")


def plot_19_comprehensive_leaderboard(results: List[Dict]):
    """Plot 19: Complete Leaderboard - All Models Ranked"""
    plt.figure(figsize=(14, max(10, len(results) * 0.4)))
    
    # Calculate composite scores
    rankings = []
    for r in results:
        m = r['final_metrics']
        composite = (m['f1_macro'] * 0.4 + m['accuracy'] * 0.3 + 
                    m['precision'] * 0.15 + m['recall'] * 0.15)
        
        paradigm = 'Centralized' if 'centralized' in r['model_name'] else 'Federated'
        
        rankings.append({
            'name': r['config']['name'],
            'paradigm': paradigm,
            'type': r['config']['model_type'].upper(),
            'f1': m['f1_macro'],
            'acc': m['accuracy'],
            'composite': composite
        })
    
    # Sort by composite score
    rankings.sort(key=lambda x: x['composite'], reverse=True)
    
    # Create leaderboard
    names = [f"{r['name']} ({r['paradigm']})" for r in rankings]
    scores = [r['composite'] for r in rankings]
    types = [r['type'] for r in rankings]
    
    type_colors = {'LLM': 'steelblue', 'VIT': 'coral', 'VLM': 'mediumseagreen'}
    colors = [type_colors.get(t, 'gray') for t in types]
    
    y_pos = np.arange(len(names))
    
    plt.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.yticks(y_pos, names, fontsize=9)
    plt.xlabel('Composite Performance Score', fontsize=12, fontweight='bold')
    plt.title('Plot 19: Complete Leaderboard - All Models Ranked', 
             fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add rank numbers
    for i, (pos, score) in enumerate(zip(y_pos, scores)):
        rank = i + 1
        plt.text(0.01, pos, f'#{rank}', va='center', ha='left', 
                fontweight='bold', fontsize=10, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        plt.text(score + 0.005, pos, f'{score:.4f}', 
                va='center', fontweight='bold', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='LLM'),
        Patch(facecolor='coral', label='ViT'),
        Patch(facecolor='mediumseagreen', label='VLM')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_19_comprehensive_leaderboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 19: Comprehensive Leaderboard")


def plot_20_model_architecture_comparison(results: List[Dict]):
    """Plot 20: Architecture-wise Performance Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Group by architecture
    architectures = {}
    for r in results:
        arch = r['config']['architecture']
        if arch not in architectures:
            architectures[arch] = []
        architectures[arch].append(r)
    
    # Average F1 by architecture
    arch_names = list(architectures.keys())
    arch_f1_avg = [np.mean([r['final_metrics']['f1_macro'] for r in architectures[a]]) 
                   for a in arch_names]
    arch_f1_std = [np.std([r['final_metrics']['f1_macro'] for r in architectures[a]]) 
                   for a in arch_names]
    
    axes[0, 0].bar(arch_names, arch_f1_avg, yerr=arch_f1_std, capsize=5, 
                  color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('Average F1 Score', fontweight='bold')
    axes[0, 0].set_title('Average F1 by Architecture', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Model count by architecture
    arch_counts = [len(architectures[a]) for a in arch_names]
    
    axes[0, 1].bar(arch_names, arch_counts, color='coral', alpha=0.7)
    axes[0, 1].set_ylabel('Number of Models', fontweight='bold')
    axes[0, 1].set_title('Model Count by Architecture', fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Best model per architecture
    best_per_arch = []
    for arch in arch_names:
        best = max(architectures[arch], key=lambda x: x['final_metrics']['f1_macro'])
        best_per_arch.append(best['final_metrics']['f1_macro'])
    
    axes[1, 0].bar(arch_names, best_per_arch, color='mediumseagreen', alpha=0.7)
    axes[1, 0].set_ylabel('Best F1 Score', fontweight='bold')
    axes[1, 0].set_title('Best Model per Architecture', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Architecture statistics table
    stats_text = "Architecture Statistics:\n\n"
    stats_text += f"{'Architecture':<20} {'Avg F1':<10} {'Best F1':<10} {'Count':<8}\n"
    stats_text += "-" * 50 + "\n"
    
    for i, arch in enumerate(arch_names):
        stats_text += f"{arch:<20} {arch_f1_avg[i]:<10.4f} {best_per_arch[i]:<10.4f} {arch_counts[i]:<8}\n"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                   family='monospace', verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.suptitle('Plot 20: Architecture-wise Performance Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_20_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 20: Architecture Comparison")


def plot_20_model_architecture_comparison(results: List[Dict]):
    """Plot 20: Architecture-wise Performance Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Group by architecture
    architectures = {}
    for r in results:
        arch = r['config']['architecture']
        if arch not in architectures:
            architectures[arch] = []
        architectures[arch].append(r)
    
    # Average F1 by architecture
    arch_names = list(architectures.keys())
    arch_f1_avg = [np.mean([r['final_metrics']['f1_macro'] for r in architectures[a]]) 
                   for a in arch_names]
    arch_f1_std = [np.std([r['final_metrics']['f1_macro'] for r in architectures[a]]) 
                   for a in arch_names]
    
    axes[0, 0].bar(arch_names, arch_f1_avg, yerr=arch_f1_std, capsize=5, 
                  color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('Average F1 Score', fontweight='bold')
    axes[0, 0].set_title('Average F1 by Architecture', fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Model count by architecture
    arch_counts = [len(architectures[a]) for a in arch_names]
    
    axes[0, 1].bar(arch_names, arch_counts, color='coral', alpha=0.7)
    axes[0, 1].set_ylabel('Number of Models', fontweight='bold')
    axes[0, 1].set_title('Model Count by Architecture', fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Best model per architecture
    best_per_arch = []
    for arch in arch_names:
        best = max(architectures[arch], key=lambda x: x['final_metrics']['f1_macro'])
        best_per_arch.append(best['final_metrics']['f1_macro'])
    
    axes[1, 0].bar(arch_names, best_per_arch, color='mediumseagreen', alpha=0.7)
    axes[1, 0].set_ylabel('Best F1 Score', fontweight='bold')
    axes[1, 0].set_title('Best Model per Architecture', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Architecture statistics table
    stats_text = "Architecture Statistics:\n\n"
    stats_text += f"{'Architecture':<20} {'Avg F1':<10} {'Best F1':<10} {'Count':<8}\n"
    stats_text += "-" * 50 + "\n"
    
    for i, arch in enumerate(arch_names):
        stats_text += f"{arch:<20} {arch_f1_avg[i]:<10.4f} {best_per_arch[i]:<10.4f} {arch_counts[i]:<8}\n"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                   family='monospace', verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.suptitle('Plot 20: Architecture-wise Performance Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_20_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 20: Architecture Comparison")


def plot_21_llm_vs_vit_vs_vlm_direct(results: List[Dict]):
    """Plot 21: Direct Comparison - LLM vs ViT vs VLM"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Filter out centralized versions for category comparison
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    
    # Group by model type
    llm_results = [r for r in fed_results if r['config']['model_type'] == 'llm']
    vit_results = [r for r in fed_results if r['config']['model_type'] == 'vit']
    vlm_results = [r for r in fed_results if r['config']['model_type'] == 'vlm']
    
    categories = ['LLM', 'ViT', 'VLM']
    category_results = [llm_results, vit_results, vlm_results]
    colors_cat = ['steelblue', 'coral', 'mediumseagreen']
    
    # Average F1 scores
    avg_f1 = [np.mean([r['final_metrics']['f1_macro'] for r in cat]) 
              for cat in category_results]
    std_f1 = [np.std([r['final_metrics']['f1_macro'] for r in cat]) 
              for cat in category_results]
    
    axes[0, 0].bar(categories, avg_f1, yerr=std_f1, capsize=10, 
                  color=colors_cat, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Average F1: LLM vs ViT vs VLM', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, (cat, score) in enumerate(zip(categories, avg_f1)):
        axes[0, 0].text(i, score + 0.01, f'{score:.4f}', ha='center', 
                       fontweight='bold', fontsize=11)
    
    # Best model from each category
    best_f1 = [max([r['final_metrics']['f1_macro'] for r in cat]) 
               for cat in category_results]
    
    axes[0, 1].bar(categories, best_f1, color=colors_cat, alpha=0.7, 
                  edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('Best F1 Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Best Model: LLM vs ViT vs VLM', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, (cat, score) in enumerate(zip(categories, best_f1)):
        axes[0, 1].text(i, score + 0.01, f'{score:.4f}', ha='center', 
                       fontweight='bold', fontsize=11)
    
    # Accuracy comparison
    avg_acc = [np.mean([r['final_metrics']['accuracy'] for r in cat]) 
               for cat in category_results]
    
    axes[0, 2].bar(categories, avg_acc, color=colors_cat, alpha=0.7, 
                  edgecolor='black', linewidth=2)
    axes[0, 2].set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('Accuracy: LLM vs ViT vs VLM', fontsize=12, fontweight='bold')
    axes[0, 2].grid(axis='y', alpha=0.3)
    for i, (cat, score) in enumerate(zip(categories, avg_acc)):
        axes[0, 2].text(i, score + 0.01, f'{score:.4f}', ha='center', 
                       fontweight='bold', fontsize=11)
    
    # Distribution of F1 scores (box plot)
    f1_distributions = [[r['final_metrics']['f1_macro'] for r in cat] 
                        for cat in category_results]
    
    bp = axes[1, 0].boxplot(f1_distributions, labels=categories, patch_artist=True,
                           widths=0.6, showmeans=True)
    for patch, color in zip(bp['boxes'], colors_cat):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('F1 Score Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('F1 Score Variance Across Categories', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Model count and average performance
    model_counts = [len(cat) for cat in category_results]
    
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, model_counts, width, label='Model Count', 
                   color='lightblue', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x_pos + width/2, avg_f1, width, label='Avg F1', 
                   color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax1.set_ylabel('Model Count', fontsize=12, fontweight='bold', color='blue')
    ax2.set_ylabel('Average F1', fontsize=12, fontweight='bold', color='red')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories)
    ax1.set_title('Model Count vs Performance', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Winner analysis
    winner_idx = np.argmax(avg_f1)
    winner_cat = categories[winner_idx]
    winner_score = avg_f1[winner_idx]
    
    best_model_idx = [i for i, cat in enumerate(category_results) 
                     for r in cat if r['final_metrics']['f1_macro'] == best_f1[winner_idx]]
    
    summary_text = f"""
    === CATEGORY COMPARISON ===
    
    Winner: {winner_cat}
    Average F1: {winner_score:.4f}
    Best F1: {best_f1[winner_idx]:.4f}
    Model Count: {model_counts[winner_idx]}
    
    === ALL CATEGORIES ===
    LLM: {avg_f1[0]:.4f} (±{std_f1[0]:.4f})
    ViT: {avg_f1[1]:.4f} (±{std_f1[1]:.4f})
    VLM: {avg_f1[2]:.4f} (±{std_f1[2]:.4f})
    
    Performance Gap:
    LLM-ViT: {avg_f1[0]-avg_f1[1]:+.4f}
    ViT-VLM: {avg_f1[1]-avg_f1[2]:+.4f}
    LLM-VLM: {avg_f1[0]-avg_f1[2]:+.4f}
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, 
                   family='monospace', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[1, 2].axis('off')
    
    plt.suptitle('Plot 21: Direct Category Comparison - LLM vs ViT vs VLM', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_21_llm_vit_vlm_direct.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 21: LLM vs ViT vs VLM Direct Comparison")


def plot_22_modality_effectiveness(results: List[Dict]):
    """Plot 22: Modality Effectiveness Analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    
    # Group by modality
    text_only = [r for r in fed_results if r['config']['model_type'] == 'llm']
    vision_only = [r for r in fed_results if r['config']['model_type'] == 'vit']
    multimodal = [r for r in fed_results if r['config']['model_type'] == 'vlm']
    
    modalities = ['Text Only\n(LLM)', 'Vision Only\n(ViT)', 'Multimodal\n(VLM)']
    modality_groups = [text_only, vision_only, multimodal]
    
    # Metrics comparison across modalities
    metrics_names = ['F1 Macro', 'Accuracy', 'Precision', 'Recall']
    metric_keys = ['f1_macro', 'accuracy', 'precision', 'recall']
    
    x = np.arange(len(modalities))
    width = 0.2
    
    for i, (metric_name, metric_key) in enumerate(zip(metrics_names, metric_keys)):
        values = [np.mean([r['final_metrics'][metric_key] for r in group]) 
                 for group in modality_groups]
        axes[0, 0].bar(x + i*width, values, width, label=metric_name, alpha=0.8)
    
    axes[0, 0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('All Metrics: Modality Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(modalities)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Convergence speed comparison
    for idx, (group, mod_name, color) in enumerate(zip(modality_groups, 
                                                        ['LLM', 'ViT', 'VLM'],
                                                        ['steelblue', 'coral', 'mediumseagreen'])):
        # Average convergence curve
        max_rounds = max([len(r['training_history'].get('rounds', [])) for r in group])
        avg_curve = []
        for round_idx in range(max_rounds):
            round_f1s = [r['training_history']['val_f1'][round_idx] 
                        for r in group 
                        if round_idx < len(r['training_history'].get('val_f1', []))]
            if round_f1s:
                avg_curve.append(np.mean(round_f1s))
        
        axes[0, 1].plot(range(len(avg_curve)), avg_curve, marker='o', 
                       label=mod_name, linewidth=2, color=color, markersize=6)
    
    axes[0, 1].set_xlabel('Training Round', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Convergence Speed by Modality', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance consistency (variance)
    variances = [np.var([r['final_metrics']['f1_macro'] for r in group]) 
                for group in modality_groups]
    
    axes[1, 0].bar(modalities, variances, color=['steelblue', 'coral', 'mediumseagreen'], 
                  alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('F1 Variance', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Performance Consistency (Lower = More Consistent)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    for i, (mod, var) in enumerate(zip(modalities, variances)):
        axes[1, 0].text(i, var + 0.0001, f'{var:.6f}', ha='center', fontweight='bold')
    
    # Statistical summary
    summary_data = []
    for mod_name, group in zip(['LLM', 'ViT', 'VLM'], modality_groups):
        f1_scores = [r['final_metrics']['f1_macro'] for r in group]
        summary_data.append({
            'Modality': mod_name,
            'Mean': np.mean(f1_scores),
            'Std': np.std(f1_scores),
            'Min': np.min(f1_scores),
            'Max': np.max(f1_scores),
            'Range': np.max(f1_scores) - np.min(f1_scores)
        })
    
    summary_text = "Statistical Summary (F1 Scores):\n\n"
    summary_text += f"{'Mod':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Range':<8}\n"
    summary_text += "-" * 52 + "\n"
    
    for s in summary_data:
        summary_text += f"{s['Modality']:<6} {s['Mean']:<8.4f} {s['Std']:<8.4f} "
        summary_text += f"{s['Min']:<8.4f} {s['Max']:<8.4f} {s['Range']:<8.4f}\n"
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                   family='monospace', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    axes[1, 1].axis('off')
    
    plt.suptitle('Plot 22: Modality Effectiveness Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_22_modality_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 22: Modality Effectiveness")


def plot_23_category_champions(results: List[Dict]):
    """Plot 23: Champion Models from Each Category"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    
    # Find best model from each category
    llm_results = [r for r in fed_results if r['config']['model_type'] == 'llm']
    vit_results = [r for r in fed_results if r['config']['model_type'] == 'vit']
    vlm_results = [r for r in fed_results if r['config']['model_type'] == 'vlm']
    
    best_llm = max(llm_results, key=lambda x: x['final_metrics']['f1_macro'])
    best_vit = max(vit_results, key=lambda x: x['final_metrics']['f1_macro'])
    best_vlm = max(vlm_results, key=lambda x: x['final_metrics']['f1_macro'])
    
    champions = [best_llm, best_vit, best_vlm]
    champion_names = [c['config']['name'] for c in champions]
    categories = ['LLM\nChampion', 'ViT\nChampion', 'VLM\nChampion']
    
    # F1 comparison
    f1_scores = [c['final_metrics']['f1_macro'] for c in champions]
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    bars = axes[0, 0].bar(categories, f1_scores, color=colors, alpha=0.7, 
                         edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Champion F1 Scores', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    for i, (bar, name, score) in enumerate(zip(bars, champion_names, f1_scores)):
        axes[0, 0].text(i, score + 0.01, f'{name}\n{score:.4f}', 
                       ha='center', fontweight='bold', fontsize=9)
    
    # All metrics radar comparison
    metrics = ['f1_macro', 'accuracy', 'precision', 'recall']
    metric_labels = ['F1', 'Acc', 'Prec', 'Rec']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    ax_radar = plt.subplot(2, 2, 2, projection='polar')
    
    for champ, color, cat_name in zip(champions, colors, ['LLM', 'ViT', 'VLM']):
        values = [champ['final_metrics'][m] for m in metrics]
        values += values[:1]
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=cat_name, color=color)
        ax_radar.fill(angles, values, alpha=0.15, color=color)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Champion Metrics Comparison', fontsize=12, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax_radar.grid(True)
    
    # Training convergence
    for champ, color, cat_name in zip(champions, colors, ['LLM', 'ViT', 'VLM']):
        history = champ['training_history']
        rounds = history.get('rounds', [])
        val_f1 = history.get('val_f1', [])
        axes[1, 0].plot(rounds, val_f1, marker='o', linewidth=2, 
                       label=cat_name, color=color, markersize=6)
    
    axes[1, 0].set_xlabel('Training Round', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Validation F1', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Champion Convergence Curves', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Champion details table
    details_text = "=== CHAMPION MODELS ===\n\n"
    
    for cat_name, champ, color_name in zip(['LLM', 'ViT', 'VLM'], champions, 
                                            ['Blue', 'Orange', 'Green']):
        details_text += f"{cat_name} Champion ({color_name}):\n"
        details_text += f"  Model: {champ['config']['name']}\n"
        details_text += f"  Architecture: {champ['config']['architecture']}\n"
        details_text += f"  F1 Score: {champ['final_metrics']['f1_macro']:.4f}\n"
        details_text += f"  Accuracy: {champ['final_metrics']['accuracy']:.4f}\n"
        details_text += f"  Precision: {champ['final_metrics']['precision']:.4f}\n"
        details_text += f"  Recall: {champ['final_metrics']['recall']:.4f}\n\n"
    
    # Overall winner
    overall_winner = max(champions, key=lambda x: x['final_metrics']['f1_macro'])
    winner_cat = ['LLM', 'ViT', 'VLM'][champions.index(overall_winner)]
    
    details_text += f"{'='*40}\n"
    details_text += f"OVERALL WINNER: {winner_cat}\n"
    details_text += f"Model: {overall_winner['config']['name']}\n"
    details_text += f"F1: {overall_winner['final_metrics']['f1_macro']:.4f}\n"
    
    axes[1, 1].text(0.05, 0.5, details_text, fontsize=10, 
                   family='monospace', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.suptitle('Plot 23: Category Champions - Best of LLM, ViT, VLM', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_23_category_champions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 23: Category Champions")


def plot_24_cross_category_detailed(results: List[Dict]):
    """Plot 24: Detailed Cross-Category Analysis"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    
    # Group by category
    llm_results = [r for r in fed_results if r['config']['model_type'] == 'llm']
    vit_results = [r for r in fed_results if r['config']['model_type'] == 'vit']
    vlm_results = [r for r in fed_results if r['config']['model_type'] == 'vlm']
    
    all_categories = [llm_results, vit_results, vlm_results]
    cat_names = ['LLM', 'ViT', 'VLM']
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    # Individual category distributions (3 histograms)
    for idx, (cat_results, name, color) in enumerate(zip(all_categories, cat_names, colors)):
        ax = fig.add_subplot(gs[0, idx])
        f1_scores = [r['final_metrics']['f1_macro'] for r in cat_results]
        
        ax.hist(f1_scores, bins=10, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(f1_scores), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(f1_scores):.4f}')
        ax.set_xlabel('F1 Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{name} F1 Distribution ({len(cat_results)} models)', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # Combined violin plot
    ax_violin = fig.add_subplot(gs[1, :])
    
    f1_data = [[r['final_metrics']['f1_macro'] for r in cat] for cat in all_categories]
    parts = ax_violin.violinplot(f1_data, positions=[1, 2, 3], showmeans=True, 
                                 showmedians=True, widths=0.7)
    
    for idx, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax_violin.set_xticks([1, 2, 3])
    ax_violin.set_xticklabels(cat_names, fontweight='bold')
    ax_violin.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax_violin.set_title('F1 Score Distribution Across Categories (Violin Plot)', 
                       fontsize=13, fontweight='bold')
    ax_violin.grid(axis='y', alpha=0.3)
    
    # Pairwise comparison matrix
    ax_matrix = fig.add_subplot(gs[2, :2])
    
    # Calculate average metrics for each category
    comparison_matrix = np.zeros((3, 4))  # 3 categories x 4 metrics
    metrics = ['f1_macro', 'accuracy', 'precision', 'recall']
    
    for i, cat_results in enumerate(all_categories):
        for j, metric in enumerate(metrics):
            comparison_matrix[i, j] = np.mean([r['final_metrics'][metric] 
                                              for r in cat_results])
    
    im = ax_matrix.imshow(comparison_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax_matrix.set_xticks(np.arange(len(metrics)))
    ax_matrix.set_yticks(np.arange(len(cat_names)))
    ax_matrix.set_xticklabels(['F1', 'Accuracy', 'Precision', 'Recall'], fontweight='bold')
    ax_matrix.set_yticklabels(cat_names, fontweight='bold')
    ax_matrix.set_title('Average Metrics Heatmap', fontsize=13, fontweight='bold')
    
    # Add values to heatmap
    for i in range(len(cat_names)):
        for j in range(len(metrics)):
            text = ax_matrix.text(j, i, f'{comparison_matrix[i, j]:.3f}',
                                ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax_matrix, label='Score')
    
    # Statistical significance summary
    ax_stats = fig.add_subplot(gs[2, 2])
    
    stats_text = "Statistical Analysis:\n\n"
    
    # Calculate pairwise differences
    llm_f1 = [r['final_metrics']['f1_macro'] for r in llm_results]
    vit_f1 = [r['final_metrics']['f1_macro'] for r in vit_results]
    vlm_f1 = [r['final_metrics']['f1_macro'] for r in vlm_results]
    
    stats_text += f"LLM vs ViT:\n"
    stats_text += f"  Δ Mean: {np.mean(llm_f1) - np.mean(vit_f1):+.4f}\n\n"
    
    stats_text += f"ViT vs VLM:\n"
    stats_text += f"  Δ Mean: {np.mean(vit_f1) - np.mean(vlm_f1):+.4f}\n\n"
    
    stats_text += f"LLM vs VLM:\n"
    stats_text += f"  Δ Mean: {np.mean(llm_f1) - np.mean(vlm_f1):+.4f}\n\n"
    
    stats_text += f"{'='*25}\n"
    stats_text += f"Category Rankings:\n"
    
    avg_scores = [np.mean(llm_f1), np.mean(vit_f1), np.mean(vlm_f1)]
    rankings = np.argsort(avg_scores)[::-1]
    
    for rank, idx in enumerate(rankings, 1):
        stats_text += f"{rank}. {cat_names[idx]}: {avg_scores[idx]:.4f}\n"
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=11, 
                 family='monospace', verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax_stats.axis('off')
    
    plt.suptitle('Plot 24: Detailed Cross-Category Analysis - LLM vs ViT vs VLM', 
                 fontsize=16, fontweight='bold')
    plt.savefig(PLOTS_DIR / 'plot_24_cross_category_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 24: Cross-Category Detailed Analysis")


def plot_25_all_models_comparison_grid(results: List[Dict]):
    """Plot 25: Complete Grid Comparison of All Models"""
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    
    # Sort by F1 score
    fed_results = sorted(fed_results, key=lambda x: x['final_metrics']['f1_macro'], 
                        reverse=True)
    
    n_models = len(fed_results)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, n_models * 0.8))
    
    if n_models == 1:
        axes = [axes]
    
    # Create horizontal bar for each model
    for idx, (ax, result) in enumerate(zip(axes, fed_results)):
        metrics = result['final_metrics']
        model_name = result['config']['name']
        model_type = result['config']['model_type'].upper()
        
        # Color by category
        if model_type == 'LLM':
            color = 'steelblue'
        elif model_type == 'VIT':
            color = 'coral'
        else:
            color = 'mediumseagreen'
        
        # Plot metrics
        metric_names = ['F1', 'Acc', 'Prec', 'Rec']
        metric_values = [metrics['f1_macro'], metrics['accuracy'], 
                        metrics['precision'], metrics['recall']]
        
        y_pos = np.arange(len(metric_names))
        ax.barh(y_pos, metric_values, color=color, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Score', fontsize=9)
        
        # Add rank and model info
        rank = idx + 1
        title = f"#{rank} {model_name} ({model_type}) - F1: {metrics['f1_macro']:.4f}"
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        
        # Add value labels
        for i, v in enumerate(metric_values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8, fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Plot 25: Complete Model Comparison Grid (All 24 Models Ranked)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_25_all_models_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 25: All Models Comparison Grid")


def plot_26_llm_models_detailed_comparison(results: List[Dict]):
    """Plot 26: Detailed Comparison Between All LLM Models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    llm_results = [r for r in fed_results if r['config']['model_type'] == 'llm']
    
    if not llm_results:
        print("[SKIP] Plot 26: No LLM results found")
        return
    
    # Sort by F1 score
    llm_results = sorted(llm_results, key=lambda x: x['final_metrics']['f1_macro'], reverse=True)
    
    names = [r['config']['name'] for r in llm_results]
    f1_scores = [r['final_metrics']['f1_macro'] for r in llm_results]
    accuracies = [r['final_metrics']['accuracy'] for r in llm_results]
    precisions = [r['final_metrics']['precision'] for r in llm_results]
    recalls = [r['final_metrics']['recall'] for r in llm_results]
    
    # F1 Score Ranking
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    axes[0, 0].barh(names, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_xlabel('F1 Score', fontweight='bold', fontsize=11)
    axes[0, 0].set_title('LLM Models - F1 Score Ranking', fontweight='bold', fontsize=12)
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, (name, score) in enumerate(zip(names, f1_scores)):
        axes[0, 0].text(score + 0.005, i, f'{score:.4f}', va='center', fontweight='bold', fontsize=9)
    
    # All Metrics Comparison
    x = np.arange(len(names))
    width = 0.2
    
    axes[0, 1].bar(x - 1.5*width, f1_scores, width, label='F1', color='steelblue', alpha=0.8)
    axes[0, 1].bar(x - 0.5*width, accuracies, width, label='Acc', color='coral', alpha=0.8)
    axes[0, 1].bar(x + 0.5*width, precisions, width, label='Prec', color='lightgreen', alpha=0.8)
    axes[0, 1].bar(x + 1.5*width, recalls, width, label='Rec', color='orange', alpha=0.8)
    
    axes[0, 1].set_ylabel('Score', fontweight='bold', fontsize=11)
    axes[0, 1].set_title('All Metrics - LLM Models', fontweight='bold', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Convergence Curves
    for result in llm_results[:5]:  # Top 5 LLMs
        history = result['training_history']
        rounds = history.get('rounds', [])
        val_f1 = history.get('val_f1', [])
        axes[0, 2].plot(rounds, val_f1, marker='o', label=result['config']['name'], linewidth=2)
    
    axes[0, 2].set_xlabel('Training Round', fontweight='bold', fontsize=11)
    axes[0, 2].set_ylabel('Validation F1', fontweight='bold', fontsize=11)
    axes[0, 2].set_title('Training Convergence - Top 5 LLMs', fontweight='bold', fontsize=12)
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Architecture-wise grouping
    arch_groups = {}
    for r in llm_results:
        arch = r['config']['architecture']
        if arch not in arch_groups:
            arch_groups[arch] = []
        arch_groups[arch].append(r['final_metrics']['f1_macro'])
    
    arch_names = list(arch_groups.keys())
    arch_avg = [np.mean(arch_groups[a]) for a in arch_names]
    arch_max = [np.max(arch_groups[a]) for a in arch_names]
    
    x_arch = np.arange(len(arch_names))
    width_arch = 0.35
    
    axes[1, 0].bar(x_arch - width_arch/2, arch_avg, width_arch, label='Average', 
                  color='steelblue', alpha=0.7)
    axes[1, 0].bar(x_arch + width_arch/2, arch_max, width_arch, label='Best', 
                  color='darkblue', alpha=0.7)
    axes[1, 0].set_ylabel('F1 Score', fontweight='bold', fontsize=11)
    axes[1, 0].set_title('LLM Architecture Comparison', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x_arch)
    axes[1, 0].set_xticklabels(arch_names, rotation=45, ha='right', fontsize=9)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Pairwise Performance Gaps
    if len(llm_results) > 1:
        gaps = []
        gap_labels = []
        for i in range(len(llm_results) - 1):
            gap = f1_scores[i] - f1_scores[i+1]
            gaps.append(gap)
            gap_labels.append(f"{names[i][:8]}\nvs\n{names[i+1][:8]}")
        
        colors_gap = ['green' if g > 0.01 else 'orange' if g > 0.005 else 'red' for g in gaps]
        axes[1, 1].bar(range(len(gaps)), gaps, color=colors_gap, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('F1 Gap', fontweight='bold', fontsize=11)
        axes[1, 1].set_title('Performance Gaps Between Consecutive Models', fontweight='bold', fontsize=12)
        axes[1, 1].set_xticks(range(len(gaps)))
        axes[1, 1].set_xticklabels(gap_labels, fontsize=7)
        axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Summary Statistics
    summary_text = f"""
    === LLM MODELS SUMMARY ===
    
    Total Models: {len(llm_results)}
    
    Best Model:
      {names[0]}
      F1: {f1_scores[0]:.4f}
      Acc: {accuracies[0]:.4f}
    
    Worst Model:
      {names[-1]}
      F1: {f1_scores[-1]:.4f}
      Acc: {accuracies[-1]:.4f}
    
    Average F1: {np.mean(f1_scores):.4f}
    Std Dev: {np.std(f1_scores):.4f}
    Range: {max(f1_scores) - min(f1_scores):.4f}
    
    Top 3 LLMs:
    1. {names[0]}: {f1_scores[0]:.4f}
    2. {names[1]}: {f1_scores[1]:.4f}
    3. {names[2]}: {f1_scores[2]:.4f}
    """
    
    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=10, 
                   family='monospace', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
    axes[1, 2].axis('off')
    
    plt.suptitle('Plot 26: Detailed Comparison Between All LLM Models', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_26_llm_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 26: LLM Models Detailed Comparison")


def plot_27_vit_models_detailed_comparison(results: List[Dict]):
    """Plot 27: Detailed Comparison Between All ViT Models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    vit_results = [r for r in fed_results if r['config']['model_type'] == 'vit']
    
    if not vit_results:
        print("[SKIP] Plot 27: No ViT results found")
        return
    
    # Sort by F1 score
    vit_results = sorted(vit_results, key=lambda x: x['final_metrics']['f1_macro'], reverse=True)
    
    names = [r['config']['name'] for r in vit_results]
    f1_scores = [r['final_metrics']['f1_macro'] for r in vit_results]
    accuracies = [r['final_metrics']['accuracy'] for r in vit_results]
    precisions = [r['final_metrics']['precision'] for r in vit_results]
    recalls = [r['final_metrics']['recall'] for r in vit_results]
    
    # F1 Score Ranking
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(names)))
    axes[0, 0].barh(names, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_xlabel('F1 Score', fontweight='bold', fontsize=11)
    axes[0, 0].set_title('ViT Models - F1 Score Ranking', fontweight='bold', fontsize=12)
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, (name, score) in enumerate(zip(names, f1_scores)):
        axes[0, 0].text(score + 0.005, i, f'{score:.4f}', va='center', fontweight='bold', fontsize=9)
    
    # All Metrics Comparison
    x = np.arange(len(names))
    width = 0.2
    
    axes[0, 1].bar(x - 1.5*width, f1_scores, width, label='F1', color='coral', alpha=0.8)
    axes[0, 1].bar(x - 0.5*width, accuracies, width, label='Acc', color='orange', alpha=0.8)
    axes[0, 1].bar(x + 0.5*width, precisions, width, label='Prec', color='tomato', alpha=0.8)
    axes[0, 1].bar(x + 1.5*width, recalls, width, label='Rec', color='darkorange', alpha=0.8)
    
    axes[0, 1].set_ylabel('Score', fontweight='bold', fontsize=11)
    axes[0, 1].set_title('All Metrics - ViT Models', fontweight='bold', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Convergence Curves
    for result in vit_results[:5]:  # Top 5 ViTs
        history = result['training_history']
        rounds = history.get('rounds', [])
        val_f1 = history.get('val_f1', [])
        axes[0, 2].plot(rounds, val_f1, marker='s', label=result['config']['name'], linewidth=2)
    
    axes[0, 2].set_xlabel('Training Round', fontweight='bold', fontsize=11)
    axes[0, 2].set_ylabel('Validation F1', fontweight='bold', fontsize=11)
    axes[0, 2].set_title('Training Convergence - Top 5 ViTs', fontweight='bold', fontsize=12)
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Architecture-wise grouping
    arch_groups = {}
    for r in vit_results:
        arch = r['config']['architecture']
        if arch not in arch_groups:
            arch_groups[arch] = []
        arch_groups[arch].append(r['final_metrics']['f1_macro'])
    
    arch_names = list(arch_groups.keys())
    arch_avg = [np.mean(arch_groups[a]) for a in arch_names]
    arch_max = [np.max(arch_groups[a]) for a in arch_names]
    
    x_arch = np.arange(len(arch_names))
    width_arch = 0.35
    
    axes[1, 0].bar(x_arch - width_arch/2, arch_avg, width_arch, label='Average', 
                  color='coral', alpha=0.7)
    axes[1, 0].bar(x_arch + width_arch/2, arch_max, width_arch, label='Best', 
                  color='darkorange', alpha=0.7)
    axes[1, 0].set_ylabel('F1 Score', fontweight='bold', fontsize=11)
    axes[1, 0].set_title('ViT Architecture Comparison', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x_arch)
    axes[1, 0].set_xticklabels(arch_names, rotation=45, ha='right', fontsize=9)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Pairwise Performance Gaps
    if len(vit_results) > 1:
        gaps = []
        gap_labels = []
        for i in range(len(vit_results) - 1):
            gap = f1_scores[i] - f1_scores[i+1]
            gaps.append(gap)
            gap_labels.append(f"{names[i][:8]}\nvs\n{names[i+1][:8]}")
        
        colors_gap = ['green' if g > 0.01 else 'orange' if g > 0.005 else 'red' for g in gaps]
        axes[1, 1].bar(range(len(gaps)), gaps, color=colors_gap, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('F1 Gap', fontweight='bold', fontsize=11)
        axes[1, 1].set_title('Performance Gaps Between Consecutive Models', fontweight='bold', fontsize=12)
        axes[1, 1].set_xticks(range(len(gaps)))
        axes[1, 1].set_xticklabels(gap_labels, fontsize=7)
        axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Summary Statistics
    summary_text = f"""
    === ViT MODELS SUMMARY ===
    
    Total Models: {len(vit_results)}
    
    Best Model:
      {names[0]}
      F1: {f1_scores[0]:.4f}
      Acc: {accuracies[0]:.4f}
    
    Worst Model:
      {names[-1]}
      F1: {f1_scores[-1]:.4f}
      Acc: {accuracies[-1]:.4f}
    
    Average F1: {np.mean(f1_scores):.4f}
    Std Dev: {np.std(f1_scores):.4f}
    Range: {max(f1_scores) - min(f1_scores):.4f}
    
    Top 3 ViTs:
    1. {names[0]}: {f1_scores[0]:.4f}
    2. {names[1]}: {f1_scores[1]:.4f}
    3. {names[2]}: {f1_scores[2]:.4f}
    """
    
    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=10, 
                   family='monospace', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
    axes[1, 2].axis('off')
    
    plt.suptitle('Plot 27: Detailed Comparison Between All ViT Models', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_27_vit_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 27: ViT Models Detailed Comparison")


def plot_28_vlm_models_detailed_comparison(results: List[Dict]):
    """Plot 28: Detailed Comparison Between All VLM Models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    fed_results = [r for r in results if 'centralized' not in r['model_name']]
    vlm_results = [r for r in fed_results if r['config']['model_type'] == 'vlm']
    
    if not vlm_results:
        print("[SKIP] Plot 28: No VLM results found")
        return
    
    # Sort by F1 score
    vlm_results = sorted(vlm_results, key=lambda x: x['final_metrics']['f1_macro'], reverse=True)
    
    names = [r['config']['name'] for r in vlm_results]
    f1_scores = [r['final_metrics']['f1_macro'] for r in vlm_results]
    accuracies = [r['final_metrics']['accuracy'] for r in vlm_results]
    precisions = [r['final_metrics']['precision'] for r in vlm_results]
    recalls = [r['final_metrics']['recall'] for r in vlm_results]
    
    # F1 Score Ranking
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(names)))
    axes[0, 0].barh(names, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_xlabel('F1 Score', fontweight='bold', fontsize=11)
    axes[0, 0].set_title('VLM Models - F1 Score Ranking', fontweight='bold', fontsize=12)
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, (name, score) in enumerate(zip(names, f1_scores)):
        axes[0, 0].text(score + 0.005, i, f'{score:.4f}', va='center', fontweight='bold', fontsize=9)
    
    # All Metrics Comparison
    x = np.arange(len(names))
    width = 0.2
    
    axes[0, 1].bar(x - 1.5*width, f1_scores, width, label='F1', color='mediumseagreen', alpha=0.8)
    axes[0, 1].bar(x - 0.5*width, accuracies, width, label='Acc', color='lightgreen', alpha=0.8)
    axes[0, 1].bar(x + 0.5*width, precisions, width, label='Prec', color='seagreen', alpha=0.8)
    axes[0, 1].bar(x + 1.5*width, recalls, width, label='Rec', color='darkgreen', alpha=0.8)
    
    axes[0, 1].set_ylabel('Score', fontweight='bold', fontsize=11)
    axes[0, 1].set_title('All Metrics - VLM Models', fontweight='bold', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Convergence Curves
    for result in vlm_results[:5]:  # Top 5 VLMs
        history = result['training_history']
        rounds = history.get('rounds', [])
        val_f1 = history.get('val_f1', [])
        axes[0, 2].plot(rounds, val_f1, marker='^', label=result['config']['name'], linewidth=2)
    
    axes[0, 2].set_xlabel('Training Round', fontweight='bold', fontsize=11)
    axes[0, 2].set_ylabel('Validation F1', fontweight='bold', fontsize=11)
    axes[0, 2].set_title('Training Convergence - Top 5 VLMs', fontweight='bold', fontsize=12)
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Architecture-wise grouping
    arch_groups = {}
    for r in vlm_results:
        arch = r['config']['architecture']
        if arch not in arch_groups:
            arch_groups[arch] = []
        arch_groups[arch].append(r['final_metrics']['f1_macro'])
    
    arch_names = list(arch_groups.keys())
    arch_avg = [np.mean(arch_groups[a]) for a in arch_names]
    arch_max = [np.max(arch_groups[a]) for a in arch_names]
    
    x_arch = np.arange(len(arch_names))
    width_arch = 0.35
    
    axes[1, 0].bar(x_arch - width_arch/2, arch_avg, width_arch, label='Average', 
                  color='mediumseagreen', alpha=0.7)
    axes[1, 0].bar(x_arch + width_arch/2, arch_max, width_arch, label='Best', 
                  color='darkgreen', alpha=0.7)
    axes[1, 0].set_ylabel('F1 Score', fontweight='bold', fontsize=11)
    axes[1, 0].set_title('VLM Architecture Comparison', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x_arch)
    axes[1, 0].set_xticklabels(arch_names, rotation=45, ha='right', fontsize=9)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Pairwise Performance Gaps
    if len(vlm_results) > 1:
        gaps = []
        gap_labels = []
        for i in range(len(vlm_results) - 1):
            gap = f1_scores[i] - f1_scores[i+1]
            gaps.append(gap)
            gap_labels.append(f"{names[i][:8]}\nvs\n{names[i+1][:8]}")
        
        colors_gap = ['green' if g > 0.01 else 'orange' if g > 0.005 else 'red' for g in gaps]
        axes[1, 1].bar(range(len(gaps)), gaps, color=colors_gap, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('F1 Gap', fontweight='bold', fontsize=11)
        axes[1, 1].set_title('Performance Gaps Between Consecutive Models', fontweight='bold', fontsize=12)
        axes[1, 1].set_xticks(range(len(gaps)))
        axes[1, 1].set_xticklabels(gap_labels, fontsize=7)
        axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Summary Statistics
    summary_text = f"""
    === VLM MODELS SUMMARY ===
    
    Total Models: {len(vlm_results)}
    
    Best Model:
      {names[0]}
      F1: {f1_scores[0]:.4f}
      Acc: {accuracies[0]:.4f}
    
    Worst Model:
      {names[-1]}
      F1: {f1_scores[-1]:.4f}
      Acc: {accuracies[-1]:.4f}
    
    Average F1: {np.mean(f1_scores):.4f}
    Std Dev: {np.std(f1_scores):.4f}
    Range: {max(f1_scores) - min(f1_scores):.4f}
    
    Top 3 VLMs:
    1. {names[0]}: {f1_scores[0]:.4f}
    2. {names[1]}: {f1_scores[1]:.4f}
    3. {names[2]}: {f1_scores[2]:.4f}
    """
    
    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=10, 
                   family='monospace', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4))
    axes[1, 2].axis('off')
    
    plt.suptitle('Plot 28: Detailed Comparison Between All VLM Models', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'plot_28_vlm_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 28: VLM Models Detailed Comparison")


def create_all_plots():
    """Create all 20+ comprehensive plots"""
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE PLOTS")
    print("="*80)
    
    results = load_results()
    
    if not results:
        print("[ERROR] No results found. Please train models first.")
        return
    
    print(f"[INFO] Loaded {len(results)} model results")
    print(f"[INFO] Generating plots in: {PLOTS_DIR}")
    
    # Create all plots
    plot_01_overall_performance(results)
    plot_02_model_type_comparison(results)
    plot_03_training_convergence(results)
    plot_04_baseline_comparison(results)
    plot_05_precision_recall_scatter(results)
    plot_06_metrics_heatmap(results)
    plot_07_federated_rounds_impact(results)
    plot_08_best_vs_worst(results)
    plot_09_improvement_over_rounds(results)
    plot_10_statistical_comparison(results)
    plot_11_radar_chart(results)
    plot_12_convergence_rate(results)
    plot_13_performance_ranking(results)
    plot_14_year_comparison(results)
    plot_15_loss_landscape(results)
    
    # New plots for expanded comparison
    plot_16_federated_vs_centralized(results)
    plot_17_model_count_comparison(results)
    plot_18_training_paradigm_efficiency(results)
    plot_19_comprehensive_leaderboard(results)
    plot_20_model_architecture_comparison(results)
    
    # Cross-category comparison plots (LLM vs ViT vs VLM)
    plot_21_llm_vs_vit_vs_vlm_direct(results)
    plot_22_modality_effectiveness(results)
    plot_23_category_champions(results)
    plot_24_cross_category_detailed(results)
    plot_25_all_models_comparison_grid(results)
    
    # Within-category detailed comparisons
    plot_26_llm_models_detailed_comparison(results)
    plot_27_vit_models_detailed_comparison(results)
    plot_28_vlm_models_detailed_comparison(results)
    
    print(f"\n{'='*80}")
    print(f"✓ Successfully created 28+ plots in {PLOTS_DIR}/")
    print(f"  - Plots 1-15: Performance, convergence, baselines")
    print(f"  - Plots 16-20: Federated vs centralized")
    print(f"  - Plots 21-25: Cross-category (LLM vs ViT vs VLM)")
    print(f"  - Plots 26-28: Within-category (individual model comparisons)")
    print(f"{'='*80}")


if __name__ == "__main__":
    create_all_plots()
