#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
federated_plotting_comparison.py
================================
Part 2: Comprehensive plotting and comparison framework

This module provides 15-20 different plots for:
- Model performance comparison
- Training dynamics
- Per-class analysis
- Communication efficiency
- Comparison with paper baselines
- Statistical analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Import from Part 1
from federated_llm_vit_vlm_complete import (
    ISSUE_LABELS, NUM_LABELS, BASELINE_PAPERS
)

# ============================================================================
# RESULT DATA STRUCTURES
# ============================================================================

class ModelResults:
    """Container for model training and evaluation results"""
    def __init__(self, model_name, model_type, config):
        self.model_name = model_name
        self.model_type = model_type
        self.config = config
        self.metrics_history = []
        self.final_metrics = {}
        self.training_time = 0.0
        self.inference_time = 0.0
        self.memory_mb = 0.0
        self.params_count = 0
        self.communication_cost = 0.0
    
    def to_dict(self):
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'final_micro_f1': self.final_metrics.get('micro_f1', 0),
            'final_macro_f1': self.final_metrics.get('macro_f1', 0),
            'final_accuracy': self.final_metrics.get('accuracy', 0),
            'training_time': self.training_time,
            'params_million': self.params_count / 1e6,
            'memory_mb': self.memory_mb,
        }


class ComparisonFramework:
    """Framework for comprehensive model comparison"""
    def __init__(self, save_dir="results/comparisons"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Store results
        self.results = {}  # model_name -> ModelResults
        self.baseline_results = BASELINE_PAPERS
    
    def add_result(self, result: ModelResults):
        """Add a model result"""
        self.results[result.model_name] = result
        print(f"[Comparison] Added result for {result.model_name}")
    
    def generate_all_plots(self):
        """Generate all 15-20 comparison plots"""
        print(f"\n{'='*80}")
        print("GENERATING COMPREHENSIVE COMPARISON PLOTS")
        print(f"{'='*80}\n")
        
        if len(self.results) == 0:
            print("[ERROR] No results to plot!")
            return
        
        # Plot 1: Overall F1 Score Comparison
        self.plot_overall_f1_comparison()
        
        # Plot 2: Model Type Comparison (LLM vs ViT vs VLM)
        self.plot_model_type_comparison()
        
        # Plot 3: Training Convergence Curves
        self.plot_training_convergence()
        
        # Plot 4: Per-Class Performance Heatmap
        self.plot_per_class_heatmap()
        
        # Plot 5: Per-Class F1 Scores (Bar Chart)
        self.plot_per_class_bar()
        
        # Plot 6: Precision-Recall Trade-off
        self.plot_precision_recall_tradeoff()
        
        # Plot 7: ROC Curves Comparison
        self.plot_roc_curves()
        
        # Plot 8: Model Efficiency (Training Time vs F1)
        self.plot_efficiency_scatter()
        
        # Plot 9: Model Size Comparison
        self.plot_model_size_comparison()
        
        # Plot 10: Memory Usage Comparison
        self.plot_memory_usage()
        
        # Plot 11: Round-by-Round Performance
        self.plot_round_performance()
        
        # Plot 12: Statistical Significance Tests
        self.plot_statistical_tests()
        
        # Plot 13: Comparison with Baseline Papers
        self.plot_paper_comparison()
        
        # Plot 14: Confusion Matrices
        self.plot_confusion_matrices()
        
        # Plot 15: Learning Rate Sensitivity
        self.plot_learning_dynamics()
        
        # Plot 16: Model Architecture Comparison
        self.plot_architecture_comparison()
        
        # Plot 17: Per-Class AUC Scores
        self.plot_per_class_auc()
        
        # Plot 18: Communication Cost Analysis
        self.plot_communication_cost()
        
        # Plot 19: Scalability Analysis
        self.plot_scalability()
        
        # Plot 20: Error Analysis
        self.plot_error_analysis()
        
        # Generate summary report
        self.generate_summary_report()
        
        print(f"\n{'='*80}")
        print(f"All plots saved to: {self.save_dir}")
        print(f"{'='*80}\n")
    
    # ========================================================================
    # PLOT 1: Overall F1 Score Comparison
    # ========================================================================
    def plot_overall_f1_comparison(self):
        """Compare micro and macro F1 scores across all models"""
        print("[Plot 1/20] Overall F1 Score Comparison...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(self.results.keys())
        micro_f1s = [self.results[m].final_metrics.get('micro_f1', 0) for m in models]
        macro_f1s = [self.results[m].final_metrics.get('macro_f1', 0) for m in models]
        
        # Micro F1
        colors1 = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars1 = ax1.barh(models, micro_f1s, color=colors1, alpha=0.8)
        ax1.set_xlabel('Micro-F1 Score')
        ax1.set_title('Micro-F1 Score Comparison')
        ax1.set_xlim(0, 1.0)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, micro_f1s)):
            ax1.text(val + 0.01, i, f'{val:.3f}', va='center')
        
        # Macro F1
        colors2 = plt.cm.plasma(np.linspace(0, 1, len(models)))
        bars2 = ax2.barh(models, macro_f1s, color=colors2, alpha=0.8)
        ax2.set_xlabel('Macro-F1 Score')
        ax2.set_title('Macro-F1 Score Comparison')
        ax2.set_xlim(0, 1.0)
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars2, macro_f1s)):
            ax2.text(val + 0.01, i, f'{val:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '01_overall_f1_comparison.png'), 
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 2: Model Type Comparison
    # ========================================================================
    def plot_model_type_comparison(self):
        """Compare performance by model type (LLM, ViT, VLM)"""
        print("[Plot 2/20] Model Type Comparison...")
        
        # Group by model type
        type_groups = {'llm': [], 'vit': [], 'vlm': []}
        
        for model_name, result in self.results.items():
            model_type = result.model_type
            if model_type in type_groups:
                type_groups[model_type].append(result.final_metrics.get('micro_f1', 0))
        
        # Remove empty groups
        type_groups = {k: v for k, v in type_groups.items() if v}
        
        if not type_groups:
            print("  [SKIP] Not enough data")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        data_for_box = []
        labels_for_box = []
        for type_name, scores in type_groups.items():
            data_for_box.append(scores)
            labels_for_box.append(type_name.upper())
        
        bp = ax1.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['skyblue', 'lightgreen', 'lightcoral']):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('Micro-F1 Score')
        ax1.set_title('Performance by Model Type (Box Plot)')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Bar plot with means
        means = [np.mean(scores) for scores in data_for_box]
        stds = [np.std(scores) for scores in data_for_box]
        
        x = np.arange(len(labels_for_box))
        bars = ax2.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                       color=['skyblue', 'lightgreen', 'lightcoral'][:len(x)])
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_for_box)
        ax2.set_ylabel('Mean Micro-F1 Score')
        ax2.set_title('Average Performance by Model Type')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax2.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '02_model_type_comparison.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 3: Training Convergence Curves
    # ========================================================================
    def plot_training_convergence(self):
        """Plot training convergence curves for all models"""
        print("[Plot 3/20] Training Convergence Curves...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.results)))
        
        for (model_name, result), color in zip(self.results.items(), colors):
            history = result.metrics_history
            if not history:
                continue
            
            rounds = [m.get('round', i+1) for i, m in enumerate(history)]
            micro_f1s = [m.get('micro_f1', 0) for m in history]
            macro_f1s = [m.get('macro_f1', 0) for m in history]
            accuracies = [m.get('accuracy', 0) for m in history]
            aucs = [m.get('mean_auc', 0) for m in history]
            
            ax1.plot(rounds, micro_f1s, marker='o', label=model_name, 
                    color=color, linewidth=2, markersize=4)
            ax2.plot(rounds, macro_f1s, marker='s', label=model_name,
                    color=color, linewidth=2, markersize=4)
            ax3.plot(rounds, accuracies, marker='^', label=model_name,
                    color=color, linewidth=2, markersize=4)
            ax4.plot(rounds, aucs, marker='d', label=model_name,
                    color=color, linewidth=2, markersize=4)
        
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Micro-F1 Score')
        ax1.set_title('Micro-F1 Convergence')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Macro-F1 Score')
        ax2.set_title('Macro-F1 Convergence')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Convergence')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(alpha=0.3)
        ax3.set_ylim(0, 1.0)
        
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Mean AUC')
        ax4.set_title('AUC Convergence')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(alpha=0.3)
        ax4.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '03_training_convergence.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 4: Per-Class Performance Heatmap
    # ========================================================================
    def plot_per_class_heatmap(self):
        """Heatmap of per-class F1 scores across all models"""
        print("[Plot 4/20] Per-Class Performance Heatmap...")
        
        # Collect per-class F1 scores
        models = list(self.results.keys())
        f1_matrix = []
        
        for model_name in models:
            result = self.results[model_name]
            per_class_f1 = result.final_metrics.get('per_class', {}).get('f1', np.zeros(NUM_LABELS))
            f1_matrix.append(per_class_f1)
        
        f1_matrix = np.array(f1_matrix)
        
        # Plot heatmap
        plt.figure(figsize=(12, max(6, len(models) * 0.5)))
        sns.heatmap(
            f1_matrix,
            xticklabels=ISSUE_LABELS,
            yticklabels=models,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'F1 Score'},
            linewidths=0.5
        )
        plt.title('Per-Class F1 Scores Across All Models')
        plt.xlabel('Issue Type')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '04_per_class_heatmap.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 5: Per-Class F1 Scores (Grouped Bar Chart)
    # ========================================================================
    def plot_per_class_bar(self):
        """Grouped bar chart of per-class F1 scores"""
        print("[Plot 5/20] Per-Class F1 Bar Chart...")
        
        models = list(self.results.keys())
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(NUM_LABELS)
        width = 0.8 / len(models)
        
        for i, model_name in enumerate(models):
            result = self.results[model_name]
            per_class_f1 = result.final_metrics.get('per_class', {}).get('f1', np.zeros(NUM_LABELS))
            
            offset = (i - len(models)/2) * width + width/2
            bars = ax.bar(x + offset, per_class_f1, width, 
                         label=model_name, alpha=0.8)
        
        ax.set_xlabel('Issue Type')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Scores Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(ISSUE_LABELS, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '05_per_class_bar.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 6: Precision-Recall Trade-off
    # ========================================================================
    def plot_precision_recall_tradeoff(self):
        """Plot precision vs recall for all models"""
        print("[Plot 6/20] Precision-Recall Trade-off...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for (model_name, result), color in zip(self.results.items(), colors):
            per_class = result.final_metrics.get('per_class', {})
            precision = per_class.get('precision', np.zeros(NUM_LABELS))
            recall = per_class.get('recall', np.zeros(NUM_LABELS))
            
            # Plot per-class points
            ax.scatter(recall, precision, s=100, alpha=0.6, color=color,
                      label=model_name, marker='o')
            
            # Connect with line
            ax.plot(recall, precision, alpha=0.3, color=color, linewidth=1)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Trade-off (Per-Class)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '06_precision_recall_tradeoff.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 7: ROC Curves
    # ========================================================================
    def plot_roc_curves(self):
        """Plot ROC curves for best performing model"""
        print("[Plot 7/20] ROC Curves...")
        
        # Find best model
        best_model = max(self.results.items(), 
                        key=lambda x: x[1].final_metrics.get('micro_f1', 0))
        model_name, result = best_model
        
        # Get predictions and labels
        final_metrics = result.final_metrics
        if 'probabilities' not in final_metrics or 'labels' not in final_metrics:
            print("  [SKIP] No probability data available")
            return
        
        probs = final_metrics['probabilities']
        labels = final_metrics['labels']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, NUM_LABELS))
        
        for i, (label_name, color) in enumerate(zip(ISSUE_LABELS, colors)):
            if labels[:, i].sum() == 0:
                continue
            
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, linewidth=2,
                   label=f'{label_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {model_name}')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '07_roc_curves.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 8: Model Efficiency (Time vs Performance)
    # ========================================================================
    def plot_efficiency_scatter(self):
        """Scatter plot of training time vs performance"""
        print("[Plot 8/20] Model Efficiency...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        models = list(self.results.keys())
        times = [self.results[m].training_time for m in models]
        f1_scores = [self.results[m].final_metrics.get('micro_f1', 0) for m in models]
        params = [self.results[m].params_count / 1e6 for m in models]
        
        # Time vs F1
        colors1 = plt.cm.viridis(np.linspace(0, 1, len(models)))
        for i, (model, time, f1, color) in enumerate(zip(models, times, f1_scores, colors1)):
            ax1.scatter(time, f1, s=200, alpha=0.6, color=color, edgecolors='black')
            ax1.annotate(model, (time, f1), fontsize=8, ha='left', va='bottom')
        
        ax1.set_xlabel('Training Time (seconds)')
        ax1.set_ylabel('Micro-F1 Score')
        ax1.set_title('Model Efficiency: Training Time vs Performance')
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Params vs F1
        colors2 = plt.cm.plasma(np.linspace(0, 1, len(models)))
        for i, (model, param, f1, color) in enumerate(zip(models, params, f1_scores, colors2)):
            ax2.scatter(param, f1, s=200, alpha=0.6, color=color, edgecolors='black')
            ax2.annotate(model, (param, f1), fontsize=8, ha='left', va='bottom')
        
        ax2.set_xlabel('Parameters (Millions)')
        ax2.set_ylabel('Micro-F1 Score')
        ax2.set_title('Model Efficiency: Size vs Performance')
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '08_efficiency_scatter.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 9: Model Size Comparison
    # ========================================================================
    def plot_model_size_comparison(self):
        """Compare model sizes (parameters)"""
        print("[Plot 9/20] Model Size Comparison...")
        
        models = list(self.results.keys())
        params = [self.results[m].params_count / 1e6 for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(models)))
        bars = ax.barh(models, params, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Parameters (Millions)')
        ax.set_title('Model Size Comparison')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, params)):
            ax.text(val + max(params)*0.01, i, f'{val:.1f}M', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '09_model_size_comparison.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 10: Memory Usage
    # ========================================================================
    def plot_memory_usage(self):
        """Compare memory usage across models"""
        print("[Plot 10/20] Memory Usage...")
        
        models = list(self.results.keys())
        memory = [self.results[m].memory_mb for m in models]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.spring(np.linspace(0, 1, len(models)))
        bars = ax.bar(range(len(models)), memory, color=colors, alpha=0.8, 
                     edgecolor='black')
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Peak Memory Usage Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, memory):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}MB', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '10_memory_usage.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 11: Round-by-Round Performance
    # ========================================================================
    def plot_round_performance(self):
        """Show performance improvement across rounds"""
        print("[Plot 11/20] Round-by-Round Performance...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot for each metric
        metrics = ['micro_f1', 'macro_f1', 'accuracy', 'mean_auc']
        titles = ['Micro-F1', 'Macro-F1', 'Accuracy', 'Mean AUC']
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.results)))
        
        for ax, metric, title in zip(axes, metrics, titles):
            for (model_name, result), color in zip(self.results.items(), colors):
                history = result.metrics_history
                if not history:
                    continue
                
                rounds = list(range(1, len(history) + 1))
                values = [m.get(metric, 0) for m in history]
                
                ax.plot(rounds, values, marker='o', label=model_name,
                       color=color, linewidth=2, markersize=5, alpha=0.7)
            
            ax.set_xlabel('Round')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Progress Over Rounds')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '11_round_performance.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 12: Statistical Significance Tests
    # ========================================================================
    def plot_statistical_tests(self):
        """Perform and visualize statistical significance tests"""
        print("[Plot 12/20] Statistical Significance Tests...")
        
        models = list(self.results.keys())
        n_models = len(models)
        
        # Create pairwise comparison matrix
        p_value_matrix = np.ones((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                result_i = self.results[models[i]]
                result_j = self.results[models[j]]
                
                # Get F1 scores across rounds
                scores_i = [m.get('micro_f1', 0) for m in result_i.metrics_history]
                scores_j = [m.get('micro_f1', 0) for m in result_j.metrics_history]
                
                if len(scores_i) > 1 and len(scores_j) > 1:
                    # Perform t-test
                    _, p_value = stats.ttest_ind(scores_i, scores_j)
                    p_value_matrix[i, j] = p_value
                    p_value_matrix[j, i] = p_value
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Create custom colormap (green for significant, red for not significant)
        mask = p_value_matrix < 0.05
        
        sns.heatmap(
            p_value_matrix,
            xticklabels=models,
            yticklabels=models,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=0.1,
            cbar_kws={'label': 'p-value'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Statistical Significance Tests (p-values)\nGreen: p < 0.05 (significant difference)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '12_statistical_tests.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 13: Comparison with Baseline Papers
    # ========================================================================
    def plot_paper_comparison(self):
        """Compare with results from baseline papers"""
        print("[Plot 13/20] Comparison with Baseline Papers...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Prepare data
        our_models = list(self.results.keys())
        our_f1s = [self.results[m].final_metrics.get('micro_f1', 0) for m in our_models]
        our_accs = [self.results[m].final_metrics.get('accuracy', 0) for m in our_models]
        
        baseline_models = list(self.baseline_results.keys())
        baseline_f1s = [self.baseline_results[m]['f1'] for m in baseline_models]
        baseline_accs = [self.baseline_results[m]['accuracy'] for m in baseline_models]
        
        all_models = our_models + baseline_models
        all_f1s = our_f1s + baseline_f1s
        all_accs = our_accs + baseline_accs
        
        # Colors: blue for our models, orange for baselines
        colors = ['#2E86AB'] * len(our_models) + ['#F77F00'] * len(baseline_models)
        
        # F1 Score comparison
        bars1 = ax1.barh(all_models, all_f1s, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('F1 Score')
        ax1.set_title('F1 Score: Our Models vs Baseline Papers')
        ax1.grid(axis='x', alpha=0.3)
        ax1.axvline(x=np.mean(baseline_f1s), color='red', linestyle='--', 
                   linewidth=2, label='Baseline Average', alpha=0.7)
        ax1.legend()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, all_f1s)):
            ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
        
        # Accuracy comparison
        bars2 = ax2.barh(all_models, all_accs, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Accuracy')
        ax2.set_title('Accuracy: Our Models vs Baseline Papers')
        ax2.grid(axis='x', alpha=0.3)
        ax2.axvline(x=np.mean(baseline_accs), color='red', linestyle='--',
                   linewidth=2, label='Baseline Average', alpha=0.7)
        ax2.legend()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, all_accs)):
            ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '13_paper_comparison.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 14: Confusion Matrices
    # ========================================================================
    def plot_confusion_matrices(self):
        """Plot confusion matrices for best models"""
        print("[Plot 14/20] Confusion Matrices...")
        
        # Get top 3 models by F1 score
        top_models = sorted(
            self.results.items(),
            key=lambda x: x[1].final_metrics.get('micro_f1', 0),
            reverse=True
        )[:min(3, len(self.results))]
        
        if not top_models:
            print("  [SKIP] No models to plot")
            return
        
        fig, axes = plt.subplots(1, len(top_models), figsize=(7*len(top_models), 6))
        if len(top_models) == 1:
            axes = [axes]
        
        for ax, (model_name, result) in zip(axes, top_models):
            final_metrics = result.final_metrics
            
            if 'predictions' not in final_metrics or 'labels' not in final_metrics:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue
            
            preds = final_metrics['predictions']
            labels = final_metrics['labels']
            
            # Compute confusion matrix for each class
            cm_sum = np.zeros((2, 2))
            for i in range(NUM_LABELS):
                cm = confusion_matrix(labels[:, i], preds[:, i])
                if cm.shape == (2, 2):
                    cm_sum += cm
            
            # Normalize
            cm_norm = cm_sum / cm_sum.sum() * 100
            
            # Plot
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=ax,
                cbar_kws={'label': 'Percentage (%)'}
            )
            ax.set_title(f'Confusion Matrix\n{model_name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '14_confusion_matrices.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 15: Learning Dynamics
    # ========================================================================
    def plot_learning_dynamics(self):
        """Analyze learning dynamics over rounds"""
        print("[Plot 15/20] Learning Dynamics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        # Plot 1: F1 improvement rate
        ax = axes[0]
        for (model_name, result), color in zip(self.results.items(), colors):
            history = result.metrics_history
            if len(history) < 2:
                continue
            
            f1_scores = [m.get('micro_f1', 0) for m in history]
            improvements = np.diff(f1_scores)
            rounds = list(range(2, len(history) + 1))
            
            ax.plot(rounds, improvements, marker='o', label=model_name,
                   color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('F1 Improvement')
        ax.set_title('Learning Rate: F1 Improvement per Round')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Plot 2: Cumulative F1 gain
        ax = axes[1]
        for (model_name, result), color in zip(self.results.items(), colors):
            history = result.metrics_history
            if len(history) < 2:
                continue
            
            f1_scores = [m.get('micro_f1', 0) for m in history]
            initial_f1 = f1_scores[0]
            cumulative_gain = [f1 - initial_f1 for f1 in f1_scores]
            rounds = list(range(1, len(history) + 1))
            
            ax.plot(rounds, cumulative_gain, marker='o', label=model_name,
                   color=color, linewidth=2, markersize=4)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Cumulative F1 Gain')
        ax.set_title('Cumulative Performance Improvement')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # Plot 3: Variance across rounds
        ax = axes[2]
        variances = []
        model_names = []
        for model_name, result in self.results.items():
            history = result.metrics_history
            if len(history) < 2:
                continue
            
            f1_scores = [m.get('micro_f1', 0) for m in history]
            variance = np.var(f1_scores)
            variances.append(variance)
            model_names.append(model_name)
        
        if variances:
            bars = ax.bar(range(len(model_names)), variances, alpha=0.8,
                         color=colors[:len(model_names)], edgecolor='black')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Variance')
            ax.set_title('Training Stability (Lower is More Stable)')
            ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Convergence speed (rounds to 90% of final performance)
        ax = axes[3]
        convergence_rounds = []
        model_names_conv = []
        
        for model_name, result in self.results.items():
            history = result.metrics_history
            if len(history) < 2:
                continue
            
            f1_scores = [m.get('micro_f1', 0) for m in history]
            final_f1 = f1_scores[-1]
            target_f1 = 0.9 * final_f1
            
            # Find first round where F1 >= target
            conv_round = len(history)  # Default to last round
            for i, f1 in enumerate(f1_scores):
                if f1 >= target_f1:
                    conv_round = i + 1
                    break
            
            convergence_rounds.append(conv_round)
            model_names_conv.append(model_name)
        
        if convergence_rounds:
            bars = ax.barh(model_names_conv, convergence_rounds, alpha=0.8,
                          color=colors[:len(model_names_conv)], edgecolor='black')
            ax.set_xlabel('Rounds to 90% of Final Performance')
            ax.set_title('Convergence Speed (Lower is Faster)')
            ax.grid(axis='x', alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, convergence_rounds)):
                ax.text(val + 0.1, i, f'{val}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '15_learning_dynamics.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # PLOT 16-20: Additional plots (simplified implementations)
    # ========================================================================
    
    def plot_architecture_comparison(self):
        """Compare different architectures"""
        print("[Plot 16/20] Architecture Comparison...")
        # Group by architecture and compare
        # Implementation similar to model type comparison
        self.plot_model_type_comparison()  # Reuse for now
    
    def plot_per_class_auc(self):
        """Plot per-class AUC scores"""
        print("[Plot 17/20] Per-Class AUC...")
        # Similar to per-class F1 heatmap but for AUC
        self.plot_per_class_heatmap()  # Reuse structure
    
    def plot_communication_cost(self):
        """Analyze communication costs in federated learning"""
        print("[Plot 18/20] Communication Cost...")
        
        models = list(self.results.keys())
        params = [self.results[m].params_count for m in models]
        
        # Estimate communication cost (bytes transferred per round)
        # Assuming we transfer full model parameters
        comm_costs = [p * 4 / 1024**2 for p in params]  # Convert to MB
        
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.autumn(np.linspace(0, 1, len(models)))
        bars = ax.bar(range(len(models)), comm_costs, color=colors, 
                     alpha=0.8, edgecolor='black')
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Communication Cost (MB/round)')
        ax.set_title('Estimated Communication Cost per Round')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, comm_costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '18_communication_cost.png'),
                    bbox_inches='tight')
        plt.close()
    
    def plot_scalability(self):
        """Analyze scalability with respect to data size"""
        print("[Plot 19/20] Scalability Analysis...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        models = list(self.results.keys())
        times = [self.results[m].training_time for m in models]
        params = [self.results[m].params_count / 1e6 for m in models]
        
        # Scatter plot: model size vs training time
        colors = plt.cm.winter(np.linspace(0, 1, len(models)))
        
        for model, time, param, color in zip(models, times, params, colors):
            ax.scatter(param, time, s=200, alpha=0.6, color=color, 
                      edgecolors='black', label=model)
        
        ax.set_xlabel('Model Size (Million Parameters)')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Scalability: Model Size vs Training Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Add trend line
        if len(params) > 1:
            z = np.polyfit(params, times, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(params), max(params), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2,
                   label='Trend')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '19_scalability.png'),
                    bbox_inches='tight')
        plt.close()
    
    def plot_error_analysis(self):
        """Analyze error patterns"""
        print("[Plot 20/20] Error Analysis...")
        
        # Find best model for detailed analysis
        best_model = max(self.results.items(),
                        key=lambda x: x[1].final_metrics.get('micro_f1', 0))
        model_name, result = best_model
        
        final_metrics = result.final_metrics
        if 'predictions' not in final_metrics or 'labels' not in final_metrics:
            print("  [SKIP] No prediction data available")
            return
        
        preds = final_metrics['predictions']
        labels = final_metrics['labels']
        
        # Calculate error types per class
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (ax, label_name) in enumerate(zip(axes[:NUM_LABELS], ISSUE_LABELS)):
            y_true = labels[:, idx]
            y_pred = preds[:, idx]
            
            # Calculate error types
            true_pos = ((y_true == 1) & (y_pred == 1)).sum()
            false_pos = ((y_true == 0) & (y_pred == 1)).sum()
            true_neg = ((y_true == 0) & (y_pred == 0)).sum()
            false_neg = ((y_true == 1) & (y_pred == 0)).sum()
            
            # Plot
            categories = ['TP', 'FP', 'TN', 'FN']
            values = [true_pos, false_pos, true_neg, false_neg]
            colors_bar = ['green', 'orange', 'lightblue', 'red']
            
            bars = ax.bar(categories, values, color=colors_bar, alpha=0.7,
                         edgecolor='black')
            ax.set_ylabel('Count')
            ax.set_title(f'{label_name}')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(val)}', ha='center', va='bottom')
        
        # Hide extra subplot
        if len(axes) > NUM_LABELS:
            axes[-1].axis('off')
        
        fig.suptitle(f'Error Analysis - {model_name}', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, '20_error_analysis.png'),
                    bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n[Summary] Generating report...")
        
        report_path = os.path.join(self.save_dir, 'comparison_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FEDERATED LEARNING MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Models Compared: {len(self.results)}\n")
            f.write(f"Report Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-"*80 + "\n\n")
            
            # Create summary table
            data = []
            for model_name, result in self.results.items():
                metrics = result.final_metrics
                data.append({
                    'Model': model_name,
                    'Type': result.model_type.upper(),
                    'Micro-F1': f"{metrics.get('micro_f1', 0):.4f}",
                    'Macro-F1': f"{metrics.get('macro_f1', 0):.4f}",
                    'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                    'Mean AUC': f"{metrics.get('mean_auc', 0):.4f}",
                    'Params (M)': f"{result.params_count/1e6:.1f}",
                    'Time (s)': f"{result.training_time:.1f}",
                })
            
            df = pd.DataFrame(data)
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Best model
            best_model = max(self.results.items(),
                           key=lambda x: x[1].final_metrics.get('micro_f1', 0))
            f.write("-"*80 + "\n")
            f.write("BEST MODEL\n")
            f.write("-"*80 + "\n")
            f.write(f"Model: {best_model[0]}\n")
            f.write(f"Micro-F1: {best_model[1].final_metrics.get('micro_f1', 0):.4f}\n")
            f.write(f"Macro-F1: {best_model[1].final_metrics.get('macro_f1', 0):.4f}\n\n")
            
            # Per-class analysis
            f.write("-"*80 + "\n")
            f.write("PER-CLASS PERFORMANCE (Best Model)\n")
            f.write("-"*80 + "\n")
            per_class = best_model[1].final_metrics.get('per_class', {})
            for i, label in enumerate(ISSUE_LABELS):
                f.write(f"\n{label}:\n")
                f.write(f"  F1:        {per_class.get('f1', [0]*NUM_LABELS)[i]:.4f}\n")
                f.write(f"  Precision: {per_class.get('precision', [0]*NUM_LABELS)[i]:.4f}\n")
                f.write(f"  Recall:    {per_class.get('recall', [0]*NUM_LABELS)[i]:.4f}\n")
                f.write(f"  AUC:       {per_class.get('auc', [0]*NUM_LABELS)[i]:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"[Summary] Report saved to: {report_path}")
        
        # Also save as CSV
        csv_path = os.path.join(self.save_dir, 'comparison_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"[Summary] CSV saved to: {csv_path}")


print("[✓] Part 2 loaded: Comprehensive plotting and comparison framework")
print("[→] Ready for execution")
