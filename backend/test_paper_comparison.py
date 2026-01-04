#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_paper_comparison.py
========================
Quick test script to verify research paper comparison framework

This script:
1. Loads the research papers database
2. Creates mock results for our models
3. Generates all 10 comparison plots
4. Produces summary statistics

Usage:
    python test_paper_comparison.py
"""

import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

# Import the research paper comparison module
from research_paper_comparison import RESEARCH_PAPERS, ResearchPaperComparator

# ============================================================================
# MOCK DATA STRUCTURES (matching federated_llm_vit_vlm_complete.py)
# ============================================================================

@dataclass
class MockModelResults:
    """Mock version of ModelResults for testing"""
    model_name: str
    model_type: str  # 'llm', 'vit', 'vlm'
    config: Dict = field(default_factory=dict)
    metrics_history: List = field(default_factory=list)
    final_metrics: Dict = field(default_factory=dict)
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_mb: float = 0.0
    params_count: int = 0
    communication_cost: float = 0.0


def create_mock_results() -> Dict[str, MockModelResults]:
    """
    Create mock results for our federated models
    Simulates realistic performance for testing
    """
    
    results = {}
    
    # ===== LLM Models =====
    llm_models = [
        ("Federated-Flan-T5-Small", "llm", 80.0, 60.0, 82.5, 78.0, 81.2, 79.5),
        ("Federated-GPT2", "llm", 150.0, 117.0, 81.8, 79.2, 82.5, 80.0),
        ("Federated-Flan-T5-Base", "llm", 250.0, 220.0, 84.5, 81.8, 85.0, 82.5),
    ]
    
    for name, mtype, params_m, memory, f1, acc, prec, rec in llm_models:
        result = MockModelResults(
            model_name=name,
            model_type=mtype,
            config={'architecture': 'seq2seq'},
            params_count=int(params_m * 1e6),
            memory_mb=memory,
            training_time=np.random.uniform(300, 600),
            inference_time=np.random.uniform(5, 15),
            final_metrics={
                'micro_f1': f1 / 100,
                'macro_f1': (f1 - 2) / 100,
                'accuracy': acc / 100,
                'precision': prec / 100,
                'recall': rec / 100,
                'predictions': np.random.randint(0, 2, (100, 5)),
                'labels': np.random.randint(0, 2, (100, 5)),
            }
        )
        results[name] = result
    
    # ===== ViT Models =====
    vit_models = [
        ("Federated-ViT-Base", "vit", 86.0, 450.0, 87.5, 85.2, 88.0, 86.5),
        ("Federated-DeiT-Base", "vit", 86.5, 420.0, 88.2, 86.0, 88.8, 87.0),
        ("Federated-ViT-Small", "vit", 22.0, 180.0, 85.0, 82.5, 85.5, 83.8),
    ]
    
    for name, mtype, params_m, memory, f1, acc, prec, rec in vit_models:
        result = MockModelResults(
            model_name=name,
            model_type=mtype,
            config={'architecture': 'vision_transformer'},
            params_count=int(params_m * 1e6),
            memory_mb=memory,
            training_time=np.random.uniform(400, 800),
            inference_time=np.random.uniform(8, 20),
            final_metrics={
                'micro_f1': f1 / 100,
                'macro_f1': (f1 - 1.5) / 100,
                'accuracy': acc / 100,
                'precision': prec / 100,
                'recall': rec / 100,
                'predictions': np.random.randint(0, 2, (100, 5)),
                'labels': np.random.randint(0, 2, (100, 5)),
            }
        )
        results[name] = result
    
    # ===== VLM (Multimodal) Models =====
    vlm_models = [
        ("Federated-CLIP-Base", "vlm", 151.0, 850.0, 86.5, 84.0, 87.0, 85.5),
        ("Federated-BLIP-Base", "vlm", 224.0, 1100.0, 88.0, 85.8, 88.5, 87.0),
        ("Federated-CLIP-ViT-L", "vlm", 427.0, 1800.0, 89.2, 87.0, 89.8, 88.5),
    ]
    
    for name, mtype, params_m, memory, f1, acc, prec, rec in vlm_models:
        result = MockModelResults(
            model_name=name,
            model_type=mtype,
            config={'architecture': 'multimodal'},
            params_count=int(params_m * 1e6),
            memory_mb=memory,
            training_time=np.random.uniform(600, 1200),
            inference_time=np.random.uniform(15, 35),
            final_metrics={
                'micro_f1': f1 / 100,
                'macro_f1': (f1 - 1) / 100,
                'accuracy': acc / 100,
                'precision': prec / 100,
                'recall': rec / 100,
                'predictions': np.random.randint(0, 2, (100, 5)),
                'labels': np.random.randint(0, 2, (100, 5)),
            }
        )
        results[name] = result
    
    return results


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def main():
    """Run the research paper comparison test"""
    
    print("\n" + "="*80)
    print("RESEARCH PAPER COMPARISON - TEST MODE")
    print("="*80 + "\n")
    
    # Display paper database info
    print(f"[Database] Total papers loaded: {len(RESEARCH_PAPERS)}")
    print("\n[Categories]:")
    categories = {}
    for name, info in RESEARCH_PAPERS.items():
        cat = info['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count} papers")
    
    # Year range
    years = [info['year'] for info in RESEARCH_PAPERS.values()]
    print(f"\n[Timeline] {min(years)} - {max(years)} ({max(years) - min(years) + 1} years)")
    
    # Top papers by F1
    print("\n[Top 5 Papers by F1 Score]:")
    top_papers = sorted(RESEARCH_PAPERS.items(), 
                       key=lambda x: x[1]['f1'], 
                       reverse=True)[:5]
    for i, (name, info) in enumerate(top_papers, 1):
        print(f"  {i}. {name}: {info['f1']*100:.1f}% ({info['year']})")
    
    # Create mock results for our models
    print("\n" + "="*80)
    print("CREATING MOCK RESULTS FOR OUR MODELS")
    print("="*80 + "\n")
    
    our_results = create_mock_results()
    print(f"[Mock Models] Created {len(our_results)} mock model results:")
    
    for name, result in our_results.items():
        f1 = result.final_metrics.get('micro_f1', 0) * 100
        acc = result.final_metrics.get('accuracy', 0) * 100
        params = result.params_count / 1e6
        print(f"  • {name:30s} | Type: {result.model_type:3s} | "
              f"F1: {f1:5.1f}% | Acc: {acc:5.1f}% | Params: {params:6.1f}M")
    
    # Calculate averages
    our_f1s = [r.final_metrics.get('micro_f1', 0) * 100 for r in our_results.values()]
    paper_f1s = [p['f1'] * 100 for p in RESEARCH_PAPERS.values()]
    
    print(f"\n[Averages]")
    print(f"  Our Models: {np.mean(our_f1s):.2f}% ± {np.std(our_f1s):.2f}%")
    print(f"  Baseline Papers: {np.mean(paper_f1s):.2f}% ± {np.std(paper_f1s):.2f}%")
    print(f"  Difference: {np.mean(our_f1s) - np.mean(paper_f1s):+.2f}%")
    
    # Create comparison framework
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80 + "\n")
    
    save_dir = "results/paper_comparison_test"
    comparator = ResearchPaperComparator(
        our_results=our_results,
        save_dir=save_dir
    )
    
    # Generate all comparisons
    comparator.generate_all_comparisons()
    
    # Success message
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n[Results] All plots saved to: {save_dir}/")
    print(f"[Plots Generated]:")
    print(f"  1. Overall F1 Score Comparison")
    print(f"  2. Accuracy Comparison")
    print(f"  3. Precision-Recall Scatter Plot")
    print(f"  4. Category-Wise Performance")
    print(f"  5. Temporal Evolution (2016-2024)")
    print(f"  6. Model Efficiency Analysis")
    print(f"  7. Multi-Metric Radar Chart")
    print(f"  8. Communication Efficiency")
    print(f"  9. Model Size vs Performance")
    print(f" 10. Category Breakdown Details")
    print(f"\n[Summary] Check summary_statistics.json for detailed metrics")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
