#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK RUNNER - Complete Federated Learning System
=================================================

This script orchestrates the complete training and evaluation pipeline:
1. Downloads and prepares datasets (text + images)
2. Trains Federated LLM, ViT, and VLM models
3. Generates 20 publication-quality comparison plots
4. Compares with state-of-the-art papers
5. Saves all results and visualizations

Usage:
    python RUN_COMPLETE_SYSTEM.py

    # For quick demo (reduced training):
    python RUN_COMPLETE_SYSTEM.py --demo

Author: FarmFederate Research Team
Date: 2026-01-07
"""

import sys
import argparse
from pathlib import Path

print("="*70)
print("🌾 FEDERATED LEARNING FOR PLANT STRESS DETECTION")
print("="*70)
print("Complete system: LLM + ViT + VLM + 20 Plots + Paper Comparisons")
print("="*70)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--demo', action='store_true', help='Run quick demo with reduced settings')
parser.add_argument('--plots-only', action='store_true', help='Generate plots only (skip training)')
parser.add_argument('--num-clients', type=int, default=5, help='Number of federated clients')
parser.add_argument('--num-rounds', type=int, default=5, help='Number of federated rounds')
args = parser.parse_args()

if args.demo:
    print("\n🎬 DEMO MODE: Using reduced settings for quick execution")
    args.num_clients = 3
    args.num_rounds = 3

# ============================================================================
# STEP 1: Generate plots (can run standalone)
# ============================================================================

if args.plots_only:
    print("\n" + "="*70)
    print("📊 GENERATING PLOTS ONLY")
    print("="*70)
    
    from comprehensive_plots import generate_all_plots
    generate_all_plots()
    
    print("\n✅ Plots generated successfully!")
    print(f"   Location: outputs_federated_complete/plots/")
    sys.exit(0)

# ============================================================================
# STEP 2: Full training pipeline
# ============================================================================

print("\n" + "="*70)
print("🚀 STARTING COMPLETE TRAINING PIPELINE")
print("="*70)

try:
    # Import main system
    from federated_complete_system import main
    
    # Run training
    results, histories = main()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Generate plots
    print("\n" + "="*70)
    print("📊 GENERATING PUBLICATION-QUALITY PLOTS")
    print("="*70)
    
    from comprehensive_plots import generate_all_plots
    generate_all_plots(results, histories)
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print("="*70)
    print("\n📁 Output locations:")
    print(f"   Results: outputs_federated_complete/results/final_results.json")
    print(f"   Plots: outputs_federated_complete/plots/ (20 plots)")
    print(f"   Models: outputs_federated_complete/models/")
    
    print("\n📊 Final Results Summary:")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  F1-Macro:  {metrics['f1_macro']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    
    print("\n" + "="*70)
    print("✨ Ready for publication submission!")
    print("="*70)

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\n🔧 Installing required packages...")
    print("Run: pip install transformers torch torchvision datasets pillow matplotlib seaborn scikit-learn")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n🔧 Generating plots with simulated data...")
    try:
        from comprehensive_plots import generate_all_plots
        generate_all_plots()
        print("\n✅ Plots generated with simulated data!")
    except:
        print("❌ Could not generate plots")
    
    sys.exit(1)
