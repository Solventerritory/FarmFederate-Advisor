"""
Quick Run Script for Comprehensive Model Comparison

Usage:
    python run_comparison.py

This will generate all 8 comparison plots and tables.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comprehensive_model_comparison import main

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   RUNNING COMPREHENSIVE MODEL COMPARISON")
    print("   " + "="*70)
    print("\n   This will generate:")
    print("   - 8 publication-quality plots (PNG, 300 DPI)")
    print("   - 1 comprehensive CSV table")
    print("   - 1 JSON file with raw results")
    print("\n   Output: plots/comparison/")
    print("   " + "="*70 + "\n")

    try:
        main()
        print("\n" + "="*70)
        print("   COMPARISON COMPLETE!")
        print("   " + "="*70)
        print("\n   Check plots/comparison/ for all outputs\n")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
