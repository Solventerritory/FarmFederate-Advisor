#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_quick_setup.py
==================
Quick verification script to test the federated learning setup

This script verifies:
1. All dependencies are installed
2. Core modules can be imported
3. Basic functionality works
4. GPU availability

Run this before starting full training to catch issues early.
"""

import sys
import importlib

def check_dependency(module_name, package_name=None):
    """Check if a Python package is installed"""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} - MISSING")
        return False

def main():
    print("\n" + "="*70)
    print("FEDERATED LEARNING SETUP VERIFICATION")
    print("="*70 + "\n")
    
    # Check Python version
    print("[1/5] Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠ WARNING: Python 3.8+ recommended")
    else:
        print("  ✓ Version OK")
    
    # Check core dependencies
    print("\n[2/5] Checking core dependencies...")
    deps_ok = True
    
    core_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("PIL", "Pillow"),
    ]
    
    for module, name in core_deps:
        if not check_dependency(module, name):
            deps_ok = False
    
    # Check optional dependencies
    print("\n[3/5] Checking optional dependencies...")
    optional_deps = [
        ("peft", "PEFT (LoRA support)"),
        ("datasets", "HuggingFace Datasets"),
    ]
    
    for module, name in optional_deps:
        check_dependency(module, name)
    
    # Check GPU availability
    print("\n[4/5] Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("  ℹ No GPU detected (will use CPU)")
            print("    Training will be slower but should work")
    except Exception as e:
        print(f"  ⚠ Error checking GPU: {e}")
    
    # Test imports of our modules
    print("\n[5/5] Testing custom module imports...")
    try:
        from federated_llm_vit_vlm_complete import (
            ModelConfig, FederatedLLM, FederatedViT, FederatedVLM,
            ISSUE_LABELS, NUM_LABELS, MODEL_CONFIGS
        )
        print("  ✓ federated_llm_vit_vlm_complete.py")
        print(f"    Found {len(MODEL_CONFIGS)} model configurations")
        print(f"    Labels: {ISSUE_LABELS}")
    except Exception as e:
        print(f"  ✗ federated_llm_vit_vlm_complete.py - {e}")
        deps_ok = False
    
    try:
        from federated_plotting_comparison import (
            ModelResults, ComparisonFramework
        )
        print("  ✓ federated_plotting_comparison.py")
    except Exception as e:
        print(f"  ✗ federated_plotting_comparison.py - {e}")
        deps_ok = False
    
    try:
        import run_federated_comprehensive
        print("  ✓ run_federated_comprehensive.py")
    except Exception as e:
        print(f"  ✗ run_federated_comprehensive.py - {e}")
        deps_ok = False
    
    # Summary
    print("\n" + "="*70)
    if deps_ok:
        print("✅ VERIFICATION PASSED")
        print("\nYou can now run:")
        print("  python run_federated_comprehensive.py --quick_test")
        print("\nOr use the Windows batch script:")
        print("  run_quick_test.bat")
    else:
        print("❌ VERIFICATION FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements_federated.txt")
        print("\nIf you have CUDA, install PyTorch with GPU support:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("="*70 + "\n")
    
    return 0 if deps_ok else 1

if __name__ == "__main__":
    exit_code = main()
    
    # On Windows, pause so user can read the output
    if sys.platform == "win32":
        input("\nPress Enter to exit...")
    
    sys.exit(exit_code)
