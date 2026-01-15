# FarmFederate Codebase Cleanup Plan

**Date:** 2026-01-15
**Purpose:** Remove redundant and outdated files to streamline the codebase

---

## Summary

- **Files to Remove:** 60+ files
- **Space to Reclaim:** ~900MB
- **Duplicate Directory:** FarmFederate-Advisor (825MB)
- **Outdated Notebooks:** 6 notebooks (replaced by comprehensive notebook)
- **Redundant Scripts:** 20+ Python files

---

## Files to Remove

### 1. Duplicate Directory (825MB)
- [x] `FarmFederate-Advisor/` - Complete duplicate

### 2. Outdated Training Notebooks (6 files)
- [ ] `backend/FarmFederate_Colab.ipynb`
- [ ] `backend/FarmFederate_Colab_Training.ipynb`
- [ ] `backend/FarmFederate_Training_Colab.ipynb`
- [ ] `backend/FarmFederate_Train_39Models.ipynb`
- [ ] `backend/FarmFederate_Training_Colab_Fixed.ipynb`
- [ ] `backend/FarmFederate_Training_Colab_Updated.ipynb`
- [ ] `backend/farm_advisor_multimodal_full.ipynb`

**Keep:** `backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb`

### 3. Test/Demo Scripts (10 files)
- [ ] `backend/test_paper_comparison.py`
- [ ] `backend/test_quick_setup.py`
- [ ] `backend/test_upload.py`
- [ ] `backend/quick_esp32_check.py`
- [ ] `backend/quick_start.py`
- [ ] `backend/simulate_esp32_usb.py`
- [ ] `backend/train_cpu.py`
- [ ] `backend/troubleshoot_esp32.py`
- [ ] `backend/verify_upload.py`
- [ ] `backend/upload_esp32_vscode.py`

### 4. Redundant Training Scripts (5 files)
- [ ] `backend/RUN_COMPLETE_SYSTEM.py`
- [ ] `backend/run_federated_comprehensive.py`
- [ ] `backend/train_federated_complete.py`

**Keep:** `backend/federated_complete_training.py`

### 5. Redundant Plotting Modules (5 files)
- [ ] `backend/comprehensive_plots.py`
- [ ] `backend/publication_plots.py`
- [ ] `backend/federated_plotting_comparison.py`

**Keep:** `backend/comprehensive_plotting.py` OR `backend/ultimate_plotting_suite.py`

### 6. Redundant Comparison Scripts (7 files)
- [ ] `backend/paper_comparison.py`
- [ ] `backend/paper_comparison_updated.py`
- [ ] `backend/ultimate_model_comparison.py`
- [ ] `backend/robust_model_comparison.py`
- [ ] `backend/robust_comparison.py`
- [ ] `backend/train_all_models_comparison.py`
- [ ] `backend/train_and_compare_all.py`

**Keep:** `backend/research_paper_comparison.py`

### 7. Farm Advisor Variants (3 files)
- [ ] `backend/farm_advisor.py`
- [ ] `backend/farm_advisor_enhanced_part2.py`
- [ ] `backend/farm_advisor_enhanced_part3.py`

**Keep:** `backend/farm_advisor_complete.py`

### 8. Test Data Directory
- [ ] `dummy-sensor-data-clean/`

---

## Files to KEEP (Core Implementation)

### Essential Training Files
✅ `backend/Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb` - NEW comprehensive notebook
✅ `backend/federated_complete_training.py` - Primary training script
✅ `backend/federated_llm_vit_vlm_complete.py` - Model architectures
✅ `backend/federated_core.py` - Core utilities
✅ `backend/datasets_loader.py` - Dataset loading
✅ `backend/master_integration.py` - Integration
✅ `backend/resume_training.py` - Checkpoint resuming

### Essential Analysis Files
✅ `backend/comprehensive_plotting.py` - Primary plotting
✅ `backend/research_paper_comparison.py` - Baseline comparisons
✅ `backend/icml_neurips_sections.py` - Publication utilities

### Essential Documentation
✅ `README.md` - Main documentation
✅ `COMPREHENSIVE_TRAINING_README.md` - Training guide
✅ `IMPLEMENTATION_SUMMARY.md` - Implementation summary

### Configuration
✅ `backend/config/` - Configuration files
✅ `backend/data/` - Dataset metadata

---

## Execution Commands

Execute these in order:
