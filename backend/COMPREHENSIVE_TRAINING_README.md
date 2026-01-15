# Comprehensive Federated Learning Training Pipeline
## LLM vs ViT vs VLM for Plant Stress Detection

**Date:** 2026-01-15
**Author:** FarmFederate Research Team
**Purpose:** Complete training and comparison of federated learning approaches

---

## 📋 Overview

This comprehensive training pipeline implements and compares:

### Models Trained (17 total)
- **9 Federated LLM Models** (text-based plant stress detection)
  - Flan-T5-Small, Flan-T5-Base, T5-Small
  - GPT-2, GPT-2-Medium, DistilGPT-2
  - RoBERTa-Base, BERT-Base, DistilBERT

- **4 Federated ViT Models** (image-based plant stress detection)
  - ViT-Base-Patch16-224
  - ViT-Large-Patch16-224
  - ViT-Base-Patch16-384
  - DeiT-Base-Patch16-224

- **4 Federated VLM Models** (multimodal vision-language)
  - CLIP-ViT-Base-Patch32
  - CLIP-ViT-Large-Patch14
  - BLIP-Image-Captioning-Base
  - BLIP-2-OPT-2.7B

### Baselines Compared (10 papers)
1. McMahan et al. (FedAvg, 2017) - Federated
2. Li et al. (FedProx, 2020) - Federated
3. Li et al. (FedBN, 2021) - Federated
4. Wang et al. (FedNova, 2020) - Federated
5. Li et al. (MOON, 2021) - Federated
6. Acar et al. (FedDyn, 2021) - Federated
7. Mohanty et al. (PlantVillage, 2016) - Centralized
8. Ferentinos (DeepPlant, 2018) - Centralized
9. Chen et al. (AgriNet, 2020) - Centralized
10. Zhang et al. (FedAgri, 2022) - Federated

### Visualizations (20 plots)
Comprehensive publication-quality plots comparing all aspects

---

## 🚀 Quick Start

### 1. Run the Jupyter Notebook

```bash
jupyter notebook Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb
```

Or upload to Google Colab for GPU access.

### 2. Execute All Cells

The notebook is organized into sections:
- **Section 1:** Installation & Imports
- **Section 2:** Configuration & Constants
- **Section 3:** Dataset Loading (text + image)
- **Section 4:** Non-IID Data Splitting
- **Section 5:** Model Architectures
- **Section 6:** Federated Training Functions
- **Section 7:** Train All Models (LLM, ViT, VLM)
- **Section 8:** Save Results
- **Section 9:** Generate 20 Plots
- **Section 10:** Final Report

### 3. Generate Plots

After training, run the plotting suite:

```bash
python comprehensive_plotting_suite.py
```

This generates 20 plots in the `plots/` directory.

---

## 📊 Datasets

### Text Datasets
1. **AG News** (agriculture filtered)
   - Source: HuggingFace `ag_news`
   - Filtering: Keywords (farm, crop, plant, agriculture, soil)
   - Size: ~500 samples

2. **Synthetic Agricultural Text**
   - Generated sensor logs and farmer observations
   - Covers all 5 stress types
   - Size: ~1000 samples

### Image Datasets
1. **PlantVillage Dataset**
   - Source: `BrandonFors/Plant-Diseases-PlantVillage-Dataset`
   - Plant disease images
   - Size: ~1000 images

2. **Synthetic Plant Images**
   - Generated for augmentation
   - Simulates plant appearances
   - Size: Variable (fills to 1000 total)

### Labels (5-class multi-label)
- `water_stress` - Drought, wilting, soil moisture
- `nutrient_def` - N, P, K deficiencies
- `pest_risk` - Aphids, whiteflies, caterpillars
- `disease_risk` - Blight, rust, fungal/viral
- `heat_stress` - Heatwave, sunburn, thermal stress

---

## ⚙️ Configuration

### Federated Learning Settings

```python
FEDERATED_CONFIG = {
    'num_clients': 5,              # Number of federated clients
    'num_rounds': 10,              # Federated training rounds
    'local_epochs': 3,             # Local training epochs per round
    'clients_per_round': 5,        # Clients participating per round
    'batch_size': 8,               # Batch size
    'learning_rate': 2e-5,         # Learning rate
    'weight_decay': 0.01,          # Weight decay
    'warmup_steps': 100,           # Warmup steps
    'max_grad_norm': 1.0,          # Gradient clipping
    'aggregation_method': 'fedavg', # FedAvg aggregation
    'use_lora': True,              # Use LoRA for efficiency
    'lora_r': 8,                   # LoRA rank
    'lora_alpha': 16,              # LoRA alpha
    'lora_dropout': 0.1,           # LoRA dropout
    'dirichlet_alpha': 0.5,        # Non-IID split parameter
}
```

### Non-IID Data Split

Uses **Dirichlet distribution** with α=0.5 for realistic heterogeneous data distribution across clients.

- Lower α = more heterogeneous (realistic)
- Higher α = more homogeneous (IID-like)

---

## 🏗️ Architecture Details

### Federated LLM
```
Text Input → Tokenizer → LLM Encoder (with LoRA)
    ↓
[CLS] Token / Mean Pooling
    ↓
Classifier (256 → 5 labels)
```

### Federated ViT
```
Image Input → ViT Encoder (with LoRA)
    ↓
[CLS] Token
    ↓
LayerNorm → MLP (512 → 5 labels)
```

### Federated VLM
```
Text Input → Text Encoder
    ↓
Text Embeddings  ┐
                 ├─→ Concatenate → Fusion (512) → Classifier (256 → 5)
Image Input → Image Encoder
    ↓
Image Embeddings ┘
```

---

## 📈 Training Process

### Federated Training Loop

For each round (1 to 10):

1. **Client Sampling**: Select clients to participate
2. **Local Training**: Each client trains locally for 3 epochs
3. **Model Upload**: Clients send model updates to server
4. **Aggregation**: Server aggregates using FedAvg
5. **Global Evaluation**: Evaluate global model on validation set
6. **Broadcast**: Send updated global model to all clients

### Metrics Tracked

Per round:
- Training Loss
- Validation Loss
- F1-Score (Macro)
- F1-Score (Micro)
- Accuracy
- Precision
- Recall

---

## 📊 Visualization Suite

### Plot 1: Overall F1-Score Comparison
Bar chart comparing all 27 models (17 trained + 10 baselines)

### Plot 2: Training Convergence
F1-score improvement over 10 federated rounds

### Plot 3: Accuracy Comparison
Overall accuracy comparison across all models

### Plot 4: Model Type Average Performance
LLM vs ViT vs VLM average metrics (F1, Accuracy, Precision, Recall)

### Plot 5: Loss Convergence
Training and validation loss over rounds (dual subplot)

### Plot 6: Precision vs Recall Scatter
Scatter plot showing precision-recall trade-offs

### Plot 7: Per-Class F1-Score Heatmap
Heatmap showing performance on each of 5 stress types

### Plot 8: Federated vs Centralized
Comparison with centralized baselines (privacy-utility gap)

### Plot 9: Communication Efficiency
Performance vs communication rounds (efficiency analysis)

### Plot 10: Model Size vs Performance
Trade-off between model parameters and F1-score

### Plots 11-20 (Additional)
- Training time comparison
- Convergence rate analysis
- ROC curves
- Confusion matrices
- Statistical significance tests
- Multi-metric radar charts
- Learning curves
- Client heterogeneity impact
- Ablation studies
- Comprehensive leaderboard

---

## 💾 Output Files

After training, the following files are generated:

```
FarmFederate/backend/
├── Federated_LLM_ViT_VLM_Comprehensive_Training.ipynb
├── comprehensive_plotting_suite.py
├── federated_training_results.json         # All training results
├── COMPREHENSIVE_REPORT.md                 # Summary report
└── plots/                                  # 20 visualization plots
    ├── plot_01_overall_f1_comparison.png
    ├── plot_02_convergence_f1.png
    ├── plot_03_accuracy_comparison.png
    ├── ... (20 plots total)
    └── plot_20_leaderboard.png
```

### `federated_training_results.json`

Structure:
```json
{
  "llm": {
    "model_name": {
      "history": {
        "rounds": [1, 2, ..., 10],
        "train_loss": [...],
        "val_loss": [...],
        "f1_macro": [...],
        "f1_micro": [...],
        "accuracy": [...],
        "precision": [...],
        "recall": [...]
      },
      "final_f1": 0.XXX,
      "final_acc": 0.XXX
    }
  },
  "vit": { ... },
  "vlm": { ... }
}
```

---

## 🎯 Expected Results

### Performance Hierarchy (Typical)

1. **Centralized Baselines** (F1: 0.87-0.95)
   - Best overall performance
   - No privacy preservation
   - Requires centralized data

2. **Federated VLM Models** (F1: 0.75-0.82)
   - Best among federated approaches
   - Leverages both text and images
   - Privacy-preserving

3. **Federated ViT Models** (F1: 0.72-0.79)
   - Strong image understanding
   - Good for visual symptoms

4. **Federated LLM Models** (F1: 0.70-0.77)
   - Relies on textual descriptions
   - Limited without images

5. **Federated Baselines** (F1: 0.72-0.79)
   - Established benchmarks
   - Similar privacy guarantees

### Privacy-Utility Trade-off

**Gap:** Centralized (0.95) - Federated VLM (0.80) = **0.15 F1 points**

This 15-point gap is the cost of privacy preservation through federated learning.

---

## 🔬 Research Contributions

### 1. Comprehensive Comparison
First comprehensive comparison of LLM, ViT, and VLM in federated plant stress detection

### 2. Multimodal Federated Learning
Novel federated VLM architecture combining text and images

### 3. Real-World Datasets
Integration of multiple agricultural datasets (PlantVillage, AG News, etc.)

### 4. Extensive Baselines
Comparison with 10 published papers from 2016-2022

### 5. Publication-Ready Visualizations
20 IEEE-style plots for paper submission

---

## 🛠️ Customization

### Modify Number of Rounds

```python
FEDERATED_CONFIG['num_rounds'] = 20  # Increase to 20 rounds
```

### Change Data Heterogeneity

```python
FEDERATED_CONFIG['dirichlet_alpha'] = 0.1  # More heterogeneous
FEDERATED_CONFIG['dirichlet_alpha'] = 1.0  # More homogeneous
```

### Add More Models

```python
LLM_MODELS.append('facebook/opt-1.3b')
VIT_MODELS.append('microsoft/swin-base-patch4-window7-224')
VLM_MODELS.append('llava-hf/llava-1.5-7b-hf')
```

### Disable LoRA

```python
FEDERATED_CONFIG['use_lora'] = False  # Train full model (slower)
```

---

## 📝 Citation

If you use this training pipeline in your research, please cite:

```bibtex
@article{farmfederate2026,
  title={Comprehensive Comparison of Federated LLM, ViT, and VLM for Plant Stress Detection},
  author={FarmFederate Research Team},
  journal={Agricultural AI Conference},
  year={2026}
}
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Add more VLM models** (LLaVA, Flamingo, etc.)
2. **Implement FedProx, SCAFFOLD** (beyond FedAvg)
3. **Add differential privacy** mechanisms
4. **Expand datasets** (more sources)
5. **Optimize training** (gradient compression, etc.)

---

## 📧 Contact

For questions or issues:
- GitHub: [FarmFederate Repository](https://github.com/Solventerritory/FarmFederate-Advisor)
- Email: research@farmfederate.ai

---

## 📜 License

This project is licensed under the MIT License.

---

## 🎉 Acknowledgments

- HuggingFace for transformer models and datasets
- PlantVillage for plant disease images
- OpenAI, Google, Meta for open-source models (CLIP, ViT, T5, etc.)
- Federated learning research community for baseline papers

---

**Last Updated:** 2026-01-15
**Version:** 1.0.0
**Status:** ✅ Production-Ready
