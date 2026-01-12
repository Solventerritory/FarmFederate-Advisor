# Ultimate Model Comparison Framework

## 🎯 Overview

This comprehensive framework compares **ALL model architectures** for the FarmFederate agricultural advisory system:

### Models Compared

#### 1. **Large Language Models (LLMs)**
- RoBERTa-Base
- BERT-Base
- DistilBERT
- DeBERTa-v3
- ELECTRA
- ALBERT

#### 2. **Vision Transformers (ViT)**
- ViT-Base-Patch16
- DeiT-Base
- Swin Transformer (if available)

#### 3. **Vision-Language Models (VLM)**
- RoBERTa + ViT (Multimodal Fusion)
- BERT + DeiT (Alternative Fusion)

#### 4. **Training Paradigms**
- **Centralized Learning**: Traditional single-server training
- **Federated Learning**: Distributed training with FedAvg aggregation

### Baseline Papers Compared

1. **PlantVillage (2018)** - Deep learning for plant disease classification
2. **SCOLD (2021)** - Smartphone-based crop disease detection
3. **FL-Weed (2022)** - Federated learning for weed detection
4. **AgriVision (2023)** - Vision transformer for agriculture
5. **FedCrop (2023)** - Federated crop monitoring
6. **FedAvg (2017)** - Original federated learning algorithm
7. **FedProx (2020)** - Federated learning with proximal term
8. **MOON (2021)** - Model-contrastive federated learning
9. **FedBN (2021)** - Federated learning with batch normalization
10. **FedDyn (2021)** - Federated learning with dynamic regularization
11. **AgriTransformer (2024)** - Transformer for agriculture
12. **PlantDoc (2020)** - Cross-domain plant disease detection
13. **Cassava (2021)** - Fine-grained disease classification
14. **FarmBERT (2023)** - BERT for agricultural text
15. **AgroVLM (2024)** - Vision-language model for agriculture

---

## 📊 Visualization Suite (25+ Plots)

The framework generates **25 comprehensive visualizations**:

### Performance Metrics (Plots 1-5)
1. **Overall Performance Comparison** - F1, Accuracy, Precision, Recall
2. **Model Type Comparison** - LLM vs ViT vs VLM
3. **Federated vs Centralized** - Training paradigm comparison
4. **Training Convergence** - Loss and metric curves over epochs/rounds
5. **Per-Class Performance** - Class-wise F1 scores

### Analysis Plots (Plots 6-10)
6. **Confusion Matrices** - Prediction error patterns
7. **ROC Curves** - True positive vs false positive rates
8. **Precision-Recall Curves** - Trade-off visualization
9. **Parameter Efficiency** - Performance vs model size
10. **Training Time Comparison** - Computational cost analysis

### Efficiency Metrics (Plots 11-13)
11. **Inference Speed** - Milliseconds per prediction
12. **Memory Usage** - RAM requirements
13. **Communication Cost** - Federated learning bandwidth

### Paper Comparisons (Plots 14-15)
14. **Paper Comparison (Bars)** - Side-by-side with SOTA
15. **Paper Comparison (Scatter)** - Performance vs parameters

### Advanced Visualizations (Plots 16-20)
16. **Radar Charts** - Multi-metric comparison
17. **Metrics Heatmap** - All metrics across all models
18. **Box Plots** - Performance distributions
19. **Violin Plots** - Metric density distributions
20. **Statistical Significance** - P-value matrix

### Specialized Analysis (Plots 21-25)
21. **Ablation Study** - Component contribution analysis
22. **Scalability Analysis** - Performance vs number of clients
23. **Robustness Analysis** - Noise, heterogeneity, missing data
24. **Error Analysis** - Failure case investigation
25. **Summary Dashboard** - Comprehensive overview

---

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to backend directory
cd FarmFederate-Advisor/backend

# Install required packages
pip install torch torchvision transformers datasets
pip install scikit-learn pandas numpy matplotlib seaborn scipy tqdm
```

### 2. Run Model Comparison

```bash
# Train and evaluate all models (takes 1-3 hours depending on hardware)
python ultimate_model_comparison.py
```

**What it does:**
- Trains 6+ LLM variants (centralized + federated)
- Trains 2+ ViT variants (centralized + federated)
- Trains 1+ VLM variants (centralized + federated)
- Evaluates all models on validation set
- Saves results to `outputs_ultimate_comparison/results/`

**Expected output:**
```
================================================================================
ULTIMATE MODEL COMPARISON FRAMEWORK
================================================================================

[INFO] Device: cuda
[INFO] Results directory: outputs_ultimate_comparison
[INFO] Creating synthetic datasets...
[INFO] Train: 800, Val: 200

================================================================================
TRAINING LLM MODELS
================================================================================

[INFO] Training RoBERTa-Base...

[Centralized Training]
[Epoch 1/3]
Training: 100%|████████████████████| 50/50 [00:45<00:00]
...
```

### 3. Generate Visualizations

```bash
# Generate all 25 plots
python ultimate_plotting_suite.py
```

**What it does:**
- Loads results from most recent comparison run
- Generates 25 publication-quality plots
- Saves to `outputs_ultimate_comparison/plots/`

**Expected output:**
```
================================================================================
GENERATING COMPREHENSIVE VISUALIZATION SUITE
================================================================================

[01/25] Generating: Overall Performance Comparison
         ✓ Saved successfully

[02/25] Generating: Model Type Comparison
         ✓ Saved successfully

...

[25/25] Generating: Summary Dashboard
         ✓ Saved successfully

ALL PLOTS GENERATED SUCCESSFULLY!
Plots saved to: outputs_ultimate_comparison/plots
```

---

## 📁 Output Structure

```
outputs_ultimate_comparison/
├── checkpoints/               # Model checkpoints
├── results/
│   ├── comparison_results_20260108_143022.json    # Full results (JSON)
│   └── comparison_results.csv                      # Summary (CSV)
├── logs/
│   └── training.log                                # Training logs
└── plots/                                          # All visualizations
    ├── 01_overall_performance.png
    ├── 02_model_type_comparison.png
    ├── 03_federated_vs_centralized.png
    ├── ...
    └── 25_summary_dashboard.png
```

---

## 🔧 Configuration Options

### Modify Training Parameters

Edit `ultimate_model_comparison.py`:

```python
# Quick test (faster, less accurate)
n_epochs = 2
n_rounds = 3
batch_size = 32

# Full training (slower, more accurate)
n_epochs = 10
n_rounds = 10
batch_size = 16

# Number of federated clients
n_clients = 5  # Default
n_clients = 10  # More realistic

# Data heterogeneity (Dirichlet alpha)
alpha = 0.5   # High heterogeneity (non-IID)
alpha = 10.0  # Low heterogeneity (nearly IID)
```

### Select Specific Models

```python
models_config = {
    'LLM': [
        ('roberta-base', 'RoBERTa-Base'),
        ('bert-base-uncased', 'BERT-Base'),
        # Add more LLMs here
    ],
    'ViT': [
        ('google/vit-base-patch16-224-in21k', 'ViT-Base'),
        # Add more ViTs here
    ],
    'VLM': [
        (('roberta-base', 'google/vit-base-patch16-224-in21k'), 'RoBERTa+ViT'),
    ]
}
```

---

## 📊 Key Metrics Explained

### Performance Metrics

- **F1-Macro**: Unweighted average of per-class F1 scores (good for imbalanced data)
- **F1-Micro**: Global F1 computed from total TP, FP, FN (emphasizes frequent classes)
- **Accuracy**: Correctly classified samples / total samples
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Hamming Loss**: Fraction of wrong labels (for multi-label)
- **Jaccard Score**: Intersection over union of predictions

### Efficiency Metrics

- **Parameters (M)**: Number of trainable parameters in millions
- **Training Time (h)**: Wall-clock time to complete training
- **Inference Time (ms)**: Average time per prediction
- **Memory (MB)**: RAM usage during training
- **Communication Cost**: Data transferred in federated learning

---

## 🔬 Use Real Data

To use your actual agricultural data instead of synthetic data:

### Option 1: CSV File

```python
# In ultimate_model_comparison.py, replace synthetic data with:
df = pd.read_csv('your_data.csv')

# Expected CSV format:
# text,labels
# "Leaves turning yellow...", "[1, 0, 1, 0, 0]"
# "Brown spots appearing...", "[0, 1, 0, 1, 0]"
```

### Option 2: Use Existing Datasets Loader

```python
# Uncomment this in ultimate_model_comparison.py:
from datasets_loader import load_farm_datasets

df_train, df_val, df_test = load_farm_datasets()
```

### Option 3: HuggingFace Datasets

```python
from datasets import load_dataset

dataset = load_dataset('your_username/farm_dataset')
df = pd.DataFrame(dataset['train'])
```

---

## 📈 Interpreting Results

### Best Model Selection

1. **Check Summary Dashboard** (`25_summary_dashboard.png`)
   - Shows best model and key statistics

2. **Review Paper Comparison** (`14_paper_comparison_bars.png`)
   - See how your models compare to published work

3. **Examine Efficiency** (`09_parameter_efficiency.png`)
   - Balance accuracy with model size

### Trade-offs

| Scenario | Recommended Model |
|----------|-------------------|
| **Highest accuracy** | VLM (Multimodal) |
| **Privacy-preserving** | Federated LLM or VLM |
| **Resource-constrained** | DistilBERT or DeiT |
| **Fast inference** | DistilBERT or Mobile-sized models |
| **Cross-domain** | Pre-trained large models |

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
batch_size = 8  # or even 4

# Use gradient accumulation
accumulation_steps = 4
```

### Download Errors

```python
# Use offline mode (download models first)
model = AutoModel.from_pretrained('roberta-base', local_files_only=True)
```

### Long Training Times

```python
# Use fewer epochs/rounds for testing
n_epochs = 2
n_rounds = 3

# Or use smaller models
'distilbert-base-uncased'  # Instead of 'roberta-base'
```

---

## 📝 Citation

If you use this comparison framework in your research, please cite:

```bibtex
@article{farmfederate2026,
  title={FarmFederate: Federated Multimodal Learning for Agricultural Advisory Systems},
  author={Your Team},
  journal={Under Review},
  year={2026}
}
```

---

## 🤝 Contributing

To add new models or plots:

1. **Add Model**: Edit `models_config` in `ultimate_model_comparison.py`
2. **Add Plot**: Add new method to `UltimatePlottingSuite` class
3. **Add Baseline**: Update `baseline_papers` dict in `ComparisonFramework`

---

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Check documentation in `COMPLETE_SYSTEM_DOCUMENTATION.md`
- Review training logs in `outputs_ultimate_comparison/logs/`

---

## ✅ Checklist

Before running experiments:

- [ ] Install all dependencies
- [ ] Verify CUDA availability (if using GPU)
- [ ] Prepare dataset (or use synthetic for testing)
- [ ] Adjust training parameters for your hardware
- [ ] Ensure sufficient disk space (~5GB for outputs)

After experiments:

- [ ] Review all 25 plots
- [ ] Check CSV results file
- [ ] Compare with baseline papers
- [ ] Document best model and settings
- [ ] Save checkpoints for deployment

---

## 🎓 Key Research Questions Answered

This framework systematically addresses:

1. **Which model architecture is best?**
   - Compare LLM, ViT, and VLM head-to-head

2. **Is federated learning worth it?**
   - Analyze performance trade-offs vs centralized

3. **How do we compare to SOTA?**
   - Direct comparison with 15+ published papers

4. **What are the efficiency trade-offs?**
   - Parameters, time, memory, communication

5. **Where does the model fail?**
   - Error analysis and robustness testing

6. **What contributes to performance?**
   - Ablation studies on components

---

## 📚 Additional Resources

- **Full System Docs**: `COMPLETE_SYSTEM_DOCUMENTATION.md`
- **Training Guide**: `COLAB_TRAINING_GUIDE.md`
- **Paper Comparison**: `PAPER_COMPARISON_COMPLETE.md`
- **Publication Guide**: `PUBLICATION_GUIDE.md`

---

**Happy Experimenting! 🌾🤖**
