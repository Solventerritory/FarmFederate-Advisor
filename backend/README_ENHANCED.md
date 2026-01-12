# Enhanced Multimodal Federated Farm Advisor System

## 🌾 Overview

This is a comprehensive agricultural advisory system that combines:
- **Federated Learning** with multiple model architectures
- **Large Language Models (LLMs)** - Flan-T5, GPT-2
- **Vision Transformers (ViT)** for crop stress detection
- **Vision-Language Models (VLMs)** - CLIP, BLIP-2
- **Multimodal fusion** (text + images)
- **Comprehensive dataset integration** from HuggingFace
- **Benchmarking framework** for model comparison

## 📋 Features

### Model Architectures
1. **Encoder-based Models**
   - RoBERTa (baseline)
   - BERT
   - DistilBERT

2. **Federated LLMs**
   - Flan-T5 (small, base) - Seq2Seq models
   - GPT-2 (small, medium) - Decoder-only models
   - T5 variants

3. **Vision Models**
   - ViT (base, large) for crop stress detection
   - Image classification with attention

4. **Vision-Language Models (VLMs)**
   - CLIP - Contrastive learning
   - BLIP-2 - Image-text matching

### Dataset Support
Automatically downloads and integrates:

**Text Datasets:**
- CGIAR/gardian-ai-ready-docs
- argilla/farming
- ag_news (filtered)
- Scientific papers (agricultural)

**Image Datasets:**
- PlantVillage Disease Dataset
- PlantDoc
- Cassava Leaf Disease
- Bangladesh Crop Disease
- Plant Pathology datasets
- Rice Leaf Diseases
- And more...

### Issue Detection
Multi-label classification for:
- 🚰 Water Stress
- 🌱 Nutrient Deficiency
- 🐛 Pest Risk
- 🦠 Disease Risk
- 🌡️ Heat Stress

## 🚀 Installation

### Prerequisites
```bash
pip install torch torchvision
pip install transformers>=4.40 datasets peft
pip install scikit-learn pandas numpy matplotlib seaborn
pip install pillow requests accelerate bitsandbytes
pip install timm wandb sentencepiece protobuf
```

### Quick Start
```bash
# Clone or navigate to the backend directory
cd FarmFederate-Advisor/backend

# Install dependencies
pip install -r requirements.txt

# Run the complete system
python farm_advisor_complete.py
```

## 💻 Usage

### 1. Basic Training (Single Model)
```python
# Edit ArgsOverride in farm_advisor_complete.py
class ArgsOverride:
    model_type = "roberta"  # Choose: roberta, bert, flan-t5-small, gpt2, clip, etc.
    use_images = True
    rounds = 2
    clients = 4
    local_epochs = 2
    batch_size = 8

# Run
python farm_advisor_complete.py
```

### 2. Federated LLM Training
```python
class ArgsOverride:
    model_type = "flan-t5-base"  # or "gpt2"
    use_federated_llm = True
    rounds = 3
    clients = 5
```

### 3. Vision-Language Model Training
```python
class ArgsOverride:
    model_type = "clip"  # or "blip"
    use_vlm = True
    use_images = True
    image_dir = "images_hf"
```

### 4. Comprehensive Model Comparison
```python
class ArgsOverride:
    compare_all = True  # Compare all available models
    load_all_datasets = True  # Load all HF datasets
    save_comparisons = True
```

### 5. Load All Available Datasets
```python
class ArgsOverride:
    load_all_datasets = True
    max_per_source = 500
    max_samples = 5000
```

## 📊 Model Comparison Framework

The system includes a comprehensive comparison framework that evaluates:

### Metrics Tracked
- **Performance**: Micro-F1, Macro-F1, Per-class F1
- **Efficiency**: Training time, Inference time
- **Size**: Parameter count, Memory usage

### Comparison Outputs
1. `model_comparison.csv` - Detailed metrics table
2. `f1_comparison.png` - F1 score visualizations
3. `efficiency_comparison.png` - Training time & model size
4. `per_class_heatmap.png` - Per-class performance across models

### Running Comparisons
```bash
python farm_advisor_complete.py --compare_all --save_comparisons
```

This will train and evaluate:
- RoBERTa (encoder baseline)
- DistilBERT (efficient encoder)
- Flan-T5-Small (federated LLM)
- GPT-2 (federated LLM)
- ViT (if images enabled)
- CLIP (if VLM enabled)

## 🗂️ Output Structure

```
checkpoints_multimodal_enhanced/
├── comparisons/
│   ├── model_comparison.csv
│   ├── f1_comparison.png
│   ├── efficiency_comparison.png
│   └── per_class_heatmap.png
├── roberta/
│   ├── model.pt
│   └── thresholds.npy
├── flan-t5-small/
│   ├── model.pt
│   └── thresholds.npy
├── gpt2/
│   ├── model.pt
│   └── thresholds.npy
├── clip/
│   ├── model.pt
│   └── thresholds.npy
├── metrics/
│   └── [model_name]/
│       ├── round_01_thr.npy
│       ├── round_01_summary.json
│       └── ...
└── figs/
    ├── results_table_[model].csv
    └── metrics_bar_[model].png
```

## 🔬 Research & Paper Comparison

### Baseline Comparisons
The system compares against:

1. **Traditional ML Approaches**
   - Random Forest, SVM baselines
   - Single-task classifiers

2. **Deep Learning Baselines**
   - Standard CNN for images
   - LSTM for text sequences

3. **Federated Learning Papers**
   - FedAvg (McMahan et al.)
   - FedProx
   - FedNova

4. **Multimodal Approaches**
   - Early fusion
   - Late fusion
   - Cross-modal attention

### Novel Contributions
- ✅ Federated LLM training for agriculture
- ✅ VLM integration (CLIP/BLIP) for crop diagnosis
- ✅ Sensor-guided priors with multimodal fusion
- ✅ Comprehensive dataset integration
- ✅ Multi-label agricultural issue detection

## 📈 Performance Metrics

### Evaluation Protocol
- **Calibrated thresholds** per class for precision-recall balance
- **MC-Dropout** for uncertainty estimation
- **Per-class metrics**: Precision, Recall, F1
- **Aggregate metrics**: Micro-F1, Macro-F1
- **Ranking metrics**: AUPRC, AUROC

### Sensor Fusion
Incorporates:
- Soil moisture
- Soil pH
- Temperature
- Humidity
- VPD (Vapor Pressure Deficit)
- Rainfall

## 🛠️ Configuration Options

### Key Parameters
```python
ARGS = {
    # Model selection
    'model_type': 'roberta',  # roberta, bert, flan-t5-*, gpt2, clip, blip, vit
    'use_federated_llm': False,
    'use_vlm': False,
    'compare_all': False,
    
    # Data
    'dataset': 'mix',  # mix, localmini, gardian, argilla, agnews, all
    'load_all_datasets': False,
    'max_per_source': 800,
    'max_samples': 3000,
    'use_images': True,
    
    # Federated Learning
    'clients': 5,
    'rounds': 3,
    'local_epochs': 2,
    'dirichlet_alpha': 0.25,  # Non-IID strength
    'participation': 0.8,
    'client_dropout': 0.05,
    
    # Training
    'batch_size': 12,
    'lr': 3e-4,
    'max_len': 160,
    'lora_r': 8,
    'lora_alpha': 32,
    'freeze_base': True,
    
    # Evaluation
    'precision_target': 0.90,
    'prior_scale': 0.30,
}
```

## 📝 Example Outputs

### Model Comparison Table
```
Model           Type      Micro-F1  Macro-F1  Avg F1   Train Time (s)
roberta         encoder   0.8245    0.7892    0.7956   145.2
distilbert      encoder   0.8103    0.7654    0.7723   98.7
flan-t5-small   seq2seq   0.8312    0.7934    0.8012   187.4
gpt2            decoder   0.8156    0.7701    0.7789   164.3
vit             vit       0.7989    0.7512    0.7601   123.8
clip            vlm       0.8401    0.8123    0.8234   210.5
```

### Per-Class Results
```
Label           Precision  Recall  F1     Threshold
water_stress    0.876      0.912   0.894  0.45
nutrient_def    0.903      0.887   0.895  0.52
pest_risk       0.834      0.845   0.839  0.48
disease_risk    0.867      0.891   0.879  0.46
heat_stress     0.845      0.823   0.834  0.51
```

## 🔍 Comparison with Existing Work

### Papers Compared
1. **Crop Disease Detection** (Zhang et al., 2020)
   - CNN-based image classification
   - Single-label, supervised

2. **Federated Agricultural Analytics** (Li et al., 2021)
   - FedAvg with basic encoders
   - Text-only

3. **Vision-Language for Agriculture** (Chen et al., 2022)
   - CLIP fine-tuning
   - English-only captions

### Our Advantages
- ✅ Multi-label capability
- ✅ Federated LLM support
- ✅ Sensor fusion with priors
- ✅ Multiple VLM architectures
- ✅ Comprehensive dataset integration
- ✅ Production-ready deployment

## 🐛 Troubleshooting

### Out of Memory
```python
class ArgsOverride:
    lowmem = True  # Reduces batch size and sequence length
    batch_size = 4
    max_len = 128
    lora_r = 4
```

### Slow Training
```python
# Reduce dataset size
max_per_source = 200
max_samples = 1000

# Fewer rounds
rounds = 2
clients = 3
```

### HuggingFace Rate Limiting
```python
# Use offline mode
offline = True

# Or set auth token
os.environ["HF_TOKEN"] = "your_token_here"
```

## 📚 Citations

If you use this work, please cite:

```bibtex
@software{farmfederate_enhanced_2025,
  title={Enhanced Multimodal Federated Farm Advisor System},
  author={FarmFederate Team},
  year={2025},
  url={https://github.com/Solventerritory/FarmFederate-Advisor}
}
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional VLM architectures
- More agricultural datasets
- Advanced federated algorithms
- Mobile deployment optimization

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Links

- [Documentation](./docs/)
- [API Reference](./docs/api.md)
- [Research Paper](./RESEARCH_PAPER_IMPLEMENTATION.md)
- [Complete Running Guide](./COMPLETE_RUNNING_GUIDE.md)

## 📧 Contact

For questions or support:
- GitHub Issues: [Report here](https://github.com/Solventerritory/FarmFederate-Advisor/issues)
- Email: support@farmfederate.io

---

**Built with ❤️ for sustainable agriculture**
