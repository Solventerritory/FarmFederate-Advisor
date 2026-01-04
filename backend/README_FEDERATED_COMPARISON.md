# Federated Learning Comprehensive Comparison
# LLM + ViT + VLM for Plant Stress Detection

This implementation provides a complete federated learning framework comparing:
- **Federated LLM** (Text-based): Flan-T5, GPT-2, T5, BERT, RoBERTa
- **Federated ViT** (Image-based): ViT, DeiT variants
- **Federated VLM** (Multimodal): CLIP, BLIP, BLIP-2

## Features

✅ **15+ Model Architectures** implemented and ready to compare
✅ **20 Comprehensive Plots** for detailed analysis
✅ **Federated Learning** with FedAvg, non-IID data distribution
✅ **Comparison with Paper Baselines** (FedAvg, FedProx, MOON, etc.)
✅ **LoRA/PEFT** for efficient fine-tuning
✅ **Multi-label Classification** for plant stress detection
✅ **Statistical Analysis** including significance tests

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_federated.txt
```

### 2. Quick Test (Recommended First Run)

```bash
python run_federated_comprehensive.py --quick_test
```

This runs a fast test with 3 models (Flan-T5-Small, ViT-Base, CLIP-Base) for 3 rounds.

### 3. Full Comparison

```bash
python run_federated_comprehensive.py --full
```

This trains all available models and generates comprehensive comparison plots.

### 4. Custom Configuration

```bash
python run_federated_comprehensive.py \
    --models flan-t5-small gpt2 vit-base clip-base \
    --rounds 10 \
    --clients 5 \
    --batch_size 16 \
    --samples 5000
```

## Available Models

### Federated LLM (Text-based)
- `flan-t5-small` - Flan-T5 Small (80M params)
- `flan-t5-base` - Flan-T5 Base (250M params)
- `t5-small` - T5 Small (60M params)
- `gpt2` - GPT-2 (124M params)
- `gpt2-medium` - GPT-2 Medium (355M params)
- `distilgpt2` - DistilGPT-2 (82M params)
- `roberta-base` - RoBERTa Base (125M params)
- `bert-base` - BERT Base (110M params)
- `distilbert` - DistilBERT (66M params)

### Federated ViT (Image-based)
- `vit-base` - Vision Transformer Base (86M params)
- `vit-large` - Vision Transformer Large (304M params)
- `vit-base-384` - ViT with 384x384 input (86M params)
- `deit-base` - Data-efficient Image Transformer (86M params)

### Federated VLM (Multimodal)
- `clip-base` - CLIP Base (151M params)
- `clip-large` - CLIP Large (428M params)
- `blip` - BLIP Base (224M params)
- `blip2` - BLIP-2 with OPT-2.7B (2.7B params)

## Generated Plots (20 Total)

1. **Overall F1 Comparison** - Micro and Macro F1 scores
2. **Model Type Comparison** - LLM vs ViT vs VLM
3. **Training Convergence** - Performance over rounds
4. **Per-Class Heatmap** - F1 scores across all classes
5. **Per-Class Bar Chart** - Detailed class-wise performance
6. **Precision-Recall Tradeoff** - Analysis of trade-offs
7. **ROC Curves** - Receiver Operating Characteristic
8. **Efficiency Scatter** - Time vs Performance
9. **Model Size Comparison** - Parameter counts
10. **Memory Usage** - Peak memory consumption
11. **Round Performance** - Metrics across rounds
12. **Statistical Tests** - Significance testing
13. **Paper Comparison** - vs Published Baselines
14. **Confusion Matrices** - Error analysis
15. **Learning Dynamics** - Convergence analysis
16. **Architecture Comparison** - Grouped by architecture
17. **Per-Class AUC** - AUC scores per class
18. **Communication Cost** - Federated learning overhead
19. **Scalability Analysis** - Size vs Time
20. **Error Analysis** - Detailed error breakdown

## Output Structure

```
results/
├── comparisons/
│   ├── 01_overall_f1_comparison.png
│   ├── 02_model_type_comparison.png
│   ├── ... (20 plots total)
│   ├── comparison_summary.txt
│   └── comparison_summary.csv
├── Flan-T5-Small/
│   ├── round_001.pt
│   ├── round_002.pt
│   ├── ...
│   └── final_model.pt
├── ViT-Base/
│   └── ...
├── CLIP-Base/
│   └── ...
└── training_summary.json
```

## Comparison with Papers

The framework compares against these baseline papers:
- **FedAvg** (McMahan et al., 2017)
- **FedProx** (Li et al., 2020)
- **FedBN** (Li et al., 2021)
- **FedNova** (Wang et al., 2020)
- **MOON** (Li et al., 2021)
- **FedDyn** (Acar et al., 2021)
- **PlantVillage** (Mohanty et al., 2016)
- **DeepPlant** (Ferentinos, 2019)
- **AgriNet** (Chen et al., 2020)
- **FedAgriculture** (Zhang et al., 2022)

## Key Components

### 1. Federated LLM (`FederatedLLM`)
- Supports seq2seq (T5), decoder (GPT), and encoder (BERT) architectures
- LoRA adapters for efficient fine-tuning
- Multi-label classification head

### 2. Federated ViT (`FederatedViT`)
- Vision Transformer backbone
- LoRA on attention layers
- Multi-scale classification

### 3. Federated VLM (`FederatedVLM`)
- Vision-language alignment (CLIP, BLIP)
- Cross-modal fusion
- Joint text-image processing

### 4. Comparison Framework (`ComparisonFramework`)
- Automatic plot generation
- Statistical analysis
- Comprehensive reporting

## Command Line Options

```
--models MODEL_NAMES      Specific models to train (space-separated)
--quick_test              Fast test with 3 models, 3 rounds
--full                    Train all available models
--rounds N                Number of federated rounds (default: 5)
--clients N               Number of federated clients (default: 5)
--batch_size N            Batch size (default: 16)
--samples N               Number of data samples (default: 5000)
--use_real_data           Try loading real HuggingFace datasets
--save_dir DIR            Output directory (default: results)
```

## Performance Tips

1. **Memory**: Start with smaller models (flan-t5-small, distilgpt2, vit-base)
2. **Speed**: Use `--quick_test` for initial testing
3. **GPU**: CUDA will be automatically used if available
4. **Batch Size**: Reduce if running out of memory

## Citation

If you use this implementation, please cite:

```bibtex
@software{farmfederate2026,
  title={Federated Learning Comprehensive Comparison: LLM, ViT, and VLM},
  author={FarmFederate Team},
  year={2026},
  url={https://github.com/your-repo/FarmFederate}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact the development team.

## Acknowledgments

- HuggingFace Transformers for model implementations
- PEFT library for LoRA support
- Agricultural dataset providers
- Federated learning research community
