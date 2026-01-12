# CPU Training Guide (VS Code)

Train FarmFederate models locally on CPU without GPUs.

## Quick Start

```powershell
cd backend
python train_cpu.py
```

## Training Options

When you run the script, you'll see:

```
1. Federated LLM (DistilBERT) - ~2-3 hours
2. Federated ViT (ViT-Base) - ~5-7 hours  
3. Federated VLM (CLIP) - ~4-5 hours
4. Train All Models - ~12-15 hours total
5. Quick Test (1 round, 2 clients) - ~30 minutes

Enter choice (1-5):
```

### Recommended: Start with Quick Test

Enter `5` to run a quick test (~30 minutes):
- 1 round instead of 5
- 2 clients instead of 4
- Validates everything works before long training

### For Production Training

Enter `4` to train all models (~12-15 hours):
- Federated LLM: DistilBERT with LoRA
- Federated ViT: ViT-Base with LoRA
- Federated VLM: CLIP multimodal

## CPU Optimizations

The script is optimized for CPU training:

| Optimization | CPU Setting | GPU Setting |
|-------------|-------------|-------------|
| Clients | 4 | 8 |
| Rounds | 5 | 10 |
| Local Epochs | 2 | 3 |
| Batch Size | 4 | 16 |
| Max Length | 128 | 512 |
| LoRA Rank | 8 | 16 |

## What Gets Trained

### 1. Federated LLM (~2-3 hours)
- Model: DistilBERT (66M params)
- Task: Plant stress detection from text
- Dataset: Synthetic text symptoms
- Output: `checkpoints_cpu/llm_distilbert_final.pt`

### 2. Federated ViT (~5-7 hours)
- Model: ViT-Base (86M params)
- Task: Plant disease detection from images
- Dataset: Synthetic plant images
- Output: `checkpoints_cpu/vit_base_final.pt`

### 3. Federated VLM (~4-5 hours)
- Model: CLIP (151M params)
- Task: Multimodal plant analysis
- Dataset: Synthetic text-image pairs
- Output: `checkpoints_cpu/vlm_clip_final.pt`

## During Training

You'll see progress like this:

```
Round 1/5
  Client 1/4... Loss: 1.2345
  Client 2/4... Loss: 1.1234
  Client 3/4... Loss: 1.0987
  Client 4/4... Loss: 1.1567
  Aggregating client models... ✓
  Round Loss: 1.1533 | Time: 24.3min

  💾 Checkpoint saved: checkpoints_cpu/llm_distilbert_round2.pt
```

## Checkpoints

Saved every 2 rounds and at the end:

```
checkpoints_cpu/
├── llm_distilbert_round2.pt
├── llm_distilbert_round4.pt
├── llm_distilbert_final.pt
├── vit_base_round2.pt
├── vit_base_round4.pt
├── vit_base_final.pt
├── vlm_clip_round2.pt
├── vlm_clip_round4.pt
├── vlm_clip_final.pt
└── training_summary.json
```

## After Training

### 1. Generate Plots

```powershell
# Update publication_plots.py to use CPU checkpoints
python publication_plots.py --checkpoint-dir checkpoints_cpu
```

### 2. View Training Summary

```powershell
cat checkpoints_cpu/training_summary.json
```

## Troubleshooting

### Out of Memory (OOM)

If you get OOM errors, reduce batch size in `train_cpu.py`:

```python
CONFIG = {
    'batch_size': 2,  # Reduce from 4 to 2
    # ...
}
```

### Slow Training

Normal on CPU. To speed up:

1. **Reduce rounds**: `'num_rounds': 3` instead of 5
2. **Reduce clients**: `'num_clients': 2` instead of 4
3. **Reduce local epochs**: `'local_epochs': 1` instead of 2

### Dependencies Missing

```powershell
pip install torch torchvision transformers peft datasets tqdm
```

### Want to Pause/Resume

Training saves checkpoints every 2 rounds. To resume:

1. Stop training (Ctrl+C)
2. Edit `train_cpu.py` to load last checkpoint
3. Restart training

## Performance Tips

### 1. Close Other Applications

Free up CPU and RAM by closing:
- Browsers with many tabs
- IDEs except VS Code
- Background applications

### 2. Run Overnight

Training takes 12-15 hours for all models:
- Start before bed
- Check progress in the morning

### 3. Monitor with Task Manager

- CPU usage should be 80-100%
- RAM usage: 4-8GB typical
- If CPU < 50%, close other apps

## Configuration

Edit `train_cpu.py` to customize:

```python
CONFIG = {
    'device': 'cpu',
    'num_clients': 4,      # More clients = more realistic federated learning
    'num_rounds': 5,       # More rounds = better convergence
    'local_epochs': 2,     # More epochs = better local training
    'batch_size': 4,       # Larger batch = faster but more memory
    'learning_rate': 2e-5, # Standard LLM learning rate
    'max_length': 128,     # Longer sequences = more context
    'image_size': 224,     # Standard ViT input size
    'lora_r': 8,           # Higher rank = more parameters
    'lora_alpha': 16,      # Higher alpha = stronger adaptation
    'checkpoint_dir': 'checkpoints_cpu',
    'save_every': 2,       # Save checkpoint frequency
}
```

## Expected Results

After training completes, you should have:

### Final Losses (approximate)
- LLM: ~0.8-1.2 (classification loss)
- ViT: ~0.9-1.3 (classification loss)
- VLM: ~0.7-1.1 (contrastive loss)

### Model Files
- Total size: ~2-3GB
- Format: PyTorch `.pt` files
- Include: model weights, optimizer state, config

### Training Summary
```json
{
  "timestamp": "2026-01-03T...",
  "total_time_hours": 14.5,
  "config": {...},
  "results": {
    "llm": {"loss": 1.05},
    "vit": {"loss": 1.12},
    "vlm": {"loss": 0.89}
  },
  "device": "cpu"
}
```

## Next Steps After Training

1. **Move checkpoints** to main checkpoint directory:
   ```powershell
   Copy-Item checkpoints_cpu/* checkpoints/
   ```

2. **Generate publication plots** with real data:
   ```powershell
   python publication_plots.py
   ```

3. **Compare with internet papers**:
   ```powershell
   python paper_comparison_updated.py
   python plot_internet_comparison.py
   ```

4. **Integrate into research paper**:
   - Use plots from `figs_publication/`
   - Use comparisons from `publication_ready/`
   - Reference training summary for methodology

## FAQ

**Q: Can I use GPU later?**
A: Yes! The same checkpoint format works on GPU. Just load with `torch.load()` and `.to('cuda')`.

**Q: Why synthetic data?**
A: Real dataset loading requires internet and ~50GB downloads. Synthetic data trains the architecture and federated learning pipeline. You can replace with real data later.

**Q: How accurate will models be?**
A: With synthetic data, accuracy metrics are not meaningful. The goal is to train the **architecture** and **federated learning** components. With real data, expect 80-90% accuracy.

**Q: Can I stop and resume?**
A: Yes, but you need to modify the script to load the last checkpoint and continue from that round. Checkpoints are saved every 2 rounds.

**Q: Different models than Colab notebook?**
A: Yes! CPU version uses smaller, faster models:
- LLM: DistilBERT instead of Flan-T5
- ViT: ViT-Base (same)
- VLM: CLIP (same) but not BLIP-2 (too large)

**Q: How to train specific model only?**
A: Choose option 1, 2, or 3 when prompted. Don't choose option 4 (train all).

## Time Estimates by System

| System | RAM | Quick Test | Full Training |
|--------|-----|------------|---------------|
| High-end (i7/Ryzen 7) | 16GB+ | 20-25 min | 10-12 hours |
| Mid-range (i5/Ryzen 5) | 8GB+ | 30-35 min | 12-15 hours |
| Low-end (i3/Ryzen 3) | 4GB+ | 40-50 min | 18-24 hours |

## Support

If training fails:
1. Check error message
2. Reduce batch size / num clients
3. Ensure dependencies installed
4. Check available RAM (need 4GB+ free)
5. Try quick test mode first

## Comparison: CPU vs Colab GPU

| Aspect | CPU (VS Code) | Colab GPU |
|--------|---------------|-----------|
| **Cost** | Free | $0-49/mo |
| **Time** | 12-15 hours | 25-60 hours |
| **Setup** | 1 minute | 5 minutes |
| **Models** | Smaller (DistilBERT) | Larger (Flan-T5) |
| **Internet** | Not required | Required |
| **Interruptions** | None | Disconnects |
| **Control** | Full control | Limited runtime |

Choose CPU if:
- ✅ You want to start NOW
- ✅ You have overnight/weekend time
- ✅ You prefer local control
- ✅ You don't want to pay for Colab Pro

Choose GPU if:
- ✅ You need larger models (Flan-T5, BLIP-2)
- ✅ You want faster per-epoch training
- ✅ You're okay with managing disconnects
- ✅ You have Colab Pro subscription

---

**Ready to train?** Run: `python train_cpu.py`
