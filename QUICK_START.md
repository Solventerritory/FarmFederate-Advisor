# Quick Start Guide - Enhanced FarmFederate

## 🚀 Getting Started with Research Features

### 1. Setup Enhanced Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Verify enhanced modules
python -c "
from federated_core import fedavg_aggregate, FederatedMetrics, krum_aggregate
from multimodal_model import MultiModalModel
print('✓ Enhanced modules loaded successfully')
"

# Run server with cross-attention enabled
export USE_CROSS_ATTENTION=true
export ENABLE_UNCERTAINTY=true
uvicorn server:app --host 0.0.0.0 --port 8000
```



### 3. Run Federated Training

```bash
cd backend

# Basic federated training with FedAvg
python train_fed_multimodal.py \
  --rounds 10 \
  --clients 5 \
  --local-epochs 3 \
  --batch-size 16

# With differential privacy
python train_fed_multimodal.py \
  --rounds 10 \
  --clients 5 \
  --use-dp \
  --noise-scale 0.01

# With Byzantine-robust aggregation (Krum)
python train_fed_multimodal.py \
  --rounds 10 \
  --clients 5 \
  --aggregation krum \
  --num-byzantine 1

# With gradient compression
python train_fed_multimodal.py \
  --rounds 10 \
  --clients 5 \
  --compression-ratio 0.1 \
  --compression-method topk

# Full enhanced training
python train_fed_multimodal.py \
  --rounds 10 \
  --clients 5 \
  --strategy importance \
  --use-dp \
  --noise-scale 0.01 \
  --compression-ratio 0.1 \
  --use-cross-attention
```

### 4. Test Enhanced Model Features

```python
# test_enhanced_model.py
import torch
from multimodal_model import MultiModalModel, build_tokenizer, build_image_processor
from PIL import Image

# Load model
model = MultiModalModel(
    use_cross_attention=True,
    dropout=0.1
)
model.eval()

# Load tokenizer and image processor
tokenizer = build_tokenizer("roberta-base")
image_processor = build_image_processor("google/vit-base-patch16-224-in21k")

# Prepare sample input
text = "Leaf showing yellow spots and wilting"
image = Image.open("sample_leaf.jpg")

# Tokenize
encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=160)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

# Process image
pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]

# Forward pass with attention weights
output = model(input_ids, attention_mask, pixel_values, return_attention=True)
logits = output.logits
attention_weights = output.attention_weights

print(f"Predictions shape: {logits.shape}")  # [1, 5]
print(f"Text-to-Image attention shape: {attention_weights['text_to_image'].shape}")
print(f"Image-to-Text attention shape: {attention_weights['image_to_text'].shape}")

# Get uncertainty estimation
mean_logits, std_logits = model.get_uncertainty(
    input_ids, attention_mask, pixel_values, n_samples=10
)

print(f"\nPredictions with uncertainty:")
labels = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
probs = torch.sigmoid(mean_logits)[0].detach().numpy()
stds = torch.sigmoid(std_logits)[0].detach().numpy()

for label, prob, std in zip(labels, probs, stds):
    print(f"  {label}: {prob:.2%} ± {std:.2%}")
```

### 5. Monitor Telemetry

```python
# monitor_telemetry.py
import requests
import time
from datetime import datetime

BACKEND_URL = "http://localhost:8000"

def check_telemetry():
    """Poll for latest telemetry data"""
    try:
        response = requests.get(f"{BACKEND_URL}/telemetry/latest")
        if response.status_code == 200:
            data = response.json()
            print(f"\n[{datetime.now()}] Telemetry:")
            print(f"  Device: {data['device_id']}")
            print(f"  Uptime: {data['uptime_ms'] / 1000:.0f}s")
            print(f"  Captures: {data['total_captures']} ({data['successful_uploads']} success)")
            print(f"  Quality: {data['avg_quality']:.2f}")
            print(f"  RSSI: {data['rssi_dbm']} dBm")
            print(f"  Free Heap: {data['free_heap_bytes'] / 1024:.1f} KB")
    except Exception as e:
        print(f"Error: {e}")

# Monitor every 30 seconds
while True:
    check_telemetry()
    time.sleep(30)
```

### 6. Use Federated Core APIs

```python
# federated_training_example.py
import torch
from federated_core import (
    fedavg_aggregate,
    krum_aggregate,
    add_differential_privacy,
    adaptive_client_sampling,
    compress_gradients,
    FederatedMetrics
)

# Initialize metrics
metrics = FederatedMetrics()

# Simulate client states (after local training)
client_states = [
    {"param1": torch.randn(100), "param2": torch.randn(50)},
    {"param1": torch.randn(100), "param2": torch.randn(50)},
    {"param1": torch.randn(100), "param2": torch.randn(50)},
    {"param1": torch.randn(100), "param2": torch.randn(50)},
    {"param1": torch.randn(100), "param2": torch.randn(50)},
]

# Client dataset sizes
data_sizes = [1000, 1500, 800, 1200, 900]

# Method 1: FedAvg with uniform weights
aggregated = fedavg_aggregate(client_states)

# Method 2: FedAvg with data-proportional weights
aggregated = fedavg_aggregate(client_states, client_weights=data_sizes)

# Method 3: Byzantine-robust Krum
aggregated = krum_aggregate(client_states, num_byzantine=1, multi_krum=True)

# Add differential privacy
private_aggregated = add_differential_privacy(
    aggregated, 
    noise_scale=0.01, 
    clip_norm=1.0
)

# Compress gradients for communication efficiency
compressed, indices = compress_gradients(
    private_aggregated,
    compression_ratio=0.1,
    method="topk"
)

print(f"Original params: {sum(p.numel() for p in private_aggregated.values())}")
print(f"Compressed params: {sum(p.numel() for p in compressed.values())}")
print(f"Compression ratio: {sum(p.numel() for p in compressed.values()) / sum(p.numel() for p in private_aggregated.values()):.2%}")

# Adaptive client selection
client_stats = [
    {"id": i, "data_size": size, "loss": 0.5, "staleness": 0}
    for i, size in enumerate(data_sizes)
]

selected_importance = adaptive_client_sampling(client_stats, 3, strategy="importance")
selected_loss = adaptive_client_sampling(client_stats, 3, strategy="loss_weighted")

print(f"\nSelected clients (importance): {selected_importance}")
print(f"Selected clients (loss-weighted): {selected_loss}")

# Log metrics
metrics.log_round(1, {
    "train_loss": 0.45,
    "val_loss": 0.52,
    "num_clients": len(client_states),
    "compression_ratio": 0.1
})

for i in range(len(client_states)):
    metrics.log_client(i, 1, {
        "local_train_loss": 0.4 + i * 0.05,
        "data_size": data_sizes[i]
    })

# Export metrics
metrics.export_to_json("federated_metrics.json")
print("\nMetrics exported to federated_metrics.json")

summary = metrics.get_summary()
print(f"\nSummary: {summary}")
```



**Normal Operation:**
```
Loop start
Loop end
Loop start
Auto capture triggered
========== MULTI-SHOT CAPTURE ==========
[SHOT 1/3] Capturing...
[QUALITY] Shot 1: 0.75
[SHOT 2/3] Capturing...
[QUALITY] Shot 2: 0.82
[SHOT 3/3] Capturing...
[QUALITY] Shot 3: 0.79
[BEST] Shot 2 selected (quality: 0.82)

[CAPTURE #1] Starting upload...
[UPLOAD] Attempt 1/3
[POST] Uploading 45678 bytes to http://192.168.208.1:8000/predict
[RESPONSE] HTTP Code: 200
[SUCCESS] Upload successful!

========== ENHANCED ANALYSIS RESULTS ==========
[DETECTED] 1 disease(s) detected:
  • disease_risk: 78.5% (±5.2% uncertainty)

[CONFIDENCE] Overall model confidence: 92.3%

[RECOMMENDATIONS]:
  • Improve airflow around plants
  • Avoid late overhead irrigation
  • Remove affected leaves promptly

[ADAPTIVE] Disease detected - increased frequency to 30s
===============================================
```

**Telemetry Output (every 10 captures):**
```
[TELEMETRY] Sending device statistics...
{
  "device_id": "device_01",
  "version": "v2.0-federated",
  "uptime_ms": 1234567,
  "total_captures": 10,
  "successful_uploads": 9,
  "failed_uploads": 1,
  "avg_quality": 0.81,
  "rssi_dbm": -67,
  "free_heap_bytes": 98304,
  "current_interval_ms": 45000
}
[TELEMETRY] Sent successfully (HTTP 200)
```

### 8. Troubleshooting

**Device not connecting to WiFi:**
```cpp
// Increase timeout in src/main.cpp
while (WiFi.status() != WL_CONNECTED && attempts < 40) {  // Was 20
    delay(500);
    Serial.print(".");
    attempts++;
}
```

**Backend out of memory:**
```bash
# Reduce batch size or use gradient checkpointing
export BATCH_SIZE=8  # Default 16
export USE_GRADIENT_CHECKPOINTING=true
```

**High uncertainty scores:**
```python
# Increase MC dropout samples
mean_logits, std_logits = model.get_uncertainty(
    input_ids, attention_mask, pixel_values, n_samples=20  # Was 10
)
```

**Slow federated convergence:**
```bash
# Adjust learning rate or increase local epochs
python train_fed_multimodal.py \
  --rounds 15 \
  --local-epochs 5 \
  --learning-rate 5e-5
```

## 📊 Monitoring Dashboard

**View metrics:**
```bash
# Install visualization dependencies
pip install matplotlib seaborn pandas

# Generate plots
python scripts/plot_federated_metrics.py metrics/federated_metrics.json
```

**Outputs:**
- `plots/train_loss_per_round.png`
- `plots/client_contributions.png`
- `plots/communication_efficiency.png`
- `plots/uncertainty_calibration.png`

## 🎯 Key Features Usage

| Feature | Command/Code |
|---------|--------------|
| Cross-attention | `model = MultiModalModel(use_cross_attention=True)` |
| Uncertainty | `mean, std = model.get_uncertainty(...)` |
| Multi-shot | `#define MULTI_SHOT_COUNT 3` |
| Adaptive interval | `#define ADAPTIVE_INTERVAL true` |
| FedAvg | `fedavg_aggregate(client_states)` |
| Krum | `krum_aggregate(client_states, num_byzantine=1)` |
| DP | `add_differential_privacy(state, noise_scale=0.01)` |
| Compression | `compress_gradients(state, compression_ratio=0.1)` |
| Metrics | `metrics = FederatedMetrics(); metrics.export_to_json(...)` |

## 🔬 Research Experiments

**Ablation Study:**
```bash
# Baseline (no enhancements)
python train_fed_multimodal.py --rounds 10 --no-cross-attention

# +Cross-attention
python train_fed_multimodal.py --rounds 10 --use-cross-attention

# +Adaptive sampling
python train_fed_multimodal.py --rounds 10 --use-cross-attention --strategy importance

# +Compression
python train_fed_multimodal.py --rounds 10 --use-cross-attention --strategy importance --compression-ratio 0.1

# Full system
python train_fed_multimodal.py --rounds 10 --use-cross-attention --strategy importance --compression-ratio 0.1 --use-dp
```

**Compare aggregation methods:**
```bash
for method in fedavg krum median; do
    python train_fed_multimodal.py --rounds 10 --aggregation $method --output results_$method.json
done
python scripts/compare_aggregations.py results_*.json
```

## 📝 Next Steps

1. ✅ Backend enhancements implemented
2. ✅ Model architecture upgraded

4. ✅ Configuration files created
5. 🔄 Frontend UI updates (partially complete)
6. 🔄 End-to-end testing (in progress)
7. ⏳ Documentation completion
8. ⏳ Performance benchmarking
9. ⏳ Deployment to production

## 🆘 Support

- Issues: Check [RESEARCH_PAPER_IMPLEMENTATION.md](RESEARCH_PAPER_IMPLEMENTATION.md)
- API docs: See `docs/API.md`


**Happy Federated Learning! 🚀🌱**
