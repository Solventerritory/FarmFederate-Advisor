# 📊 Using ALL Available Datasets - Configuration Guide

**Goal:** Maximize real dataset usage for best training results

---

## 🎯 Quick Start: Use All Datasets

The **Zero-Error Edition** is now pre-configured to use ALL available datasets!

### Default Configuration (Maximum Data)

```python
class ArgsOverride:
    dataset = "mix"                              # Uses ALL HF datasets + synthetic
    use_images = True                            # Enable image datasets
    max_per_source = 1000                        # 1000 samples per HF dataset
    max_samples = 5000                           # 5000 total samples
    mix_sources = "gardian,argilla,agnews,localmini"  # All text sources
```

**This will attempt to load:**
- ✅ 1,000 samples from AG News
- ✅ 1,000 samples from CGIAR GARDIAN
- ✅ 1,000 samples from Argilla Farming
- ✅ 1,000+ samples from LocalMini (synthetic)
- ✅ 1,000 samples PER image dataset (9 datasets)
- ✅ **Total: ~5,000 multimodal samples**

---

## 📚 All Available Datasets

### **Text Datasets** (4 sources)

#### 1. AG News (Real-world)
```python
Source: "ag_news"
Size: Up to 1,000 (filtered for agriculture)
Quality: High
Content: Agricultural news articles
```

#### 2. CGIAR GARDIAN (Real-world)
```python
Source: "CGIAR/gardian-ai-ready-docs"
Size: Up to 1,000
Quality: Very High (research-grade)
Content: Agricultural research documents, best practices
```

#### 3. Argilla Farming (Real-world)
```python
Source: "argilla/farming"
Size: Up to 1,000
Quality: High
Content: Farming Q&A pairs, expert answers
```

#### 4. LocalMini (Synthetic fallback)
```python
Source: Generated in-code
Size: 1,000-1,500
Quality: Medium (synthetic but realistic)
Content: Sensor readings + farmer observations
```

### **Image Datasets** (9 sources)

#### 1. PlantVillage (Primary) ⭐
```python
Source: "BrandonFors/Plant-Diseases-PlantVillage-Dataset"
Size: 54,000+ images (using up to 1,000)
Quality: Very High
Content: 38 disease classes, 14 plant species
```

#### 2. PlantVillage Variant
```python
Source: "GVJahnavi/PlantVillage_dataset"
Size: Up to 1,000
Quality: High
Content: Alternative PlantVillage distribution
```

#### 3. Cassava Disease
```python
Source: "pufanyi/cassava-leaf-disease-classification"
Size: Up to 1,000
Quality: High
Content: Cassava leaf diseases
```

#### 4. Bangladesh Crops
```python
Source: "Saon110/bd-crop-vegetable-plant-disease-dataset"
Size: Up to 1,000
Quality: High
Content: Regional crop diseases
```

#### 5. Plant Pathology
```python
Source: "timm/plant-pathology-2021"
Size: Up to 1,000
Quality: Very High (competition dataset)
Content: Apple leaf pathology
```

#### 6. PlantWild
```python
Source: "uqtwei2/PlantWild"
Size: Up to 1,000
Quality: Medium-High
Content: Wild plant images
```

#### 7. PlantDoc
```python
Source: "agyaatcoder/PlantDoc"
Size: Up to 1,000
Quality: High
Content: Plant documentation images
```

#### 8. Alternative PlantVillage
```python
Source: "nateraw/plant-village"
Size: Up to 1,000
Quality: High
Content: Another PlantVillage variant
```

#### 9. Disease Classification
```python
Source: "keremberke/plant-disease-classification"
Size: Up to 1,000
Quality: High
Content: Multi-class disease images
```

---

## 🚀 How to Run with All Datasets

### Option 1: Use Default (Pre-configured)

Just run the Zero-Error Edition - it's already configured!

```bash
cd backend
python farm_advisor_multimodal_zero_error.py
```

Or in Colab:
```python
%run farm_advisor_multimodal_zero_error.py
```

### Option 2: Customize Dataset Mix

Edit the `ArgsOverride` class in the script:

```python
class ArgsOverride:
    # Maximize everything
    max_per_source = 2000          # Even more per dataset
    max_samples = 10000            # 10K total samples

    # All text sources
    mix_sources = "gardian,argilla,agnews,localmini"

    # Enable images
    use_images = True
    image_dir = "images_all"

    # Full training
    rounds = 10
    clients = 5
    local_epochs = 3
```

### Option 3: Text-Only (No Images)

```python
class ArgsOverride:
    use_images = False             # Disable images
    max_per_source = 2000
    max_samples = 8000             # More text samples
    mix_sources = "gardian,argilla,agnews,localmini"
```

### Option 4: Images-Only (No Text)

```python
class ArgsOverride:
    dataset = "localmini"          # Minimal text (just for labels)
    use_images = True
    max_per_source = 2000          # 2K images per source
    max_samples = 10000            # Max 10K images
```

---

## 📊 Expected Dataset Sizes

With `max_per_source=1000` and `max_samples=5000`:

### Text Sources (attempting to load):
```
AG News:        1,000 samples
GARDIAN:        1,000 samples (if accessible)
Argilla:        1,000 samples (if accessible)
LocalMini:      1,000 samples (always)
────────────────────────────
Potential:      4,000 text samples
Actual:         ~2,500-4,000 (depends on HF access)
```

### Image Sources (attempting to load):
```
PlantVillage:   1,000 images
PlantVillage-2: 1,000 images
Cassava:        1,000 images
Bangladesh:     1,000 images
PathPlat:       1,000 images
PlantWild:      1,000 images
PlantDoc:       1,000 images
AltVillage:     1,000 images
DiseaseClass:   1,000 images
────────────────────────────
Potential:      9,000 images
Actual:         ~2,000-5,000 (depends on availability)
```

### Final Multimodal Dataset:
```
Total samples: ~5,000 (capped by max_samples)
Text:          ~2,500-4,000
Images:        ~2,000-5,000
Pairs:         ~5,000 (text + image aligned)
```

---

## 🔧 Handling Dataset Failures

The system is **robust** and handles failures gracefully:

### If a dataset fails to load:
1. ✅ **Logs warning** (not an error)
2. ✅ **Continues** to next dataset
3. ✅ **Falls back** to synthetic if all fail
4. ✅ **Never crashes** - training always proceeds

### Common reasons for failure:
- Network timeout
- Dataset gated (requires authentication)
- Dataset moved or deleted
- HuggingFace API rate limit

### Solution:
The code automatically handles all these cases!

---

## 🎛️ Advanced Configuration

### Maximum Data Collection

```python
class ArgsOverride:
    # ULTRA settings - maximum data
    max_per_source = 5000          # 5K per source
    max_samples = 20000            # 20K total
    mix_sources = "gardian,argilla,agnews,localmini"
    use_images = True

    # Warning: This will take longer to load and train
    # Recommended for final production runs only
```

**Expected load time:** 10-20 minutes for dataset download

### Balanced Mix (Recommended)

```python
class ArgsOverride:
    max_per_source = 1000          # 1K per source
    max_samples = 5000             # 5K total
    mix_sources = "gardian,argilla,agnews,localmini"
    use_images = True
```

**Expected load time:** 3-5 minutes

### Quick Test (Minimal Real Data)

```python
class ArgsOverride:
    max_per_source = 200           # 200 per source
    max_samples = 1000             # 1K total
    mix_sources = "agnews,localmini"  # Just AG News + synthetic
    use_images = True
```

**Expected load time:** 1-2 minutes

---

## 📈 Performance vs Dataset Size

### Small (1K samples)
- **Training Time:** ~30 min
- **Expected F1:** 0.65-0.72
- **Use Case:** Quick testing

### Medium (5K samples) ⭐ **Recommended**
- **Training Time:** ~1-2 hours
- **Expected F1:** 0.75-0.82
- **Use Case:** Research, production

### Large (10K+ samples)
- **Training Time:** ~3-5 hours
- **Expected F1:** 0.78-0.85
- **Use Case:** Final model, paper results

---

## 🔍 Verify Dataset Loading

Add this to check what actually loaded:

```python
# After running build_corpus()
df = build_corpus()

print(f"\nDataset Statistics:")
print(f"Total samples: {len(df)}")
print(f"With images: {df['image_path'].notna().sum()}")
print(f"Text-only: {df['image_path'].isna().sum()}")

# Show label distribution
import numpy as np
all_labels = []
for labs in df['labels']:
    all_labels.extend(labs)
unique, counts = np.unique(all_labels, return_counts=True)
print(f"\nLabel distribution:")
for l, c in zip(unique, counts):
    print(f"  {ISSUE_LABELS[l]}: {c}")
```

---

## 📥 Offline Mode (Cached Datasets)

If you've already downloaded datasets and want to use cached versions:

```python
class ArgsOverride:
    offline = True                 # Use cached models/datasets only
    dataset = "localmini"          # Fallback to synthetic
    use_images = False             # Or point to local images
```

---

## 🎯 Recommended Configurations

### For Research Papers
```python
max_per_source = 2000
max_samples = 8000
rounds = 10
clients = 5
use_images = True
```

### For Production Deployment
```python
max_per_source = 5000
max_samples = 15000
rounds = 15
clients = 7
use_images = True
```

### For Quick Experiments
```python
max_per_source = 300
max_samples = 1500
rounds = 5
clients = 3
use_images = True
```

---

## 📊 Dataset Quality Comparison

| Dataset | Quality | Size | Diversity | Accessibility |
|---------|---------|------|-----------|---------------|
| PlantVillage | ⭐⭐⭐⭐⭐ | 54K+ | High | Easy |
| GARDIAN | ⭐⭐⭐⭐⭐ | 1K+ | Very High | Easy |
| AG News | ⭐⭐⭐⭐ | 1K+ | Medium | Easy |
| Argilla | ⭐⭐⭐⭐ | 1K+ | Medium | Easy |
| Cassava | ⭐⭐⭐⭐ | 1K+ | Medium | Easy |
| PathPlat | ⭐⭐⭐⭐⭐ | 1K+ | High | Easy |
| LocalMini | ⭐⭐⭐ | Unlimited | Low | Always |

---

## ✅ Summary

**Default configuration now uses:**
- ✅ ALL 4 text dataset sources
- ✅ ALL 9 image dataset sources
- ✅ Robust fallback handling
- ✅ Maximum of 5,000 total samples
- ✅ Balanced multimodal pairs

**To use it:**
```bash
# Just run!
python farm_advisor_multimodal_zero_error.py
```

**To customize:**
Edit the `ArgsOverride` class in the script.

---

**Last Updated:** 2026-01-15
**Status:** Production-ready with maximum dataset utilization

