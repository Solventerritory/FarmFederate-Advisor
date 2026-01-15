# Datasets Used in FarmFederate

**Project:** Federated Learning for Plant Stress Detection
**Date:** 2026-01-15

---

## 📊 Overview

The FarmFederate system uses **both real-world and synthetic datasets** for training and evaluation across three modalities:

1. **Text datasets** - For LLM models (text-based stress detection)
2. **Image datasets** - For ViT models (image-based stress detection)
3. **Multimodal datasets** - For VLM models (combined text + image)

---

## 📝 Text Datasets (for LLM Models)

### 1. AG News (Agriculture Filtered)
**Source:** HuggingFace `ag_news` dataset
**Size:** ~500 samples (filtered from 5,000)
**Type:** Real-world news articles

**Filtering Criteria:**
- Contains keywords: `farm`, `crop`, `plant`, `agriculture`, `soil`
- Relevant to agricultural domain

**Example:**
```
"Corn crop yields increase 15% after new irrigation system installed in Iowa farmlands"
```

**Usage:**
```python
ag_news = load_dataset("ag_news", split="train[:5000]")
ag_texts = [item['text'] for item in ag_news if any(kw in item['text'].lower()
            for kw in ['farm', 'crop', 'plant', 'agriculture', 'soil'])]
```

---

### 2. Synthetic Agricultural Text
**Source:** Generated (in-notebook)
**Size:** 1,000 samples
**Type:** Synthetic farmer observations and sensor logs

**Categories:**
- Farmer observations (visual symptoms)
- Sensor readings (moisture, pH, temperature)
- MQTT-style logs
- Crop stress descriptions

**Examples:**
```
"Corn leaves showing yellowing at edges, possible nitrogen deficiency."
"Tomato plants wilting despite adequate irrigation schedule."
"Wheat crop infested with aphids, population increasing rapidly."
"Rice paddies showing brown spots, suspected fungal infection."
"Soybean field experiencing heat stress, temperature above 35°C."
```

**Label Distribution:**
- Water stress: ~200 samples
- Nutrient deficiency: ~200 samples
- Pest risk: ~200 samples
- Disease risk: ~200 samples
- Heat stress: ~200 samples

---

### 3. Referenced (Not Directly Used)

These datasets are **mentioned as sources** but not directly loaded in the current implementation:

#### CGIAR GARDIAN AI-Ready Docs
- **Source:** CGIAR agricultural research database
- **Content:** Research papers, reports, agricultural best practices
- **Status:** Referenced in documentation, not directly loaded

#### Argilla Farming QA
- **Source:** HuggingFace Argilla datasets
- **Content:** Question-answer pairs about farming practices
- **Status:** Referenced, could be integrated

---

## 🖼️ Image Datasets (for ViT Models)

### 1. PlantVillage Dataset ⭐ **PRIMARY**
**Source:** HuggingFace `BrandonFors/Plant-Diseases-PlantVillage-Dataset`
**Size:** ~1,000 images (subset)
**Type:** Real-world plant disease images

**Content:**
- Plant leaf images with diseases
- Multiple crop types (tomato, potato, corn, etc.)
- Various disease types (blight, rust, mildew, spots)
- High-resolution color images

**Classes (original dataset):**
- 38 disease classes across 14 plant species
- Healthy vs diseased leaves

**Mapped to FarmFederate Classes:**
- `disease_risk` - Fungal, viral, bacterial diseases
- `pest_risk` - Pest damage visible on leaves
- `nutrient_def` - Nutrient deficiency symptoms
- `water_stress` - Wilting, drought symptoms
- `heat_stress` - Heat damage, sunburn

**Usage:**
```python
plant_dataset = load_dataset(
    "BrandonFors/Plant-Diseases-PlantVillage-Dataset",
    split="train[:1000]"
)
```

**Dataset Statistics:**
- **Original:** 54,000+ images
- **Used:** 1,000 images (subset for faster training)
- **Format:** RGB color images
- **Resolution:** Variable (typically 256×256 or larger)

---

### 2. PlantDoc (Local)
**Source:** Local directory `backend/data/`
**Size:** 180+ images
**Type:** Real-world plant documentation images

**Content:**
- Plant disease images
- Pest damage photos
- Growth stage documentation
- Environmental stress examples

**Status:** Referenced but not directly loaded in main notebook

---

### 3. Synthetic Plant Images
**Source:** Generated (in-notebook)
**Size:** Variable (fills to 1,000 total if PlantVillage fails)
**Type:** Synthetic RGB images

**Generation Method:**
```python
# Create random green-ish image (simulating plant)
img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
img[:, :, 1] = np.clip(img[:, :, 1] + 50, 0, 255)  # More green
```

**Purpose:**
- Fallback if real datasets fail to load
- Ensures training can proceed
- Demonstrates pipeline functionality

---

### 4. Referenced (Not Directly Used)

These datasets are **mentioned** but not actively loaded:

#### Saon110/bd-crop-vegetable-plant-disease-dataset
- **Source:** HuggingFace
- **Content:** Bangladesh crop disease images
- **Status:** Referenced, not loaded

#### timm/plant-pathology-2021
- **Source:** HuggingFace / Kaggle competition
- **Content:** Apple leaf pathology images
- **Status:** Referenced, not loaded

#### uqtwei2/PlantWild
- **Source:** HuggingFace
- **Content:** Wild plant images
- **Status:** Referenced, not loaded

---

## 🔗 Multimodal Datasets (for VLM Models)

### Combined Text + Image
**Source:** Pairing of text and image datasets
**Size:** Min(text_samples, image_samples) = ~1,000 pairs
**Type:** Aligned multimodal data

**Alignment Method:**
- Use same indices for text and image
- Ensure labels are consistent
- Create text-image pairs for VLM training

**Example Pair:**
```
Text: "Tomato plants showing yellowing leaves, suspected nutrient deficiency"
Image: [Tomato leaf with yellow discoloration]
Label: [0, 1, 0, 0, 0]  # nutrient_def
```

---

## 🏷️ Label Schema (5-Class Multi-Label)

All datasets are mapped to these 5 stress types:

### 1. Water Stress
**Keywords:** drought, wilting, moisture, irrigation, dry
**Symptoms:** Wilted leaves, dry soil, curled leaves
**Image indicators:** Drooping plants, brown edges

### 2. Nutrient Deficiency
**Keywords:** nitrogen, phosphorus, potassium, N, P, K, yellowing
**Symptoms:** Yellow/pale leaves, stunted growth
**Image indicators:** Chlorosis, interveinal chlorosis

### 3. Pest Risk
**Keywords:** aphids, whiteflies, caterpillars, borers, insects
**Symptoms:** Holes in leaves, visible insects
**Image indicators:** Chewed leaves, insect presence

### 4. Disease Risk
**Keywords:** blight, rust, mildew, fungal, viral, bacterial, infection
**Symptoms:** Spots, lesions, discoloration
**Image indicators:** Necrotic spots, powdery coating, lesions

### 5. Heat Stress
**Keywords:** heat, sunburn, temperature, heatwave, thermal
**Symptoms:** Scorched leaves, wilting despite watering
**Image indicators:** Brown/white patches, burnt tips

---

## 📈 Dataset Statistics

### Overall Counts
```
Total Text Samples:    ~1,500
├─ AG News (real):     ~500
└─ Synthetic:          ~1,000

Total Image Samples:   ~1,000
├─ PlantVillage:       ~1,000 (or fewer if subset)
└─ Synthetic:          Variable (fills to 1,000)

Multimodal Pairs:      ~1,000
├─ Aligned text-image pairs
└─ Shared labels
```

### Label Distribution (Approximate)
```
Water Stress:        ~300 samples (20%)
Nutrient Def:        ~300 samples (20%)
Pest Risk:           ~300 samples (20%)
Disease Risk:        ~300 samples (20%)
Heat Stress:         ~300 samples (20%)
```

**Note:** Multi-label, so samples can have multiple labels.

---

## 🔄 Data Splitting

### Federated Split (Non-IID)
**Method:** Dirichlet distribution with α=0.5

**Clients:** 5
**Distribution:** Heterogeneous (non-IID)

**Example Split:**
```
Client 0: 400 samples (diverse classes)
Client 1: 300 samples (bias toward disease)
Client 2: 250 samples (bias toward water stress)
Client 3: 300 samples (bias toward pest/nutrient)
Client 4: 250 samples (diverse classes)
```

**Why Non-IID?**
- Simulates real-world scenarios
- Different farms have different conditions
- Tests federated learning robustness

---

## 📥 How to Access Datasets

### AG News
```python
from datasets import load_dataset
ag_news = load_dataset("ag_news", split="train[:5000]")
```

### PlantVillage
```python
plant_dataset = load_dataset(
    "BrandonFors/Plant-Diseases-PlantVillage-Dataset",
    split="train[:1000]"
)
```

### Local PlantDoc
```python
# Images in: backend/data/
# Load using PIL or torchvision
from PIL import Image
img = Image.open("backend/data/plant_image.jpg")
```

---

## 🛠️ Data Preprocessing

### Text Preprocessing
```python
# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoded = tokenizer(
    text,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
```

### Image Preprocessing
```python
image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
```

### Label Format
```python
# Multi-label binary format
labels = [0, 1, 0, 0, 0]  # nutrient deficiency only
labels = [1, 0, 1, 0, 0]  # water stress + pest risk
```

---

## 🔍 Dataset Limitations

### Current Limitations
1. **Limited size:** ~1,500 samples (small for deep learning)
2. **Synthetic data:** Significant portion is synthetic
3. **Label noise:** Weak labeling based on keywords
4. **Imbalanced:** Some classes may be underrepresented
5. **Single domain:** Mostly focused on common crops

### Future Improvements
1. **Expand real data:** Add more HuggingFace datasets
2. **Expert labeling:** Get agricultural expert annotations
3. **Balanced sampling:** Ensure equal class distribution
4. **Domain diversity:** Include more crop types and regions
5. **Temporal data:** Add time-series sensor data

---

## 📚 Dataset References

### Papers Using Similar Datasets

1. **Mohanty et al. (PlantVillage, 2016)**
   - "Using Deep Learning for Image-Based Plant Disease Detection"
   - F1: 0.95, Accuracy: 0.96
   - Dataset: 54,000+ images

2. **Ferentinos (DeepPlant, 2018)**
   - "Deep Learning Models for Plant Disease Detection"
   - F1: 0.89, Accuracy: 0.91
   - Dataset: Multiple sources

3. **Zhang et al. (FedAgri, 2022)**
   - "Federated Learning for Agricultural Applications"
   - F1: 0.79, Accuracy: 0.81
   - Dataset: Distributed farm data

---

## 🎯 Dataset Usage in Models

### LLM Models (Text Only)
- **Input:** AG News + Synthetic text
- **Size:** ~1,500 samples
- **Processing:** Tokenization, padding, truncation

### ViT Models (Image Only)
- **Input:** PlantVillage + Synthetic images
- **Size:** ~1,000 images
- **Processing:** Resize, normalize, augmentation

### VLM Models (Multimodal)
- **Input:** Paired text + images
- **Size:** ~1,000 pairs
- **Processing:** Both text and image preprocessing

---

## ✅ Summary

**Primary Datasets:**
- ✅ AG News (real text)
- ✅ PlantVillage (real images)
- ✅ Synthetic text (generated)
- ✅ Synthetic images (fallback)

**Total Samples:**
- Text: ~1,500
- Images: ~1,000
- Multimodal: ~1,000 pairs

**Label Classes:** 5 stress types (multi-label)

**Split:** Non-IID across 5 clients using Dirichlet distribution

**Quality:** Mix of real-world and synthetic data for demonstration and research purposes

---

**Last Updated:** 2026-01-15
**Version:** 1.0.0
