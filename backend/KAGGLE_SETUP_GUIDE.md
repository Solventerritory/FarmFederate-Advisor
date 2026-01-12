# Kaggle API Setup Guide

## Overview
To download real datasets from Kaggle, you need to set up the Kaggle API credentials.

## Setup Steps

### 1. Create Kaggle Account
- Go to https://www.kaggle.com
- Sign up for a free account

### 2. Get API Token
1. Log in to Kaggle
2. Go to your Account Settings: https://www.kaggle.com/settings
3. Scroll down to **API** section
4. Click **"Create New API Token"**
5. This downloads `kaggle.json` file

### 3. Place API Token

#### Windows:
```bash
# Create directory
mkdir %USERPROFILE%\.kaggle

# Copy kaggle.json to directory
copy kaggle.json %USERPROFILE%\.kaggle\
```

#### Linux/Mac:
```bash
# Create directory
mkdir -p ~/.kaggle

# Copy kaggle.json
cp kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Verify Installation
```bash
kaggle --version
```

### 5. Accept Dataset Terms
Before downloading, you must accept the dataset terms on Kaggle website:
- PlantVillage: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Plant Disease: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Click **"Download"** button to accept terms (you don't need to actually download manually)

## Datasets Available

### 1. PlantVillage Dataset
- **Source**: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **Size**: ~800 MB
- **Samples**: 54,303 images
- **Classes**: 38 plant disease classes
- **Crops**: Tomato, Potato, Pepper, Corn, Grape, etc.

### 2. New Plant Diseases Dataset
- **Source**: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- **Size**: ~2 GB
- **Samples**: 87,000+ images (includes augmented versions)
- **Classes**: 38 classes
- **Structure**: train/test/valid splits

## Running the Download Script

Once Kaggle API is configured:

```bash
cd backend
python download_real_datasets.py
```

The script will:
1. **Try downloading from Kaggle** (PlantVillage + Plant Disease datasets)
2. **Try downloading from HuggingFace** (agricultural text datasets)
3. **Fall back to synthetic data** if downloads fail
4. **Process and combine** all data sources
5. **Save in standardized format** for training

## Fallback Mode

If Kaggle API is not configured, the script automatically:
- ✓ Creates synthetic PlantVillage-style dataset (1000 samples)
- ✓ Generates crop stress text descriptions (2000 samples)
- ✓ Creates multimodal pairs (1000 samples)
- ✓ All stress categories properly labeled

## Troubleshooting

### Error: "401 Unauthorized"
- **Solution**: Re-download kaggle.json from Kaggle website
- Make sure file is in correct location: `~/.kaggle/kaggle.json`

### Error: "403 Forbidden"
- **Solution**: Accept dataset terms on Kaggle website
- Visit dataset page and click "Download" button

### Error: "Could not find kaggle.json"
- **Solution**: Place kaggle.json in:
  - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
  - Linux/Mac: `~/.kaggle/kaggle.json`

### Slow Downloads
- Kaggle datasets are large (800 MB - 2 GB)
- First run will take 10-30 minutes depending on internet speed
- Subsequent runs use cached data

## Manual Alternative

If automated download doesn't work:

1. **Download manually from Kaggle**:
   - Go to dataset page
   - Click "Download" button
   - Extract to `backend/data/real_datasets/`

2. **Expected structure**:
```
backend/data/real_datasets/
├── plantvillage_kaggle/
│   └── PlantVillage/
│       ├── Tomato___Early_blight/
│       ├── Potato___Late_blight/
│       └── ...
├── plant_disease_kaggle/
│   ├── train/
│   ├── test/
│   └── valid/
└── dataset_summary.json
```

3. **Run script** - it will detect existing files

## Success Indicators

After successful download:
```
✓ PlantVillage downloaded successfully!
✓ Processed 54,303 real images
✓ Plant Disease dataset downloaded!
✓ Processed 87,000 real images
✓ HuggingFace dataset downloaded!

Total samples: 143,303+ REAL images from internet
```

## Quick Test

Test if Kaggle API works:
```bash
kaggle datasets list
```

Should show list of Kaggle datasets.
