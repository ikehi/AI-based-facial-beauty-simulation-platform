# AI Model Download Instructions

## Required Models

This project requires several AI models that are too large to include in the GitHub repository. Please download them manually:

### 1. Dlib Face Detection Models

Download from: https://github.com/davisking/dlib-models

**Required files:**
- `shape_predictor_68_face_landmarks.dat` (95MB) - Place in `models/dlib/`
- `mmod_human_face_detector.dat.bz2` (520KB) - Place in `models/dlib/`

### 2. Advanced Makeup Model

**File:** `advanced_makeup_model.pth` (43MB)
**Location:** `models/makeup_ai/`

This is a pre-trained PyTorch model for advanced makeup effects.

### 3. Advanced Hair Model

**File:** `advanced_hair_model.pth` (466MB)
**Location:** `models/hair_ai/`

This is a pre-trained PyTorch model for advanced hair transformation effects.

## Quick Setup

1. Create the model directories:
   ```bash
   mkdir -p models/dlib models/makeup_ai models/hair_ai
   ```

2. Download the models and place them in the correct directories

3. Extract the dlib models if they are compressed:
   ```bash
   # For Windows
   tar -xf models/dlib/shape_predictor_68_face_landmarks.dat.bz2
   tar -xf models/dlib/mmod_human_face_detector.dat.bz2
   ```

4. Run the application:
   ```bash
   python ai_enhanced_ui.py
   ```

## Alternative: Use Download Script

You can also use the provided download script:

```bash
python download_pretrained_models.py
```

This script will attempt to download the models automatically, but you may need to manually download them if the URLs are not accessible.

## Model Sources

- **Dlib Models:** Official dlib repository
- **Makeup Model:** Custom trained model for beauty enhancement
- **Hair Model:** Custom trained model for hair transformation

## Note

The models are excluded from Git tracking due to their large size. This is standard practice for AI projects to keep repository sizes manageable. 