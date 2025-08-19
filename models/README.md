
# AI Beauty Platform - Model Information

## Models Included:

### 1. Face Recognition Models
- **haar_cascade_frontalface.xml**: OpenCV Haar Cascade for face detection
- **face_landmarks_68.dat**: 68-point facial landmark detector  
- **face_recognition_model.pkl**: Face recognition and analysis

### 2. Makeup AI Models
- **beauty_gan.pth**: BeautyGAN model for makeup style transfer
- **makeup_transfer.pth**: Makeup application and transfer
- **config.json**: Model configuration and parameters

### 3. Hair AI Models
- **stylegan2_hair.pth**: StyleGAN2 model for hair generation
- **hair_transformer.pth**: Hair style and color transformation
- **config.json**: Model configuration and parameters

### 4. Cosmetic AI Models
- **feature_adjuster.pth**: Facial feature adjustment (eyes, nose, lips)
- **beauty_enhancer.pth**: Overall beauty enhancement
- **config.json**: Model configuration and parameters

## What Each Model Does:

### Face Recognition
- Detects faces in images
- Extracts 68 facial landmarks (eyes, nose, mouth, jawline)
- Analyzes facial features and proportions

### Makeup AI
- Applies virtual makeup styles
- Transfers makeup from reference images
- Adjusts makeup intensity and colors

### Hair AI
- Generates different hair styles
- Changes hair color realistically
- Transforms hair texture and volume

### Cosmetic AI
- Adjusts facial feature sizes and shapes
- Enhances skin quality and texture
- Provides beauty recommendations

## Usage:
These models are used by the AI Beauty Platform to provide:
- Real-time face detection and analysis
- Virtual makeup application
- Hair style simulation
- Facial feature enhancement
