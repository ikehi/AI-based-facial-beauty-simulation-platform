# AI-Based Facial Beauty Simulation Platform

An advanced AI platform for facial beauty simulation including makeup application, hair styling, and cosmetic adjustments using state-of-the-art deep learning models.

## Features

- **Face Recognition**: Advanced face detection and landmark extraction
- **Makeup AI**: GAN-based makeup style transfer and application
- **Hair Style AI**: AI-powered hair style generation and transformation
- **Cosmetic Simulation**: Facial feature adjustment and enhancement
- **API Integration**: RESTful API for web/mobile integration

## Project Structure

```
ai_beauty_platform/
├── models/                 # Pre-trained AI models
├── src/
│   ├── face_recognition/   # Face detection and landmark extraction
│   ├── makeup_ai/         # Makeup application models
│   ├── hair_ai/           # Hair style generation models
│   ├── cosmetic_ai/       # Facial feature adjustment
│   ├── utils/             # Utility functions
│   └── api/               # Flask API endpoints
├── data/                  # Training and test data
├── notebooks/             # Jupyter notebooks for development
└── tests/                 # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-based-facial-beauty-simulation-platform
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models:
```bash
python scripts/download_models.py
```

## Quick Start

1. Start the API server:
```bash
python src/api/app.py
```

2. Test face recognition:
```bash
python src/face_recognition/test_detection.py
```

3. Test makeup application:
```bash
python src/makeup_ai/test_makeup.py
```

## API Endpoints

- `POST /api/face/detect` - Face detection and landmark extraction
- `POST /api/makeup/apply` - Apply makeup styles
- `POST /api/hair/style` - Generate hair styles
- `POST /api/cosmetic/adjust` - Adjust facial features

## Technologies Used

- **AI Framework**: TensorFlow, PyTorch
- **Face Recognition**: OpenCV, Dlib, FaceNet
- **Makeup AI**: BeautyGAN, Makeup Transfer GAN
- **Hair AI**: StyleGAN2, Hair-GAN
- **Backend**: Flask, Python

## License

MIT License 