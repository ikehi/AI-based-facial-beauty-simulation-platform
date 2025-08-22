"""
Configuration file for AI Beauty Platform
"""

import os
from typing import Dict, Any

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'static/uploads'
    RESULTS_FOLDER = 'static/results'
    
    # AI Model settings
    AI_MODELS_PATH = 'models'
    DEVICE = os.environ.get('DEVICE', 'cpu')  # 'cpu' or 'cuda'
    
    # Face Detection settings
    FACE_DETECTION_CONFIDENCE = 0.5
    FACE_DETECTION_MIN_SIZE = (30, 30)
    
    # Makeup settings
    DEFAULT_MAKEUP_STYLE = 'natural'
    DEFAULT_MAKEUP_INTENSITY = 0.7
    
    # Hair settings
    DEFAULT_HAIR_STYLE = 'straight'
    DEFAULT_HAIR_COLOR = 'brown'
    DEFAULT_HAIR_INTENSITY = 0.8
    
    # Image processing settings
    DEFAULT_IMAGE_SIZE = (256, 256)
    IMAGE_QUALITY = 95  # JPEG quality
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 30  # seconds
    
    # Security settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    @classmethod
    def get_model_paths(cls) -> Dict[str, str]:
        """Get paths to AI models"""
        return {
            'face_detection': {
                'haar_cascade': os.path.join(cls.AI_MODELS_PATH, 'face_recognition', 'haar_cascade_frontalface.xml'),
                'dlib_landmarks': os.path.join(cls.AI_MODELS_PATH, 'face_recognition', 'shape_predictor_68_face_landmarks.dat'),
                'mmod_detector': os.path.join(cls.AI_MODELS_PATH, 'face_recognition', 'mmod_human_face_detector.dat')
            },
            'makeup': {
                'beautygan': os.path.join(cls.AI_MODELS_PATH, 'makeup_ai', 'beautygan_model.pth'),
                'advanced': os.path.join(cls.AI_MODELS_PATH, 'makeup_ai', 'advanced_makeup_model.pth')
            },
            'hair': {
                'stylegan2': os.path.join(cls.AI_MODELS_PATH, 'hair_ai', 'stylegan2_model.pth'),
                'advanced': os.path.join(cls.AI_MODELS_PATH, 'hair_ai', 'advanced_hair_model.pth')
            },
            'cosmetic': {
                'config': os.path.join(cls.AI_MODELS_PATH, 'cosmetic_ai', 'config.json')
            }
        }
    
    @classmethod
    def get_available_styles(cls) -> Dict[str, list]:
        """Get available styles for different features"""
        return {
            'makeup': ['natural', 'glamorous', 'casual', 'evening', 'party'],
            'hair_style': ['straight', 'wavy', 'curly', 'coily', 'updo'],
            'hair_color': ['black', 'brown', 'blonde', 'red', 'gray', 'white']
        }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: str = None) -> Config:
    """Get configuration class by name"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    return config.get(config_name, config['default'])
