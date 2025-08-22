"""
Real AI Beauty Platform API
Integrates advanced AI models for face detection, makeup transfer, and hair transformation
"""

import os
import sys
import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import json
import traceback

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io

# Import real AI modules
try:
    from face_recognition.robust_face_detector import RobustFaceDetector, FaceDetection
    from makeup_ai.real_makeup_transfer import RealMakeupTransfer
    from hair_ai.real_hair_transformer import RealHairTransformer
    REAL_AI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Real AI modules not available: {e}")
    REAL_AI_AVAILABLE = False

# Import fallback modules
try:
    from face_recognition.enhanced_face_detector import EnhancedFaceDetector
    from makeup_ai.enhanced_makeup_transfer import EnhancedMakeupTransfer
    from hair_ai.enhanced_hair_styler import EnhancedHairStyler
    FALLBACK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Fallback modules not available: {e}")
    FALLBACK_AVAILABLE = False

# Import utility modules
try:
    from utils.logger import setup_logger
    from utils.error_handler import AIBeautyError, handle_api_errors
    from config import Config
except ImportError as e:
    logging.warning(f"Utility modules not available: {e}")
    # Create basic fallbacks
    setup_logger = lambda: logging.basicConfig(level=logging.INFO)
    AIBeautyError = Exception
    handle_api_errors = lambda f: f
    Config = type('Config', (), {})

# Setup logging
logger = setup_logger()

class RealAIBeautyAPI:
    """
    Real AI Beauty Platform API with production features
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Real AI Beauty API"""
        self.config = config or Config()
        self.logger = logger
        
        # Initialize AI modules
        self.face_detector = None
        self.makeup_transfer = None
        self.hair_transformer = None
        
        self._init_ai_modules()
        
        # Performance tracking
        self.api_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'response_times': [],
            'endpoint_usage': {}
        }
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        self._setup_routes()
        
        self.logger.info("Real AI Beauty API initialized successfully")
    
    def _init_ai_modules(self):
        """Initialize AI modules with fallback support"""
        try:
            if REAL_AI_AVAILABLE:
                # Initialize real AI modules
                self.face_detector = RobustFaceDetector()
                self.makeup_transfer = RealMakeupTransfer()
                self.hair_transformer = RealHairTransformer()
                self.logger.info("Real AI modules initialized successfully")
            elif FALLBACK_AVAILABLE:
                # Initialize fallback modules
                self.face_detector = EnhancedFaceDetector()
                self.makeup_transfer = EnhancedMakeupTransfer()
                self.hair_transformer = EnhancedHairStyler()
                self.logger.info("Fallback modules initialized successfully")
            else:
                self.logger.error("No AI modules available")
                raise Exception("No AI modules available")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI modules: {e}")
            raise
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'ai_modules': {
                    'face_detection': self.face_detector is not None,
                    'makeup_transfer': self.makeup_transfer is not None,
                    'hair_transformation': self.hair_transformer is not None
                },
                'real_ai_available': REAL_AI_AVAILABLE
            })
        
        @self.app.route('/api/face/detect', methods=['POST'])
        @handle_api_errors
        def detect_faces():
            """Detect faces in an image"""
            start_time = time.time()
            
            # Get image from request
            if 'image' not in request.files:
                raise AIBeautyError("No image provided")
            
            image_file = request.files['image']
            image = self._load_image(image_file)
            
            # Detect faces
            detections = self.face_detector.detect_faces(image)
            
            # Convert detections to serializable format
            detection_data = []
            for detection in detections:
                detection_data.append({
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'method': detection.method,
                    'quality_score': detection.quality_score,
                    'landmarks': detection.landmarks
                })
            
            # Update statistics
            self._update_stats('detect_faces', time.time() - start_time, True)
            
            return jsonify({
                'success': True,
                'faces_detected': len(detections),
                'detections': detection_data,
                'processing_time': time.time() - start_time
            })
        
        @self.app.route('/api/makeup/apply', methods=['POST'])
        @handle_api_errors
        def apply_makeup():
            """Apply makeup style to an image"""
            start_time = time.time()
            
            # Get parameters from request
            if 'image' not in request.files:
                raise AIBeautyError("No image provided")
            
            image_file = request.files['image']
            style_name = request.form.get('style', 'natural')
            intensity = float(request.form.get('intensity', 0.5))
            
            # Load and process image
            image = self._load_image(image_file)
            
            # Apply makeup
            result_image = self.makeup_transfer.apply_makeup_style(image, style_name, intensity)
            
            # Convert result to bytes
            result_bytes = self._image_to_bytes(result_image)
            
            # Update statistics
            self._update_stats('apply_makeup', time.time() - start_time, True)
            
            # Return image
            return send_file(
                io.BytesIO(result_bytes),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name='makeup_result.jpg'
            )
        
        @self.app.route('/api/hair/style', methods=['POST'])
        @handle_api_errors
        def style_hair():
            """Apply hair styling to an image"""
            start_time = time.time()
            
            # Get parameters from request
            if 'image' not in request.files:
                raise AIBeautyError("No image provided")
            
            image_file = request.files['image']
            style_name = request.form.get('style', 'straight')
            intensity = float(request.form.get('intensity', 0.5))
            
            # Load and process image
            image = self._load_image(image_file)
            
            # Apply hair styling
            result_image = self.hair_transformer.transform_hair_style(image, style_name, intensity)
            
            # Convert result to bytes
            result_bytes = self._image_to_bytes(result_image)
            
            # Update statistics
            self._update_stats('style_hair', time.time() - start_time, True)
            
            # Return image
            return send_file(
                io.BytesIO(result_bytes),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name='hair_style_result.jpg'
            )
        
        @self.app.route('/api/hair/color', methods=['POST'])
        @handle_api_errors
        def change_hair_color():
            """Change hair color in an image"""
            start_time = time.time()
            
            # Get parameters from request
            if 'image' not in request.files:
                raise AIBeautyError("No image provided")
            
            image_file = request.files['image']
            color_name = request.form.get('color', 'brown')
            intensity = float(request.form.get('intensity', 0.5))
            
            # Load and process image
            image = self._load_image(image_file)
            
            # Change hair color
            result_image = self.hair_transformer.change_hair_color(image, color_name, intensity)
            
            # Convert result to bytes
            result_bytes = self._image_to_bytes(result_image)
            
            # Update statistics
            self._update_stats('change_hair_color', time.time() - start_time, True)
            
            # Return image
            return send_file(
                io.BytesIO(result_bytes),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name='hair_color_result.jpg'
            )
        
        @self.app.route('/api/beauty/full', methods=['POST'])
        @handle_api_errors
        def full_beauty_transformation():
            """Apply full beauty transformation (makeup + hair)"""
            start_time = time.time()
            
            # Get parameters from request
            if 'image' not in request.files:
                raise AIBeautyError("No image provided")
            
            image_file = request.files['image']
            makeup_style = request.form.get('makeup_style', 'natural')
            makeup_intensity = float(request.form.get('makeup_intensity', 0.5))
            hair_style = request.form.get('hair_style', 'straight')
            hair_intensity = float(request.form.get('hair_intensity', 0.5))
            hair_color = request.form.get('hair_color', 'brown')
            color_intensity = float(request.form.get('color_intensity', 0.5))
            
            # Load and process image
            image = self._load_image(image_file)
            
            # Apply makeup first
            image = self.makeup_transfer.apply_makeup_style(image, makeup_style, makeup_intensity)
            
            # Apply hair styling
            image = self.hair_transformer.transform_hair_style(image, hair_style, hair_intensity)
            
            # Apply hair color
            image = self.hair_transformer.change_hair_color(image, hair_color, color_intensity)
            
            # Convert result to bytes
            result_bytes = self._image_to_bytes(image)
            
            # Update statistics
            self._update_stats('full_beauty', time.time() - start_time, True)
            
            # Return image
            return send_file(
                io.BytesIO(result_bytes),
                mimetype='image/jpeg',
                as_attachment=True,
                download_name='full_beauty_result.jpg'
            )
        
        @self.app.route('/api/makeup/styles', methods=['GET'])
        def get_makeup_styles():
            """Get available makeup styles"""
            try:
                styles = self.makeup_transfer.get_available_styles()
                style_info = {}
                
                for style_name in styles:
                    style = self.makeup_transfer.get_style_info(style_name)
                    if style:
                        style_info[style_name] = {
                            'name': style.name,
                            'description': style.description,
                            'intensity': style.intensity,
                            'effects': style.effects
                        }
                
                return jsonify({
                    'success': True,
                    'styles': style_info
                })
            except Exception as e:
                self.logger.error(f"Failed to get makeup styles: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/hair/styles', methods=['GET'])
        def get_hair_styles():
            """Get available hair styles"""
            try:
                styles = self.hair_transformer.get_available_styles()
                colors = self.hair_transformer.get_available_colors()
                
                style_info = {}
                for style_name in styles:
                    style = self.hair_transformer.get_style_info(style_name)
                    if style:
                        style_info[style_name] = {
                            'name': style.name,
                            'description': style.description,
                            'complexity': style.complexity,
                            'effects': style.effects
                        }
                
                color_info = {}
                for color_name in colors:
                    color = self.hair_transformer.get_color_info(color_name)
                    if color:
                        color_info[color_name] = {
                            'name': color.name,
                            'rgb_value': color.rgb_value,
                            'intensity': color.intensity,
                            'highlights': color.highlights,
                            'shadows': color.shadows
                        }
                
                return jsonify({
                    'success': True,
                    'styles': style_info,
                    'colors': color_info
                })
            except Exception as e:
                self.logger.error(f"Failed to get hair styles: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/system/info', methods=['GET'])
        def get_system_info():
            """Get system information and statistics"""
            try:
                # Get AI module statistics
                face_stats = self.face_detector.get_detection_stats() if self.face_detector else {}
                makeup_stats = self.makeup_transfer.get_processing_stats() if self.makeup_transfer else {}
                hair_stats = self.hair_transformer.get_processing_stats() if self.hair_transformer else {}
                
                return jsonify({
                    'success': True,
                    'system_info': {
                        'real_ai_available': REAL_AI_AVAILABLE,
                        'fallback_available': FALLBACK_AVAILABLE,
                        'api_stats': self.api_stats,
                        'face_detection_stats': face_stats,
                        'makeup_transfer_stats': makeup_stats,
                        'hair_transformation_stats': hair_stats
                    }
                })
            except Exception as e:
                self.logger.error(f"Failed to get system info: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/system/stats', methods=['GET'])
        def get_api_stats():
            """Get API performance statistics"""
            return jsonify({
                'success': True,
                'stats': self.api_stats
            })
    
    def _load_image(self, image_file) -> np.ndarray:
        """Load image from file and convert to numpy array"""
        try:
            # Read image file
            image_bytes = image_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise AIBeautyError("Invalid image format")
            
            return image
        except Exception as e:
            raise AIBeautyError(f"Failed to load image: {e}")
    
    def _image_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy image array to bytes"""
        try:
            # Encode image to JPEG
            success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if not success:
                raise AIBeautyError("Failed to encode image")
            
            return buffer.tobytes()
        except Exception as e:
            raise AIBeautyError(f"Failed to convert image to bytes: {e}")
    
    def _update_stats(self, endpoint: str, response_time: float, success: bool):
        """Update API statistics"""
        self.api_stats['total_requests'] += 1
        self.api_stats['response_times'].append(response_time)
        
        if success:
            self.api_stats['successful_requests'] += 1
        else:
            self.api_stats['failed_requests'] += 1
        
        # Update endpoint usage
        self.api_stats['endpoint_usage'][endpoint] = self.api_stats['endpoint_usage'].get(endpoint, 0) + 1
        
        # Calculate average response time
        self.api_stats['average_response_time'] = sum(self.api_stats['response_times']) / len(self.api_stats['response_times'])
        
        # Keep only last 1000 response times to prevent memory issues
        if len(self.api_stats['response_times']) > 1000:
            self.api_stats['response_times'] = self.api_stats['response_times'][-1000:]
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the Flask application"""
        self.logger.info(f"Starting Real AI Beauty API on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def create_app(config: Optional[Config] = None):
    """Factory function to create the Flask app"""
    api = RealAIBeautyAPI(config)
    return api.app

if __name__ == '__main__':
    # Create and run the API
    api = RealAIBeautyAPI()
    api.run(debug=True)
