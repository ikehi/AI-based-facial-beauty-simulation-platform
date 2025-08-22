"""
Enhanced Flask API for AI Beauty Platform
Uses enhanced AI modules for better performance and effects
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
import base64
import logging
import os
import sys
from PIL import Image
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import enhanced AI modules with error handling
try:
    from face_recognition.enhanced_face_detector import EnhancedFaceDetector
    from makeup_ai.enhanced_makeup_transfer import EnhancedMakeupTransfer
    from hair_ai.enhanced_hair_styler import EnhancedHairStyler
    print("✅ All enhanced AI modules imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Some enhanced AI modules failed to import: {e}")
    # Fallback to basic modules
    try:
        from face_recognition.face_detector import FaceDetector
        from makeup_ai.makeup_transfer import MakeupTransferGAN
        from hair_ai.hair_stylegan import HairStyleGAN
        print("✅ Using basic AI modules as fallback")
    except ImportError as e2:
        print(f"❌ Critical: No AI modules available: {e2}")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize AI models with error handling
try:
    # Try enhanced models first
    if 'EnhancedFaceDetector' in globals():
        face_detector = EnhancedFaceDetector(confidence_threshold=0.5)
        logger.info("✅ Enhanced Face Detector initialized")
    else:
        face_detector = FaceDetector()
        logger.info("✅ Basic Face Detector initialized")
    
    if 'EnhancedMakeupTransfer' in globals():
        makeup_gan = EnhancedMakeupTransfer()
        logger.info("✅ Enhanced Makeup Transfer initialized")
    else:
        makeup_gan = MakeupTransferGAN()
        logger.info("✅ Basic Makeup Transfer initialized")
    
    if 'EnhancedHairStyler' in globals():
        hair_gan = EnhancedHairStyler()
        logger.info("✅ Enhanced Hair Styler initialized")
    else:
        hair_gan = HairStyleGAN()
        logger.info("✅ Basic Hair Styler initialized")
        
except Exception as e:
    logger.error(f"❌ Error initializing AI models: {e}")
    sys.exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    try:
        # Get model statistics
        face_stats = face_detector.get_detection_stats() if hasattr(face_detector, 'get_detection_stats') else {}
        makeup_stats = makeup_gan.get_makeup_stats() if hasattr(makeup_gan, 'get_makeup_stats') else {}
        hair_stats = hair_gan.get_hair_stats() if hasattr(hair_gan, 'get_hair_stats') else {}
        
        return jsonify({
            'status': 'healthy',
            'message': 'AI Beauty Platform API is running',
            'timestamp': time.time(),
            'models': {
                'face_detection': {
                    'type': 'Enhanced' if 'EnhancedFaceDetector' in globals() else 'Basic',
                    'stats': face_stats
                },
                'makeup_transfer': {
                    'type': 'Enhanced' if 'EnhancedMakeupTransfer' in globals() else 'Basic',
                    'stats': makeup_stats
                },
                'hair_styling': {
                    'type': 'Enhanced' if 'EnhancedHairStyler' in globals() else 'Basic',
                    'stats': hair_stats
                }
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/face/detect', methods=['POST'])
def detect_faces():
    """
    Enhanced face detection endpoint
    
    Expected input:
    - image: base64 encoded image or multipart form data
    
    Returns:
    - faces: list of face bounding boxes with confidence scores
    - landmarks: facial landmarks for each face
    - quality_scores: quality assessment for each face
    """
    start_time = time.time()
    
    try:
        # Get image from request
        if 'image' in request.files:
            # Multipart form data
            file = request.files['image']
            image_array = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif 'image' in request.json:
            # Base64 encoded image
            image_data = request.json['image']
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Detect faces with enhanced detector
        faces = face_detector.detect_faces(image)
        
        # Extract additional information if available
        face_details = []
        for face_box in faces:
            face_info = {
                'bbox': face_box[:4],  # x, y, w, h
                'confidence': face_box[4] if len(face_box) > 4 else 1.0
            }
            
            # Get quality score if available
            if hasattr(face_detector, 'get_face_quality_score'):
                try:
                    # Crop face for quality assessment
                    cropped_face = face_detector.crop_face(image, face_box)
                    quality_score = face_detector.get_face_quality_score(cropped_face)
                    face_info['quality_score'] = quality_score
                except Exception as e:
                    logger.warning(f"Quality assessment failed: {e}")
                    face_info['quality_score'] = 0.5
            
            face_details.append(face_info)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            'faces': face_details,
            'num_faces': len(faces),
            'processing_time': processing_time,
            'image_dimensions': {
                'width': image.shape[1],
                'height': image.shape[0]
            }
        })
    
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/makeup/apply', methods=['POST'])
def apply_makeup():
    """
    Enhanced makeup application endpoint
    
    Expected input:
    - image: base64 encoded image
    - style: makeup style name (optional)
    - intensity: makeup intensity 0.0-1.0 (optional)
    
    Returns:
    - result_image: base64 encoded result image
    - style_info: detailed style information
    - processing_time: time taken for processing
    """
    start_time = time.time()
    
    try:
        data = request.json
        
        # Get parameters
        image_data = data['image']
        style = data.get('style', 'natural')
        intensity = float(data.get('intensity', 1.0))
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Detect face
        face_box = face_detector.get_largest_face(image)
        if face_box is None:
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Crop face
        face_image = face_detector.crop_face(image, face_box)
        
        # Apply makeup with enhanced system
        result_image = makeup_gan.apply_makeup_style(face_image, style, intensity)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get style information if available
        style_info = {}
        if hasattr(makeup_gan, 'get_style_info'):
            style_info = makeup_gan.get_style_info(style) or {}
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            'result_image': result_base64,
            'style': style,
            'intensity': intensity,
            'style_info': style_info,
            'processing_time': processing_time,
            'face_bbox': face_box[:4] if len(face_box) > 4 else face_box
        })
    
    except Exception as e:
        logger.error(f"Error in makeup application: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/hair/style', methods=['POST'])
def apply_hair_style():
    """
    Enhanced hair styling endpoint
    
    Expected input:
    - image: base64 encoded image
    - style: hair style name
    - color: hair color (optional)
    - intensity: transformation intensity 0.0-1.0 (optional)
    
    Returns:
    - result_image: base64 encoded result image
    - style_info: detailed style information
    - color_info: color palette information
    - processing_time: time taken for processing
    """
    start_time = time.time()
    
    try:
        data = request.json
        
        # Get parameters
        image_data = data['image']
        style = data['style']
        color = data.get('color', 'brown')
        intensity = float(data.get('intensity', 1.0))
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Apply hair style with enhanced system
        result_image = hair_gan.transform_hair_style(image, style, color, intensity)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get style and color information if available
        style_info = {}
        color_info = {}
        
        if hasattr(hair_gan, 'get_style_info'):
            style_info = hair_gan.get_style_info(style) or {}
        
        if hasattr(hair_gan, 'get_color_palette'):
            color_info = hair_gan.get_color_palette(color) or {}
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            'result_image': result_base64,
            'style': style,
            'color': color,
            'intensity': intensity,
            'style_info': style_info,
            'color_info': color_info,
            'processing_time': processing_time
        })
    
    except Exception as e:
        logger.error(f"Error in hair style application: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/beauty/full', methods=['POST'])
def apply_full_beauty():
    """
    Enhanced full beauty transformation endpoint
    
    Expected input:
    - image: base64 encoded image
    - makeup_style: makeup style name
    - makeup_intensity: makeup intensity 0.0-1.0
    - hair_style: hair style name
    - hair_color: hair color
    - hair_intensity: hair transformation intensity 0.0-1.0
    
    Returns:
    - result_image: base64 encoded result image
    - transformation_details: detailed transformation information
    - processing_time: time taken for processing
    """
    start_time = time.time()
    
    try:
        data = request.json
        
        # Get parameters
        image_data = data['image']
        makeup_style = data.get('makeup_style', 'natural')
        makeup_intensity = float(data.get('makeup_intensity', 1.0))
        hair_style = data.get('hair_style', 'straight')
        hair_color = data.get('hair_color', 'brown')
        hair_intensity = float(data.get('hair_intensity', 1.0))
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Detect face
        face_box = face_detector.get_largest_face(image)
        if face_box is None:
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Apply makeup first
        face_image = face_detector.crop_face(image, face_box)
        makeup_result = makeup_gan.apply_makeup_style(face_image, makeup_style, makeup_intensity)
        
        # Apply hair style to the full image
        result_image = hair_gan.transform_hair_style(image, hair_style, hair_color, hair_intensity)
        
        # Combine makeup and hair results
        # This is a simplified combination - in a real implementation, you'd want more sophisticated blending
        if face_box is not None:
            x, y, w, h = face_box[:4] if len(face_box) > 4 else face_box
            # Replace the face area with makeup result
            result_image[y:y+h, x:x+w] = makeup_result
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get transformation details
        transformation_details = {
            'makeup': {
                'style': makeup_style,
                'intensity': makeup_intensity,
                'style_info': makeup_gan.get_style_info(makeup_style) if hasattr(makeup_gan, 'get_style_info') else {}
            },
            'hair': {
                'style': hair_style,
                'color': hair_color,
                'intensity': hair_intensity,
                'style_info': hair_gan.get_style_info(hair_style) if hasattr(hair_gan, 'get_style_info') else {},
                'color_info': hair_gan.get_color_palette(hair_color) if hasattr(hair_gan, 'get_color_palette') else {}
            }
        }
        
        return jsonify({
            'result_image': result_base64,
            'transformation_details': transformation_details,
            'processing_time': processing_time,
            'face_bbox': face_box[:4] if len(face_box) > 4 else face_box
        })
    
    except Exception as e:
        logger.error(f"Error in full beauty transformation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/makeup/styles', methods=['GET'])
def get_makeup_styles():
    """Get available makeup styles with detailed information"""
    try:
        if hasattr(makeup_gan, 'get_available_styles'):
            styles = makeup_gan.get_available_styles()
            style_details = {}
            
            for style in styles:
                if hasattr(makeup_gan, 'get_style_info'):
                    style_details[style] = makeup_gan.get_style_info(style)
                else:
                    style_details[style] = {'name': style}
            
            return jsonify({
                'styles': styles,
                'style_details': style_details,
                'count': len(styles)
            })
        else:
            # Fallback for basic modules
            return jsonify({
                'styles': ['natural', 'glamorous', 'casual', 'evening', 'party'],
                'count': 5
            })
    except Exception as e:
        logger.error(f"Error getting makeup styles: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/hair/styles', methods=['GET'])
def get_hair_styles():
    """Get available hair styles and colors with detailed information"""
    try:
        if hasattr(hair_gan, 'get_available_styles'):
            styles = hair_gan.get_available_styles()
            colors = hair_gan.get_available_colors() if hasattr(hair_gan, 'get_available_colors') else []
            
            style_details = {}
            color_details = {}
            
            for style in styles:
                if hasattr(hair_gan, 'get_style_info'):
                    style_details[style] = hair_gan.get_style_info(style)
                else:
                    style_details[style] = {'name': style}
            
            for color in colors:
                if hasattr(hair_gan, 'get_color_palette'):
                    color_details[color] = hair_gan.get_color_palette(color)
                else:
                    color_details[color] = {'name': color}
            
            return jsonify({
                'styles': styles,
                'colors': colors,
                'style_details': style_details,
                'color_details': color_details,
                'style_count': len(styles),
                'color_count': len(colors)
            })
        else:
            # Fallback for basic modules
            return jsonify({
                'styles': ['straight', 'wavy', 'curly', 'coily'],
                'colors': ['black', 'brown', 'blonde', 'red', 'gray', 'white'],
                'style_count': 4,
                'color_count': 6
            })
    except Exception as e:
        logger.error(f"Error getting hair styles: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/info', methods=['GET'])
def get_system_info():
    """Get comprehensive system information"""
    try:
        system_info = {
            'api_version': '2.0.0',
            'enhanced_modules': {
                'face_detection': 'EnhancedFaceDetector' in globals(),
                'makeup_transfer': 'EnhancedMakeupTransfer' in globals(),
                'hair_styling': 'EnhancedHairStyler' in globals()
            },
            'model_capabilities': {}
        }
        
        # Add model-specific capabilities
        if hasattr(face_detector, 'get_detection_stats'):
            system_info['model_capabilities']['face_detection'] = face_detector.get_detection_stats()
        
        if hasattr(makeup_gan, 'get_makeup_stats'):
            system_info['model_capabilities']['makeup_transfer'] = makeup_gan.get_makeup_stats()
        
        if hasattr(hair_gan, 'get_hair_stats'):
            system_info['model_capabilities']['hair_styling'] = hair_gan.get_hair_stats()
        
        return jsonify(system_info)
    
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 Enhanced AI Beauty Platform API Starting...")
    print(f"✅ Face Detection: {'Enhanced' if 'EnhancedFaceDetector' in globals() else 'Basic'}")
    print(f"✅ Makeup Transfer: {'Enhanced' if 'EnhancedMakeupTransfer' in globals() else 'Basic'}")
    print(f"✅ Hair Styling: {'Enhanced' if 'EnhancedHairStyler' in globals() else 'Basic'}")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
