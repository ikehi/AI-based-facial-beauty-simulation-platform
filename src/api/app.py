"""
Flask API for AI Beauty Platform
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

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import AI modules with error handling
try:
    from face_recognition.face_detector import FaceDetector
    from face_recognition.landmark_extractor import LandmarkExtractor
    from makeup_ai.makeup_transfer import MakeupTransferGAN
    from hair_ai.hair_stylegan import HairStyleGAN
    print("✅ All AI modules imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Some AI modules failed to import: {e}")
    # Create fallback classes
    class FaceDetector:
        def __init__(self):
            print("⚠️ Using fallback FaceDetector")
        def detect_faces(self, image):
            # Basic OpenCV fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return [(x, y, w, h) for (x, y, w, h) in faces]
        def get_largest_face(self, image):
            faces = self.detect_faces(image)
            if not faces:
                return None
            return max(faces, key=lambda x: x[2] * x[3])
        def crop_face(self, image, face_box, padding=0.2):
            x, y, w, h = face_box
            height, width = image.shape[:2]
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + w + pad_x)
            y2 = min(height, y + h + pad_y)
            return image[y1:y2, x1:x2]
    
    class LandmarkExtractor:
        def __init__(self):
            print("⚠️ Using fallback LandmarkExtractor")
        def extract_landmarks_from_box(self, image, face_box):
            # Return estimated landmarks
            x, y, w, h = face_box
            landmarks = []
            for i in range(68):
                if i < 17:  # Jawline
                    angle = (i / 16) * np.pi
                    lx = x + w//2 + int(0.4 * w * np.cos(angle))
                    ly = y + h//2 + int(0.4 * h * np.sin(angle))
                elif i < 22:  # Right eyebrow
                    lx = x + w//2 + int(0.2 * w * (i - 17) / 4)
                    ly = y + h//2 - int(0.3 * h)
                elif i < 27:  # Left eyebrow
                    lx = x + w//2 - int(0.2 * w * (i - 22) / 4)
                    ly = y + h//2 - int(0.3 * h)
                elif i < 31:  # Nose bridge
                    lx = x + w//2
                    ly = y + h//2 - int(0.1 * h * (i - 27) / 3)
                elif i < 36:  # Nose tip
                    lx = x + w//2 + int(0.1 * w * np.sin((i - 31) * np.pi / 4))
                    ly = y + h//2 + int(0.1 * h * (i - 31) / 4)
                elif i < 42:  # Right eye
                    angle = (i - 36) * np.pi / 3
                    lx = x + w//2 + int(0.15 * w) + int(0.05 * w * np.cos(angle))
                    ly = y + h//2 - int(0.1 * h) + int(0.05 * h * np.sin(angle))
                elif i < 48:  # Left eye
                    angle = (i - 42) * np.pi / 3
                    lx = x + w//2 - int(0.15 * w) + int(0.05 * w * np.cos(angle))
                    ly = y + h//2 - int(0.1 * h) + int(0.05 * h * np.sin(angle))
                elif i < 60:  # Outer mouth
                    angle = (i - 48) * np.pi / 6
                    lx = x + w//2 + int(0.2 * w * np.cos(angle))
                    ly = y + h//2 + int(0.2 * h * np.sin(angle))
                else:  # Inner mouth
                    angle = (i - 60) * np.pi / 4
                    lx = x + w//2 + int(0.1 * w * np.cos(angle))
                    ly = y + h//2 + int(0.1 * h * np.sin(angle))
                landmarks.append([lx, ly])
            return np.array(landmarks)
    
    class MakeupTransferGAN:
        def __init__(self):
            print("⚠️ Using fallback MakeupTransferGAN")
        def apply_makeup_style(self, face_image, style, intensity):
            # Basic makeup effect
            result = face_image.copy()
            if style == 'natural':
                # Subtle enhancement
                result = cv2.addWeighted(result, 1.0, result, 0.1, 10)
            elif style == 'glamorous':
                # More dramatic effect
                result = cv2.addWeighted(result, 1.0, result, 0.2, 20)
            return result
        def get_available_styles(self):
            return ['natural', 'glamorous', 'casual', 'evening', 'party']
    
    class HairStyleGAN:
        def __init__(self):
            print("⚠️ Using fallback HairStyleGAN")
        def transform_hair_style(self, image, style, intensity):
            # Basic hair effect
            result = image.copy()
            if style == 'curly':
                # Add some texture
                kernel = np.ones((3,3), np.uint8)
                result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
            return result
        def get_available_styles(self):
            return ['straight', 'wavy', 'curly', 'coily']
        def get_available_colors(self):
            return ['black', 'brown', 'blonde', 'red', 'gray', 'white']

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize AI models with error handling
try:
    face_detector = FaceDetector()
    landmark_extractor = LandmarkExtractor()
    makeup_gan = MakeupTransferGAN()
    hair_gan = HairStyleGAN()
    logger.info("✅ All AI models initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing AI models: {e}")
    # Use fallback models
    face_detector = FaceDetector()
    landmark_extractor = LandmarkExtractor()
    makeup_gan = MakeupTransferGAN()
    hair_gan = HairStyleGAN()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'AI Beauty Platform API is running'
    })

@app.route('/api/face/detect', methods=['POST'])
def detect_faces():
    """
    Detect faces in uploaded image
    
    Expected input:
    - image: base64 encoded image or multipart form data
    
    Returns:
    - faces: list of face bounding boxes
    - landmarks: facial landmarks for each face
    """
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
        
        # Detect faces
        faces = face_detector.detect_faces(image)
        
        # Extract landmarks for each face
        landmarks_list = []
        for face_box in faces:
            landmarks = landmark_extractor.extract_landmarks_from_box(image, face_box)
            if landmarks is not None:
                landmarks_list.append(landmarks.tolist())
            else:
                landmarks_list.append(None)
        
        return jsonify({
            'faces': faces,
            'landmarks': landmarks_list,
            'num_faces': len(faces)
        })
    
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/makeup/apply', methods=['POST'])
def apply_makeup():
    """
    Apply makeup to face image
    
    Expected input:
    - image: base64 encoded image
    - style: makeup style name (optional)
    - intensity: makeup intensity 0.0-1.0 (optional)
    
    Returns:
    - result_image: base64 encoded result image
    """
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
        
        # Detect face
        face_box = face_detector.get_largest_face(image)
        if face_box is None:
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Crop face
        face_image = face_detector.crop_face(image, face_box)
        
        # Apply makeup
        result_image = makeup_gan.apply_makeup_style(face_image, style, intensity)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'result_image': result_base64,
            'style': style,
            'intensity': intensity
        })
    
    except Exception as e:
        logger.error(f"Error in makeup application: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/hair/style', methods=['POST'])
def apply_hair_style():
    """
    Apply hair style to face image
    
    Expected input:
    - image: base64 encoded image
    - style: hair style name
    - color: hair color (optional)
    - intensity: transformation intensity 0.0-1.0 (optional)
    
    Returns:
    - result_image: base64 encoded result image
    """
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
        
        # Apply hair style
        result_image = hair_gan.transform_hair_style(image, style, intensity)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'result_image': result_base64,
            'style': style,
            'color': color,
            'intensity': intensity
        })
    
    except Exception as e:
        logger.error(f"Error in hair style application: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/makeup/styles', methods=['GET'])
def get_makeup_styles():
    """Get available makeup styles"""
    try:
        styles = makeup_gan.get_available_styles()
        return jsonify({
            'styles': styles,
            'count': len(styles)
        })
    except Exception as e:
        logger.error(f"Error getting makeup styles: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/hair/styles', methods=['GET'])
def get_hair_styles():
    """Get available hair styles"""
    try:
        styles = hair_gan.get_available_styles()
        colors = hair_gan.get_available_colors()
        return jsonify({
            'styles': styles,
            'colors': colors,
            'style_count': len(styles),
            'color_count': len(colors)
        })
    except Exception as e:
        logger.error(f"Error getting hair styles: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/beauty/full', methods=['POST'])
def apply_full_beauty():
    """
    Apply full beauty transformation (makeup + hair)
    
    Expected input:
    - image: base64 encoded image
    - makeup_style: makeup style name
    - makeup_intensity: makeup intensity 0.0-1.0
    - hair_style: hair style name
    - hair_color: hair color
    - hair_intensity: hair transformation intensity 0.0-1.0
    
    Returns:
    - result_image: base64 encoded result image
    """
    try:
        data = request.json
        
        # Get parameters
        image_data = data['image']
        makeup_style = data.get('makeup_style', 'natural')
        makeup_intensity = float(data.get('makeup_intensity', 1.0))
        hair_style = data.get('hair_style', 'straight_long')
        hair_color = data.get('hair_color', 'brown')
        hair_intensity = float(data.get('hair_intensity', 1.0))
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Detect face
        face_box = face_detector.get_largest_face(image)
        if face_box is None:
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Apply makeup first
        face_image = face_detector.crop_face(image, face_box)
        makeup_result = makeup_gan.apply_makeup_style(face_image, makeup_style, makeup_intensity)
        
        # Apply hair style
        result_image = hair_gan.transform_hair_style(image, hair_style, hair_intensity)
        
        # Encode result
        _, buffer = cv2.imencode('.jpg', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'result_image': result_base64,
            'makeup_style': makeup_style,
            'makeup_intensity': makeup_intensity,
            'hair_style': hair_style,
            'hair_color': hair_color,
            'hair_intensity': hair_intensity
        })
    
    except Exception as e:
        logger.error(f"Error in full beauty transformation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('styles', exist_ok=True)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True) 