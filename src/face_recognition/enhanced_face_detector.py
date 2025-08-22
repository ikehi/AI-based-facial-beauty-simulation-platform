"""
Enhanced Face Detection Module
Combines multiple detection algorithms for better accuracy
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class EnhancedFaceDetector:
    """
    Enhanced face detector using multiple algorithms
    - OpenCV Haar Cascade (fast, good for frontal faces)
    - OpenCV DNN (deep learning, better accuracy)
    - Fallback geometric detection (basic but reliable)
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the enhanced face detector
        
        Args:
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        self.detection_methods = []
        
        # Initialize detection methods
        self._init_haar_cascade()
        self._init_dnn_detector()
        
        logger.info(f"Enhanced Face Detector initialized with {len(self.detection_methods)} methods")
    
    def _init_haar_cascade(self):
        """Initialize Haar Cascade detector"""
        try:
            # Try to use the model from our models directory first
            model_path = Path("models/face_recognition/haar_cascade_frontalface.xml")
            if model_path.exists():
                self.haar_cascade = cv2.CascadeClassifier(str(model_path))
            else:
                # Fallback to OpenCV's built-in model
                self.haar_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            
            if self.haar_cascade.empty():
                logger.warning("Haar Cascade model failed to load")
                return
            
            self.detection_methods.append(('haar_cascade', self._detect_haar_cascade))
            logger.info("✅ Haar Cascade detector initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Haar Cascade: {e}")
    
    def _init_dnn_detector(self):
        """Initialize DNN face detector"""
        try:
            # Try to use our custom DNN model
            model_path = Path("models/face_recognition/face_detection_yunet_2022mar.onnx")
            if model_path.exists():
                self.dnn_detector = cv2.dnn.readNet(str(model_path))
                self.dnn_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.dnn_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                self.detection_methods.append(('dnn', self._detect_dnn))
                logger.info("✅ DNN detector initialized")
            else:
                logger.info("DNN model not found, skipping DNN detection")
                
        except Exception as e:
            logger.warning(f"Failed to initialize DNN detector: {e}")
    
    def _detect_haar_cascade(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with different scale factors
        faces = []
        for scale_factor in [1.1, 1.05, 1.15]:
            detected = self.haar_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale_factor, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in detected:
                # Calculate confidence based on face size and position
                confidence = min(1.0, (w * h) / (image.shape[0] * image.shape[1]) * 100)
                if confidence >= self.confidence_threshold:
                    faces.append((x, y, w, h, confidence))
        
        return faces
    
    def _detect_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces using DNN"""
        try:
            height, width = image.shape[:2]
            
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (320, 320), (104, 177, 123), swapRB=True, crop=False
            )
            
            self.dnn_detector.setInput(blob)
            detections = self.dnn_detector.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence >= self.confidence_threshold:
                    # Get bounding box coordinates
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    w, h = x2 - x1, y2 - y1
                    faces.append((x1, y1, w, h, confidence))
            
            return faces
            
        except Exception as e:
            logger.warning(f"DNN detection failed: {e}")
            return []
    
    def _detect_geometric(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Geometric face detection as fallback"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            faces = []
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly face-shaped (4-6 sides, reasonable aspect ratio)
                if 4 <= len(approx) <= 6:
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Filter by reasonable face proportions
                    aspect_ratio = w / h
                    if 0.7 <= aspect_ratio <= 1.3 and w >= 30 and h >= 30:
                        # Calculate confidence based on contour area
                        area = cv2.contourArea(contour)
                        confidence = min(1.0, area / (image.shape[0] * image.shape[1]) * 200)
                        
                        if confidence >= self.confidence_threshold:
                            faces.append((x, y, w, h, confidence))
            
            return faces
            
        except Exception as e:
            logger.warning(f"Geometric detection failed: {e}")
            return []
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using multiple methods and combine results
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of (x, y, w, h, confidence) tuples
        """
        start_time = time.time()
        
        all_faces = []
        
        # Run all available detection methods
        for method_name, method_func in self.detection_methods:
            try:
                faces = method_func(image)
                logger.debug(f"{method_name} detected {len(faces)} faces")
                all_faces.extend(faces)
            except Exception as e:
                logger.warning(f"{method_name} detection failed: {e}")
        
        # Add geometric detection as fallback if no other methods worked
        if not all_faces:
            logger.info("No faces detected by primary methods, trying geometric detection")
            all_faces = self._detect_geometric(image)
        
        # Remove duplicate detections using non-maximum suppression
        final_faces = self._non_maximum_suppression(all_faces)
        
        detection_time = time.time() - start_time
        logger.info(f"Face detection completed in {detection_time:.3f}s: {len(final_faces)} faces found")
        
        return final_faces
    
    def _non_maximum_suppression(self, faces: List[Tuple[int, int, int, int, float]], 
                                overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
        """
        Remove overlapping face detections
        
        Args:
            faces: List of (x, y, w, h, confidence) tuples
            overlap_threshold: IoU threshold for considering faces as overlapping
            
        Returns:
            Filtered list of faces
        """
        if not faces:
            return []
        
        # Sort by confidence (highest first)
        faces = sorted(faces, key=lambda x: x[4], reverse=True)
        
        final_faces = []
        
        for current_face in faces:
            should_add = True
            
            for existing_face in final_faces:
                if self._calculate_iou(current_face, existing_face) > overlap_threshold:
                    should_add = False
                    break
            
            if should_add:
                final_faces.append(current_face)
        
        return final_faces
    
    def _calculate_iou(self, face1: Tuple[int, int, int, int, float], 
                      face2: Tuple[int, int, int, int, float]) -> float:
        """Calculate Intersection over Union between two face bounding boxes"""
        x1, y1, w1, h1, _ = face1
        x2, y2, w2, h2, _ = face2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def get_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """Get the largest face in the image"""
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Return face with largest area
        return max(faces, key=lambda x: x[2] * x[3])
    
    def crop_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int, float], 
                  padding: float = 0.2) -> np.ndarray:
        """
        Crop face from image with padding
        
        Args:
            image: Input image
            face_box: (x, y, w, h, confidence) tuple
            padding: Padding factor (0.2 = 20% padding)
            
        Returns:
            Cropped face image
        """
        x, y, w, h, _ = face_box
        height, width = image.shape[:2]
        
        # Calculate padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        # Calculate crop boundaries
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        
        return image[y1:y2, x1:x2]
    
    def get_face_quality_score(self, face_image: np.ndarray) -> float:
        """
        Calculate face quality score based on various factors
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Quality score (0.0 - 1.0)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate various quality metrics
            metrics = {}
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'] = min(1.0, laplacian_var / 500.0)
            
            # 2. Brightness (mean intensity)
            mean_intensity = np.mean(gray)
            metrics['brightness'] = 1.0 - abs(mean_intensity - 128) / 128
            
            # 3. Contrast (standard deviation)
            std_intensity = np.std(gray)
            metrics['contrast'] = min(1.0, std_intensity / 50.0)
            
            # 4. Face size (larger faces are generally better)
            height, width = face_image.shape[:2]
            size_score = min(1.0, (height * width) / (256 * 256))
            metrics['size'] = size_score
            
            # Calculate weighted average
            weights = {'sharpness': 0.4, 'brightness': 0.2, 'contrast': 0.2, 'size': 0.2}
            quality_score = sum(metrics[key] * weights[key] for key in weights)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate face quality: {e}")
            return 0.5  # Default quality score
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get statistics about detection methods"""
        return {
            'available_methods': [method[0] for method in self.detection_methods],
            'method_count': len(self.detection_methods),
            'confidence_threshold': self.confidence_threshold
        }
