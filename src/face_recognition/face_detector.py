"""
Face Detection Module using OpenCV and Dlib
"""

import cv2
import dlib
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Advanced face detector using multiple detection methods for robustness
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize face detector with OpenCV and Dlib
        
        Args:
            confidence_threshold: Minimum confidence for face detection
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize OpenCV face detector
        self.opencv_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize Dlib face detector
        self.dlib_detector = dlib.get_frontal_face_detector()
        
        # Load CNN-based face detector for better accuracy
        try:
            self.cnn_detector = dlib.cnn_face_detection_model_v1(
                'models/mmod_human_face_detector.dat'
            )
            self.use_cnn = True
        except:
            logger.warning("CNN face detector model not found, using HOG detector")
            self.use_cnn = False
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the image using multiple methods
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face bounding boxes (x, y, width, height)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = []
        
        # Method 1: OpenCV Haar Cascade
        opencv_faces = self.opencv_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Method 2: Dlib HOG detector
        dlib_faces = self.dlib_detector(gray)
        
        # Method 3: Dlib CNN detector (if available)
        if self.use_cnn:
            cnn_faces = self.cnn_detector(gray)
        
        # Combine and filter results
        for (x, y, w, h) in opencv_faces:
            faces.append((x, y, w, h))
        
        for face in dlib_faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            faces.append((x, y, w, h))
        
        if self.use_cnn:
            for face in cnn_faces:
                x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
                if face.confidence > self.confidence_threshold:
                    faces.append((x, y, w, h))
        
        # Remove duplicate detections using IoU
        faces = self._remove_duplicates(faces)
        
        return faces
    
    def _remove_duplicates(self, faces: List[Tuple[int, int, int, int]], 
                          iou_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Remove duplicate face detections using IoU
        
        Args:
            faces: List of face bounding boxes
            iou_threshold: IoU threshold for considering duplicates
            
        Returns:
            Filtered list of unique face detections
        """
        if len(faces) <= 1:
            return faces
        
        # Sort by area (largest first)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        unique_faces = []
        
        for face in faces:
            is_duplicate = False
            for unique_face in unique_faces:
                if self._calculate_iou(face, unique_face) > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            box1: First bounding box (x, y, width, height)
            box2: Second bounding box (x, y, width, height)
            
        Returns:
            IoU value
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the largest face in the image
        
        Args:
            image: Input image
            
        Returns:
            Largest face bounding box or None if no faces detected
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Return the face with largest area
        return max(faces, key=lambda x: x[2] * x[3])
    
    def crop_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int], 
                  padding: float = 0.2) -> np.ndarray:
        """
        Crop face from image with padding
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, width, height)
            padding: Padding ratio around the face
            
        Returns:
            Cropped face image
        """
        x, y, w, h = face_box
        height, width = image.shape[:2]
        
        # Calculate padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        # Calculate crop coordinates
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        
        return image[y1:y2, x1:x2] 