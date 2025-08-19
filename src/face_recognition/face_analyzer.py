"""
Face Analyzer for AI Beauty Platform
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FaceAnalyzer:
    """
    Analyze facial features and characteristics
    """
    
    def __init__(self):
        """
        Initialize face analyzer
        """
        pass
    
    def analyze_face_shape(self, landmarks: np.ndarray) -> str:
        """
        Analyze face shape based on landmarks
        
        Args:
            landmarks: 68-point facial landmarks
            
        Returns:
            Face shape classification
        """
        # Extract jaw landmarks
        jaw_landmarks = landmarks[:17]
        
        # Calculate face measurements
        face_width = np.max(jaw_landmarks[:, 0]) - np.min(jaw_landmarks[:, 0])
        face_height = np.max(jaw_landmarks[:, 1]) - np.min(jaw_landmarks[:, 1])
        
        # Calculate ratios
        width_height_ratio = face_width / face_height
        
        # Classify face shape
        if width_height_ratio > 0.85:
            return "round"
        elif width_height_ratio < 0.75:
            return "oval"
        else:
            return "square"
    
    def analyze_skin_tone(self, image: np.ndarray, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Analyze skin tone from face image
        
        Args:
            image: Face image
            landmarks: Facial landmarks
            
        Returns:
            Skin tone analysis results
        """
        # Create face mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        jaw_points = landmarks[:17].astype(np.int32)
        cv2.fillPoly(mask, [jaw_points], 255)
        
        # Extract skin region
        skin_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(skin_region, cv2.COLOR_BGR2LAB)
        
        # Calculate average skin tone
        skin_pixels = skin_region[mask > 0]
        if len(skin_pixels) > 0:
            avg_bgr = np.mean(skin_pixels, axis=0)
            avg_hsv = np.mean(hsv[mask > 0], axis=0)
            avg_lab = np.mean(lab[mask > 0], axis=0)
            
            return {
                'bgr': avg_bgr.tolist(),
                'hsv': avg_hsv.tolist(),
                'lab': avg_lab.tolist(),
                'brightness': avg_hsv[2],
                'saturation': avg_hsv[1]
            }
        
        return {}
    
    def analyze_facial_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Analyze facial feature proportions
        
        Args:
            landmarks: 68-point facial landmarks
            
        Returns:
            Facial feature analysis
        """
        # Extract feature landmarks
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        nose = landmarks[27:36]
        mouth = landmarks[48:68]
        
        # Calculate measurements
        eye_width = np.mean([
            np.max(left_eye[:, 0]) - np.min(left_eye[:, 0]),
            np.max(right_eye[:, 0]) - np.min(right_eye[:, 0])
        ])
        
        eye_height = np.mean([
            np.max(left_eye[:, 1]) - np.min(left_eye[:, 1]),
            np.max(right_eye[:, 1]) - np.min(right_eye[:, 1])
        ])
        
        nose_width = np.max(nose[:, 0]) - np.min(nose[:, 0])
        nose_height = np.max(nose[:, 1]) - np.min(nose[:, 1])
        
        mouth_width = np.max(mouth[:, 0]) - np.min(mouth[:, 0])
        mouth_height = np.max(mouth[:, 1]) - np.min(mouth[:, 1])
        
        # Calculate ratios
        eye_ratio = eye_width / eye_height
        nose_ratio = nose_width / nose_height
        mouth_ratio = mouth_width / mouth_height
        
        return {
            'eye_ratio': eye_ratio,
            'nose_ratio': nose_ratio,
            'mouth_ratio': mouth_ratio,
            'eye_width': eye_width,
            'eye_height': eye_height,
            'nose_width': nose_width,
            'nose_height': nose_height,
            'mouth_width': mouth_width,
            'mouth_height': mouth_height
        }
    
    def get_beauty_recommendations(self, analysis: Dict) -> List[str]:
        """
        Get beauty recommendations based on analysis
        
        Args:
            analysis: Face analysis results
            
        Returns:
            List of beauty recommendations
        """
        recommendations = []
        
        # Face shape recommendations
        if 'face_shape' in analysis:
            face_shape = analysis['face_shape']
            if face_shape == 'round':
                recommendations.append("Consider angular makeup to add definition")
                recommendations.append("Try hairstyles that add height and volume")
            elif face_shape == 'square':
                recommendations.append("Soft, rounded makeup looks work well")
                recommendations.append("Consider layered hairstyles to soften angles")
            elif face_shape == 'oval':
                recommendations.append("Most makeup styles suit your face shape")
                recommendations.append("You can experiment with various hairstyles")
        
        # Skin tone recommendations
        if 'skin_tone' in analysis:
            skin_tone = analysis['skin_tone']
            if skin_tone.get('brightness', 0) < 100:
                recommendations.append("Consider brightening makeup products")
            if skin_tone.get('saturation', 0) < 50:
                recommendations.append("Try adding warmth with bronzer or blush")
        
        return recommendations 