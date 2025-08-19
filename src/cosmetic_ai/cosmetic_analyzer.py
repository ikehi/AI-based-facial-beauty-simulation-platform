"""
Cosmetic Analyzer for AI Beauty Platform
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CosmeticAnalyzer:
    """
    Analyze cosmetic features and provide recommendations
    """
    
    def __init__(self):
        """
        Initialize cosmetic analyzer
        """
        pass
    
    def analyze_cosmetic_features(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze cosmetic features in the image
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Cosmetic analysis results
        """
        analysis = {}
        
        # Analyze different cosmetic features
        analysis['skin_quality'] = self._analyze_skin_quality(image, landmarks)
        analysis['facial_symmetry'] = self._analyze_facial_symmetry(landmarks)
        analysis['overall_beauty'] = self._analyze_overall_beauty(image, landmarks)
        
        return analysis
    
    def _analyze_skin_quality(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze skin quality
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Skin quality analysis
        """
        # Create face mask
        face_mask = self._create_face_mask(image, landmarks)
        
        # Extract face region
        face_region = cv2.bitwise_and(image, image, mask=face_mask)
        
        # Convert to HSV
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Calculate skin quality metrics
        skin_pixels = face_region[face_mask > 0]
        if len(skin_pixels) > 0:
            # Calculate texture (using standard deviation)
            texture_score = np.std(skin_pixels)
            
            # Calculate evenness (using color variance)
            color_variance = np.var(skin_pixels, axis=0)
            evenness_score = 1.0 / (1.0 + np.mean(color_variance))
            
            # Calculate brightness
            brightness = np.mean(hsv[face_mask > 0, 2])
            
            # Determine skin quality grade
            if texture_score < 20 and evenness_score > 0.8:
                quality_grade = 'excellent'
            elif texture_score < 30 and evenness_score > 0.6:
                quality_grade = 'good'
            elif texture_score < 40 and evenness_score > 0.4:
                quality_grade = 'fair'
            else:
                quality_grade = 'poor'
            
            return {
                'quality_grade': quality_grade,
                'texture_score': float(texture_score),
                'evenness_score': float(evenness_score),
                'brightness': float(brightness)
            }
        
        return {'quality_grade': 'unknown'}
    
    def _analyze_facial_symmetry(self, landmarks: np.ndarray) -> Dict:
        """
        Analyze facial symmetry
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Facial symmetry analysis
        """
        # Calculate face center line
        face_center_x = np.mean(landmarks[:, 0])
        
        # Analyze eye symmetry
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        eye_symmetry = self._calculate_symmetry_score(left_eye_center, right_eye_center, face_center_x)
        
        # Determine symmetry grade
        if eye_symmetry > 0.9:
            symmetry_grade = 'excellent'
        elif eye_symmetry > 0.8:
            symmetry_grade = 'good'
        elif eye_symmetry > 0.7:
            symmetry_grade = 'fair'
        else:
            symmetry_grade = 'poor'
        
        return {
            'overall_symmetry': float(eye_symmetry),
            'symmetry_grade': symmetry_grade
        }
    
    def _analyze_overall_beauty(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze overall beauty score
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Overall beauty analysis
        """
        # Combine different analysis components
        skin_quality = self._analyze_skin_quality(image, landmarks)
        facial_symmetry = self._analyze_facial_symmetry(landmarks)
        
        # Convert grades to numerical scores
        grade_scores = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.3,
            'unknown': 0.5
        }
        
        skin_score = grade_scores.get(skin_quality.get('quality_grade', 'unknown'), 0.5)
        symmetry_score = facial_symmetry.get('overall_symmetry', 0.5)
        
        # Calculate beauty score
        beauty_score = (skin_score + symmetry_score) / 2
        
        # Determine beauty grade
        if beauty_score > 0.8:
            beauty_grade = 'excellent'
        elif beauty_score > 0.7:
            beauty_grade = 'good'
        elif beauty_score > 0.6:
            beauty_grade = 'fair'
        else:
            beauty_grade = 'poor'
        
        return {
            'beauty_score': float(beauty_score),
            'beauty_grade': beauty_grade
        }
    
    def _create_face_mask(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Create face mask from landmarks
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Face mask
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Use jaw landmarks for face outline
        jaw_points = landmarks[:17].astype(np.int32)
        
        # Create convex hull for face shape
        hull = cv2.convexHull(jaw_points)
        cv2.fillPoly(mask, [hull], 255)
        
        return mask
    
    def _calculate_symmetry_score(self, left_point: np.ndarray, right_point: np.ndarray, 
                                center_x: float) -> float:
        """
        Calculate symmetry score between two points
        
        Args:
            left_point: Left feature point
            right_point: Right feature point
            center_x: Face center x-coordinate
            
        Returns:
            Symmetry score (0-1)
        """
        # Calculate expected positions based on center
        left_expected_x = center_x - (right_point[0] - center_x)
        right_expected_x = center_x + (center_x - left_point[0])
        
        # Calculate actual vs expected positions
        left_error = abs(left_point[0] - left_expected_x)
        right_error = abs(right_point[0] - right_expected_x)
        
        # Calculate symmetry score
        total_error = left_error + right_error
        face_width = 2 * abs(right_point[0] - center_x)
        
        if face_width > 0:
            symmetry_score = 1.0 - (total_error / face_width)
            return max(0, min(1, symmetry_score))
        
        return 0.5
    
    def get_cosmetic_recommendations(self, analysis: Dict) -> List[str]:
        """
        Get cosmetic recommendations based on analysis
        
        Args:
            analysis: Cosmetic analysis results
            
        Returns:
            List of cosmetic recommendations
        """
        recommendations = []
        
        # Skin quality recommendations
        if 'skin_quality' in analysis:
            skin_grade = analysis['skin_quality'].get('quality_grade', 'unknown')
            if skin_grade == 'poor':
                recommendations.append("Consider a skincare routine to improve skin texture")
            elif skin_grade == 'fair':
                recommendations.append("A gentle exfoliation routine could improve skin quality")
        
        # Facial symmetry recommendations
        if 'facial_symmetry' in analysis:
            symmetry_grade = analysis['facial_symmetry'].get('symmetry_grade', 'unknown')
            if symmetry_grade == 'poor':
                recommendations.append("Consider makeup techniques to balance facial features")
        
        # Overall beauty recommendations
        if 'overall_beauty' in analysis:
            beauty_grade = analysis['overall_beauty'].get('beauty_grade', 'unknown')
            if beauty_grade == 'excellent':
                recommendations.append("Your natural beauty is already well-balanced")
            elif beauty_grade == 'good':
                recommendations.append("Minor enhancements could further improve your appearance")
            elif beauty_grade == 'fair':
                recommendations.append("Consider a comprehensive beauty consultation")
            elif beauty_grade == 'poor':
                recommendations.append("Professional beauty consultation recommended")
        
        return recommendations 