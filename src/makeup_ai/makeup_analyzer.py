"""
Makeup Analyzer for AI Beauty Platform
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MakeupAnalyzer:
    """
    Analyze makeup styles and effects in images
    """
    
    def __init__(self):
        """
        Initialize makeup analyzer
        """
        pass
    
    def analyze_makeup_style(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze makeup style in the image
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Makeup analysis results
        """
        analysis = {}
        
        # Analyze different makeup components
        analysis['lipstick'] = self._analyze_lipstick(image, landmarks)
        analysis['eyeshadow'] = self._analyze_eyeshadow(image, landmarks)
        analysis['blush'] = self._analyze_blush(image, landmarks)
        analysis['foundation'] = self._analyze_foundation(image, landmarks)
        
        return analysis
    
    def _analyze_lipstick(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze lipstick color and intensity
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Lipstick analysis
        """
        # Extract mouth region
        mouth_landmarks = landmarks[48:68]
        
        # Create mouth mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mouth_points = mouth_landmarks.astype(np.int32)
        cv2.fillPoly(mask, [mouth_points], 255)
        
        # Extract lip region
        lip_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Analyze color
        lip_pixels = lip_region[mask > 0]
        if len(lip_pixels) > 0:
            avg_color = np.mean(lip_pixels, axis=0)
            color_std = np.std(lip_pixels, axis=0)
            
            # Determine lipstick color
            color_name = self._classify_lipstick_color(avg_color)
            
            return {
                'color': avg_color.tolist(),
                'color_name': color_name,
                'intensity': np.mean(color_std),
                'present': True
            }
        
        return {'present': False}
    
    def _analyze_eyeshadow(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze eyeshadow color and intensity
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Eyeshadow analysis
        """
        # Extract eye regions
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Create eye region mask (extended area for eyeshadow)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Extend eye regions upward for eyeshadow
        for eye in [left_eye, right_eye]:
            eye_center = np.mean(eye, axis=0)
            eye_radius = np.max(eye[:, 0]) - np.min(eye[:, 0])
            
            # Create extended region
            top_y = int(eye_center[1] - eye_radius * 1.5)
            bottom_y = int(eye_center[1] + eye_radius * 0.5)
            left_x = int(eye_center[0] - eye_radius * 1.2)
            right_x = int(eye_center[0] + eye_radius * 1.2)
            
            cv2.rectangle(mask, (left_x, top_y), (right_x, bottom_y), 255, -1)
        
        # Extract eyeshadow region
        eyeshadow_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Analyze color
        eyeshadow_pixels = eyeshadow_region[mask > 0]
        if len(eyeshadow_pixels) > 0:
            avg_color = np.mean(eyeshadow_pixels, axis=0)
            color_std = np.std(eyeshadow_pixels, axis=0)
            
            return {
                'color': avg_color.tolist(),
                'intensity': np.mean(color_std),
                'present': True
            }
        
        return {'present': False}
    
    def _analyze_blush(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze blush color and intensity
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Blush analysis
        """
        # Extract cheek regions (approximate)
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        # Define cheek regions
        left_cheek_center = np.array([
            np.min(landmarks[:, 0]) + face_width * 0.2,
            np.min(landmarks[:, 1]) + face_height * 0.6
        ])
        right_cheek_center = np.array([
            np.max(landmarks[:, 0]) - face_width * 0.2,
            np.min(landmarks[:, 1]) + face_height * 0.6
        ])
        
        # Create cheek masks
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cheek_radius = int(face_width * 0.15)
        
        cv2.circle(mask, tuple(left_cheek_center.astype(int)), cheek_radius, 255, -1)
        cv2.circle(mask, tuple(right_cheek_center.astype(int)), cheek_radius, 255, -1)
        
        # Extract blush region
        blush_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Analyze color
        blush_pixels = blush_region[mask > 0]
        if len(blush_pixels) > 0:
            avg_color = np.mean(blush_pixels, axis=0)
            color_std = np.std(blush_pixels, axis=0)
            
            return {
                'color': avg_color.tolist(),
                'intensity': np.mean(color_std),
                'present': True
            }
        
        return {'present': False}
    
    def _analyze_foundation(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze foundation coverage and tone
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Foundation analysis
        """
        # Create face mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        jaw_points = landmarks[:17].astype(np.int32)
        cv2.fillPoly(mask, [jaw_points], 255)
        
        # Extract face region
        face_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Analyze skin tone uniformity
        face_pixels = face_region[mask > 0]
        if len(face_pixels) > 0:
            color_std = np.std(face_pixels, axis=0)
            uniformity = 1.0 / (1.0 + np.mean(color_std))
            
            return {
                'uniformity': uniformity,
                'coverage': 'high' if uniformity > 0.7 else 'medium' if uniformity > 0.4 else 'low',
                'present': True
            }
        
        return {'present': False}
    
    def _classify_lipstick_color(self, color: np.ndarray) -> str:
        """
        Classify lipstick color
        
        Args:
            color: BGR color values
            
        Returns:
            Color name
        """
        b, g, r = color
        
        # Simple color classification
        if r > 150 and g < 100 and b < 100:
            return 'red'
        elif r > 150 and g > 100 and b < 100:
            return 'coral'
        elif r > 100 and g > 100 and b > 100:
            return 'nude'
        elif r < 100 and g < 100 and b > 100:
            return 'purple'
        elif r > 200 and g > 200 and b < 100:
            return 'pink'
        else:
            return 'natural'
    
    def get_makeup_recommendations(self, analysis: Dict) -> List[str]:
        """
        Get makeup recommendations based on analysis
        
        Args:
            analysis: Makeup analysis results
            
        Returns:
            List of makeup recommendations
        """
        recommendations = []
        
        # Lipstick recommendations
        if 'lipstick' in analysis and analysis['lipstick']['present']:
            color = analysis['lipstick'].get('color_name', 'unknown')
            if color == 'red':
                recommendations.append("Bold red lipstick detected - great for evening looks")
            elif color == 'nude':
                recommendations.append("Natural nude lipstick - perfect for everyday wear")
        
        # Eyeshadow recommendations
        if 'eyeshadow' in analysis and analysis['eyeshadow']['present']:
            intensity = analysis['eyeshadow'].get('intensity', 0)
            if intensity > 50:
                recommendations.append("Strong eyeshadow detected - consider softer shades for daytime")
            else:
                recommendations.append("Subtle eyeshadow - you could try bolder colors for evening")
        
        # Blush recommendations
        if 'blush' in analysis and analysis['blush']['present']:
            recommendations.append("Blush detected - adds nice warmth to your complexion")
        else:
            recommendations.append("Consider adding blush for a healthy glow")
        
        # Foundation recommendations
        if 'foundation' in analysis and analysis['foundation']['present']:
            coverage = analysis['foundation'].get('coverage', 'unknown')
            if coverage == 'high':
                recommendations.append("Full coverage foundation detected - great for special occasions")
            elif coverage == 'medium':
                recommendations.append("Medium coverage foundation - good for everyday wear")
        
        return recommendations 