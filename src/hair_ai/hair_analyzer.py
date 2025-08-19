"""
Hair Analyzer for AI Beauty Platform
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class HairAnalyzer:
    """
    Analyze hair characteristics and provide recommendations
    """
    
    def __init__(self):
        """
        Initialize hair analyzer
        """
        pass
    
    def analyze_hair_characteristics(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze hair characteristics in the image
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Hair analysis results
        """
        analysis = {}
        
        # Analyze different hair characteristics
        analysis['color'] = self._analyze_hair_color(image, landmarks)
        analysis['length'] = self._analyze_hair_length(image, landmarks)
        analysis['texture'] = self._analyze_hair_texture(image, landmarks)
        analysis['volume'] = self._analyze_hair_volume(image, landmarks)
        
        return analysis
    
    def _analyze_hair_color(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze hair color
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Hair color analysis
        """
        # Define hair region (top portion of face)
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        # Hair region coordinates
        hair_top = max(0, int(np.min(landmarks[:, 1]) - face_height * 0.3))
        hair_bottom = int(np.min(landmarks[:, 1]) + face_height * 0.1)
        hair_left = int(np.min(landmarks[:, 0]) - face_width * 0.1)
        hair_right = int(np.max(landmarks[:, 0]) + face_width * 0.1)
        
        # Extract hair region
        hair_region = image[hair_top:hair_bottom, hair_left:hair_right]
        
        if hair_region.size == 0:
            return {'detected': False}
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(hair_region, cv2.COLOR_BGR2LAB)
        
        # Calculate average color
        avg_bgr = np.mean(hair_region, axis=(0, 1))
        avg_hsv = np.mean(hsv, axis=(0, 1))
        avg_lab = np.mean(lab, axis=(0, 1))
        
        # Classify hair color
        color_name = self._classify_hair_color(avg_bgr, avg_hsv)
        
        return {
            'detected': True,
            'color': avg_bgr.tolist(),
            'color_name': color_name,
            'brightness': avg_hsv[2],
            'saturation': avg_hsv[1]
        }
    
    def _analyze_hair_length(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze hair length
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Hair length analysis
        """
        # Calculate face dimensions
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        # Define hair region
        hair_top = max(0, int(np.min(landmarks[:, 1]) - face_height * 0.5))
        hair_bottom = int(np.min(landmarks[:, 1]) + face_height * 0.2)
        
        # Calculate hair length ratio
        hair_length = hair_bottom - hair_top
        length_ratio = hair_length / face_height
        
        # Classify length
        if length_ratio < 0.3:
            length_category = 'short'
        elif length_ratio < 0.6:
            length_category = 'medium'
        else:
            length_category = 'long'
        
        return {
            'length_ratio': length_ratio,
            'category': length_category,
            'pixels': hair_length
        }
    
    def _analyze_hair_texture(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze hair texture (straight, wavy, curly)
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Hair texture analysis
        """
        # Define hair region
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        hair_top = max(0, int(np.min(landmarks[:, 1]) - face_height * 0.3))
        hair_bottom = int(np.min(landmarks[:, 1]) + face_height * 0.1)
        hair_left = int(np.min(landmarks[:, 0]) - face_width * 0.1)
        hair_right = int(np.max(landmarks[:, 0]) + face_width * 0.1)
        
        # Extract hair region
        hair_region = image[hair_top:hair_bottom, hair_left:hair_right]
        
        if hair_region.size == 0:
            return {'detected': False}
        
        # Convert to grayscale
        gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features using edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        # Classify texture
        if edge_density < 0.05 and avg_gradient < 20:
            texture = 'straight'
        elif edge_density < 0.1 and avg_gradient < 40:
            texture = 'wavy'
        else:
            texture = 'curly'
        
        return {
            'detected': True,
            'texture': texture,
            'edge_density': edge_density,
            'gradient_magnitude': avg_gradient
        }
    
    def _analyze_hair_volume(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """
        Analyze hair volume
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Hair volume analysis
        """
        # Define hair region
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        hair_top = max(0, int(np.min(landmarks[:, 1]) - face_height * 0.4))
        hair_bottom = int(np.min(landmarks[:, 1]) + face_height * 0.1)
        hair_left = int(np.min(landmarks[:, 0]) - face_width * 0.2)
        hair_right = int(np.max(landmarks[:, 0]) + face_width * 0.2)
        
        # Extract hair region
        hair_region = image[hair_top:hair_bottom, hair_left:hair_right]
        
        if hair_region.size == 0:
            return {'detected': False}
        
        # Convert to grayscale
        gray = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate hair density using thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        hair_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        density = hair_pixels / total_pixels
        
        # Classify volume
        if density < 0.3:
            volume = 'thin'
        elif density < 0.6:
            volume = 'medium'
        else:
            volume = 'thick'
        
        return {
            'detected': True,
            'volume': volume,
            'density': density,
            'hair_pixels': hair_pixels
        }
    
    def _classify_hair_color(self, bgr_color: np.ndarray, hsv_color: np.ndarray) -> str:
        """
        Classify hair color based on BGR and HSV values
        
        Args:
            bgr_color: BGR color values
            hsv_color: HSV color values
            
        Returns:
            Color name
        """
        b, g, r = bgr_color
        h, s, v = hsv_color
        
        # Simple color classification
        if v < 50:
            return 'black'
        elif v < 100 and s < 50:
            return 'gray'
        elif v > 200 and s < 30:
            return 'white'
        elif r > 150 and g < 100 and b < 100:
            return 'red'
        elif r > 150 and g > 120 and b < 100:
            return 'blonde'
        elif r < 100 and g < 100 and b < 100:
            return 'black'
        else:
            return 'brown'
    
    def get_hair_recommendations(self, analysis: Dict) -> List[str]:
        """
        Get hair recommendations based on analysis
        
        Args:
            analysis: Hair analysis results
            
        Returns:
            List of hair recommendations
        """
        recommendations = []
        
        # Color recommendations
        if 'color' in analysis and analysis['color']['detected']:
            color = analysis['color'].get('color_name', 'unknown')
            if color == 'black':
                recommendations.append("Dark hair detected - consider highlights for dimension")
            elif color == 'blonde':
                recommendations.append("Blonde hair detected - lowlights can add depth")
            elif color == 'red':
                recommendations.append("Red hair detected - consider complementary tones")
        
        # Length recommendations
        if 'length' in analysis:
            length = analysis['length'].get('category', 'unknown')
            if length == 'short':
                recommendations.append("Short hair - try texturizing products for volume")
            elif length == 'long':
                recommendations.append("Long hair - consider layers for movement")
        
        # Texture recommendations
        if 'texture' in analysis and analysis['texture']['detected']:
            texture = analysis['texture'].get('texture', 'unknown')
            if texture == 'straight':
                recommendations.append("Straight hair - try curling products for texture")
            elif texture == 'curly':
                recommendations.append("Curly hair - use smoothing products for definition")
        
        # Volume recommendations
        if 'volume' in analysis and analysis['volume']['detected']:
            volume = analysis['volume'].get('volume', 'unknown')
            if volume == 'thin':
                recommendations.append("Thin hair - volumizing products can add body")
            elif volume == 'thick':
                recommendations.append("Thick hair - smoothing products can help manage")
        
        return recommendations 