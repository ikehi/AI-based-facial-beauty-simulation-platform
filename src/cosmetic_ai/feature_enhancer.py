"""
Feature Enhancer for AI Beauty Platform
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEnhancer:
    """
    Enhance facial features using AI-based techniques
    """
    
    def __init__(self):
        """
        Initialize feature enhancer
        """
        pass
    
    def enhance_features(self, image: np.ndarray, landmarks: np.ndarray, 
                        enhancements: Dict) -> np.ndarray:
        """
        Enhance facial features based on specified parameters
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            enhancements: Dictionary of enhancements to apply
            
        Returns:
            Image with enhanced features
        """
        result = image.copy()
        
        # Apply different enhancements
        if 'skin' in enhancements:
            result = self._enhance_skin(result, landmarks, enhancements['skin'])
        
        if 'eyes' in enhancements:
            result = self._enhance_eyes(result, landmarks, enhancements['eyes'])
        
        if 'lips' in enhancements:
            result = self._enhance_lips(result, landmarks, enhancements['lips'])
        
        if 'overall' in enhancements:
            result = self._enhance_overall(result, landmarks, enhancements['overall'])
        
        return result
    
    def _enhance_skin(self, image: np.ndarray, landmarks: np.ndarray, 
                     enhancements: Dict) -> np.ndarray:
        """
        Enhance skin features
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            enhancements: Skin enhancement parameters
            
        Returns:
            Image with enhanced skin
        """
        result = image.copy()
        
        # Create face mask
        face_mask = self._create_face_mask(image, landmarks)
        
        # Apply skin smoothing
        if 'smoothing' in enhancements:
            smoothing_factor = enhancements['smoothing']
            result = self._apply_skin_smoothing(result, face_mask, smoothing_factor)
        
        # Apply skin brightening
        if 'brightening' in enhancements:
            brightening_factor = enhancements['brightening']
            result = self._apply_skin_brightening(result, face_mask, brightening_factor)
        
        return result
    
    def _enhance_eyes(self, image: np.ndarray, landmarks: np.ndarray, 
                     enhancements: Dict) -> np.ndarray:
        """
        Enhance eye features
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            enhancements: Eye enhancement parameters
            
        Returns:
            Image with enhanced eyes
        """
        result = image.copy()
        
        # Extract eye landmarks
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Apply eye whitening
        if 'whitening' in enhancements:
            whitening_factor = enhancements['whitening']
            result = self._apply_eye_whitening(result, left_eye, right_eye, whitening_factor)
        
        return result
    
    def _enhance_lips(self, image: np.ndarray, landmarks: np.ndarray, 
                     enhancements: Dict) -> np.ndarray:
        """
        Enhance lip features
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            enhancements: Lip enhancement parameters
            
        Returns:
            Image with enhanced lips
        """
        result = image.copy()
        
        # Extract mouth landmarks
        mouth = landmarks[48:68]
        
        # Apply lip color enhancement
        if 'color_enhancement' in enhancements:
            color_factor = enhancements['color_enhancement']
            result = self._apply_lip_color_enhancement(result, mouth, color_factor)
        
        return result
    
    def _enhance_overall(self, image: np.ndarray, landmarks: np.ndarray, 
                        enhancements: Dict) -> np.ndarray:
        """
        Apply overall enhancements
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            enhancements: Overall enhancement parameters
            
        Returns:
            Image with overall enhancements
        """
        result = image.copy()
        
        # Apply brightness adjustment
        if 'brightness' in enhancements:
            brightness_factor = enhancements['brightness']
            result = cv2.convertScaleAbs(result, alpha=brightness_factor, beta=0)
        
        # Apply contrast adjustment
        if 'contrast' in enhancements:
            contrast_factor = enhancements['contrast']
            result = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=0)
        
        return result
    
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
    
    def _apply_skin_smoothing(self, image: np.ndarray, mask: np.ndarray, 
                            factor: float) -> np.ndarray:
        """
        Apply skin smoothing
        
        Args:
            image: Input image
            mask: Face mask
            factor: Smoothing factor
            
        Returns:
            Image with smoothed skin
        """
        result = image.copy()
        
        # Apply bilateral filter for skin smoothing
        smoothed = cv2.bilateralFilter(image, 15, 50, 50)
        
        # Blend with original based on factor
        result = cv2.addWeighted(result, 1 - factor, smoothed, factor, 0)
        
        # Apply mask to keep only face region
        result = cv2.bitwise_and(result, result, mask=mask)
        result = cv2.add(result, cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)))
        
        return result
    
    def _apply_skin_brightening(self, image: np.ndarray, mask: np.ndarray, 
                              factor: float) -> np.ndarray:
        """
        Apply skin brightening
        
        Args:
            image: Input image
            mask: Face mask
            factor: Brightening factor
            
        Returns:
            Image with brightened skin
        """
        result = image.copy()
        
        # Create brightened version
        brightened = cv2.convertScaleAbs(image, alpha=1.0, beta=factor * 30)
        
        # Blend with original based on factor
        result = cv2.addWeighted(result, 1 - factor, brightened, factor, 0)
        
        # Apply mask to keep only face region
        result = cv2.bitwise_and(result, result, mask=mask)
        result = cv2.add(result, cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)))
        
        return result
    
    def _apply_eye_whitening(self, image: np.ndarray, left_eye: np.ndarray, 
                           right_eye: np.ndarray, factor: float) -> np.ndarray:
        """
        Apply eye whitening
        
        Args:
            image: Input image
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
            factor: Whitening factor
            
        Returns:
            Image with whitened eyes
        """
        result = image.copy()
        
        for eye in [left_eye, right_eye]:
            # Calculate eye center and radius
            eye_center = np.mean(eye, axis=0).astype(int)
            eye_radius = int(np.max(eye[:, 0]) - np.min(eye[:, 0])) // 2
            
            # Create mask for eye region
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, tuple(eye_center), eye_radius, 255, -1)
            
            # Apply whitening
            eye_region = cv2.bitwise_and(image, image, mask=mask)
            whitened = cv2.convertScaleAbs(eye_region, alpha=1.0, beta=factor * 20)
            
            # Blend with original
            result = cv2.addWeighted(result, 1 - 0.5, whitened, 0.5, 0)
        
        return result
    
    def _apply_lip_color_enhancement(self, image: np.ndarray, mouth: np.ndarray, 
                                   factor: float) -> np.ndarray:
        """
        Apply lip color enhancement
        
        Args:
            image: Input image
            mouth: Mouth landmarks
            factor: Color enhancement factor
            
        Returns:
            Image with enhanced lip color
        """
        result = image.copy()
        
        # Create mouth mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mouth_points = mouth.astype(np.int32)
        cv2.fillPoly(mask, [mouth_points], 255)
        
        # Apply color enhancement
        mouth_region = cv2.bitwise_and(image, image, mask=mask)
        enhanced = cv2.convertScaleAbs(mouth_region, alpha=factor, beta=0)
        
        # Blend with original
        result = cv2.addWeighted(result, 1 - 0.3, enhanced, 0.3, 0)
        
        return result 