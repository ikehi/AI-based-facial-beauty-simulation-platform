"""
Image utility functions for AI Beauty Platform
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class ImageUtils:
    """
    Utility class for image processing operations
    """
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create canvas with target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Center the resized image
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            return canvas
        else:
            return cv2.resize(image, target_size)
    
    @staticmethod
    def normalize_image(image: np.ndarray, mean: List[float] = None, 
                       std: List[float] = None) -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            
        Returns:
            Normalized image
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet mean
        if std is None:
            std = [0.229, 0.224, 0.225]   # ImageNet std
        
        # Convert to float
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image
    
    @staticmethod
    def denormalize_image(image: np.ndarray, mean: List[float] = None, 
                         std: List[float] = None) -> np.ndarray:
        """
        Denormalize image pixel values
        
        Args:
            image: Input normalized image
            mean: Mean values used for normalization
            std: Standard deviation values used for normalization
            
        Returns:
            Denormalized image
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet mean
        if std is None:
            std = [0.229, 0.224, 0.225]   # ImageNet std
        
        # Denormalize
        for i in range(3):
            image[:, :, i] = image[:, :, i] * std[i] + mean[i]
        
        # Clip to [0, 1] and convert to uint8
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        
        return image
    
    @staticmethod
    def enhance_image(image: np.ndarray, brightness: float = 1.0, 
                     contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
        """
        Enhance image with brightness, contrast, and saturation adjustments
        
        Args:
            image: Input image
            brightness: Brightness multiplier
            contrast: Contrast multiplier
            saturation: Saturation multiplier
            
        Returns:
            Enhanced image
        """
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Adjust brightness and contrast
        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=(brightness - 1) * 100)
        
        return enhanced
    
    @staticmethod
    def remove_background(image: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        Remove background from image using simple thresholding
        
        Args:
            image: Input image
            threshold: Threshold value for background removal
            
        Returns:
            Image with background removed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply mask
        result = image.copy()
        result[mask == 0] = [0, 0, 0]
        
        return result
    
    @staticmethod
    def blend_images(image1: np.ndarray, image2: np.ndarray, 
                    alpha: float = 0.5) -> np.ndarray:
        """
        Blend two images with specified alpha
        
        Args:
            image1: First image
            image2: Second image
            alpha: Blending factor (0.0 to 1.0)
            
        Returns:
            Blended image
        """
        # Ensure same size
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # Blend images
        blended = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
        
        return blended
    
    @staticmethod
    def create_mask_from_landmarks(landmarks: np.ndarray, image_shape: Tuple[int, int], 
                                 region: str = 'face') -> np.ndarray:
        """
        Create mask from facial landmarks
        
        Args:
            landmarks: Facial landmarks
            image_shape: Image shape (height, width)
            region: Region to mask ('face', 'eyes', 'mouth', etc.)
            
        Returns:
            Binary mask
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        if region == 'face':
            # Use jaw landmarks for face mask
            jaw_points = landmarks[:17]
        elif region == 'eyes':
            # Use eye landmarks
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            jaw_points = np.vstack([left_eye, right_eye])
        elif region == 'mouth':
            # Use mouth landmarks
            jaw_points = landmarks[48:68]
        else:
            # Use all landmarks
            jaw_points = landmarks
        
        # Convert to integer points
        points = jaw_points.astype(np.int32)
        
        # Create convex hull
        hull = cv2.convexHull(points)
        
        # Fill mask
        cv2.fillPoly(mask, [hull], 255)
        
        return mask
    
    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray, 
                  background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Apply mask to image
        
        Args:
            image: Input image
            mask: Binary mask
            background_color: Background color for masked areas
            
        Returns:
            Masked image
        """
        result = image.copy()
        result[mask == 0] = background_color
        return result
    
    @staticmethod
    def save_image(image: np.ndarray, filepath: str, quality: int = 95) -> bool:
        """
        Save image to file
        
        Args:
            image: Image to save
            filepath: Output file path
            quality: JPEG quality (1-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine file format
            if filepath.lower().endswith('.jpg') or filepath.lower().endswith('.jpeg'):
                # Save as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                cv2.imwrite(filepath, image, encode_param)
            else:
                # Save as other format
                cv2.imwrite(filepath, image)
            
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def load_image(filepath: str) -> Optional[np.ndarray]:
        """
        Load image from file
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded image or None if failed
        """
        try:
            image = cv2.imread(filepath)
            if image is None:
                logger.error(f"Could not load image: {filepath}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None 