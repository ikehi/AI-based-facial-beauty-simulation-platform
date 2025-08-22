"""
Enhanced Makeup Transfer Module
Advanced makeup application with multiple styles and effects
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import random

logger = logging.getLogger(__name__)

class EnhancedMakeupTransfer:
    """
    Enhanced makeup transfer with multiple styles and advanced effects
    - Natural makeup enhancement
    - Glamorous makeup styles
    - Color correction and skin smoothing
    - Fallback implementations for reliability
    """
    
    def __init__(self):
        """Initialize the enhanced makeup transfer system"""
        self.styles = {
            'natural': {
                'description': 'Subtle, everyday makeup',
                'intensity_range': (0.3, 0.7),
                'effects': ['foundation', 'concealer', 'mascara', 'lip_balm']
            },
            'glamorous': {
                'description': 'Dramatic, evening makeup',
                'intensity_range': (0.6, 1.0),
                'effects': ['foundation', 'concealer', 'eyeshadow', 'eyeliner', 'mascara', 'lipstick', 'blush']
            },
            'casual': {
                'description': 'Relaxed, weekend makeup',
                'intensity_range': (0.4, 0.8),
                'effects': ['foundation', 'mascara', 'lip_gloss', 'light_blush']
            },
            'evening': {
                'description': 'Sophisticated night makeup',
                'intensity_range': (0.7, 1.0),
                'effects': ['foundation', 'concealer', 'eyeshadow', 'eyeliner', 'mascara', 'lipstick', 'contour', 'highlight']
            },
            'party': {
                'description': 'Bold, fun party makeup',
                'intensity_range': (0.8, 1.0),
                'effects': ['foundation', 'concealer', 'bright_eyeshadow', 'eyeliner', 'mascara', 'bright_lipstick', 'glitter']
            }
        }
        
        # Initialize color palettes for different styles
        self._init_color_palettes()
        
        logger.info("Enhanced Makeup Transfer initialized")
    
    def _init_color_palettes(self):
        """Initialize color palettes for different makeup styles"""
        self.color_palettes = {
            'natural': {
                'foundation': [(220, 200, 180), (200, 180, 160), (180, 160, 140)],
                'lipstick': [(180, 120, 100), (200, 140, 120), (220, 160, 140)],
                'eyeshadow': [(180, 160, 140), (200, 180, 160), (220, 200, 180)],
                'blush': [(220, 180, 180), (200, 160, 160), (180, 140, 140)]
            },
            'glamorous': {
                'foundation': [(240, 220, 200), (220, 200, 180), (200, 180, 160)],
                'lipstick': [(160, 60, 80), (180, 80, 100), (200, 100, 120)],
                'eyeshadow': [(120, 80, 120), (140, 100, 140), (160, 120, 160)],
                'blush': [(240, 200, 200), (220, 180, 180), (200, 160, 160)]
            },
            'casual': {
                'foundation': [(220, 200, 180), (200, 180, 160), (180, 160, 140)],
                'lipstick': [(200, 140, 120), (220, 160, 140), (240, 180, 160)],
                'eyeshadow': [(200, 180, 160), (220, 200, 180), (240, 220, 200)],
                'blush': [(220, 180, 180), (200, 160, 160), (180, 140, 140)]
            },
            'evening': {
                'foundation': [(240, 220, 200), (220, 200, 180), (200, 180, 160)],
                'lipstick': [(140, 40, 60), (160, 60, 80), (180, 80, 100)],
                'eyeshadow': [(100, 60, 100), (120, 80, 120), (140, 100, 140)],
                'blush': [(240, 200, 200), (220, 180, 180), (200, 160, 160)]
            },
            'party': {
                'foundation': [(240, 220, 200), (220, 200, 180), (200, 180, 160)],
                'lipstick': [(200, 40, 80), (220, 60, 100), (240, 80, 120)],
                'eyeshadow': [(80, 40, 120), (100, 60, 140), (120, 80, 160)],
                'blush': [(240, 180, 200), (220, 160, 180), (200, 140, 160)]
            }
        }
    
    def apply_makeup_style(self, face_image: np.ndarray, style: str, 
                          intensity: float = 1.0) -> np.ndarray:
        """
        Apply makeup style to face image
        
        Args:
            face_image: Input face image
            style: Makeup style name
            intensity: Makeup intensity (0.0 - 1.0)
            
        Returns:
            Image with makeup applied
        """
        start_time = time.time()
        
        if style not in self.styles:
            logger.warning(f"Unknown style '{style}', using 'natural'")
            style = 'natural'
        
        # Validate intensity
        intensity = max(0.0, min(1.0, intensity))
        
        # Get style configuration
        style_config = self.styles[style]
        min_intensity, max_intensity = style_config['intensity_range']
        
        # Adjust intensity to style range
        adjusted_intensity = min_intensity + (max_intensity - min_intensity) * intensity
        
        logger.info(f"Applying {style} makeup with intensity {adjusted_intensity:.2f}")
        
        # Start with original image
        result = face_image.copy()
        
        # Apply foundation first (base layer)
        result = self._apply_foundation(result, style, adjusted_intensity)
        
        # Apply other effects based on style
        for effect in style_config['effects']:
            if effect != 'foundation':  # Already applied
                result = self._apply_effect(result, effect, style, adjusted_intensity)
        
        # Final adjustments
        result = self._apply_final_adjustments(result, style, adjusted_intensity)
        
        processing_time = time.time() - start_time
        logger.info(f"Makeup application completed in {processing_time:.3f}s")
        
        return result
    
    def _apply_foundation(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply foundation to the face"""
        try:
            # Get foundation color from palette
            colors = self.color_palettes[style]['foundation']
            base_color = random.choice(colors)
            
            # Create foundation mask (focus on central face area)
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create elliptical mask for face area
            center = (width // 2, height // 2)
            axes = (int(width * 0.4), int(height * 0.5))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            
            # Apply foundation with intensity
            foundation_color = np.array(base_color, dtype=np.uint8)
            foundation_layer = np.full_like(image, foundation_color)
            
            # Blend foundation with original
            alpha = intensity * 0.3  # Foundation is subtle
            result = cv2.addWeighted(image, 1 - alpha, foundation_layer, alpha, 0)
            
            # Apply foundation only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Foundation application failed: {e}")
            return image
    
    def _apply_effect(self, image: np.ndarray, effect: str, style: str, intensity: float) -> np.ndarray:
        """Apply specific makeup effect"""
        try:
            if effect == 'concealer':
                return self._apply_concealer(image, style, intensity)
            elif effect == 'eyeshadow':
                return self._apply_eyeshadow(image, style, intensity)
            elif effect == 'eyeliner':
                return self._apply_eyeliner(image, style, intensity)
            elif effect == 'mascara':
                return self._apply_mascara(image, style, intensity)
            elif effect == 'lipstick':
                return self._apply_lipstick(image, style, intensity)
            elif effect == 'blush':
                return self._apply_blush(image, style, intensity)
            elif effect == 'contour':
                return self._apply_contour(image, style, intensity)
            elif effect == 'highlight':
                return self._apply_highlight(image, style, intensity)
            elif effect == 'lip_balm':
                return self._apply_lip_balm(image, style, intensity)
            elif effect == 'lip_gloss':
                return self._apply_lip_gloss(image, style, intensity)
            elif effect == 'glitter':
                return self._apply_glitter(image, style, intensity)
            else:
                logger.warning(f"Unknown effect: {effect}")
                return image
                
        except Exception as e:
            logger.warning(f"Effect {effect} application failed: {e}")
            return image
    
    def _apply_concealer(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply concealer to reduce dark circles and blemishes"""
        try:
            height, width = image.shape[:2]
            
            # Create concealer mask (under eyes and central face)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Under eye areas
            left_eye_center = (int(width * 0.35), int(height * 0.4))
            right_eye_center = (int(width * 0.65), int(height * 0.4))
            eye_radius = int(min(width, height) * 0.08)
            
            cv2.circle(mask, left_eye_center, eye_radius, 255, -1)
            cv2.circle(mask, right_eye_center, eye_radius, 255, -1)
            
            # Central face area for blemishes
            center = (width // 2, height // 2)
            axes = (int(width * 0.25), int(height * 0.3))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            
            # Get concealer color (slightly lighter than foundation)
            colors = self.color_palettes[style]['foundation']
            base_color = random.choice(colors)
            concealer_color = tuple(min(255, c + 20) for c in base_color)
            
            # Apply concealer
            concealer_layer = np.full_like(image, concealer_color)
            alpha = intensity * 0.4
            
            result = cv2.addWeighted(image, 1 - alpha, concealer_layer, alpha, 0)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Concealer application failed: {e}")
            return image
    
    def _apply_eyeshadow(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply eyeshadow to the eyelids"""
        try:
            height, width = image.shape[:2]
            
            # Create eyeshadow mask (eyelid areas)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Left eye area
            left_eye_center = (int(width * 0.35), int(height * 0.4))
            left_eye_radius = int(min(width, height) * 0.12)
            cv2.circle(mask, left_eye_center, left_eye_radius, 255, -1)
            
            # Right eye area
            right_eye_center = (int(width * 0.65), int(height * 0.4))
            right_eye_radius = int(min(width, height) * 0.12)
            cv2.circle(mask, right_eye_center, right_eye_radius, 255, -1)
            
            # Get eyeshadow color
            colors = self.color_palettes[style]['eyeshadow']
            eyeshadow_color = random.choice(colors)
            
            # Apply eyeshadow
            eyeshadow_layer = np.full_like(image, eyeshadow_color)
            alpha = intensity * 0.6
            
            result = cv2.addWeighted(image, 1 - alpha, eyeshadow_layer, alpha, 0)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Eyeshadow application failed: {e}")
            return image
    
    def _apply_eyeliner(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply eyeliner to the eyes"""
        try:
            height, width = image.shape[:2]
            
            # Create eyeliner mask (eyelash line)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Left eye eyeliner
            left_eye_center = (int(width * 0.35), int(height * 0.4))
            left_eye_radius = int(min(width, height) * 0.08)
            
            # Draw eyeliner along the eye contour
            for angle in range(0, 360, 10):
                rad = np.radians(angle)
                x = int(left_eye_center[0] + left_eye_radius * np.cos(rad))
                y = int(left_eye_center[1] + left_eye_radius * np.sin(rad))
                
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(mask, (x, y), 2, 255, -1)
            
            # Right eye eyeliner
            right_eye_center = (int(width * 0.65), int(height * 0.4))
            right_eye_radius = int(min(width, height) * 0.08)
            
            for angle in range(0, 360, 10):
                rad = np.radians(angle)
                x = int(right_eye_center[0] + right_eye_radius * np.cos(rad))
                y = int(right_eye_center[1] + right_eye_radius * np.sin(rad))
                
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(mask, (x, y), 2, 255, -1)
            
            # Apply eyeliner (black or dark color)
            eyeliner_color = (0, 0, 0)  # Black
            alpha = intensity * 0.8
            
            # Create eyeliner layer
            eyeliner_layer = np.full_like(image, eyeliner_color)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d * alpha) + eyeliner_layer * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Eyeliner application failed: {e}")
            return image
    
    def _apply_mascara(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply mascara to enhance eyelashes"""
        try:
            height, width = image.shape[:2]
            
            # Create mascara mask (eyelash areas)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Left eye lashes
            left_eye_center = (int(width * 0.35), int(height * 0.4))
            left_eye_radius = int(min(width, height) * 0.08)
            
            # Draw individual lashes
            for i in range(20):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(left_eye_radius * 0.8, left_eye_radius * 1.2)
                x = int(left_eye_center[0] + radius * np.cos(angle))
                y = int(left_eye_center[1] + radius * np.sin(angle))
                
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(mask, (x, y), 1, 255, -1)
            
            # Right eye lashes
            right_eye_center = (int(width * 0.65), int(height * 0.4))
            right_eye_radius = int(min(width, height) * 0.08)
            
            for i in range(20):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(right_eye_radius * 0.8, right_eye_radius * 1.2)
                x = int(right_eye_center[0] + radius * np.cos(angle))
                y = int(right_eye_center[1] + radius * np.sin(angle))
                
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(mask, (x, y), 1, 255, -1)
            
            # Apply mascara (dark color)
            mascara_color = (20, 20, 20)  # Dark gray
            alpha = intensity * 0.6
            
            # Create mascara layer
            mascara_layer = np.full_like(image, mascara_color)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d * alpha) + mascara_layer * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Mascara application failed: {e}")
            return image
    
    def _apply_lipstick(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply lipstick to the lips"""
        try:
            height, width = image.shape[:2]
            
            # Create lip mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Lip area (oval shape)
            lip_center = (width // 2, int(height * 0.65))
            lip_axes = (int(width * 0.15), int(height * 0.08))
            cv2.ellipse(mask, lip_center, lip_axes, 0, 0, 360, 255, -1)
            
            # Get lipstick color
            colors = self.color_palettes[style]['lipstick']
            lipstick_color = random.choice(colors)
            
            # Apply lipstick
            lipstick_layer = np.full_like(image, lipstick_color)
            alpha = intensity * 0.7
            
            result = cv2.addWeighted(image, 1 - alpha, lipstick_layer, alpha, 0)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Lipstick application failed: {e}")
            return image
    
    def _apply_blush(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply blush to the cheeks"""
        try:
            height, width = image.shape[:2]
            
            # Create blush mask (cheek areas)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Left cheek
            left_cheek_center = (int(width * 0.25), int(height * 0.55))
            left_cheek_radius = int(min(width, height) * 0.12)
            cv2.circle(mask, left_cheek_center, left_cheek_radius, 255, -1)
            
            # Right cheek
            right_cheek_center = (int(width * 0.75), int(height * 0.55))
            right_cheek_radius = int(min(width, height) * 0.12)
            cv2.circle(mask, right_cheek_center, right_cheek_radius, 255, -1)
            
            # Get blush color
            colors = self.color_palettes[style]['blush']
            blush_color = random.choice(colors)
            
            # Apply blush
            blush_layer = np.full_like(image, blush_color)
            alpha = intensity * 0.5
            
            result = cv2.addWeighted(image, 1 - alpha, blush_layer, alpha, 0)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Blush application failed: {e}")
            return image
    
    def _apply_contour(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply contouring to define facial features"""
        try:
            height, width = image.shape[:2]
            
            # Create contour mask (jawline, nose, cheekbones)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Jawline contour
            jaw_points = [
                (int(width * 0.2), int(height * 0.7)),
                (int(width * 0.3), int(height * 0.75)),
                (int(width * 0.5), int(height * 0.8)),
                (int(width * 0.7), int(height * 0.75)),
                (int(width * 0.8), int(height * 0.7))
            ]
            
            for i in range(len(jaw_points) - 1):
                cv2.line(mask, jaw_points[i], jaw_points[i + 1], 255, 3)
            
            # Nose contour
            nose_center = (width // 2, int(height * 0.5))
            cv2.line(mask, (nose_center[0], nose_center[1] - 20), 
                     (nose_center[0], nose_center[1] + 20), 255, 2)
            
            # Apply contour (darker color)
            contour_color = (100, 80, 60)  # Dark brown
            alpha = intensity * 0.4
            
            contour_layer = np.full_like(image, contour_color)
            result = cv2.addWeighted(image, 1 - alpha, contour_layer, alpha, 0)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Contour application failed: {e}")
            return image
    
    def _apply_highlight(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply highlighting to brighten facial features"""
        try:
            height, width = image.shape[:2]
            
            # Create highlight mask (cheekbones, nose bridge, cupid's bow)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Cheekbone highlights
            left_highlight = (int(width * 0.3), int(height * 0.45))
            right_highlight = (int(width * 0.7), int(height * 0.45))
            
            cv2.circle(mask, left_highlight, 15, 255, -1)
            cv2.circle(mask, right_highlight, 15, 255, -1)
            
            # Nose bridge highlight
            nose_highlight = (width // 2, int(height * 0.45))
            cv2.circle(mask, nose_highlight, 8, 255, -1)
            
            # Cupid's bow highlight
            lip_highlight = (width // 2, int(height * 0.6))
            cv2.circle(mask, lip_highlight, 6, 255, -1)
            
            # Apply highlight (lighter color)
            highlight_color = (255, 240, 220)  # Light beige
            alpha = intensity * 0.5
            
            highlight_layer = np.full_like(image, highlight_color)
            result = cv2.addWeighted(image, 1 - alpha, highlight_layer, alpha, 0)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Highlight application failed: {e}")
            return image
    
    def _apply_lip_balm(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply lip balm for subtle lip enhancement"""
        try:
            height, width = image.shape[:2]
            
            # Create lip mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Lip area
            lip_center = (width // 2, int(height * 0.65))
            lip_axes = (int(width * 0.15), int(height * 0.08))
            cv2.ellipse(mask, lip_center, lip_axes, 0, 0, 360, 255, -1)
            
            # Apply subtle lip enhancement
            lip_color = (220, 200, 180)  # Natural lip color
            alpha = intensity * 0.3
            
            lip_layer = np.full_like(image, lip_color)
            result = cv2.addWeighted(image, 1 - alpha, lip_layer, alpha, 0)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Lip balm application failed: {e}")
            return image
    
    def _apply_lip_gloss(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply lip gloss for shiny lip effect"""
        try:
            # First apply lipstick
            result = self._apply_lipstick(image, style, intensity)
            
            # Add gloss effect (shiny overlay)
            height, width = result.shape[:2]
            
            # Create gloss mask (lip area)
            mask = np.zeros((height, width), dtype=np.uint8)
            lip_center = (width // 2, int(height * 0.65))
            lip_axes = (int(width * 0.15), int(height * 0.08))
            cv2.ellipse(mask, lip_center, lip_axes, 0, 0, 360, 255, -1)
            
            # Add shine effect
            shine_color = (255, 255, 255)  # White for shine
            alpha = intensity * 0.2
            
            shine_layer = np.full_like(result, shine_color)
            result = cv2.addWeighted(result, 1 - alpha, shine_layer, alpha, 0)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Lip gloss application failed: {e}")
            return image
    
    def _apply_glitter(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply glitter effect for party makeup"""
        try:
            height, width = image.shape[:2]
            
            # Create glitter mask (cheekbones, eyes, lips)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Cheekbone glitter
            left_glitter = (int(width * 0.3), int(height * 0.45))
            right_glitter = (int(width * 0.7), int(height * 0.45))
            
            cv2.circle(mask, left_glitter, 20, 255, -1)
            cv2.circle(mask, right_glitter, 20, 255, -1)
            
            # Eye area glitter
            left_eye = (int(width * 0.35), int(height * 0.4))
            right_eye = (int(width * 0.65), int(height * 0.4))
            
            cv2.circle(mask, left_eye, 15, 255, -1)
            cv2.circle(mask, right_eye, 15, 255, -1)
            
            # Add random glitter particles
            for _ in range(int(50 * intensity)):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                cv2.circle(mask, (x, y), 1, 255, -1)
            
            # Apply glitter (sparkly effect)
            glitter_color = (255, 255, 200)  # Light yellow for sparkle
            alpha = intensity * 0.4
            
            glitter_layer = np.full_like(image, glitter_color)
            result = cv2.addWeighted(image, 1 - alpha, glitter_layer, alpha, 0)
            
            # Apply only to masked area
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = image * (1 - mask_3d) + result * mask_3d
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Glitter application failed: {e}")
            return image
    
    def _apply_final_adjustments(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply final adjustments to the makeup"""
        try:
            result = image.copy()
            
            # Skin smoothing
            if intensity > 0.5:
                # Apply bilateral filter for skin smoothing
                result = cv2.bilateralFilter(result, 9, 75, 75)
            
            # Color correction
            if style in ['glamorous', 'evening', 'party']:
                # Enhance saturation slightly
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.1)  # Increase saturation
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Final brightness adjustment
            if intensity > 0.7:
                # Slight brightness boost for dramatic looks
                result = cv2.addWeighted(result, 1.0, result, 0.1, 5)
            
            return result
            
        except Exception as e:
            logger.warning(f"Final adjustments failed: {e}")
            return image
    
    def get_available_styles(self) -> List[str]:
        """Get list of available makeup styles"""
        return list(self.styles.keys())
    
    def get_style_info(self, style: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific style"""
        return self.styles.get(style)
    
    def get_color_palette(self, style: str) -> Optional[Dict[str, List[Tuple[int, int, int]]]]:
        """Get color palette for a specific style"""
        return self.color_palettes.get(style)
    
    def get_makeup_stats(self) -> Dict[str, Any]:
        """Get statistics about the makeup system"""
        return {
            'total_styles': len(self.styles),
            'available_styles': list(self.styles.keys()),
            'color_palettes': list(self.color_palettes.keys())
        }
