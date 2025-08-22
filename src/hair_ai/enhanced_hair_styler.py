"""
Enhanced Hair Styling Module
Advanced hair style transformation with multiple effects
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import random
import math

logger = logging.getLogger(__name__)

class EnhancedHairStyler:
    """
    Enhanced hair styling with multiple styles and effects
    - Multiple hair styles (straight, wavy, curly, coily)
    - Hair color transformation
    - Texture and volume effects
    - Realistic hair physics simulation
    """
    
    def __init__(self):
        """Initialize the enhanced hair styling system"""
        self.styles = {
            'straight': {
                'description': 'Sleek, straight hair',
                'texture': 'smooth',
                'volume': 'low',
                'effects': ['smoothing', 'shine', 'straightening']
            },
            'wavy': {
                'description': 'Natural, wavy hair',
                'texture': 'medium',
                'volume': 'medium',
                'effects': ['wave_pattern', 'texture', 'volume_boost']
            },
            'curly': {
                'description': 'Bouncy, curly hair',
                'texture': 'high',
                'volume': 'high',
                'effects': ['curl_pattern', 'texture', 'volume_boost', 'bounce']
            },
            'coily': {
                'description': 'Tight, coily hair',
                'texture': 'very_high',
                'volume': 'very_high',
                'effects': ['coil_pattern', 'texture', 'volume_boost', 'definition']
            },
            'updo': {
                'description': 'Elegant updo style',
                'texture': 'medium',
                'volume': 'medium',
                'effects': ['updo_shape', 'texture', 'volume_control']
            },
            'braided': {
                'description': 'Beautiful braided hair',
                'texture': 'medium',
                'volume': 'medium',
                'effects': ['braid_pattern', 'texture', 'volume_control']
            }
        }
        
        # Initialize color palettes
        self._init_color_palettes()
        
        # Initialize texture patterns
        self._init_texture_patterns()
        
        logger.info("Enhanced Hair Styler initialized")
    
    def _init_color_palettes(self):
        """Initialize hair color palettes"""
        self.color_palettes = {
            'black': {
                'base': (20, 20, 20),
                'highlights': [(40, 40, 40), (60, 60, 60)],
                'shadows': [(10, 10, 10), (5, 5, 5)]
            },
            'brown': {
                'base': (80, 50, 30),
                'highlights': [(120, 80, 50), (160, 110, 70)],
                'shadows': [(50, 30, 20), (30, 20, 15)]
            },
            'blonde': {
                'base': (200, 180, 120),
                'highlights': [(220, 200, 140), (240, 220, 160)],
                'shadows': [(180, 160, 100), (160, 140, 80)]
            },
            'red': {
                'base': (120, 40, 20),
                'highlights': [(160, 60, 30), (200, 80, 40)],
                'shadows': [(80, 30, 15), (60, 20, 10)]
            },
            'gray': {
                'base': (120, 120, 120),
                'highlights': [(160, 160, 160), (200, 200, 200)],
                'shadows': [(80, 80, 80), (60, 60, 60)]
            },
            'white': {
                'base': (220, 220, 220),
                'highlights': [(240, 240, 240), (255, 255, 255)],
                'shadows': [(200, 200, 200), (180, 180, 180)]
            }
        }
    
    def _init_texture_patterns(self):
        """Initialize texture patterns for different hair styles"""
        self.texture_patterns = {
            'smooth': {
                'noise_level': 0.1,
                'smoothing_factor': 0.8,
                'shine_factor': 0.9
            },
            'medium': {
                'noise_level': 0.3,
                'smoothing_factor': 0.5,
                'shine_factor': 0.6
            },
            'high': {
                'noise_level': 0.6,
                'smoothing_factor': 0.2,
                'shine_factor': 0.3
            },
            'very_high': {
                'noise_level': 0.8,
                'smoothing_factor': 0.1,
                'shine_factor': 0.2
            }
        }
    
    def transform_hair_style(self, image: np.ndarray, style: str, 
                           color: str = 'brown', intensity: float = 1.0) -> np.ndarray:
        """
        Transform hair style and color
        
        Args:
            image: Input image
            style: Hair style name
            color: Hair color name
            intensity: Transformation intensity (0.0 - 1.0)
            
        Returns:
            Image with transformed hair
        """
        start_time = time.time()
        
        if style not in self.styles:
            logger.warning(f"Unknown style '{style}', using 'straight'")
            style = 'straight'
        
        if color not in self.color_palettes:
            logger.warning(f"Unknown color '{color}', using 'brown'")
            color = 'brown'
        
        # Validate intensity
        intensity = max(0.0, min(1.0, intensity))
        
        logger.info(f"Transforming hair to {style} style with {color} color, intensity {intensity:.2f}")
        
        # Start with original image
        result = image.copy()
        
        # Create hair mask
        hair_mask = self._create_hair_mask(image)
        
        # Apply style transformation
        result = self._apply_hair_style(result, style, hair_mask, intensity)
        
        # Apply color transformation
        result = self._apply_hair_color(result, color, hair_mask, intensity)
        
        # Apply texture effects
        result = self._apply_texture_effects(result, style, hair_mask, intensity)
        
        # Final adjustments
        result = self._apply_final_adjustments(result, style, intensity)
        
        processing_time = time.time() - start_time
        logger.info(f"Hair transformation completed in {processing_time:.3f}s")
        
        return result
    
    def _create_hair_mask(self, image: np.ndarray) -> np.ndarray:
        """Create mask for hair areas"""
        try:
            height, width = image.shape[:2]
            
            # Create hair mask (top and sides of head)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Top hair area (elliptical)
            top_center = (width // 2, int(height * 0.15))
            top_axes = (int(width * 0.4), int(height * 0.25))
            cv2.ellipse(mask, top_center, top_axes, 0, 0, 360, 255, -1)
            
            # Side hair areas
            left_side_center = (int(width * 0.15), int(height * 0.3))
            left_side_axes = (int(width * 0.15), int(height * 0.4))
            cv2.ellipse(mask, left_side_center, left_side_axes, 0, 0, 360, 255, -1)
            
            right_side_center = (int(width * 0.85), int(height * 0.3))
            right_side_axes = (int(width * 0.15), int(height * 0.4))
            cv2.ellipse(mask, right_side_center, right_side_axes, 0, 0, 360, 255, -1)
            
            # Back hair area
            back_center = (width // 2, int(height * 0.35))
            back_axes = (int(width * 0.35), int(height * 0.3))
            cv2.ellipse(mask, back_center, back_axes, 0, 0, 360, 255, -1)
            
            # Smooth the mask
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            return mask
            
        except Exception as e:
            logger.warning(f"Hair mask creation failed: {e}")
            # Return simple mask as fallback
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (width // 2, height // 2), min(width, height) // 3, 255, -1)
            return mask
    
    def _apply_hair_style(self, image: np.ndarray, style: str, 
                         hair_mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply specific hair style transformation"""
        try:
            if style == 'straight':
                return self._apply_straight_style(image, hair_mask, intensity)
            elif style == 'wavy':
                return self._apply_wavy_style(image, hair_mask, intensity)
            elif style == 'curly':
                return self._apply_curly_style(image, hair_mask, intensity)
            elif style == 'coily':
                return self._apply_coily_style(image, hair_mask, intensity)
            elif style == 'updo':
                return self._apply_updo_style(image, hair_mask, intensity)
            elif style == 'braided':
                return self._apply_braided_style(image, hair_mask, intensity)
            else:
                logger.warning(f"Unknown hair style: {style}")
                return image
                
        except Exception as e:
            logger.warning(f"Hair style application failed: {e}")
            return image
    
    def _apply_straight_style(self, image: np.ndarray, hair_mask: np.ndarray, 
                             intensity: float) -> np.ndarray:
        """Apply straight hair style"""
        try:
            result = image.copy()
            
            # Smooth the hair area
            smoothed = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Apply smoothing only to hair area
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = intensity * 0.6
            
            result = image * (1 - mask_3d * alpha) + smoothed * mask_3d * alpha
            
            # Add shine effect
            shine_effect = self._create_shine_effect(image, hair_mask)
            shine_alpha = intensity * 0.3
            
            result = result * (1 - mask_3d * shine_alpha) + shine_effect * mask_3d * shine_alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Straight style application failed: {e}")
            return image
    
    def _apply_wavy_style(self, image: np.ndarray, hair_mask: np.ndarray, 
                          intensity: float) -> np.ndarray:
        """Apply wavy hair style"""
        try:
            result = image.copy()
            height, width = image.shape[:2]
            
            # Create wave pattern
            wave_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Generate multiple wave lines
            for i in range(5):
                y_offset = int(height * 0.2) + i * int(height * 0.15)
                
                # Create wavy line
                points = []
                for x in range(0, width, 5):
                    wave_x = x
                    wave_y = y_offset + int(20 * intensity * math.sin(x * 0.02))
                    if 0 <= wave_y < height:
                        points.append((wave_x, wave_y))
                
                if len(points) > 1:
                    # Draw wavy line
                    for j in range(len(points) - 1):
                        cv2.line(wave_mask, points[j], points[j + 1], 255, 3)
            
            # Apply wave effect
            wave_effect = self._create_wave_effect(image, wave_mask)
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = intensity * 0.5
            
            result = image * (1 - mask_3d * alpha) + wave_effect * mask_3d * alpha
            
            # Add texture
            texture_effect = self._create_texture_effect(image, hair_mask, 'medium')
            texture_alpha = intensity * 0.4
            
            result = result * (1 - mask_3d * texture_alpha) + texture_effect * mask_3d * texture_alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Wavy style application failed: {e}")
            return image
    
    def _apply_curly_style(self, image: np.ndarray, hair_mask: np.ndarray, 
                           intensity: float) -> np.ndarray:
        """Apply curly hair style"""
        try:
            result = image.copy()
            height, width = image.shape[:2]
            
            # Create curl pattern
            curl_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Generate multiple curl areas
            for i in range(8):
                x_center = int(width * 0.2) + i * int(width * 0.1)
                y_center = int(height * 0.25) + (i % 2) * int(height * 0.1)
                
                # Create spiral curl
                radius = int(min(width, height) * 0.08)
                for angle in range(0, 720, 10):  # 2 full rotations
                    rad = math.radians(angle)
                    spiral_radius = radius * (1 - angle / 720)
                    
                    x = int(x_center + spiral_radius * math.cos(rad))
                    y = int(y_center + spiral_radius * math.sin(rad))
                    
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(curl_mask, (x, y), 2, 255, -1)
            
            # Apply curl effect
            curl_effect = self._create_curl_effect(image, curl_mask)
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = intensity * 0.7
            
            result = image * (1 - mask_3d * alpha) + curl_effect * mask_3d * alpha
            
            # Add high texture
            texture_effect = self._create_texture_effect(image, hair_mask, 'high')
            texture_alpha = intensity * 0.6
            
            result = result * (1 - mask_3d * texture_alpha) + texture_effect * mask_3d * texture_alpha
            
            # Add volume
            volume_effect = self._create_volume_effect(image, hair_mask)
            volume_alpha = intensity * 0.4
            
            result = result * (1 - mask_3d * volume_alpha) + volume_effect * mask_3d * volume_alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Curly style application failed: {e}")
            return image
    
    def _apply_coily_style(self, image: np.ndarray, hair_mask: np.ndarray, 
                           intensity: float) -> np.ndarray:
        """Apply coily hair style"""
        try:
            result = image.copy()
            height, width = image.shape[:2]
            
            # Create tight coil pattern
            coil_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Generate multiple tight coils
            for i in range(12):
                x_center = int(width * 0.15) + i * int(width * 0.07)
                y_center = int(height * 0.2) + (i % 3) * int(height * 0.08)
                
                # Create tight spiral coil
                radius = int(min(width, height) * 0.05)
                for angle in range(0, 1080, 15):  # 3 full rotations
                    rad = math.radians(angle)
                    spiral_radius = radius * (1 - angle / 1080)
                    
                    x = int(x_center + spiral_radius * math.cos(rad))
                    y = int(y_center + spiral_radius * math.sin(rad))
                    
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(coil_mask, (x, y), 1, 255, -1)
            
            # Apply coil effect
            coil_effect = self._create_coil_effect(image, coil_mask)
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = intensity * 0.8
            
            result = image * (1 - mask_3d * alpha) + coil_effect * mask_3d * alpha
            
            # Add very high texture
            texture_effect = self._create_texture_effect(image, hair_mask, 'very_high')
            texture_alpha = intensity * 0.8
            
            result = result * (1 - mask_3d * texture_alpha) + texture_effect * mask_3d * texture_alpha
            
            # Add maximum volume
            volume_effect = self._create_volume_effect(image, hair_mask)
            volume_alpha = intensity * 0.7
            
            result = result * (1 - mask_3d * volume_alpha) + volume_effect * mask_3d * volume_alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Coily style application failed: {e}")
            return image
    
    def _apply_updo_style(self, image: np.ndarray, hair_mask: np.ndarray, 
                          intensity: float) -> np.ndarray:
        """Apply updo hair style"""
        try:
            result = image.copy()
            height, width = image.shape[:2]
            
            # Create updo shape mask
            updo_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Top bun area
            bun_center = (width // 2, int(height * 0.2))
            bun_radius = int(min(width, height) * 0.12)
            cv2.circle(updo_mask, bun_center, bun_radius, 255, -1)
            
            # Hair pulled back area
            back_center = (width // 2, int(height * 0.35))
            back_axes = (int(width * 0.25), int(height * 0.2))
            cv2.ellipse(updo_mask, back_center, back_axes, 0, 0, 360, 255, -1)
            
            # Apply updo effect
            updo_effect = self._create_updo_effect(image, updo_mask)
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = intensity * 0.6
            
            result = image * (1 - mask_3d * alpha) + updo_effect * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Updo style application failed: {e}")
            return image
    
    def _apply_braided_style(self, image: np.ndarray, hair_mask: np.ndarray, 
                             intensity: float) -> np.ndarray:
        """Apply braided hair style"""
        try:
            result = image.copy()
            height, width = image.shape[:2]
            
            # Create braid pattern mask
            braid_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Generate braid pattern
            braid_center = (width // 2, int(height * 0.3))
            braid_length = int(height * 0.4)
            
            # Create braid strands
            for i in range(3):  # 3 strands
                strand_offset = (i - 1) * 8
                strand_points = []
                
                for y in range(int(height * 0.3), int(height * 0.7), 5):
                    # Wavy braid pattern
                    x = braid_center[0] + strand_offset + int(10 * intensity * math.sin(y * 0.05))
                    if 0 <= x < width:
                        strand_points.append((x, y))
                
                if len(strand_points) > 1:
                    # Draw braid strand
                    for j in range(len(strand_points) - 1):
                        cv2.line(braid_mask, strand_points[j], strand_points[j + 1], 255, 3)
            
            # Apply braid effect
            braid_effect = self._create_braid_effect(image, braid_mask)
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = intensity * 0.7
            
            result = image * (1 - mask_3d * alpha) + braid_effect * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Braided style application failed: {e}")
            return image
    
    def _apply_hair_color(self, image: np.ndarray, color: str, 
                         hair_mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply hair color transformation"""
        try:
            result = image.copy()
            
            # Get color palette
            palette = self.color_palettes[color]
            base_color = np.array(palette['base'], dtype=np.uint8)
            
            # Create color overlay
            color_layer = np.full_like(image, base_color)
            
            # Apply color transformation
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = intensity * 0.8
            
            result = image * (1 - mask_3d * alpha) + color_layer * mask_3d * alpha
            
            # Add highlights and shadows
            highlights = self._create_highlights(image, hair_mask, palette['highlights'])
            shadows = self._create_shadows(image, hair_mask, palette['shadows'])
            
            # Apply highlights
            highlight_alpha = intensity * 0.3
            result = result * (1 - mask_3d * highlight_alpha) + highlights * mask_3d * highlight_alpha
            
            # Apply shadows
            shadow_alpha = intensity * 0.2
            result = result * (1 - mask_3d * shadow_alpha) + shadows * mask_3d * shadow_alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Hair color application failed: {e}")
            return image
    
    def _apply_texture_effects(self, image: np.ndarray, style: str, 
                             hair_mask: np.ndarray, intensity: float) -> np.ndarray:
        """Apply texture effects based on hair style"""
        try:
            style_config = self.styles[style]
            texture_type = style_config['texture']
            
            # Get texture pattern
            pattern = self.texture_patterns[texture_type]
            
            # Apply texture
            texture_effect = self._create_texture_effect(image, hair_mask, texture_type)
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = intensity * pattern['noise_level']
            
            result = image * (1 - mask_3d * alpha) + texture_effect * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Texture effects application failed: {e}")
            return image
    
    def _create_shine_effect(self, image: np.ndarray, hair_mask: np.ndarray) -> np.ndarray:
        """Create shine effect for hair"""
        try:
            height, width = image.shape[:2]
            
            # Create shine mask (bright areas)
            shine_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Add bright highlights
            for i in range(5):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                cv2.circle(shine_mask, (x, y), random.randint(10, 20), 255, -1)
            
            # Create shine effect
            shine_effect = np.full_like(image, (255, 255, 255))
            
            return shine_effect
            
        except Exception as e:
            logger.warning(f"Shine effect creation failed: {e}")
            return image
    
    def _create_wave_effect(self, image: np.ndarray, wave_mask: np.ndarray) -> np.ndarray:
        """Create wave effect for hair"""
        try:
            # Apply slight distortion to create wave effect
            height, width = image.shape[:2]
            
            # Create displacement map
            displacement_x = np.zeros((height, width), dtype=np.float32)
            displacement_y = np.zeros((height, width), dtype=np.float32)
            
            # Add wave displacement
            for y in range(height):
                for x in range(width):
                    if wave_mask[y, x] > 0:
                        displacement_x[y, x] = 5 * math.sin(y * 0.1)
                        displacement_y[y, x] = 3 * math.cos(x * 0.1)
            
            # Apply displacement
            result = cv2.remap(image, displacement_x, displacement_y, cv2.INTER_LINEAR)
            
            return result
            
        except Exception as e:
            logger.warning(f"Wave effect creation failed: {e}")
            return image
    
    def _create_curl_effect(self, image: np.ndarray, curl_mask: np.ndarray) -> np.ndarray:
        """Create curl effect for hair"""
        try:
            # Apply more pronounced distortion for curls
            height, width = image.shape[:2]
            
            # Create displacement map
            displacement_x = np.zeros((height, width), dtype=np.float32)
            displacement_y = np.zeros((height, width), dtype=np.float32)
            
            # Add curl displacement
            for y in range(height):
                for x in range(width):
                    if curl_mask[y, x] > 0:
                        angle = math.atan2(y - height//2, x - width//2)
                        radius = math.sqrt((x - width//2)**2 + (y - height//2)**2)
                        displacement_x[y, x] = 8 * math.cos(angle + radius * 0.1)
                        displacement_y[y, x] = 8 * math.sin(angle + radius * 0.1)
            
            # Apply displacement
            result = cv2.remap(image, displacement_x, displacement_y, cv2.INTER_LINEAR)
            
            return result
            
        except Exception as e:
            logger.warning(f"Curl effect creation failed: {e}")
            return image
    
    def _create_coil_effect(self, image: np.ndarray, coil_mask: np.ndarray) -> np.ndarray:
        """Create coil effect for hair"""
        try:
            # Apply tight spiral distortion for coils
            height, width = image.shape[:2]
            
            # Create displacement map
            displacement_x = np.zeros((height, width), dtype=np.float32)
            displacement_y = np.zeros((height, width), dtype=np.float32)
            
            # Add coil displacement
            for y in range(height):
                for x in range(width):
                    if coil_mask[y, x] > 0:
                        angle = math.atan2(y - height//2, x - width//2)
                        radius = math.sqrt((x - width//2)**2 + (y - height//2)**2)
                        displacement_x[y, x] = 12 * math.cos(angle + radius * 0.2)
                        displacement_y[y, x] = 12 * math.sin(angle + radius * 0.2)
            
            # Apply displacement
            result = cv2.remap(image, displacement_x, displacement_y, cv2.INTER_LINEAR)
            
            return result
            
        except Exception as e:
            logger.warning(f"Coil effect creation failed: {e}")
            return image
    
    def _create_updo_effect(self, image: np.ndarray, updo_mask: np.ndarray) -> np.ndarray:
        """Create updo effect for hair"""
        try:
            # Apply volume and texture for updo
            result = image.copy()
            
            # Add volume by lightening the updo area
            mask_3d = cv2.cvtColor(updo_mask, cv2.COLOR_GRAY2BGR) / 255.0
            lightened = cv2.addWeighted(image, 1.0, image, 0.2, 20)
            
            alpha = 0.4
            result = result * (1 - mask_3d * alpha) + lightened * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Updo effect creation failed: {e}")
            return image
    
    def _create_braid_effect(self, image: np.ndarray, braid_mask: np.ndarray) -> np.ndarray:
        """Create braid effect for hair"""
        try:
            # Apply texture and definition for braids
            result = image.copy()
            
            # Add texture by darkening the braid lines
            mask_3d = cv2.cvtColor(braid_mask, cv2.COLOR_GRAY2BGR) / 255.0
            darkened = cv2.addWeighted(image, 0.8, image, 0, -20)
            
            alpha = 0.6
            result = result * (1 - mask_3d * alpha) + darkened * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Braid effect creation failed: {e}")
            return image
    
    def _create_texture_effect(self, image: np.ndarray, hair_mask: np.ndarray, 
                              texture_type: str) -> np.ndarray:
        """Create texture effect for hair"""
        try:
            height, width = image.shape[:2]
            
            # Create noise pattern
            noise = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            noise = cv2.GaussianBlur(noise, (5, 5), 0)
            
            # Apply noise to hair area
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            noise_3d = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
            
            # Blend noise with original
            alpha = 0.3
            result = image * (1 - mask_3d * alpha) + noise_3d * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Texture effect creation failed: {e}")
            return image
    
    def _create_volume_effect(self, image: np.ndarray, hair_mask: np.ndarray) -> np.ndarray:
        """Create volume effect for hair"""
        try:
            # Add volume by lightening and adding highlights
            result = image.copy()
            
            # Lighten hair area slightly
            mask_3d = cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) / 255.0
            lightened = cv2.addWeighted(image, 1.0, image, 0.1, 10)
            
            alpha = 0.3
            result = result * (1 - mask_3d * alpha) + lightened * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Volume effect creation failed: {e}")
            return image
    
    def _create_highlights(self, image: np.ndarray, hair_mask: np.ndarray, 
                          highlight_colors: List[Tuple[int, int, int]]) -> np.ndarray:
        """Create highlight effect for hair"""
        try:
            result = image.copy()
            
            # Create highlight mask (random bright spots)
            highlight_mask = np.zeros_like(hair_mask)
            
            for _ in range(20):
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                if hair_mask[y, x] > 0:
                    cv2.circle(highlight_mask, (x, y), random.randint(5, 15), 255, -1)
            
            # Apply highlights
            highlight_color = random.choice(highlight_colors)
            highlight_layer = np.full_like(image, highlight_color)
            
            mask_3d = cv2.cvtColor(highlight_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = 0.4
            
            result = result * (1 - mask_3d * alpha) + highlight_layer * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Highlight creation failed: {e}")
            return image
    
    def _create_shadows(self, image: np.ndarray, hair_mask: np.ndarray, 
                       shadow_colors: List[Tuple[int, int, int]]) -> np.ndarray:
        """Create shadow effect for hair"""
        try:
            result = image.copy()
            
            # Create shadow mask (darker areas)
            shadow_mask = np.zeros_like(hair_mask)
            
            for _ in range(15):
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                if hair_mask[y, x] > 0:
                    cv2.circle(shadow_mask, (x, y), random.randint(8, 20), 255, -1)
            
            # Apply shadows
            shadow_color = random.choice(shadow_colors)
            shadow_layer = np.full_like(image, shadow_color)
            
            mask_3d = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR) / 255.0
            alpha = 0.3
            
            result = result * (1 - mask_3d * alpha) + shadow_layer * mask_3d * alpha
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Shadow creation failed: {e}")
            return image
    
    def _apply_final_adjustments(self, image: np.ndarray, style: str, intensity: float) -> np.ndarray:
        """Apply final adjustments to the hair transformation"""
        try:
            result = image.copy()
            
            # Style-specific adjustments
            if style in ['curly', 'coily']:
                # Enhance contrast for textured styles
                result = cv2.addWeighted(result, 1.1, result, 0, -10)
            
            elif style == 'straight':
                # Smooth edges for sleek styles
                result = cv2.bilateralFilter(result, 9, 75, 75)
            
            # Final brightness adjustment
            if intensity > 0.7:
                # Slight brightness boost for dramatic styles
                result = cv2.addWeighted(result, 1.0, result, 0.05, 3)
            
            return result
            
        except Exception as e:
            logger.warning(f"Final adjustments failed: {e}")
            return image
    
    def get_available_styles(self) -> List[str]:
        """Get list of available hair styles"""
        return list(self.styles.keys())
    
    def get_available_colors(self) -> List[str]:
        """Get list of available hair colors"""
        return list(self.color_palettes.keys())
    
    def get_style_info(self, style: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific style"""
        return self.styles.get(style)
    
    def get_color_palette(self, color: str) -> Optional[Dict[str, List[Tuple[int, int, int]]]]:
        """Get color palette for a specific color"""
        return self.color_palettes.get(color)
    
    def get_hair_stats(self) -> Dict[str, Any]:
        """Get statistics about the hair styling system"""
        return {
            'total_styles': len(self.styles),
            'total_colors': len(self.color_palettes),
            'available_styles': list(self.styles.keys()),
            'available_colors': list(self.color_palettes.keys())
        }
