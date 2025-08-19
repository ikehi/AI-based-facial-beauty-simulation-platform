"""
Hair StyleGAN2 Implementation for Hair Style Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class HairStyleGAN:
    """
    Hair StyleGAN2 for hair style generation and transformation
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize Hair StyleGAN2
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize generator (simplified for now)
        self.generator = self._build_generator()
        self.generator.to(self.device)
        
        # Load pre-trained weights if available
        if model_path:
            self.load_model(model_path)
        
        # Set to evaluation mode
        self.generator.eval()
        
        # Predefined hair styles
        self.hair_styles = {
            'straight_long': 'styles/straight_long.jpg',
            'curly_short': 'styles/curly_short.jpg',
            'wavy_medium': 'styles/wavy_medium.jpg',
            'bob': 'styles/bob.jpg',
            'pixie': 'styles/pixie.jpg'
        }
        
        # Hair color options
        self.hair_colors = {
            'black': [0, 0, 0],
            'brown': [139, 69, 19],
            'blonde': [255, 215, 0],
            'red': [255, 0, 0],
            'gray': [128, 128, 128]
        }
    
    def _build_generator(self) -> nn.Module:
        """
        Build simplified StyleGAN2 generator architecture
        
        Returns:
            Generator network
        """
        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Mapping network
                self.mapping = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 512),
                    nn.LeakyReLU(0.2)
                )
                
                # Synthesis network
                self.synthesis = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 4, 1, 0),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(32, 16, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(16, 3, 4, 2, 1),
                    nn.Tanh()
                )
            
            def forward(self, z, truncation=0.7):
                # Mapping network
                w = self.mapping(z)
                
                # Truncation trick
                if truncation < 1.0:
                    w_avg = w.mean(dim=0, keepdim=True)
                    w = w_avg + truncation * (w - w_avg)
                
                # Reshape for synthesis
                x = w.view(w.shape[0], -1, 1, 1)
                
                # Synthesis network
                x = self.synthesis(x)
                
                return x
        
        return Generator()
    
    def load_model(self, model_path: str):
        """
        Load pre-trained model weights
        
        Args:
            model_path: Path to model weights
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def generate_hair_style(self, style_name: str, color: str = 'brown', 
                          truncation: float = 0.7) -> np.ndarray:
        """
        Generate hair style image
        
        Args:
            style_name: Name of hair style
            color: Hair color
            truncation: Truncation parameter for generation
            
        Returns:
            Generated hair style image
        """
        # Generate random latent vector
        z = torch.randn(1, 512).to(self.device)
        
        # Generate hair image
        with torch.no_grad():
            hair_image = self.generator(z, truncation)
        
        # Convert to numpy
        hair_image = hair_image.squeeze(0).cpu().detach().numpy()
        hair_image = np.transpose(hair_image, (1, 2, 0))
        hair_image = (hair_image + 1.0) * 127.5
        hair_image = np.clip(hair_image, 0, 255).astype(np.uint8)
        
        # Apply color transformation
        if color in self.hair_colors:
            hair_image = self._apply_hair_color(hair_image, self.hair_colors[color])
        
        return hair_image
    
    def _apply_hair_color(self, image: np.ndarray, color: List[int]) -> np.ndarray:
        """
        Apply hair color to image
        
        Args:
            image: Input image
            color: Target color [R, G, B]
            
        Returns:
            Image with applied hair color
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create color mask (assuming hair is in the middle brightness range)
        mask = cv2.inRange(hsv[:, :, 2], 50, 200)
        
        # Apply color
        result = image.copy()
        result[mask > 0] = color
        
        return result
    
    def transform_hair_style(self, source_image: np.ndarray, target_style: str, 
                           intensity: float = 1.0) -> np.ndarray:
        """
        Transform hair style in source image
        
        Args:
            source_image: Source image with face
            target_style: Target hair style name
            intensity: Transformation intensity (0.0 to 1.0)
            
        Returns:
            Image with transformed hair style
        """
        # Generate target hair style
        target_hair = self.generate_hair_style(target_style)
        
        # Resize target hair to match source image
        target_hair = cv2.resize(target_hair, (source_image.shape[1], source_image.shape[0]))
        
        # Blend hair regions
        result = self._blend_hair_regions(source_image, target_hair, intensity)
        
        return result
    
    def _blend_hair_regions(self, source: np.ndarray, target: np.ndarray, 
                           intensity: float) -> np.ndarray:
        """
        Blend hair regions between source and target images
        
        Args:
            source: Source image
            target: Target hair image
            intensity: Blend intensity
            
        Returns:
            Blended image
        """
        # Simple blending - in practice, you'd use more sophisticated hair segmentation
        result = source.copy()
        
        # Blend upper portion of image (hair region)
        hair_region_height = int(source.shape[0] * 0.4)  # Assume top 40% is hair
        
        # Create gradient mask for smooth blending
        mask = np.zeros((hair_region_height, source.shape[1]))
        for i in range(hair_region_height):
            mask[i, :] = (i / hair_region_height) * intensity
        
        mask = np.stack([mask] * 3, axis=2)
        
        # Blend hair region
        result[:hair_region_height] = (
            source[:hair_region_height] * (1 - mask) + 
            target[:hair_region_height] * mask
        ).astype(np.uint8)
        
        return result
    
    def get_available_styles(self) -> List[str]:
        """
        Get list of available hair styles
        
        Returns:
            List of available style names
        """
        return list(self.hair_styles.keys())
    
    def get_available_colors(self) -> List[str]:
        """
        Get list of available hair colors
        
        Returns:
            List of available color names
        """
        return list(self.hair_colors.keys()) 