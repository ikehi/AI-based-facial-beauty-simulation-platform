"""
Makeup Transfer GAN Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    """
    Generator network for makeup transfer
    """
    
    def __init__(self, input_channels: int = 6, output_channels: int = 3):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(input_channels, 64, 7, stride=1, padding=3)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Residual blocks
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)
        self.res5 = ResidualBlock(256)
        self.res6 = ResidualBlock(256)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Conv2d(64, output_channels, 7, stride=1, padding=3)
        
        # Normalization
        self.norm1 = nn.InstanceNorm2d(64)
        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(256)
        self.norm4 = nn.InstanceNorm2d(128)
        self.norm5 = nn.InstanceNorm2d(64)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.norm1(self.enc1(x)))
        x = F.relu(self.norm2(self.enc2(x)))
        x = F.relu(self.norm3(self.enc3(x)))
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        
        # Decoder
        x = F.relu(self.norm4(self.dec1(x)))
        x = F.relu(self.norm5(self.dec2(x)))
        x = torch.tanh(self.dec3(x))
        
        return x

class ResidualBlock(nn.Module):
    """
    Residual block for the generator
    """
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = x + residual
        return F.relu(x)

class MakeupTransferGAN:
    """
    Makeup Transfer GAN for applying makeup styles to faces
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize Makeup Transfer GAN
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.generator = Generator().to(self.device)
        
        # Load pre-trained weights if available
        if model_path:
            self.load_model(model_path)
        
        # Set to evaluation mode
        self.generator.eval()
        
        # Predefined makeup styles
        self.makeup_styles = {
            'natural': 'styles/natural_makeup.jpg',
            'glamorous': 'styles/glamorous_makeup.jpg',
            'casual': 'styles/casual_makeup.jpg',
            'evening': 'styles/evening_makeup.jpg',
            'party': 'styles/party_makeup.jpg'
        }
    
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
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
        """
        Preprocess image for the model
        
        Args:
            image: Input image as numpy array
            target_size: Target size for the model
            
        Returns:
            Preprocessed image as torch tensor
        """
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        
        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.to(self.device)
    
    def postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output to image
        
        Args:
            tensor: Model output tensor
            
        Returns:
            Postprocessed image as numpy array
        """
        # Remove batch dimension and convert to numpy
        image = tensor.squeeze(0).cpu().detach().numpy()
        
        # Convert from (C, H, W) to (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        
        # Denormalize from [-1, 1] to [0, 255]
        image = (image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def apply_makeup(self, source_image: np.ndarray, reference_image: np.ndarray, 
                    intensity: float = 1.0) -> np.ndarray:
        """
        Apply makeup from reference image to source image
        
        Args:
            source_image: Source face image
            reference_image: Reference makeup image
            intensity: Makeup intensity (0.0 to 1.0)
            
        Returns:
            Image with applied makeup
        """
        # Preprocess images
        source_tensor = self.preprocess_image(source_image)
        reference_tensor = self.preprocess_image(reference_image)
        
        # Concatenate source and reference
        input_tensor = torch.cat([source_tensor, reference_tensor], dim=1)
        
        # Generate makeup transfer
        with torch.no_grad():
            output_tensor = self.generator(input_tensor)
        
        # Blend with original based on intensity
        if intensity < 1.0:
            output_tensor = source_tensor * (1 - intensity) + output_tensor * intensity
        
        # Postprocess
        result = self.postprocess_image(output_tensor)
        
        return result
    
    def apply_makeup_style(self, source_image: np.ndarray, style_name: str, 
                          intensity: float = 1.0) -> np.ndarray:
        """
        Apply predefined makeup style to source image
        
        Args:
            source_image: Source face image
            style_name: Name of makeup style
            intensity: Makeup intensity (0.0 to 1.0)
            
        Returns:
            Image with applied makeup style
        """
        if style_name not in self.makeup_styles:
            raise ValueError(f"Unknown makeup style: {style_name}")
        
        # Load reference style image
        reference_path = self.makeup_styles[style_name]
        try:
            reference_image = cv2.imread(reference_path)
            if reference_image is None:
                raise FileNotFoundError(f"Style image not found: {reference_path}")
        except Exception as e:
            logger.error(f"Error loading style image: {e}")
            return source_image
        
        return self.apply_makeup(source_image, reference_image, intensity)
    
    def get_available_styles(self) -> List[str]:
        """
        Get list of available makeup styles
        
        Returns:
            List of available style names
        """
        return list(self.makeup_styles.keys())
    
    def add_style(self, style_name: str, style_image_path: str):
        """
        Add a new makeup style
        
        Args:
            style_name: Name of the new style
            style_image_path: Path to the style reference image
        """
        self.makeup_styles[style_name] = style_image_path
        logger.info(f"Added new makeup style: {style_name}")
    
    def remove_style(self, style_name: str):
        """
        Remove a makeup style
        
        Args:
            style_name: Name of the style to remove
        """
        if style_name in self.makeup_styles:
            del self.makeup_styles[style_name]
            logger.info(f"Removed makeup style: {style_name}")
        else:
            logger.warning(f"Style {style_name} not found") 