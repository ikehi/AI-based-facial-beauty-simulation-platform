"""
Hair Transformer for AI Beauty Platform
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class HairTransformer:
    """
    Hair Transformer for style and color transformation
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize Hair Transformer
        
        Args:
            model_path: Path to pre-trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize transformer network
        self.transformer = self._build_transformer()
        self.transformer.to(self.device)
        
        # Load pre-trained weights if available
        if model_path:
            self.load_model(model_path)
        
        # Set to evaluation mode
        self.transformer.eval()
        
        # Hair transformation options
        self.transformation_types = {
            'style': ['straight', 'wavy', 'curly', 'coily'],
            'length': ['short', 'medium', 'long'],
            'color': ['black', 'brown', 'blonde', 'red', 'gray', 'white']
        }
    
    def _build_transformer(self) -> nn.Module:
        """
        Build simplified hair transformer network
        
        Returns:
            Transformer network
        """
        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Simple encoder-decoder architecture
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=1, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
                
                # Style embedding
                self.style_embedding = nn.Embedding(10, 256)
                self.length_embedding = nn.Embedding(3, 256)
                self.color_embedding = nn.Embedding(6, 256)
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, 7, stride=1, padding=3),
                    nn.Tanh()
                )
            
            def forward(self, x, style_id, length_id, color_id):
                # Encode input
                features = self.encoder(x)
                
                # Get style embeddings
                style_emb = self.style_embedding(style_id).unsqueeze(-1).unsqueeze(-1)
                length_emb = self.length_embedding(length_id).unsqueeze(-1).unsqueeze(-1)
                color_emb = self.color_embedding(color_id).unsqueeze(-1).unsqueeze(-1)
                
                # Combine embeddings
                combined_emb = style_emb + length_emb + color_emb
                
                # Add to features
                features = features + combined_emb
                
                # Decode
                output = self.decoder(features)
                
                return output
        
        return Transformer()
    
    def load_model(self, model_path: str):
        """
        Load pre-trained model weights
        
        Args:
            model_path: Path to model weights
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def transform_hair(self, image: np.ndarray, style: str, length: str, 
                      color: str, intensity: float = 1.0) -> np.ndarray:
        """
        Transform hair in the image
        
        Args:
            image: Input image
            style: Hair style ('straight', 'wavy', 'curly', 'coily')
            length: Hair length ('short', 'medium', 'long')
            color: Hair color ('black', 'brown', 'blonde', 'red', 'gray', 'white')
            intensity: Transformation intensity (0.0 to 1.0)
            
        Returns:
            Image with transformed hair
        """
        # Convert style parameters to IDs
        style_id = self._get_style_id(style)
        length_id = self._get_length_id(length)
        color_id = self._get_color_id(color)
        
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        
        # Transform hair
        with torch.no_grad():
            transformed_tensor = self.transformer(
                image_tensor, 
                torch.tensor([style_id]).to(self.device),
                torch.tensor([length_id]).to(self.device),
                torch.tensor([color_id]).to(self.device)
            )
        
        # Postprocess image
        transformed_image = self._postprocess_image(transformed_tensor)
        
        # Blend with original based on intensity
        if intensity < 1.0:
            transformed_image = cv2.addWeighted(
                image, 1 - intensity, transformed_image, intensity, 0
            )
        
        return transformed_image
    
    def _get_style_id(self, style: str) -> int:
        """Get style ID from style name"""
        style_map = {
            'straight': 0, 'wavy': 1, 'curly': 2, 'coily': 3
        }
        return style_map.get(style, 0)
    
    def _get_length_id(self, length: str) -> int:
        """Get length ID from length name"""
        length_map = {
            'short': 0, 'medium': 1, 'long': 2
        }
        return length_map.get(length, 1)
    
    def _get_color_id(self, color: str) -> int:
        """Get color ID from color name"""
        color_map = {
            'black': 0, 'brown': 1, 'blonde': 2, 'red': 3, 'gray': 4, 'white': 5
        }
        return color_map.get(color, 1)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for the model
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image tensor
        """
        # Resize to 256x256
        image = cv2.resize(image, (256, 256))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def _postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output to image
        
        Args:
            tensor: Model output tensor
            
        Returns:
            Postprocessed image
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
    
    def get_available_transformations(self) -> Dict[str, List[str]]:
        """
        Get available transformation options
        
        Returns:
            Dictionary of available transformations
        """
        return self.transformation_types 