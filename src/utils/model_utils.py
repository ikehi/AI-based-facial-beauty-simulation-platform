"""
Model utilities for AI Beauty Platform
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)

class ModelUtils:
    """
    Utility class for managing AI models
    """
    
    @staticmethod
    def load_model_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                            device: str = 'cuda') -> bool:
        """
        Load model checkpoint from file
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            device: Device to load model on
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint file not found: {checkpoint_path}")
                return False
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info(f"Model loaded from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model checkpoint: {e}")
            return False
    
    @staticmethod
    def save_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, 
                            additional_info: Dict = None):
        """
        Save model checkpoint to file
        
        Args:
            model: Model to save
            checkpoint_path: Path to save checkpoint
            additional_info: Additional information to save
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            checkpoint = {
                'state_dict': model.state_dict(),
                'model_info': {
                    'type': type(model).__name__,
                    'parameters': sum(p.numel() for p in model.parameters())
                }
            }
            
            if additional_info:
                checkpoint.update(additional_info)
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Model saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving model checkpoint: {e}")
    
    @staticmethod
    def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_type': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'device': next(model.parameters()).device
        }
    
    @staticmethod
    def create_model_directory(base_path: str = 'models') -> str:
        """
        Create model directory structure
        
        Args:
            base_path: Base path for models
            
        Returns:
            Path to created directory
        """
        model_path = os.path.join(base_path, 'ai_beauty_platform')
        os.makedirs(model_path, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['face_recognition', 'makeup_ai', 'hair_ai', 'cosmetic_ai']
        for subdir in subdirs:
            os.makedirs(os.path.join(model_path, subdir), exist_ok=True)
        
        return model_path
    
    @staticmethod
    def list_available_models(model_path: str = 'models') -> Dict[str, List[str]]:
        """
        List available models in the model directory
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Dictionary of available models by category
        """
        available_models = {}
        
        if not os.path.exists(model_path):
            return available_models
        
        for category in os.listdir(model_path):
            category_path = os.path.join(model_path, category)
            if os.path.isdir(category_path):
                models = []
                for file in os.listdir(category_path):
                    if file.endswith(('.pth', '.pt', '.h5', '.pb')):
                        models.append(file)
                available_models[category] = models
        
        return available_models
    
    @staticmethod
    def validate_model_file(file_path: str) -> bool:
        """
        Validate if a model file is valid
        
        Args:
            file_path: Path to model file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False
            
            # Try to load the model file
            if file_path.endswith(('.pth', '.pt')):
                checkpoint = torch.load(file_path, map_location='cpu')
                if not isinstance(checkpoint, dict):
                    return False
                if 'state_dict' not in checkpoint and 'model_state_dict' not in checkpoint:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating model file {file_path}: {e}")
            return False
    
    @staticmethod
    def get_model_config(model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a model type
        
        Args:
            model_type: Type of model ('face_detector', 'makeup_gan', etc.)
            
        Returns:
            Model configuration dictionary
        """
        configs = {
            'face_detector': {
                'confidence_threshold': 0.5,
                'min_face_size': 30,
                'max_faces': 10
            },
            'makeup_gan': {
                'input_size': (256, 256),
                'batch_size': 1,
                'device': 'cuda'
            },
            'hair_gan': {
                'input_size': (256, 256),
                'latent_dim': 512,
                'truncation': 0.7
            },
            'cosmetic_adjuster': {
                'adjustment_range': (-0.5, 0.5),
                'smoothing_factor': 0.1
            }
        }
        
        return configs.get(model_type, {})
    
    @staticmethod
    def save_model_config(config: Dict[str, Any], config_path: str):
        """
        Save model configuration to file
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    @staticmethod
    def load_model_config(config_path: str) -> Dict[str, Any]:
        """
        Load model configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Configuration file not found: {config_path}")
                return {}
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    @staticmethod
    def check_gpu_availability() -> Dict[str, Any]:
        """
        Check GPU availability and information
        
        Returns:
            Dictionary with GPU information
        """
        gpu_info = {
            'available': torch.cuda.is_available(),
            'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        return gpu_info
    
    @staticmethod
    def optimize_model_for_inference(model: torch.nn.Module, 
                                   device: str = 'cuda') -> torch.nn.Module:
        """
        Optimize model for inference
        
        Args:
            model: Model to optimize
            device: Device to optimize for
            
        Returns:
            Optimized model
        """
        model.eval()
        model = model.to(device)
        
        # Use torch.jit for optimization if available
        try:
            if hasattr(torch.jit, 'script'):
                model = torch.jit.script(model)
                logger.info("Model optimized with TorchScript")
        except Exception as e:
            logger.warning(f"Could not optimize model with TorchScript: {e}")
        
        return model 