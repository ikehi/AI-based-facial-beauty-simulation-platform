"""
Error handling utilities for AI Beauty Platform
"""

import traceback
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from flask import jsonify, request

logger = logging.getLogger(__name__)

class AIBeautyError(Exception):
    """Base exception class for AI Beauty Platform"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'UNKNOWN_ERROR'
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format"""
        return {
            'error': True,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }

class ModelLoadError(AIBeautyError):
    """Raised when AI model fails to load"""
    pass

class FaceDetectionError(AIBeautyError):
    """Raised when face detection fails"""
    pass

class ImageProcessingError(AIBeautyError):
    """Raised when image processing fails"""
    pass

class ValidationError(AIBeautyError):
    """Raised when input validation fails"""
    pass

class ResourceNotFoundError(AIBeautyError):
    """Raised when a required resource is not found"""
    pass

def handle_api_errors(f: Callable) -> Callable:
    """Decorator to handle errors in API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except AIBeautyError as e:
            logger.warning(f"Handled error in {f.__name__}: {e.message}")
            return jsonify(e.to_dict()), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Don't expose internal errors in production
            if logger.level <= logging.DEBUG:
                error_details = {
                    'error': True,
                    'error_code': 'INTERNAL_ERROR',
                    'message': 'An unexpected error occurred',
                    'details': {
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'traceback': traceback.format_exc()
                    }
                }
            else:
                error_details = {
                    'error': True,
                    'error_code': 'INTERNAL_ERROR',
                    'message': 'An unexpected error occurred'
                }
            
            return jsonify(error_details), 500
    
    return decorated_function

def validate_image_input(image_data: str) -> bool:
    """
    Validate image input data
    
    Args:
        image_data: Base64 encoded image data
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValidationError: If validation fails
    """
    if not image_data:
        raise ValidationError("Image data is required", "MISSING_IMAGE")
    
    if not isinstance(image_data, str):
        raise ValidationError("Image data must be a string", "INVALID_IMAGE_TYPE")
    
    # Check if it's a valid base64 string
    try:
        import base64
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Try to decode
        base64.b64decode(image_data)
    except Exception:
        raise ValidationError("Invalid base64 image data", "INVALID_BASE64")
    
    return True

def validate_makeup_parameters(style: str, intensity: float) -> bool:
    """
    Validate makeup parameters
    
    Args:
        style: Makeup style name
        intensity: Makeup intensity (0.0-1.0)
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValidationError: If validation fails
    """
    valid_styles = ['natural', 'glamorous', 'casual', 'evening', 'party']
    
    if style not in valid_styles:
        raise ValidationError(
            f"Invalid makeup style. Must be one of: {valid_styles}",
            "INVALID_MAKEUP_STYLE"
        )
    
    if not isinstance(intensity, (int, float)) or not 0.0 <= intensity <= 1.0:
        raise ValidationError(
            "Makeup intensity must be a number between 0.0 and 1.0",
            "INVALID_MAKEUP_INTENSITY"
        )
    
    return True

def validate_hair_parameters(style: str, color: str, intensity: float) -> bool:
    """
    Validate hair parameters
    
    Args:
        style: Hair style name
        color: Hair color name
        intensity: Hair transformation intensity (0.0-1.0)
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValidationError: If validation fails
    """
    valid_styles = ['straight', 'wavy', 'curly', 'coily', 'updo']
    valid_colors = ['black', 'brown', 'blonde', 'red', 'gray', 'white']
    
    if style not in valid_styles:
        raise ValidationError(
            f"Invalid hair style. Must be one of: {valid_styles}",
            "INVALID_HAIR_STYLE"
        )
    
    if color not in valid_colors:
        raise ValidationError(
            f"Invalid hair color. Must be one of: {valid_colors}",
            "INVALID_HAIR_COLOR"
        )
    
    if not isinstance(intensity, (int, float)) or not 0.0 <= intensity <= 1.0:
        raise ValidationError(
            "Hair intensity must be a number between 0.0 and 1.0",
            "INVALID_HAIR_INTENSITY"
        )
    
    return True

def safe_image_processing(func: Callable) -> Callable:
    """Decorator to safely handle image processing operations"""
    @wraps(func)
    def decorated_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Image processing error in {func.__name__}: {str(e)}")
            raise ImageProcessingError(
                f"Failed to process image: {str(e)}",
                "IMAGE_PROCESSING_FAILED"
            )
    
    return decorated_function

def log_request_info(f: Callable) -> Callable:
    """Decorator to log request information"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.info(f"Request to {f.__name__} from {request.remote_addr}")
        logger.debug(f"Request data: {request.get_json(silent=True)}")
        
        try:
            result = f(*args, **kwargs)
            logger.info(f"Request to {f.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Request to {f.__name__} failed: {str(e)}")
            raise
    
    return decorated_function

def create_error_response(
    message: str,
    error_code: str = 'UNKNOWN_ERROR',
    status_code: int = 400,
    details: Dict[str, Any] = None
) -> tuple:
    """
    Create a standardized error response
    
    Args:
        message: Error message
        error_code: Error code
        status_code: HTTP status code
        details: Additional error details
        
    Returns:
        Tuple of (response, status_code)
    """
    error_data = {
        'error': True,
        'error_code': error_code,
        'message': message
    }
    
    if details:
        error_data['details'] = details
    
    return jsonify(error_data), status_code

def handle_model_loading_error(model_name: str, error: Exception) -> None:
    """
    Handle AI model loading errors gracefully
    
    Args:
        model_name: Name of the model that failed to load
        error: The error that occurred
    """
    logger.error(f"Failed to load {model_name} model: {str(error)}")
    
    if "CUDA" in str(error) or "GPU" in str(error):
        logger.warning(f"{model_name} model failed to load on GPU, falling back to CPU")
    elif "file not found" in str(error).lower():
        logger.error(f"{model_name} model file not found. Please check the model path.")
    else:
        logger.error(f"Unknown error loading {model_name} model: {str(error)}")
    
    # Log additional context
    logger.debug(f"Model loading error details: {traceback.format_exc()}")

# Global error handlers for common scenarios
def setup_global_error_handlers(app):
    """Set up global error handlers for the Flask app"""
    
    @app.errorhandler(404)
    def not_found(error):
        return create_error_response(
            "Resource not found",
            "RESOURCE_NOT_FOUND",
            404
        )
    
    @app.errorhandler(413)
    def too_large(error):
        return create_error_response(
            "File too large",
            "FILE_TOO_LARGE",
            413
        )
    
    @app.errorhandler(500)
    def internal_error(error):
        return create_error_response(
            "Internal server error",
            "INTERNAL_ERROR",
            500
        )
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {str(error)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return create_error_response(
            "An unexpected error occurred",
            "INTERNAL_ERROR",
            500
        )
