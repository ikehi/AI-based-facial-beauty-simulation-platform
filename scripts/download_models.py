"""
Download pre-trained models for AI Beauty Platform
"""

import os
import requests
import zipfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs and file paths
MODELS = {
    'shape_predictor_68_face_landmarks.dat': {
        'url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'path': 'models/shape_predictor_68_face_landmarks.dat',
        'compressed': True
    },
    'mmod_human_face_detector.dat': {
        'url': 'http://dlib.net/files/mmod_human_face_detector.dat.bz2',
        'path': 'models/mmod_human_face_detector.dat',
        'compressed': True
    }
}

def download_file(url: str, filepath: str, chunk_size: int = 8192):
    """
    Download a file from URL
    
    Args:
        url: Download URL
        filepath: Local file path
        chunk_size: Chunk size for download
    """
    try:
        logger.info(f"Downloading {url} to {filepath}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Download completed: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def extract_bz2_file(filepath: str):
    """
    Extract .bz2 compressed file
    
    Args:
        filepath: Path to compressed file
    """
    try:
        import bz2
        
        logger.info(f"Extracting {filepath}")
        
        # Read compressed file
        with bz2.open(filepath, 'rb') as f_in:
            content = f_in.read()
        
        # Write uncompressed file
        output_path = filepath.replace('.bz2', '')
        with open(output_path, 'wb') as f_out:
            f_out.write(content)
        
        # Remove compressed file
        os.remove(filepath)
        
        logger.info(f"Extraction completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting {filepath}: {e}")
        return False

def download_models():
    """
    Download all required models
    """
    logger.info("Starting model download...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    success_count = 0
    total_count = len(MODELS)
    
    for model_name, model_info in MODELS.items():
        url = model_info['url']
        filepath = model_info['path']
        compressed = model_info.get('compressed', False)
        
        # Check if file already exists
        if os.path.exists(filepath):
            logger.info(f"Model already exists: {filepath}")
            success_count += 1
            continue
        
        # Download file
        if compressed:
            # Download compressed file
            compressed_path = filepath + '.bz2'
            if download_file(url, compressed_path):
                # Extract file
                if extract_bz2_file(compressed_path):
                    success_count += 1
        else:
            # Download uncompressed file
            if download_file(url, filepath):
                success_count += 1
    
    logger.info(f"Download completed: {success_count}/{total_count} models downloaded successfully")
    
    if success_count == total_count:
        logger.info("All models downloaded successfully!")
    else:
        logger.warning(f"Some models failed to download. {total_count - success_count} models missing.")

def create_placeholder_models():
    """
    Create placeholder model files for testing
    """
    logger.info("Creating placeholder models for testing...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Create placeholder files
    placeholder_models = [
        'models/shape_predictor_68_face_landmarks.dat',
        'models/mmod_human_face_detector.dat'
    ]
    
    for model_path in placeholder_models:
        if not os.path.exists(model_path):
            # Create empty file
            with open(model_path, 'wb') as f:
                f.write(b'PLACEHOLDER_MODEL')
            logger.info(f"Created placeholder: {model_path}")

def main():
    """
    Main function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Download AI Beauty Platform models')
    parser.add_argument('--placeholder', action='store_true', 
                       help='Create placeholder models for testing')
    parser.add_argument('--download', action='store_true', 
                       help='Download actual models')
    
    args = parser.parse_args()
    
    if args.placeholder:
        create_placeholder_models()
    elif args.download:
        download_models()
    else:
        # Default: try to download, fall back to placeholders
        logger.info("Attempting to download models...")
        download_models()
        
        # Check if any models are missing
        missing_models = []
        for model_name, model_info in MODELS.items():
            if not os.path.exists(model_info['path']):
                missing_models.append(model_name)
        
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            logger.info("Creating placeholder models for testing...")
            create_placeholder_models()

if __name__ == "__main__":
    main() 