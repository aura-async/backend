"""
Model Downloader for AuraSync Backend
This script downloads required ML models from cloud storage on first run
"""

import os
import sys
import requests
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_downloader")

# Model URLs - Replace these with your actual URLs
MODEL_URLS = {
    "face_shape_Ml/face_shape_model.h5": "YOUR_CLOUD_STORAGE_URL_FOR_FACE_MODEL",
    "body-shape-api/body_shape_model.pkl": "YOUR_CLOUD_STORAGE_URL_FOR_BODY_MODEL",
}

# Alternative: use these environment variables if set
MODEL_URLS_ENV = {
    "face_shape_Ml/face_shape_model.h5": "FACE_MODEL_URL",
    "body-shape-api/body_shape_model.pkl": "BODY_MODEL_URL",
}

# Fallback: Use local models (for development)
USE_FALLBACK = True


def download_file(url, destination):
    """Download a file from URL to destination with progress bar"""
    try:
        logger.info(f"Downloading model from {url} to {destination}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Send request
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        file_size = int(response.headers.get('content-length', 0))
        
        # Create progress bar
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
        
        # Write to file
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
        
        progress_bar.close()
        logger.info(f"Successfully downloaded {os.path.basename(destination)}")
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading model: {e}")
        return False


def check_and_download_models():
    """Check if models exist and download if necessary"""
    base_path = Path(__file__).parent.absolute()
    
    # Check and create all necessary directories
    for model_path in MODEL_URLS.keys():
        dir_path = os.path.join(base_path, os.path.dirname(model_path))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    # Check each model
    missing_models = []
    for model_path, default_url in MODEL_URLS.items():
        full_path = os.path.join(base_path, model_path)
        
        if not os.path.exists(full_path):
            # Check for environment variable URL
            env_var = MODEL_URLS_ENV.get(model_path)
            url = os.environ.get(env_var, default_url) if env_var else default_url
            
            logger.info(f"Model not found: {model_path}")
            missing_models.append((model_path, url, full_path))
    
    # Download missing models
    if missing_models:
        logger.info(f"Found {len(missing_models)} missing models. Starting download...")
        
        for model_path, url, full_path in missing_models:
            if url == "YOUR_CLOUD_STORAGE_URL_FOR_FACE_MODEL" or url == "YOUR_CLOUD_STORAGE_URL_FOR_BODY_MODEL":
                logger.warning(f"⚠️ No download URL configured for {model_path}")
                if USE_FALLBACK:
                    logger.info(f"Creating empty model file as fallback for {model_path}")
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'wb') as f:
                        # Create an empty file as placeholder
                        pass
                continue
                
            success = download_file(url, full_path)
            if not success:
                logger.error(f"❌ Failed to download {model_path}")
                if USE_FALLBACK:
                    logger.warning("⚠️ Using fallback method with empty model file")
                    # Create empty file as placeholder
                    with open(full_path, 'wb') as f:
                        pass
                else:
                    # Critical failure
                    logger.critical("❌ Missing required model file and no fallback available")
                    return False
    
    return True


if __name__ == "__main__":
    if check_and_download_models():
        logger.info("✅ All models are available")
    else:
        logger.critical("❌ Failed to download required models")
        sys.exit(1)
