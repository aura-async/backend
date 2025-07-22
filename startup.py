"""
Startup script for AuraSync Backend API
This script handles model checking and starts the FastAPI app
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("aurasync_startup")

# Import FastAPI app
from main import app

# Check for models
def check_model_files():
    """Check if model files exist and log their status"""
    base_path = Path(__file__).parent.absolute()
    
    model_files = [
        "face_shape_Ml/face_shape_model.h5",
        "body-shape-api/body_shape_model.pkl"
    ]
    
    missing = []
    for model_file in model_files:
        full_path = base_path / model_file
        if not full_path.exists():
            missing.append(model_file)
            logger.warning(f"⚠️ Model file missing: {model_file}")
        else:
            size_mb = full_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Model file found: {model_file} ({size_mb:.2f} MB)")
    
    if missing:
        logger.warning(f"⚠️ {len(missing)}/{len(model_files)} model files are missing. API will use fallbacks.")
    else:
        logger.info("✅ All model files found")
    
    return len(missing) == 0

# Check for models on startup
check_model_files()

# Run the app
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting AuraSync API on port {port}")
    uvicorn.run("startup:app", host="0.0.0.0", port=port, log_level="info")
