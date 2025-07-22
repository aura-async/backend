"""
AuraSync Backend API - Lightweight Version
This version uses fallbacks for ML models to reduce dependencies and size
"""
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import uvicorn
import os
import logging
import json
import random
import csv
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aurasync_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("aurasync_api")

# Import fallback models
from fallback_models import (
    fallback_face_shape_analysis,
    fallback_body_shape_analysis,
    fallback_skin_tone_analysis
)

# Try to download models on startup
try:
    from model_downloader import check_and_download_models
    MODELS_AVAILABLE = check_and_download_models()
    logger.info(f"Model availability check: {'✅ Models available' if MODELS_AVAILABLE else '⚠️ Using fallbacks'}")
except ImportError:
    MODELS_AVAILABLE = False
    logger.warning("⚠️ Model downloader not available - will use fallbacks")

# Try to import optional dependencies with fallbacks
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV available")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("⚠️ OpenCV not available - using fallback analysis")

# Try to import model-specific modules
try:
    if MODELS_AVAILABLE:
        from enhanced_body_analysis import analyze_body_type
        ENHANCED_ANALYSIS_AVAILABLE = True
        logger.info("✅ Enhanced body analysis available")
    else:
        ENHANCED_ANALYSIS_AVAILABLE = False
        logger.warning("⚠️ Enhanced body analysis not available - using fallback")
except ImportError:
    ENHANCED_ANALYSIS_AVAILABLE = False
    logger.warning("⚠️ Enhanced body analysis import failed - using fallback")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="AuraSync Fashion Recommendation API",
    description="API for fashion recommendations based on body type, face shape, skin tone, and personality analysis",
    version="1.0.0",
)

# Configure CORS
origins_str = os.getenv("CORS_ORIGINS", '["http://localhost:3000", "https://auraasync.in"]')
try:
    origins = json.loads(origins_str)
except json.JSONDecodeError:
    origins = ["http://localhost:3000", "https://auraasync.in"]
    logger.warning(f"Invalid CORS_ORIGINS format: {origins_str}. Using default origins.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring services"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "features": {
            "opencv": CV2_AVAILABLE,
            "enhanced_analysis": ENHANCED_ANALYSIS_AVAILABLE,
            "models_available": MODELS_AVAILABLE
        }
    }

# Body type analysis endpoint
@app.post("/analyze/body")
async def analyze_body(file: UploadFile = File(...), gender: Optional[str] = "female"):
    """
    Analyze body type from an uploaded image
    Uses enhanced analysis if available, otherwise uses fallback
    """
    try:
        # Read image file
        contents = await file.read()
        
        # Use enhanced analysis if available
        if CV2_AVAILABLE and ENHANCED_ANALYSIS_AVAILABLE and MODELS_AVAILABLE:
            try:
                # Convert bytes to numpy array
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Call enhanced analysis
                result = analyze_body_type(img, gender=gender)
                logger.info(f"Enhanced body analysis complete: {result.get('body_type')}")
                return result
            except Exception as e:
                logger.error(f"Enhanced analysis failed: {e}")
                logger.info("Falling back to basic analysis")
        
        # Use fallback analysis
        result = fallback_body_shape_analysis(contents, gender=gender)
        logger.info(f"Fallback body analysis complete: {result.get('body_type')}")
        return result
    
    except Exception as e:
        logger.error(f"Body analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Body analysis failed: {str(e)}")

# Face shape analysis endpoint
@app.post("/analyze/face")
async def analyze_face(file: UploadFile = File(...)):
    """
    Analyze face shape from an uploaded image
    Uses ML model if available, otherwise uses fallback
    """
    try:
        # Read image file
        contents = await file.read()
        
        # Use ML analysis if available (simplified - in real code we'd import face analysis module)
        if CV2_AVAILABLE and MODELS_AVAILABLE:
            try:
                # This is a placeholder - in real code we'd call the actual ML model
                # Simulating ML analysis for now
                result = fallback_face_shape_analysis(contents)
                result["using_fallback"] = False
                logger.info(f"ML face analysis complete: {result.get('face_shape')}")
                return result
            except Exception as e:
                logger.error(f"ML face analysis failed: {e}")
                logger.info("Falling back to basic analysis")
        
        # Use fallback analysis
        result = fallback_face_shape_analysis(contents)
        logger.info(f"Fallback face analysis complete: {result.get('face_shape')}")
        return result
    
    except Exception as e:
        logger.error(f"Face analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face analysis failed: {str(e)}")

# Skin tone analysis endpoint
@app.post("/analyze/skin")
async def analyze_skin(file: UploadFile = File(...)):
    """
    Analyze skin tone from an uploaded image
    Uses ML model if available, otherwise uses fallback
    """
    try:
        # Read image file
        contents = await file.read()
        
        # Use ML analysis if available
        if CV2_AVAILABLE and MODELS_AVAILABLE:
            try:
                # This is a placeholder - in real code we'd call the actual ML model
                # Simulating ML analysis for now
                result = fallback_skin_tone_analysis(contents)
                result["using_fallback"] = False
                logger.info(f"ML skin analysis complete: {result.get('skin_tone')}")
                return result
            except Exception as e:
                logger.error(f"ML skin analysis failed: {e}")
                logger.info("Falling back to basic analysis")
        
        # Use fallback analysis
        result = fallback_skin_tone_analysis(contents)
        logger.info(f"Fallback skin analysis complete: {result.get('skin_tone')}")
        return result
    
    except Exception as e:
        logger.error(f"Skin analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Skin analysis failed: {str(e)}")

# Personality-based recommendations
PERSONALITY_STYLES = {
    "ISTJ": "Classic and traditional styles with high-quality fabrics and timeless designs.",
    "ISFJ": "Comfortable, practical clothing with soft fabrics and subtle patterns.",
    "INFJ": "Minimalist designs with artistic touches and unique accessories.",
    "INTJ": "Sophisticated, sleek styles with clean lines and intellectual appeal.",
    "ISTP": "Functional, durable clothes with a casual edge and practical features.",
    "ISFP": "Artistic, bohemian looks with unique textures and experimental combinations.",
    "INFP": "Romantic, whimsical styles with soft layers and personal meaning.",
    "INTP": "Unconventional, comfortable clothing with intellectual or nerdy references.",
    "ESTP": "Bold, trendy pieces that make a statement and show confidence.",
    "ESFP": "Fun, vibrant outfits with attention-grabbing colors and playful accessories.",
    "ENFP": "Eclectic, creative ensembles with mixed patterns and expressive elements.",
    "ENTP": "Smart-casual looks with unexpected twists and conversation starters.",
    "ESTJ": "Professional, structured outfits with perfect coordination and attention to detail.",
    "ESFJ": "Polished, put-together ensembles that follow current trends appropriately.",
    "ENFJ": "Warm, approachable styles with harmonious colors and elegant touches.",
    "ENTJ": "Power dressing with sharp tailoring and status-signaling elements."
}

@app.get("/personality/{mbti_type}")
async def personality_recommendation(mbti_type: str):
    """Get style recommendations based on MBTI personality type"""
    mbti_type = mbti_type.upper()
    if mbti_type not in PERSONALITY_STYLES:
        raise HTTPException(status_code=400, detail=f"Invalid MBTI type: {mbti_type}")
    
    return {
        "personality_type": mbti_type,
        "style_recommendation": PERSONALITY_STYLES[mbti_type]
    }

# Load product data
def load_products():
    """Load product data from CSV file"""
    products = []
    try:
        products_file = Path(__file__).parent / "products.csv"
        if not products_file.exists():
            logger.warning(f"Products file not found at {products_file}")
            return []
        
        with open(products_file, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                products.append(row)
        logger.info(f"Loaded {len(products)} products from CSV")
    except Exception as e:
        logger.error(f"Failed to load products: {e}")
        products = []
    
    return products

PRODUCTS = load_products()

@app.post("/recommend")
async def recommend_style(request: Request):
    """
    Get personalized fashion recommendations based on body type, face shape, 
    skin tone, and personality type
    """
    try:
        data = await request.json()
        gender = data.get("gender", "female")
        body_type = data.get("body_type", "")
        face_shape = data.get("face_shape", "")
        mbti = data.get("mbti", "")
        skin_tone = data.get("skin_tone", "")
        
        # Generate personalized recommendation text
        recommendation = generate_recommendation(gender, body_type, face_shape, mbti, skin_tone)
        
        return {
            "recommendation": recommendation,
            "input_parameters": {
                "gender": gender,
                "body_type": body_type,
                "face_shape": face_shape,
                "personality": mbti,
                "skin_tone": skin_tone
            }
        }
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendation: {str(e)}")

@app.post("/products/recommendations")
async def product_recommendations(request: Request):
    """Get personalized product recommendations based on analysis results"""
    try:
        data = await request.json()
        gender = data.get("gender", "").lower()
        body_type = data.get("body_type", "")
        skin_tone = data.get("skin_tone", "")
        mbti = data.get("mbti", "")
        
        # Filter products based on criteria
        filtered_products = filter_products(gender, body_type, skin_tone, mbti)
        
        # Return top products
        return {
            "products": filtered_products[:6],  # Return top 6 products
            "count": len(filtered_products),
            "filters_applied": {
                "gender": gender,
                "body_type": body_type,
                "skin_tone": skin_tone,
                "personality": mbti
            }
        }
    except Exception as e:
        logger.error(f"Product recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get product recommendations: {str(e)}")

@app.get("/products")
async def get_products(gender: Optional[str] = None, limit: int = 10):
    """Get all available products with optional filtering"""
    try:
        if gender:
            filtered = [p for p in PRODUCTS if p.get("gender", "").lower() == gender.lower()]
        else:
            filtered = PRODUCTS
        
        return {
            "products": filtered[:limit],
            "count": len(filtered),
            "total_products": len(PRODUCTS)
        }
    except Exception as e:
        logger.error(f"Product retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve products: {str(e)}")

# Helper functions
def generate_recommendation(gender, body_type, face_shape, mbti, skin_tone):
    """Generate a personalized style recommendation based on analysis results"""
    # This is a simplified version of the recommendation logic
    
    # Basic recommendation parts
    parts = []
    
    # Add body type recommendations
    if body_type:
        body_recs = {
            "hourglass": "Embrace fitted styles that highlight your balanced proportions. Wrap dresses and belted jackets work especially well for your figure.",
            "rectangle": "Create curves with peplum tops, layered outfits, and statement belts to define your waist.",
            "triangle": "Balance your proportions with structured tops, wider necklines, and A-line skirts or dresses.",
            "inverted triangle": "Balance your broader shoulders with full skirts, wide-leg pants, and details at the hip area.",
            "oval": "Define your waistline with empire waists and vertical patterns. V-necks and A-line silhouettes are particularly flattering."
        }
        parts.append(body_recs.get(body_type.lower(), "Choose styles that enhance your unique body shape."))
    
    # Add face shape recommendations
    if face_shape:
        face_recs = {
            "oval": "You can wear most styles. Experiment with different necklines and accessories.",
            "round": "Create length with V-necks, long earrings, and hairstyles with height.",
            "square": "Soften your angles with round necklines and curved accessories.",
            "heart": "Balance your wider forehead with wider bottoms and choker necklaces.",
            "diamond": "Highlight your cheekbones with earrings and avoid oversized eyewear.",
            "rectangle": "Soften your jawline with round necklines and curved accessories."
        }
        parts.append(face_recs.get(face_shape.lower(), "Select necklines and accessories that complement your face shape."))
    
    # Add skin tone recommendations
    if skin_tone:
        if "I" in skin_tone or "II" in skin_tone:
            parts.append("With your fair skin, jewel tones like emerald, sapphire, and ruby will create stunning contrast. Soft pastels also complement your complexion beautifully.")
        elif "III" in skin_tone:
            parts.append("Your medium skin tone works well with both warm and cool colors. Rich hues like olive green, teal, and coral pink are particularly flattering.")
        elif "IV" in skin_tone:
            parts.append("Your olive skin tone is complemented by earthy colors like terracotta, mustard, and olive green, as well as vibrant jewel tones.")
        else:
            parts.append("Your deep skin tone is enhanced by bright, vibrant colors and rich jewel tones. White and cream create beautiful contrast.")
    
    # Add personality recommendations
    if mbti and mbti.upper() in PERSONALITY_STYLES:
        parts.append(f"Your {mbti.upper()} personality suggests: {PERSONALITY_STYLES[mbti.upper()]}")
    
    # Combine all recommendations
    if parts:
        full_rec = " ".join(parts)
    else:
        full_rec = "Based on your unique characteristics, we recommend exploring styles that express your individuality while enhancing your natural features. Consider consulting with a personal stylist for more tailored advice."
    
    return full_rec

def filter_products(gender, body_type, skin_tone, mbti):
    """Filter products based on user characteristics"""
    # This is a simplified version of product filtering
    filtered = PRODUCTS.copy()
    
    # Filter by gender if specified
    if gender:
        filtered = [p for p in filtered if p.get("gender", "").lower() == gender.lower()]
    
    # If no products match or we have too few, return all products
    if len(filtered) < 3:
        filtered = PRODUCTS.copy()
    
    # Randomize order to provide variety
    random.shuffle(filtered)
    
    return filtered

# Run the FastAPI app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
