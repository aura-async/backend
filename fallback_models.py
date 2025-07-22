"""
Fallback implementations for ML models when actual models are not available
This allows the API to run in a limited capacity even without the full models
"""
import random
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

logger = logging.getLogger("fallback_models")

# ===== FACE SHAPE ANALYSIS FALLBACKS =====

def fallback_face_shape_analysis(image_data) -> Dict[str, Any]:
    """Fallback face shape analysis that returns realistic but random data"""
    # Define possible face shapes with realistic probabilities
    face_shapes = [
        ("Oval", 0.25),
        ("Round", 0.20),
        ("Square", 0.15),
        ("Heart", 0.15),
        ("Diamond", 0.10),
        ("Rectangle", 0.10),
        ("Triangle", 0.05)
    ]
    
    # Weighted random selection
    selected_shape, _ = weighted_random_selection(face_shapes)
    confidence = random.uniform(0.75, 0.95)  # High confidence
    
    # Generate realistic facial features measurements
    features = {
        "face_width": random.uniform(120, 160),
        "face_height": random.uniform(180, 220),
        "jaw_width": random.uniform(110, 150),
        "forehead_width": random.uniform(115, 155),
        "chin_prominence": random.uniform(0.1, 0.5),
        "cheekbone_width": random.uniform(120, 165)
    }
    
    logger.info(f"Using fallback face analysis: {selected_shape} with {confidence:.2f} confidence")
    
    return {
        "face_shape": selected_shape,
        "confidence": confidence,
        "features": features,
        "using_fallback": True
    }

# ===== BODY SHAPE ANALYSIS FALLBACKS =====

def fallback_body_shape_analysis(image_data, gender: str = "female") -> Dict[str, Any]:
    """Fallback body shape analysis that returns realistic but random data"""
    
    if gender.lower() in ["male", "m"]:
        # Male body shapes
        body_shapes = [
            ("Trapezoid", 0.3),
            ("Rectangle", 0.25),
            ("Triangle", 0.2),
            ("Oval", 0.15),
            ("Inverted Triangle", 0.1)
        ]
    else:
        # Female body shapes (default)
        body_shapes = [
            ("Hourglass", 0.25),
            ("Rectangle", 0.25),
            ("Pear", 0.20),
            ("Apple", 0.15),
            ("Inverted Triangle", 0.15)
        ]
    
    # Weighted random selection
    selected_shape, _ = weighted_random_selection(body_shapes)
    confidence = random.uniform(0.8, 0.95)
    
    # Generate realistic body measurement ratios
    measurements = {
        "shoulder_to_waist_ratio": random.uniform(0.8, 1.4),
        "waist_to_hip_ratio": random.uniform(0.7, 1.1),
        "inseam_to_height_ratio": random.uniform(0.4, 0.5),
        "shoulder_width": random.uniform(36, 50),
        "hip_width": random.uniform(34, 48)
    }
    
    logger.info(f"Using fallback body analysis: {selected_shape} with {confidence:.2f} confidence")
    
    return {
        "body_type": selected_shape,
        "confidence": confidence,
        "measurements": measurements,
        "using_fallback": True
    }

# ===== SKIN TONE ANALYSIS FALLBACKS =====

def fallback_skin_tone_analysis(image_data) -> Dict[str, Any]:
    """Fallback skin tone analysis that returns realistic but random data"""
    
    # Fitzpatrick scale skin types
    skin_tones = [
        ("Type I - Very fair", 0.1),
        ("Type II - Fair", 0.2),
        ("Type III - Medium", 0.3),
        ("Type IV - Olive", 0.2),
        ("Type V - Brown", 0.15),
        ("Type VI - Dark brown to black", 0.05)
    ]
    
    # Color families
    undertones = [
        ("Cool", 0.3),
        ("Neutral", 0.4),
        ("Warm", 0.3)
    ]
    
    selected_tone, _ = weighted_random_selection(skin_tones)
    selected_undertone, _ = weighted_random_selection(undertones)
    confidence = random.uniform(0.75, 0.95)
    
    logger.info(f"Using fallback skin analysis: {selected_tone} with {selected_undertone} undertone")
    
    return {
        "skin_tone": selected_tone,
        "undertone": selected_undertone,
        "confidence": confidence,
        "rgb_value": generate_realistic_skin_rgb(selected_tone, selected_undertone),
        "using_fallback": True
    }

# ===== HELPER FUNCTIONS =====

def weighted_random_selection(items_with_weights: List[Tuple]) -> Tuple:
    """Select an item based on its weight"""
    items, weights = zip(*items_with_weights)
    total = sum(weights)
    normalized_weights = [w/total for w in weights]
    return random.choices(items, normalized_weights, k=1)[0], normalized_weights

def generate_realistic_skin_rgb(skin_tone: str, undertone: str) -> Dict[str, int]:
    """Generate a realistic RGB value for the given skin tone and undertone"""
    
    # Base RGB ranges for different skin tones
    base_ranges = {
        "Type I - Very fair": ((240, 255), (220, 240), (200, 225)),
        "Type II - Fair": ((225, 245), (200, 220), (175, 200)),
        "Type III - Medium": ((200, 225), (170, 200), (140, 170)),
        "Type IV - Olive": ((180, 200), (150, 180), (120, 150)),
        "Type V - Brown": ((150, 180), (120, 150), (90, 120)),
        "Type VI - Dark brown to black": ((90, 120), (70, 90), (60, 80))
    }
    
    # Undertone adjustments
    undertone_adjustments = {
        "Cool": (-10, 0, 10),
        "Neutral": (0, 0, 0),
        "Warm": (10, 5, -10)
    }
    
    # Get base range and undertone adjustment
    (r_min, r_max), (g_min, g_max), (b_min, b_max) = base_ranges[skin_tone]
    r_adj, g_adj, b_adj = undertone_adjustments[undertone]
    
    # Generate RGB values with adjustments
    r = max(0, min(255, random.randint(r_min, r_max) + r_adj))
    g = max(0, min(255, random.randint(g_min, g_max) + g_adj))
    b = max(0, min(255, random.randint(b_min, b_max) + b_adj))
    
    return {"r": r, "g": g, "b": b}
