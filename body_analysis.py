import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

def analyze_body_shape_from_image(image_path: str) -> Dict:
    """
    Analyze body shape from uploaded image using computer vision
    Returns body type classification with confidence scores
    """
    try:
        # Load and validate image
        img = cv2.imread(image_path)
        if img is None:
            return {
                "body_type": "Rectangle",
                "confidence": 0.5,
                "error": "Could not read image",
                "measurements": {},
                "recommendations": get_body_type_recommendations("Rectangle")
            }
        
        # Detect person silhouette
        measurements = extract_body_measurements(img)
        
        if not measurements or "error" in measurements:
            # Fallback to default analysis
            return {
                "body_type": "Rectangle",
                "confidence": 0.6,
                "message": "Using default body type - upload a clear full-body image for accurate analysis",
                "measurements": {},
                "recommendations": get_body_type_recommendations("Rectangle")
            }
        
        # Classify body type based on measurements
        classification = classify_body_type_from_measurements(measurements)
        
        # Add recommendations
        classification["recommendations"] = get_body_type_recommendations(classification["body_type"])
        classification["measurements"] = measurements
        
        return classification
        
    except Exception as e:
        return {
            "body_type": "Rectangle",
            "confidence": 0.5,
            "error": str(e),
            "measurements": {},
            "recommendations": get_body_type_recommendations("Rectangle")
        }

def extract_body_measurements(img: np.ndarray) -> Dict:
    """
    Extract body measurements using contour detection and analysis
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use edge detection to find contours
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"error": "No contours found"}
        
        # Find the largest contour (assumed to be the person)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate approximate body measurements
        measurements = calculate_approximate_measurements(largest_contour, x, y, w, h)
        
        return measurements
        
    except Exception as e:
        return {"error": str(e)}

def calculate_approximate_measurements(contour: np.ndarray, x: int, y: int, w: int, h: int) -> Dict:
    """
    Calculate approximate body measurements from contour
    """
    try:
        # Divide body into sections for measurement
        total_height = h
        
        # Approximate body sections (these are rough estimates)
        shoulder_region = y + int(0.15 * h)  # Top 15% for shoulders
        bust_region = y + int(0.25 * h)      # 25% for bust
        waist_region = y + int(0.45 * h)     # 45% for waist
        hip_region = y + int(0.65 * h)       # 65% for hips
        
        # Extract width measurements at different heights
        shoulder_width = get_width_at_height(contour, shoulder_region)
        bust_width = get_width_at_height(contour, bust_region)
        waist_width = get_width_at_height(contour, waist_region)
        hip_width = get_width_at_height(contour, hip_region)
        
        # Calculate ratios for body type classification
        measurements = {
            "shoulder_width": shoulder_width,
            "bust_width": bust_width,
            "waist_width": waist_width,
            "hip_width": hip_width,
            "total_height": total_height,
            "ratios": {
                "shoulder_to_hip": shoulder_width / hip_width if hip_width > 0 else 1,
                "bust_to_waist": bust_width / waist_width if waist_width > 0 else 1,
                "waist_to_hip": waist_width / hip_width if hip_width > 0 else 1,
                "shoulder_to_waist": shoulder_width / waist_width if waist_width > 0 else 1
            }
        }
        
        return measurements
        
    except Exception as e:
        return {"error": str(e)}

def get_width_at_height(contour: np.ndarray, target_height: int) -> float:
    """
    Get the width of contour at a specific height
    """
    try:
        # Find all points at approximately the target height
        tolerance = 5  # pixels
        points_at_height = []
        
        for point in contour:
            x, y = point[0]
            if abs(y - target_height) <= tolerance:
                points_at_height.append(x)
        
        if len(points_at_height) < 2:
            return 0
        
        # Return width as difference between leftmost and rightmost points
        return max(points_at_height) - min(points_at_height)
        
    except Exception:
        return 0

def classify_body_type_from_measurements(measurements: Dict) -> Dict:
    """
    Classify body type based on extracted measurements
    """
    try:
        ratios = measurements.get("ratios", {})
        
        shoulder_to_hip = ratios.get("shoulder_to_hip", 1)
        bust_to_waist = ratios.get("bust_to_waist", 1)
        waist_to_hip = ratios.get("waist_to_hip", 1)
        shoulder_to_waist = ratios.get("shoulder_to_waist", 1)
        
        confidence = 0.7
        
        # Classification logic based on body proportion ratios
        if shoulder_to_hip > 1.15 and waist_to_hip < 0.85:
            body_type = "Inverted Triangle"
            confidence = min(0.9, 0.7 + (shoulder_to_hip - 1.15) * 2)
            
        elif waist_to_hip > 0.85 and shoulder_to_hip < 0.95:
            body_type = "Pear"
            confidence = min(0.9, 0.7 + (waist_to_hip - 0.85) * 3)
            
        elif (abs(shoulder_to_hip - 1) < 0.1 and 
              waist_to_hip < 0.8 and 
              bust_to_waist > 1.1):
            body_type = "Hourglass"
            confidence = min(0.95, 0.8 + (bust_to_waist - 1.1) * 2)
            
        elif waist_to_hip > 0.95 and abs(shoulder_to_hip - 1) < 0.1:
            body_type = "Rectangle"
            confidence = 0.8
            
        elif waist_to_hip > 0.9 and shoulder_to_hip > 1.05:
            body_type = "Apple"
            confidence = min(0.9, 0.7 + (waist_to_hip - 0.9) * 4)
            
        else:
            # Default classification based on dominant ratio
            if shoulder_to_hip > 1.1:
                body_type = "Inverted Triangle"
            elif waist_to_hip > 0.9:
                body_type = "Apple" if shoulder_to_hip > 1 else "Pear"
            else:
                body_type = "Hourglass" if bust_to_waist > 1.1 else "Rectangle"
            confidence = 0.6
        
        return {
            "body_type": body_type,
            "confidence": round(confidence, 2),
            "analysis_ratios": {k: round(v, 2) for k, v in ratios.items()},
            "success": True
        }
        
    except Exception as e:
        return {
            "body_type": "Rectangle",
            "confidence": 0.5,
            "error": str(e)
        }

def get_body_type_recommendations(body_type: str) -> Dict:
    """
    Get comprehensive styling recommendations for each body type
    """
    recommendations = {
        "Pear": {
            "description": "Hips are wider than shoulders, with a defined waist",
            "goal": "Balance the lower body by adding volume to shoulders and drawing attention upward",
            "tops": {
                "best": ["Boat necks", "Off-shoulder", "Horizontal stripes", "Bright colors", "Statement sleeves"],
                "avoid": ["Tight fitting tops", "Vertical stripes", "Dark colors on top"]
            },
            "bottoms": {
                "best": ["A-line skirts", "Straight-leg pants", "Dark colors", "Bootcut jeans"],
                "avoid": ["Skinny jeans", "Pencil skirts", "Light colored bottoms", "Cargo pants"]
            },
            "dresses": ["A-line dresses", "Fit-and-flare", "Empire waist", "Wrap dresses"],
            "accessories": ["Statement necklaces", "Scarves", "Bright belts", "Shoulder bags"],
            "styling_tips": "Draw attention to your upper body with interesting necklines and colors"
        },
        "Apple": {
            "description": "Fuller midsection with shoulders and hips roughly the same width",
            "goal": "Create waist definition and draw attention away from the midsection",
            "tops": {
                "best": ["V-necks", "Scoop necks", "Wrap tops", "Empire waist", "Flowy fabrics"],
                "avoid": ["Tight fitted tops", "Crew necks", "Horizontal stripes", "Clingy materials"]
            },
            "bottoms": {
                "best": ["High-waisted pants", "A-line skirts", "Straight-leg jeans", "Flare pants"],
                "avoid": ["Low-rise pants", "Skinny jeans", "Tight skirts"]
            },
            "dresses": ["Empire waist dresses", "A-line dresses", "Wrap dresses", "Shift dresses"],
            "accessories": ["Long necklaces", "Statement earrings", "Belts worn high", "Structured bags"],
            "styling_tips": "Emphasize your legs and décolletage while creating waist definition"
        },
        "Hourglass": {
            "description": "Balanced shoulders and hips with a defined waist",
            "goal": "Highlight curves and emphasize the natural waistline",
            "tops": {
                "best": ["Fitted tops", "Wrap styles", "V-necks", "Scoop necks", "Tailored shirts"],
                "avoid": ["Boxy tops", "Loose-fitting clothes", "High necklines that hide curves"]
            },
            "bottoms": {
                "best": ["High-waisted pants", "Pencil skirts", "Fitted jeans", "A-line skirts"],
                "avoid": ["Baggy pants", "Low-rise jeans", "Shapeless skirts"]
            },
            "dresses": ["Bodycon dresses", "Wrap dresses", "Fit-and-flare", "Sheath dresses"],
            "accessories": ["Belts to define waist", "Statement jewelry", "Fitted blazers"],
            "styling_tips": "Show off your curves with fitted, tailored pieces that highlight your waist"
        },
        "Rectangle": {
            "description": "Shoulders, waist, and hips are roughly the same width",
            "goal": "Create curves and add shape to the silhouette",
            "tops": {
                "best": ["Peplum tops", "Ruffled blouses", "Layered looks", "Textured fabrics", "Horizontal stripes"],
                "avoid": ["Straight, boxy cuts", "Shapeless tunics"]
            },
            "bottoms": {
                "best": ["Wide-leg pants", "Flared skirts", "Pleated styles", "Textured fabrics"],
                "avoid": ["Straight-leg pants", "Pencil skirts", "Tight-fitting bottoms"]
            },
            "dresses": ["Fit-and-flare dresses", "Tiered dresses", "Belted styles", "A-line dresses"],
            "accessories": ["Wide belts", "Statement jewelry", "Layered necklaces", "Textured bags"],
            "styling_tips": "Add volume and texture to create the illusion of curves"
        },
        "Inverted Triangle": {
            "description": "Shoulders are wider than hips, with little waist definition",
            "goal": "Balance broad shoulders by adding volume to the lower body",
            "tops": {
                "best": ["V-necks", "Scoop necks", "Simple styles", "Dark colors", "Minimal details"],
                "avoid": ["Shoulder pads", "Horizontal stripes", "Boat necks", "Off-shoulder styles"]
            },
            "bottoms": {
                "best": ["Wide-leg pants", "Flared jeans", "Pleated skirts", "Light colors", "Bold patterns"],
                "avoid": ["Skinny jeans", "Tight skirts", "Dark colored bottoms"]
            },
            "dresses": ["A-line dresses", "Fit-and-flare", "Empire waist", "Drop waist dresses"],
            "accessories": ["Hip belts", "Statement shoes", "Bold bottom pieces", "Crossbody bags"],
            "styling_tips": "Keep tops simple and add interest to your lower half"
        },
        "Athletic": {
            "description": "Muscular build with defined shoulders and minimal waist curve",
            "goal": "Soften angular lines and add feminine curves",
            "tops": {
                "best": ["Ruffled blouses", "Soft fabrics", "Draped tops", "Feminine details", "Flowing materials"],
                "avoid": ["Structured blazers", "Tight athletic wear", "Masculine cuts"]
            },
            "bottoms": {
                "best": ["Flowy skirts", "Wide-leg pants", "Soft fabrics", "Feminine cuts"],
                "avoid": ["Athletic shorts", "Cargo pants", "Structured pieces"]
            },
            "dresses": ["Maxi dresses", "Flowy dresses", "Soft A-line", "Romantic styles"],
            "accessories": ["Delicate jewelry", "Soft scarves", "Feminine bags", "Flowing cardigans"],
            "styling_tips": "Choose soft, flowing fabrics and feminine details to contrast your athletic build"
        }
    }
    
    return recommendations.get(body_type, recommendations["Rectangle"])

def enhanced_body_analysis(measurements: Dict) -> Dict:
    """
    Provide enhanced body analysis with detailed insights
    """
    try:
        if "ratios" not in measurements:
            return {"error": "Invalid measurements provided"}
        
        ratios = measurements["ratios"]
        
        # Calculate body shape score for each type
        scores = {
            "Pear": calculate_pear_score(ratios),
            "Apple": calculate_apple_score(ratios),
            "Hourglass": calculate_hourglass_score(ratios),
            "Rectangle": calculate_rectangle_score(ratios),
            "Inverted Triangle": calculate_inverted_triangle_score(ratios)
        }
        
        # Find best match
        best_match = max(scores, key=scores.get)
        confidence = scores[best_match]
        
        # Get second best for comparison
        scores_copy = scores.copy()
        del scores_copy[best_match]
        second_best = max(scores_copy, key=scores_copy.get) if scores_copy else None
        
        return {
            "primary_body_type": best_match,
            "confidence": round(confidence, 2),
            "secondary_type": second_best,
            "all_scores": {k: round(v, 2) for k, v in scores.items()},
            "analysis_notes": generate_analysis_notes(ratios, best_match)
        }
        
    except Exception as e:
        return {"error": str(e)}

def calculate_pear_score(ratios: Dict) -> float:
    """Calculate how well measurements match pear body type"""
    shoulder_to_hip = ratios.get("shoulder_to_hip", 1)
    waist_to_hip = ratios.get("waist_to_hip", 1)
    
    score = 0
    if shoulder_to_hip < 0.95: score += 0.3
    if waist_to_hip < 0.85: score += 0.4
    if shoulder_to_hip < 0.9: score += 0.3
    
    return min(1.0, score)

def calculate_apple_score(ratios: Dict) -> float:
    """Calculate how well measurements match apple body type"""
    waist_to_hip = ratios.get("waist_to_hip", 1)
    shoulder_to_hip = ratios.get("shoulder_to_hip", 1)
    
    score = 0
    if waist_to_hip > 0.9: score += 0.4
    if 0.95 <= shoulder_to_hip <= 1.05: score += 0.3
    if waist_to_hip > 0.95: score += 0.3
    
    return min(1.0, score)

def calculate_hourglass_score(ratios: Dict) -> float:
    """Calculate how well measurements match hourglass body type"""
    shoulder_to_hip = ratios.get("shoulder_to_hip", 1)
    waist_to_hip = ratios.get("waist_to_hip", 1)
    bust_to_waist = ratios.get("bust_to_waist", 1)
    
    score = 0
    if 0.95 <= shoulder_to_hip <= 1.05: score += 0.3
    if waist_to_hip < 0.8: score += 0.4
    if bust_to_waist > 1.1: score += 0.3
    
    return min(1.0, score)

def calculate_rectangle_score(ratios: Dict) -> float:
    """Calculate how well measurements match rectangle body type"""
    shoulder_to_hip = ratios.get("shoulder_to_hip", 1)
    waist_to_hip = ratios.get("waist_to_hip", 1)
    
    score = 0
    if 0.95 <= shoulder_to_hip <= 1.05: score += 0.4
    if waist_to_hip > 0.85: score += 0.4
    if 0.9 <= waist_to_hip <= 1.0: score += 0.2
    
    return min(1.0, score)

def calculate_inverted_triangle_score(ratios: Dict) -> float:
    """Calculate how well measurements match inverted triangle body type"""
    shoulder_to_hip = ratios.get("shoulder_to_hip", 1)
    waist_to_hip = ratios.get("waist_to_hip", 1)
    
    score = 0
    if shoulder_to_hip > 1.1: score += 0.4
    if waist_to_hip < 0.9: score += 0.3
    if shoulder_to_hip > 1.2: score += 0.3
    
    return min(1.0, score)

def generate_analysis_notes(ratios: Dict, body_type: str) -> List[str]:
    """Generate personalized analysis notes"""
    notes = []
    
    shoulder_to_hip = ratios.get("shoulder_to_hip", 1)
    waist_to_hip = ratios.get("waist_to_hip", 1)
    
    if shoulder_to_hip > 1.15:
        notes.append("You have broader shoulders which create a strong upper body silhouette")
    elif shoulder_to_hip < 0.9:
        notes.append("Your hip area is your widest point, creating a feminine lower body curve")
    
    if waist_to_hip < 0.75:
        notes.append("You have a very defined waist which is your best asset to highlight")
    elif waist_to_hip > 0.95:
        notes.append("Creating waist definition will enhance your overall silhouette")
    
    notes.append(f"Your {body_type} body type has many styling advantages when dressed correctly")
    
    return notes
