import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

def analyze_skin_tone_advanced(image_path: str) -> Dict:
    """
    Advanced skin tone analysis using multiple color space analysis
    Returns comprehensive skin tone information with confidence scores
    """
    try:
        # Load and validate image
        img = cv2.imread(image_path)
        if img is None:
            return {"skin_tone": "Neutral", "confidence": 0.5, "error": "Could not read image"}
        
        # Convert to different color spaces for analysis
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Detect face region for skin analysis
        skin_region = extract_skin_region(img_rgb)
        
        if skin_region is None or skin_region.size == 0:
            # Fallback to center region analysis
            h, w = img_rgb.shape[:2]
            center_h, center_w = h//3, w//3
            skin_region = img_rgb[center_h:2*center_h, center_w:2*center_w]
        
        # Perform multi-space color analysis
        rgb_analysis = analyze_rgb_skin_tone(skin_region)
        hsv_analysis = analyze_hsv_skin_tone(cv2.cvtColor(skin_region, cv2.COLOR_RGB2HSV))
        lab_analysis = analyze_lab_skin_tone(cv2.cvtColor(skin_region, cv2.COLOR_RGB2LAB))
        
        # Combine analyses for final classification
        final_classification = combine_skin_tone_analyses(rgb_analysis, hsv_analysis, lab_analysis)
        
        # Add color palette recommendations
        final_classification["color_recommendations"] = get_skin_tone_color_palette(
            final_classification["skin_tone"]
        )
        
        return final_classification
        
    except Exception as e:
        return {"skin_tone": "Neutral", "confidence": 0.5, "error": str(e)}

def extract_skin_region(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract skin region from face detection
    """
    try:
        # Convert to BGR for OpenCV face detection
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Get largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region with padding for better skin sampling
        padding = 0.1
        x_pad = int(w * padding)
        y_pad = int(h * padding)
        
        x_start = max(0, x + x_pad)
        y_start = max(0, y + y_pad)
        x_end = min(img_rgb.shape[1], x + w - x_pad)
        y_end = min(img_rgb.shape[0], y + h - y_pad)
        
        face_region = img_rgb[y_start:y_end, x_start:x_end]
        
        # Further filter to central face area (cheek region)
        center_h, center_w = face_region.shape[:2]
        quarter_h, quarter_w = center_h//4, center_w//4
        
        skin_region = face_region[quarter_h:3*quarter_h, quarter_w:3*quarter_w]
        
        return skin_region
        
    except Exception:
        return None

def analyze_rgb_skin_tone(skin_region: np.ndarray) -> Dict:
    """
    Analyze skin tone using RGB color space
    """
    try:
        # Calculate average RGB values
        avg_color = np.mean(skin_region.reshape(-1, 3), axis=0)
        r, g, b = avg_color
        
        # Calculate color indices
        total_color = r + g + b
        red_ratio = r / total_color if total_color > 0 else 0.33
        green_ratio = g / total_color if total_color > 0 else 0.33
        blue_ratio = b / total_color if total_color > 0 else 0.33
        
        # Calculate warmth index (red-blue difference)
        warmth_index = (r - b) / total_color if total_color > 0 else 0
        
        # Calculate undertone indicators
        yellow_undertone = (r + g - 2*b) / total_color if total_color > 0 else 0
        pink_undertone = (r + b - 2*g) / total_color if total_color > 0 else 0
        
        # Classify based on RGB analysis
        confidence = 0.7
        
        if warmth_index > 0.08 and yellow_undertone > 0.05:
            skin_tone = "Warm"
            confidence = min(0.95, 0.7 + warmth_index * 3)
        elif warmth_index < -0.05 and pink_undertone > 0.03:
            skin_tone = "Cool"
            confidence = min(0.95, 0.7 + abs(warmth_index) * 3)
        else:
            skin_tone = "Neutral"
            confidence = 0.75
        
        return {
            "method": "RGB",
            "skin_tone": skin_tone,
            "confidence": round(confidence, 2),
            "metrics": {
                "avg_r": round(r, 1),
                "avg_g": round(g, 1),
                "avg_b": round(b, 1),
                "warmth_index": round(warmth_index, 3),
                "yellow_undertone": round(yellow_undertone, 3),
                "pink_undertone": round(pink_undertone, 3)
            }
        }
        
    except Exception as e:
        return {"method": "RGB", "skin_tone": "Neutral", "confidence": 0.5, "error": str(e)}

def analyze_hsv_skin_tone(skin_hsv: np.ndarray) -> Dict:
    """
    Analyze skin tone using HSV color space
    """
    try:
        # Calculate average HSV values
        avg_hsv = np.mean(skin_hsv.reshape(-1, 3), axis=0)
        h, s, v = avg_hsv
        
        # Normalize hue to 0-360 range
        hue_degrees = h * 2  # OpenCV uses 0-179 range
        
        # Analyze hue for undertones
        # Warm undertones: 20-50 degrees (yellow-orange)
        # Cool undertones: 300-360, 0-20 degrees (red-pink)
        # Neutral: 50-300 degrees
        
        confidence = 0.7
        
        if 20 <= hue_degrees <= 50 and s > 30:  # Yellow-orange range with sufficient saturation
            skin_tone = "Warm"
            confidence = min(0.9, 0.7 + (s / 255) * 0.3)
        elif (hue_degrees >= 300 or hue_degrees <= 20) and s > 25:  # Red-pink range
            skin_tone = "Cool"
            confidence = min(0.9, 0.7 + (s / 255) * 0.3)
        else:
            skin_tone = "Neutral"
            confidence = 0.75
        
        return {
            "method": "HSV",
            "skin_tone": skin_tone,
            "confidence": round(confidence, 2),
            "metrics": {
                "hue_degrees": round(hue_degrees, 1),
                "saturation": round(s, 1),
                "value": round(v, 1)
            }
        }
        
    except Exception as e:
        return {"method": "HSV", "skin_tone": "Neutral", "confidence": 0.5, "error": str(e)}

def analyze_lab_skin_tone(skin_lab: np.ndarray) -> Dict:
    """
    Analyze skin tone using LAB color space (most accurate for skin tone)
    """
    try:
        # Calculate average LAB values
        avg_lab = np.mean(skin_lab.reshape(-1, 3), axis=0)
        l, a, b = avg_lab
        
        # Convert to standard LAB range
        l_norm = l * 100 / 255  # L: 0-100
        a_norm = a - 128        # a: -128 to +127 (green to red)
        b_norm = b - 128        # b: -128 to +127 (blue to yellow)
        
        # Analyze a and b channels for undertones
        # Positive a = red undertones, negative a = green undertones
        # Positive b = yellow undertones, negative b = blue undertones
        
        confidence = 0.8  # LAB is generally more reliable for skin analysis
        
        # Enhanced classification using both a and b channels
        if b_norm > 8 and a_norm > 2:  # Strong yellow and slight red
            skin_tone = "Warm"
            confidence = min(0.95, 0.8 + (b_norm / 30) * 0.2)
        elif b_norm < -3 and a_norm > 5:  # Blue undertones with red
            skin_tone = "Cool"
            confidence = min(0.95, 0.8 + (abs(b_norm) / 30) * 0.2)
        elif abs(b_norm) <= 8 and abs(a_norm) <= 8:  # Balanced
            skin_tone = "Neutral"
            confidence = 0.85
        else:
            # Edge cases - use dominant channel
            if abs(b_norm) > abs(a_norm):
                skin_tone = "Warm" if b_norm > 0 else "Cool"
            else:
                skin_tone = "Cool" if a_norm > 0 else "Neutral"
            confidence = 0.7
        
        return {
            "method": "LAB",
            "skin_tone": skin_tone,
            "confidence": round(confidence, 2),
            "metrics": {
                "lightness": round(l_norm, 1),
                "a_channel": round(a_norm, 1),
                "b_channel": round(b_norm, 1)
            }
        }
        
    except Exception as e:
        return {"method": "LAB", "skin_tone": "Neutral", "confidence": 0.5, "error": str(e)}

def combine_skin_tone_analyses(rgb_result: Dict, hsv_result: Dict, lab_result: Dict) -> Dict:
    """
    Combine results from multiple color space analyses
    """
    try:
        # Weight the different methods (LAB is most reliable for skin)
        weights = {"RGB": 0.25, "HSV": 0.25, "LAB": 0.50}
        
        # Count votes for each skin tone
        votes = {"Warm": 0, "Cool": 0, "Neutral": 0}
        confidence_sum = 0
        
        for result, weight in [(rgb_result, weights["RGB"]), 
                              (hsv_result, weights["HSV"]), 
                              (lab_result, weights["LAB"])]:
            if "skin_tone" in result and "confidence" in result:
                skin_tone = result["skin_tone"]
                confidence = result["confidence"]
                votes[skin_tone] += weight * confidence
                confidence_sum += confidence * weight
        
        # Determine final classification
        final_skin_tone = max(votes, key=votes.get)
        final_confidence = confidence_sum / sum(weights.values())
        
        # Compile detailed analysis
        return {
            "skin_tone": final_skin_tone,
            "confidence": round(final_confidence, 2),
            "analysis_methods": {
                "rgb": rgb_result,
                "hsv": hsv_result, 
                "lab": lab_result
            },
            "vote_distribution": {k: round(v, 2) for k, v in votes.items()},
            "success": True
        }
        
    except Exception as e:
        return {
            "skin_tone": "Neutral",
            "confidence": 0.5,
            "error": f"Analysis combination failed: {str(e)}"
        }

def get_skin_tone_color_palette(skin_tone: str) -> Dict:
    """
    Get comprehensive color palette and styling recommendations
    """
    palettes = {
        "Warm": {
            "best_colors": {
                "earth_tones": ["Rust", "Terracotta", "Burnt Orange", "Mustard Yellow", "Olive Green"],
                "warm_neutrals": ["Cream", "Camel", "Taupe", "Warm Beige", "Cognac Brown"],
                "vibrant_colors": ["Coral", "Peach", "Turquoise", "Teal", "Golden Yellow"],
                "classic_colors": ["Warm Red", "Burgundy", "Forest Green", "Navy with warm undertones"]
            },
            "avoid_colors": ["Icy colors", "Cool grays", "Stark white", "Magenta", "Cool blues"],
            "metal_preference": "Gold, brass, bronze, copper",
            "jewelry_stones": "Amber, citrine, tiger's eye, coral, turquoise",
            "makeup_undertones": "Golden, peachy, warm coral",
            "seasonal_palette": "Autumn/Spring colors work best"
        },
        "Cool": {
            "best_colors": {
                "jewel_tones": ["Emerald", "Sapphire", "Ruby", "Amethyst", "Cool Garnet"],
                "cool_neutrals": ["Pure White", "Cool Gray", "Charcoal", "Navy", "Cool Beige"],
                "pastels": ["Lavender", "Mint Green", "Icy Pink", "Cool Yellow", "Sky Blue"],
                "vibrant_colors": ["Fuchsia", "Royal Blue", "Cool Red", "Purple", "Cool Green"]
            },
            "avoid_colors": ["Orange", "Rust", "Mustard yellow", "Warm browns", "Golden colors"],
            "metal_preference": "Silver, platinum, white gold, stainless steel",
            "jewelry_stones": "Diamond, sapphire, emerald, amethyst, aquamarine",
            "makeup_undertones": "Pink, rose, berry, cool coral",
            "seasonal_palette": "Winter/Summer colors work best"
        },
        "Neutral": {
            "best_colors": {
                "versatile_colors": ["Soft Peach", "Jade Green", "Dusty Pink", "Teal", "Plum"],
                "neutral_base": ["Off-white", "Greige", "Taupe", "Soft Gray", "Cream"],
                "balanced_tones": ["True Red", "Navy", "Charcoal", "Soft Yellow", "Mint"],
                "earth_meets_cool": ["Dusty Blue", "Sage Green", "Mauve", "Soft Coral", "Lavender Gray"]
            },
            "avoid_colors": ["Colors that are too extreme in temperature", "Overly bright neons"],
            "metal_preference": "Both gold and silver work well - mix metals freely",
            "jewelry_stones": "Pearl, moonstone, labradorite, rose quartz, mixed metals",
            "makeup_undertones": "Balanced tones, peachy-pink, neutral coral",
            "seasonal_palette": "Can wear colors from all seasons with proper styling"
        }
    }
    
    return palettes.get(skin_tone, palettes["Neutral"])

def calculate_color_temperature(r: float, g: float, b: float) -> Dict:
    """
    Calculate color temperature and provide detailed color analysis
    """
    try:
        # Calculate color temperature using RGB ratios
        total = r + g + b
        if total == 0:
            return {"temperature": "neutral", "warmth_score": 0}
        
        # Warmth indicators
        red_dominance = r / total
        blue_dominance = b / total
        yellow_indicator = (r + g) / (2 * total)
        
        # Calculate warmth score (-1 to +1, where +1 is very warm, -1 is very cool)
        warmth_score = (red_dominance - blue_dominance) + (yellow_indicator - 0.33) * 2
        
        # Classify temperature
        if warmth_score > 0.15:
            temperature = "warm"
        elif warmth_score < -0.15:
            temperature = "cool"
        else:
            temperature = "neutral"
        
        return {
            "temperature": temperature,
            "warmth_score": round(warmth_score, 3),
            "red_dominance": round(red_dominance, 3),
            "blue_dominance": round(blue_dominance, 3),
            "yellow_indicator": round(yellow_indicator, 3)
        }
        
    except Exception as e:
        return {"temperature": "neutral", "warmth_score": 0, "error": str(e)}
