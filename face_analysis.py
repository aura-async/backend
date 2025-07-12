import cv2
import numpy as np
from typing import Tuple, Dict, Optional

def detect_face_landmarks(image_path: str) -> Tuple[str, Optional[str]]:
    """
    Detect face shape using OpenCV Haar Cascade and geometric analysis
    Returns (face_shape, warning_message)
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return "Oval", "Could not read image file"
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return "Oval", "No face detected in image - using default shape"
        
        # Get the largest face (assumed to be the main subject)
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region for detailed analysis
        face_region = gray[y:y+h, x:x+w]
        
        # Calculate face measurements
        face_ratio = w / h
        face_area = w * h
        image_area = img.shape[0] * img.shape[1]
        face_area_ratio = face_area / image_area
        
        # Analyze face proportions for shape classification
        face_shape = classify_face_shape(face_ratio, face_area_ratio, w, h)
        
        return face_shape, None
        
    except Exception as e:
        return "Oval", f"Face analysis error: {str(e)}"

def classify_face_shape(face_ratio: float, area_ratio: float, width: int, height: int) -> str:
    """
    Classify face shape based on geometric measurements
    """
    # Enhanced classification logic
    if face_ratio > 1.25:
        return "Rectangle"  # Much wider than tall
    elif face_ratio > 1.15:
        return "Oval"       # Slightly wider than tall (classic oval)
    elif face_ratio > 1.05:
        # Check for round vs heart based on area
        if area_ratio > 0.15:  # Large face in image suggests round
            return "Round"
        else:
            return "Heart"
    elif face_ratio > 0.90:
        return "Diamond"    # Slightly taller than wide
    else:
        return "Heart"      # Much taller than wide

def analyze_face_symmetry(image_path: str) -> Dict:
    """
    Analyze facial symmetry and proportions
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"symmetry_score": 0.5, "error": "Could not read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {"symmetry_score": 0.5, "error": "No face detected"}
        
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region
        face_region = gray[y:y+h, x:x+w]
        
        # Simple symmetry analysis by comparing left and right halves
        mid_point = w // 2
        left_half = face_region[:, :mid_point]
        right_half = cv2.flip(face_region[:, mid_point:], 1)  # Flip right half
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate similarity using normalized cross-correlation
        correlation = cv2.matchTemplate(left_half.astype(np.float32), 
                                       right_half.astype(np.float32), 
                                       cv2.TM_CCOEFF_NORMED)
        
        symmetry_score = float(np.max(correlation))
        
        return {
            "symmetry_score": round(max(0, min(1, symmetry_score)), 2),
            "face_width": w,
            "face_height": h,
            "face_ratio": round(w/h, 2)
        }
        
    except Exception as e:
        return {"symmetry_score": 0.5, "error": str(e)}

def detect_facial_features(image_path: str) -> Dict:
    """
    Detect and analyze specific facial features
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"features": {}, "error": "Could not read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load different cascade classifiers
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {"features": {}, "error": "No face detected"}
        
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Extract face region for feature detection
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        
        # Detect smile
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        
        features = {
            "eyes_detected": len(eyes),
            "eye_positions": eyes.tolist() if len(eyes) > 0 else [],
            "smile_detected": len(smiles) > 0,
            "smile_positions": smiles.tolist() if len(smiles) > 0 else []
        }
        
        # Calculate eye distance if two eyes detected
        if len(eyes) >= 2:
            eye1_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            eye2_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            eye_distance = np.sqrt((eye1_center[0] - eye2_center[0])**2 + 
                                 (eye1_center[1] - eye2_center[1])**2)
            features["eye_distance"] = round(eye_distance, 2)
            features["eye_to_face_ratio"] = round(eye_distance / w, 2)
        
        return {"features": features}
        
    except Exception as e:
        return {"features": {}, "error": str(e)}

def get_face_shape_recommendations(face_shape: str) -> Dict:
    """
    Get styling recommendations based on face shape
    """
    recommendations = {
        "Oval": {
            "description": "Oval faces are well-balanced and can wear most styles",
            "best_hairstyles": ["Most styles work", "Side parts", "Layered cuts", "Bangs"],
            "best_glasses": ["Any frame shape", "Avoid oversized frames"],
            "makeup_tips": ["Balanced contouring", "Focus on eyes or lips", "Natural blush placement"],
            "accessories": ["Most earring styles", "Avoid very long dangling earrings"]
        },
        "Round": {
            "description": "Round faces benefit from angular styles that add length",
            "best_hairstyles": ["Long layers", "Side parts", "Volume on top", "Avoid chin-length cuts"],
            "best_glasses": ["Rectangular frames", "Angular shapes", "Avoid round frames"],
            "makeup_tips": ["Contour temples and jawline", "Highlight center of face", "Angular eyebrow shape"],
            "accessories": ["Long, angular earrings", "Avoid large round earrings"]
        },
        "Heart": {
            "description": "Heart faces have wider foreheads and benefit from balance at the jawline",
            "best_hairstyles": ["Chin-length bobs", "Side-swept bangs", "Volume at jaw level"],
            "best_glasses": ["Bottom-heavy frames", "Cat-eye shapes", "Avoid top-heavy frames"],
            "makeup_tips": ["Contour forehead", "Highlight jaw and chin", "Fuller bottom lip"],
            "accessories": ["Statement earrings", "Bottom-heavy designs"]
        },
        "Diamond": {
            "description": "Diamond faces have wide cheekbones and narrow forehead/jaw",
            "best_hairstyles": ["Full bangs", "Chin-length styles", "Volume at forehead and jaw"],
            "best_glasses": ["Oval or round frames", "Rimless styles", "Avoid narrow frames"],
            "makeup_tips": ["Highlight forehead and chin", "Contour cheekbones", "Soften angular features"],
            "accessories": ["Studs or small earrings", "Avoid wide statement pieces"]
        },
        "Rectangle": {
            "description": "Rectangle faces are longer than wide and benefit from width-adding styles",
            "best_hairstyles": ["Layered cuts", "Side parts", "Curls and waves", "Avoid long straight hair"],
            "best_glasses": ["Wide frames", "Oversized styles", "Decorative temples"],
            "makeup_tips": ["Horizontal contouring", "Wide eyeshadow application", "Fuller cheeks"],
            "accessories": ["Wide, statement earrings", "Choker necklaces"]
        }
    }
    
    return recommendations.get(face_shape, recommendations["Oval"])
