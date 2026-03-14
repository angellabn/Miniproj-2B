import cv2
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model

# Initialize MediaPipe Face Detection
mp_face_detection = None
face_detection = None
USE_MEDIAPIPE = False

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        USE_MEDIAPIPE = True
    else:
        print("Warning: MediaPipe 'solutions' not found (likely Python 3.13+).")
        print("Falling back to OpenCV Haar Cascades for Face Detection temporarily.")
except ImportError:
    print("Warning: MediaPipe not found. Falling back to OpenCV Haar Cascades.")

# OpenCV Haar Cascade Fallback
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Model Loading (TensorFlow/Keras) ---
# When you have your trained model, replace these with your actual model files.

def load_video_model():
    """Load the trained CNN+LSTM model for video."""
    print("Loading video model...")
    # -------------------------------------------------------------
    # TODO: Load your trained TensorFlow weights when ready!
    # e.g.:
    # from tensorflow.keras.models import load_model
    # model = load_model("video_model.h5") 
    # -------------------------------------------------------------
    
    # TEMPORARY DUMMY FALLBACK
    # Hardcoded dummy model that predicts Fake (Class 1) 
    model = lambda x: np.array([[0.3, 0.7]]) 
    
    return model

def load_image_model():
    """Load the trained model for single images."""
    print("Loading image model...")
    # -------------------------------------------------------------
    # TODO: Load your trained TensorFlow weights when ready!
    # e.g.:
    # from tensorflow.keras.models import load_model
    # model = load_model("image_model.h5") 
    # -------------------------------------------------------------
    
    # TEMPORARY DUMMY FALLBACK
    model = load_model("Img_detector.h5")
    
    return model

# Load models globally so they're ready when API starts
VIDEO_MODEL = load_video_model()
IMAGE_MODEL = load_image_model()

# --- Preprocessing Pipeline ---

def detect_and_crop_face(image_rgb):
    """
    Use MediaPipe to detect a face in the RGB image. 
    If MediaPipe is unavailable or fails, fallback to OpenCV Haar Cascade.
    If no face is found at all, returns a center crop.
    """
    h, w, _ = image_rgb.shape
    
    # Try MediaPipe if available
    if USE_MEDIAPIPE and face_detection is not None:
        results = face_detection.process(image_rgb)
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            return pad_and_crop(image_rgb, x, y, width, height)
            
    # Try OpenCV Haar Cascade as fallback
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, width, height = faces[0]
        return pad_and_crop(image_rgb, x, y, width, height)

    # Fallback to center crop if no face detected
    size = min(h, w)
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    return image_rgb[start_y:start_y+size, start_x:start_x+size]

def pad_and_crop(image_rgb, x, y, width, height):
    """Helper to add margin and safely crop within image bounds."""
    h, w, _ = image_rgb.shape
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    
    x_min = max(0, x - margin_x)
    y_min = max(0, y - margin_y)
    x_max = min(w, x + width + margin_x)
    y_max = min(h, y + height + margin_y)
    
    face_crop = image_rgb[y_min:y_max, x_min:x_max]
    
    if face_crop.size > 0:
        return face_crop
        
    # Ultimate fallback if crop fails
    size = min(h, w)
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    return image_rgb[start_y:start_y+size, start_x:start_x+size]


def preprocess_frame(frame_rgb, target_size=(224, 224)):
    """Convert a raw frame to model-ready tensor."""
    face = detect_and_crop_face(frame_rgb)
    
    # Resize face to target input size
    face_resized = cv2.resize(face, target_size)
    
    # Convert back to PIL for standard torchvision transforms, or do numpy operations
    # Normalize (standard ImageNet normalization typically used)
    # [0, 255] -> [0, 1]
    face_normalized = face_resized.astype(np.float32) / 255.0
    
    # Standardize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    face_normalized = (face_normalized - mean) / std
    
    # TensorFlow/Keras expects (H, W, C) -> channels last. 
    # Do NOT transpose to (C, H, W) like PyTorch does.
    return face_normalized


def extract_frames(video_path, num_frames=15):
    """
    Extract perfectly distributed `num_frames` from the video file.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError("Could not read video frames")
        
    frames_to_extract = []
    
    # Calculate step to extract frames evenly
    step = max(1, total_frames // num_frames)
    
    # Ensure exactly num_frames using evenly spaced indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    extracted_faces = []
    
    current_frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while cap.isOpened() and len(extracted_faces) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in indices:
            # OpenCV loads as BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_face = preprocess_frame(frame_rgb)
            extracted_faces.append(processed_face)
            
        current_frame += 1
        
    cap.release()
    
    # Handle short videos (pad with last frame if needed)
    while len(extracted_faces) < num_frames and len(extracted_faces) > 0:
        extracted_faces.append(extracted_faces[-1])
        
    if len(extracted_faces) == 0:
        raise ValueError("Failed to extract any frames from the video.")
        
    # Shape becomes (frames, channels, height, width) : (15, 3, 224, 224)
    return np.array(extracted_faces)


# --- Inference Execution ---

def run_video_inference(video_path):
    """
    Run pipeline on a video file using the global VIDEO_MODEL.
    """
    try:
        print(f"Starting video inference pipeline for {video_path}...")
        
        # 1. Extract 15 frames + 2. Crop Face + 3. Resize/Normalize
        # Shape: (15, 3, 224, 224)
        sequence_data = extract_frames(video_path, num_frames=15)
        
        # Ensure it's a tensor and add batch dimension using numpy (for TensorFlow)
        # Shape: (1, 15, 224, 224, 3) 
        sequence_tensor = np.expand_dims(sequence_data, axis=0)
        
        print(f"Sequence shape created for TF: {sequence_tensor.shape}")
        
        # 4. Model Prediction
        # For actual TF model: outputs = VIDEO_MODEL.predict(sequence_tensor)
        # For our dummy lambda:
        outputs = VIDEO_MODEL(sequence_tensor)
        
        # Assuming output is probabilities or logits that can be directly parsed
        # Real = outputs[0][0], Fake = outputs[0][1]
        fake_prob = float(outputs[0][1])
        real_prob = float(outputs[0][0])
        
        is_deepfake = fake_prob > 0.5
        confidence = round(max(fake_prob, real_prob) * 100, 2)
        
        print(f"Video result: Deepfake={is_deepfake}, Confidence={confidence}%")
        
        return {
            "isDeepfake": is_deepfake,
            "confidence": confidence,
            "label": "Fake" if is_deepfake else "Real",
            "probabilities": {
                "real": round(real_prob * 100, 2),
                "fake": round(fake_prob * 100, 2)
            }
        }
    
    except Exception as e:
        print(f"Error in video inference pipeline: {e}")
        raise e

def run_image_inference(image_bytes):
    """
    Run pipeline on a single image.
    """
    try:
        print("Starting image inference pipeline...")
        
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        
        # Detect face and preprocess
        # Shape: (3, 224, 224)
        processed_image = preprocess_frame(image_np)
        
        # Add batch dimension using numpy (for TensorFlow)
        # Shape: (1, 224, 224, 3)
        image_tensor = np.expand_dims(processed_image, axis=0)
        
        print(f"Image tensor shape created for TF: {image_tensor.shape}")
        
        # Model Prediction
        # For actual TF model: outputs = IMAGE_MODEL.predict(image_tensor)
        outputs = IMAGE_MODEL(image_tensor)
            
        # Safely parse output depending on if model has 1 output unit (sigmoid) or 2 (softmax)
        preds = outputs[0]
        if len(preds) == 1:
            # Single output (sigmoid): usually probability of being Fake
            fake_prob = float(preds[0])
            real_prob = 1.0 - fake_prob
        else:
            # Two outputs (softmax): [Real, Fake]
            fake_prob = float(preds[1])
            real_prob = float(preds[0])
        
        is_deepfake = fake_prob > 0.5
        confidence = round(max(fake_prob, real_prob) * 100, 2)
        
        print(f"Image result: Deepfake={is_deepfake}, Confidence={confidence}%")
        
        return {
            "isDeepfake": is_deepfake,
            "confidence": confidence,
            "label": "Fake" if is_deepfake else "Real"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in image inference pipeline: {e}")
        raise e
