import cv2
import numpy as np
import os
import sys
import joblib
from numpy.typing import NDArray
from typing import Optional, Any

# --- Add src to Python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports from our Clean Architecture ---
from src.domain.entities.hand import Hand
from src.domain.interfaces.hand_detector import IHandDetector
from src.infrastructure.detectors.mediapipe_detector import MediaPipeHandDetector
from src.application.services.feature_extractor import FeatureExtractor
# --- NEW: Import our smoother and alert state ---
from src.application.services.prediction_smoother import PredictionSmoother, AlertState

# --- Constants ---
MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'gesture_model.joblib')
ImageType = NDArray[np.uint8]

def load_model(path: str) -> Any:
    if not os.path.exists(path):
        print(f"Error: Model file not found at {path}")
        print("Please run 'scripts/train_model.py' first.")
        return None
    try:
        model_data = joblib.load(path)
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def draw_alert(frame: ImageType, state: AlertState) -> None:
    """Draws the visual alert based on the 3-phase state."""
    h, w, _ = frame.shape
    
    if state == AlertState.CONFIRMING:
        # Phase 2: Confirmation
        text = "Confirming HELP signal..."
        color = (0, 165, 255) # Orange
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, text, (int(w * 0.1), h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    elif state == AlertState.ACTIVE:
        # Phase 3: Active Alert
        text = "EMERGENCY: HELP SIGNAL DETECTED"
        color = (0, 0, 255) # Red
        # Full red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
        # Blend the overlay
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, text, (int(w * 0.1), int(h / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

def main():
    print("🚀 Starting Real-Time Gesture Recognition (v2 with Alert)...")
    
    # 1. --- Load Model ---
    model_data = load_model(MODEL_FILE)
    if model_data is None: return
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    
    print(f"✅ Model loaded. Detecting {len(label_encoder.classes_)} gestures:")
    print(f"   {label_encoder.classes_}")

    # 2. --- Setup Components (DI) ---
    detector: IHandDetector = MediaPipeHandDetector()
    extractor: FeatureExtractor = FeatureExtractor()
    # --- NEW: Initialize the smoother ---
    smoother = PredictionSmoother(
        window_size=10, 
        stability_threshold=7,
        alert_confirm_time=2.0,  # 2 seconds to confirm
        alert_active_time=4.0    # 4 seconds to activate
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    print("-" * 30)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # 3. --- Detect Hand ---
        hand: Optional[Hand] = detector.detect(frame)
        
        raw_prediction = "NONE" # Default to NONE
        text_color = (0, 0, 255) # Red

        if hand:
            # 4. --- Extract, Predict ---
            features = extractor.extract_features(hand)
            features_2d = features.reshape(1, -1)
            prediction_index = model.predict(features_2d)[0]
            raw_prediction = label_encoder.inverse_transform([prediction_index])[0]

            # --- Visual Feedback (Hand) ---
            h, w, _ = frame.shape
            for landmark in hand.landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        
        # 5. --- NEW: Update and Use Smoother ---
        smoother.update(raw_prediction)
        stable_prediction = smoother.get_stable_prediction()
        alert_state = smoother.get_alert_state()

        # 6. --- NEW: Update Display Logic ---
        if stable_prediction != "NONE":
            display_text = stable_prediction.upper()
            text_color = (0, 255, 0) # Green
        else:
            display_text = "..." # Show ... instead of "NONE"
            text_color = (255, 255, 255) # White

        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1) # Black bar
        cv2.putText(frame, display_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

        # --- NEW: Draw the alert on top ---
        draw_alert(frame, alert_state)

        cv2.imshow("Gesture Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    print("Shutting down...")
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()