import cv2
import numpy as np
import os
import sys
import argparse  # We use this to get the gesture label from the command line
import csv
from numpy.typing import NDArray
from typing import Optional, List

# --- Add src to Python path ---
# This is the same trick as our test script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports from our Clean Architecture ---
from src.domain.entities.hand import Hand
from src.domain.interfaces.hand_detector import IHandDetector
from src.infrastructure.detectors.mediapipe_detector import MediaPipeHandDetector
from src.application.services.feature_extractor import FeatureExtractor, TOTAL_FEATURES

# --- Constants ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
DATA_FILE = os.path.join(DATA_PATH, 'gestures.csv')

# Make sure the /data/processed directory exists
os.makedirs(DATA_PATH, exist_ok=True)

ImageType = NDArray[np.uint8]

def setup_csv_file() -> None:
    """
    Creates the CSV file and writes the header if it doesn't exist.
    The header will be 'label' + 'f1', 'f2', ..., 'f20'
    """
    if not os.path.exists(DATA_FILE):
        header = ['label'] + [f'f{i+1}' for i in range(TOTAL_FEATURES)]
        with open(DATA_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    print(f"✅ Data will be saved to: {DATA_FILE}")

def save_data(label: str, features: NDArray[np.float64]) -> None:
    """
    Appends a new row of data to the CSV file.
    """
    # Create a row with the label first, then all features
    row: List[str | float] = [label] + list(features)
    
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main(gesture_label: str) -> None:
    """
    Main loop for data collection.
    """
    print(f"🚀 Starting Data Collection for gesture: '{gesture_label}'")
    print(f"    Press [s] to save a data point.")
    print(f"    Press [q] to quit.")
    print("-" * 30)

    # 1. --- Setup our components (DI) ---
    detector: IHandDetector = MediaPipeHandDetector()
    extractor: FeatureExtractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    data_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # 2. --- Use our detection system ---
        hand: Optional[Hand] = detector.detect(frame)

        if hand:
            # 3. --- Extract Features ---
            # We get the 20-point feature vector
            features = extractor.extract_features(hand)
            
            # --- Visual Feedback ---
            h, w, _ = frame.shape
            for landmark in hand.landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            
            # Display text showing detection is active
            cv2.putText(frame, f"Detecting '{gesture_label}'", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Press [s] to save | Count: {data_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        else:
            # Hand not detected
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        # 4. --- Save Data on Key Press ---
        if key == ord('s'):
            if hand:
                save_data(gesture_label, features)
                data_count += 1
                print(f"    [Saved] Sample {data_count} for '{gesture_label}'")
            else:
                print("    [Warning] No hand detected. Data not saved.")
        
        if key == ord('q'):
            break

    # 5. --- Cleanup ---
    print("-" * 30)
    print(f"✅ Collection finished. Saved {data_count} samples for '{gesture_label}'.")
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- Argument Parsing ---
    # This lets us run: python scripts/collect_data.py my_gesture
    parser = argparse.ArgumentParser(description="Data collection script for hand gestures.")
    parser.add_argument(
        'label', 
        type=str, 
        help="The label for the gesture you are collecting (e.g., 'fist', 'peace')."
    )
    args = parser.parse_args()
    
    setup_csv_file()
    main(args.label)