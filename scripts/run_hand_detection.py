import cv2
import numpy as np
import sys
from typing import Optional
from numpy.typing import NDArray

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.domain.entities.hand import Hand
from src.domain.interfaces.hand_detector import IHandDetector
from src.infrastructure.detectors.mediapipe_detector import MediaPipeHandDetector

# Type alias for our image
ImageType = NDArray[np.uint8]

def main():
    print("🚀 Starting Hand Detection Test...")
    print("Press 'q' to quit.")

    # 1. --- DEPENDENCY INJECTION ---
    # We type-hint the interface (IHandDetector)
    # but initialize the concrete adapter (MediaPipeHandDetector).
    # This is the "magic" of Clean Architecture!
    detector: IHandDetector = MediaPipeHandDetector()

    # 2. --- VIDEO SOURCE (from Week 1) ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        # 3. --- USE OUR ADAPTER ---
        # This is the ONLY line that calls our system.
        # We pass in a frame, we get back our 'Hand' entity.
        hand: Optional[Hand] = detector.detect(frame) # type: ignore

        # 4. --- FEEDBACK (Console) ---
        if hand:
            print(f"Hand Detected! Left: {hand.is_left}, "
                  f"Confidence: {hand.confidence:.2%}")
            
            # 5. --- FEEDBACK (Visual) ---
            # We can now use our 'Hand' entity to draw.
            # We convert normalized (0.0-1.0) coords to pixel (0-640) coords.
            h, w, _ = frame.shape
            for landmark in hand.landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        cv2.imshow("Hand Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. --- CLEANUP ---
    print("Shutting down...")
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()