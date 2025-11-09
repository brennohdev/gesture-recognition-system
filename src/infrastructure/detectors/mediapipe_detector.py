import cv2
import mediapipe as mp  # type: ignore
import numpy as np
from typing import Optional, Any
from numpy.typing import NDArray

# Imports from our OWN domain
from src.domain.entities.hand import Hand, LANDMARK_COUNT
from src.domain.value_objects.coordinate import Coordinate
from src.domain.interfaces.hand_detector import IHandDetector

class MediaPipeHandDetector(IHandDetector):
    """
    An Adapter that implements the IHandDetector interface using
    Google's MediaPipe library.
    """

    _hands_model: Any  # MediaPipe Hands model (type ignored for Pylance)

    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.5):
        """
        Initializes the MediaPipe Hands model.
        """
        self._mp_hands = mp.solutions.hands  # type: ignore

        self._hands_model = self._mp_hands.Hands(  # pyright: ignore[reportUnknownMemberType]
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame: NDArray[np.uint8]) -> Optional[Hand]:
        """
        Detects a single hand in a given image frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results: Any = self._hands_model.process(rgb_frame)  # type: ignore
        
        if not results.multi_hand_landmarks:
            return None

        mp_hand_landmarks = results.multi_hand_landmarks[0]
        mp_handedness = results.multi_handedness[0]

        try:
            domain_landmarks = tuple(
                Coordinate(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z
                )
                for landmark in mp_hand_landmarks.landmark
            )
        except Exception:
            return None

        if len(domain_landmarks) != LANDMARK_COUNT:
            return None
            
        confidence = mp_handedness.classification[0].score
        is_left = mp_handedness.classification[0].label.lower() == 'left'

        return Hand(
            landmarks=domain_landmarks,
            is_left=is_left,
            confidence=confidence
        )

    def close(self) -> None:
        """
        Cleans up the MediaPipe Hands model.
        """
        self._hands_model.close()  # type: ignore
