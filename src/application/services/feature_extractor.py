import numpy as np
from numpy.typing import NDArray

from src.domain.entities.hand import Hand

# These are the standard MediaPipe landmark indices for each finger
# We define them here to make the code readable
FINGER_JOINTS_INDICES = {
    'thumb': [0, 1, 2, 3, 4],
    'index': [0, 5, 6, 7, 8],
    'middle': [0, 9, 10, 11, 12],
    'ring': [0, 13, 14, 15, 16],
    'pinky': [0, 17, 18, 19, 20],
}
# We will calculate 4 angles for each of the 5 fingers = 20 features
TOTAL_FEATURES = 20


class FeatureExtractor:
    """
    A service that extracts a feature vector (a list of numbers)
    from a 'Hand' entity.
    
    This is the core of our gesture-invariant ML approach.
    """

    def _calculate_angle(self, p0_idx: int, p1_idx: int, p2_idx: int, hand: Hand) -> float:
        """
        Calculates the angle (in radians) at p1 formed by p0-p1-p2.
        This implements the vector math we just discussed.
        """
        # 1. Get the 3D coordinate arrays for each point
        p0 = hand.get_landmark(p0_idx).to_array()
        p1 = hand.get_landmark(p1_idx).to_array()
        p2 = hand.get_landmark(p2_idx).to_array()

        # 2. Get the two vectors (A and B)
        # Vector A = P0 -> P1
        vec_a = p1 - p0
        # Vector B = P1 -> P2
        vec_b = p2 - p1

        # 3. Calculate the angle using the dot product formula
        # cos(theta) = (A . B) / (|A| * |B|)
        
        # We use np.dot for the dot product
        dot_product = np.dot(vec_a, vec_b)
        
        # We use np.linalg.norm for the magnitudes |A| and |B|
        mag_a = np.linalg.norm(vec_a)
        mag_b = np.linalg.norm(vec_b)

        # Prevent division by zero if a vector has zero length
        if mag_a == 0 or mag_b == 0:
            return 0.0

        # Clamp the value to [-1.0, 1.0] to avoid floating point errors
        # with np.arccos
        cosine_angle = np.clip(dot_product / (mag_a * mag_b), -1.0, 1.0)
        
        # 4. Solve for theta
        angle = np.arccos(cosine_angle)
        
        # Return the angle in radians
        return angle

    def extract_features(self, hand: Hand) -> NDArray[np.float64]:
        """
        Creates a 1D feature vector (a "fingerprint") for a given hand.
        We will calculate 4 angles for each of the 5 fingers.
        """
        features: list[float] = []
        
        for finger_name, indices in FINGER_JOINTS_INDICES.items(): # type: ignore
            # We calculate 4 angles for each finger
            # e.g., for 'index':
            # Angle 1: 0-5-6 (Wrist -> MCP -> PIP)
            # Angle 2: 5-6-7 (MCP -> PIP -> DIP)
            # Angle 3: 6-7-8 (PIP -> DIP -> TIP)
            #
            # The thumb is slightly different, but the principle is the same.
            # We use 4 points to create 3 angles.
            for i in range(len(indices) - 2):
                p0_idx = indices[i]
                p1_idx = indices[i+1]
                p2_idx = indices[i+2]
                
                angle = self._calculate_angle(p0_idx, p1_idx, p2_idx, hand)
                features.append(angle)

        # For the thumb, we add a 4th angle (e.g., 2-3-4)
        thumb_indices = FINGER_JOINTS_INDICES['thumb']
        angle = self._calculate_angle(
            thumb_indices[2], thumb_indices[3], thumb_indices[4], hand
        )
        features.append(angle)

        return np.array(features, dtype=np.float64)