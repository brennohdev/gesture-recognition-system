import math
from dataclasses import dataclass
from typing import Tuple

from src.domain.value_objects.coordinate import Coordinate

LANDMARK_COUNT = 21

LandmarkList = Tuple[Coordinate, ...]

@dataclass(frozen=True)
class Hand:
    landmarks: LandmarkList
    
    is_left: bool
    
    confidence: float
    
    def __post_init__(self) -> None:
        if len(self.landmarks) != LANDMARK_COUNT:
            raise ValueError(
                f"Expected {LANDMARK_COUNT} landmarks, but got {len(self.landmarks)}"
            )
            
    def get_landmark(self, index: int) -> Coordinate:
        if not (0 <= index < LANDMARK_COUNT):
            raise IndexError(f"Landmark index {index} out of range")
        return self.landmarks[index]
    
    def distance_between(self, start_index: int, end_index: int) -> float:
        start = self.get_landmark(start_index)
        end = self.get_landmark(end_index)
        return math.sqrt(
            (end.x - start.x) ** 2 +
            (end.y - start.y) ** 2 +
            (end.z - start.z) ** 2
        )
    
    