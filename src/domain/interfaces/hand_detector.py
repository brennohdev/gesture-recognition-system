import numpy as np
from numpy.typing import NDArray
from typing import Protocol, Optional
from src.domain.entities.hand import Hand

class IHandDetector(Protocol):
    """
    An Interface (Port) defining the contract for any hand detection
    implementation.

    It decouples the application from any specific library.
    """
    
    def detect(self, frame: NDArray[np.uint8]) -> Optional[Hand]:
        """
        Detects a hand in the given frame.

        Args:
            frame (NDArray[np.uint8]): The input image frame in which to detect hands.
            
        Returns:
            A hand domain entity if a hand is detected,
            otherwise None.
        """
        ...
        
    def close(self) -> None:
        """
        Releases any resources held by the detector.
        """
        ...