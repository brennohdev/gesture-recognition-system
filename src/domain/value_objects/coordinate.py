from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class Coordinate:
    """
    A value object representing a single 3D point in normalized space.
    
    - frozen=True makes the dataclass immutable.
    """
    
    x: float
    y: float
    z: float
    def to_array(self) -> NDArray[np.float64]:
        """
        Converts the Coordinate to a NumPy array.
        """
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> 'Coordinate':
        if arr.shape != (3,):
            raise ValueError("Array must be of shape (3,)")
        return cls(x=arr[0], y=arr[1], z=arr[2])
