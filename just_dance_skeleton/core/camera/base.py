from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class CameraSource(ABC):
    """Abstract base class for camera input sources"""
    
    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.is_opened = False
        
    @abstractmethod
    def open(self) -> bool:
        """Open the camera source. Returns True if successful."""
        pass
    
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        Returns (success, frame) tuple.
        Frame is BGR format numpy array or None if failed.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the camera source and release resources."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the camera source is available and working."""
        pass
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get the current resolution (width, height)."""
        return (self.width, self.height)
    
    def get_fps(self) -> int:
        """Get the target FPS."""
        return self.fps
    
    def __enter__(self):
        """Context manager entry."""
        if not self.open():
            raise RuntimeError("Failed to open camera source")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()