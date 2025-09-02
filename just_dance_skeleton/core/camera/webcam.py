import cv2
import numpy as np
from typing import Tuple, Optional
from .base import CameraSource

class WebcamSource(CameraSource):
    """Webcam camera source implementation using OpenCV"""
    
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        super().__init__(width, height, fps)
        self.camera_id = camera_id
        self.cap = None
        
    def open(self) -> bool:
        """Open the webcam"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings were applied
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"Webcam opened: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_opened = True
            return True
            
        except Exception as e:
            print(f"Failed to open webcam {self.camera_id}: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the webcam"""
        if not self.is_opened or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Flip horizontally for mirror effect (more natural for users)
                frame = cv2.flip(frame, 1)
                return True, frame
            return False, None
            
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def close(self) -> None:
        """Close the webcam and release resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        print("Webcam closed")
    
    def is_available(self) -> bool:
        """Check if webcam is available"""
        if self.is_opened and self.cap is not None:
            return self.cap.isOpened()
        
        # Try to open temporarily to check availability
        temp_cap = cv2.VideoCapture(self.camera_id)
        if temp_cap.isOpened():
            temp_cap.release()
            return True
        return False
    
    def get_actual_resolution(self) -> Tuple[int, int]:
        """Get the actual resolution being used by the camera"""
        if self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return self.get_resolution()
    
    def get_actual_fps(self) -> float:
        """Get the actual FPS being used by the camera"""
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return float(self.fps)
    
    @staticmethod
    def list_available_cameras() -> list:
        """List all available camera indices"""
        available_cameras = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras