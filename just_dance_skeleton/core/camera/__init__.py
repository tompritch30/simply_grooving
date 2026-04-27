"""Camera source implementations"""

from .base import CameraSource
from .kinect import KinectSource
from .webcam import WebcamSource

__all__ = ["CameraSource", "WebcamSource", "KinectSource"]
