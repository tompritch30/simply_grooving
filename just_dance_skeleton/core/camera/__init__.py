"""Camera source implementations"""

from .base import CameraSource
from .webcam import WebcamSource
from .kinect import KinectSource

__all__ = ['CameraSource', 'WebcamSource', 'KinectSource']