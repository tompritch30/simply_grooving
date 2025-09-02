"""
Just Dance - Skeleton Tracking PoC
Phase 1 & 2 Implementation

Entry point for the Just Dance skeleton tracking application.
Implements real-time pose detection and matching against predefined dance poses.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.camera.webcam import WebcamSource
from core.camera.kinect import KinectSource
from gui.opencv_display import OpenCVDisplay
from config.settings import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, WEBCAM_ID
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Just Dance - Skeleton Tracking PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Use default webcam
  python main.py --camera 1         # Use camera index 1
  python main.py --kinect           # Use Kinect (if available)
  python main.py --resolution 640 480  # Use 640x480 resolution
  python main.py --fps 15           # Use 15 FPS
        """
    )
    
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=WEBCAM_ID,
        help=f"Camera index to use (default: {WEBCAM_ID})"
    )
    
    parser.add_argument(
        "--kinect", "-k",
        action="store_true",
        help="Use Kinect instead of webcam (Phase 5 feature, not yet implemented)"
    )
    
    parser.add_argument(
        "--resolution", "-r",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=[CAMERA_WIDTH, CAMERA_HEIGHT],
        help=f"Camera resolution (default: {CAMERA_WIDTH}x{CAMERA_HEIGHT})"
    )
    
    parser.add_argument(
        "--fps", "-f",
        type=int,
        default=CAMERA_FPS,
        help=f"Target FPS (default: {CAMERA_FPS})"
    )
    
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit"
    )
    
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI (for testing/debugging)"
    )
    
    return parser.parse_args()

def list_available_cameras():
    """List all available camera sources"""
    print("Scanning for available cameras...")
    
    # Check webcams
    webcam_indices = WebcamSource.list_available_cameras()
    if webcam_indices:
        print(f"\nWebcams found: {webcam_indices}")
        for idx in webcam_indices:
            print(f"  Camera {idx}: Available")
    else:
        print("\nNo webcams found")
    
    # Check Kinect
    kinect_devices = KinectSource.list_available_kinects()
    if kinect_devices:
        print(f"\nKinect devices found: {len(kinect_devices)}")
        for i, device in enumerate(kinect_devices):
            print(f"  Kinect {i}: Available")
    else:
        print("\nNo Kinect devices found (or libfreenect not available)")

def create_camera_source(args):
    """Create appropriate camera source based on arguments"""
    width, height = args.resolution
    fps = args.fps
    
    if args.kinect:
        print("Attempting to use Kinect...")
        camera = KinectSource(width=width, height=height, fps=fps)
        if not camera.is_available():
            print("Kinect not available, falling back to webcam")
            camera = WebcamSource(args.camera, width, height, fps)
    else:
        print(f"Using webcam {args.camera}")
        camera = WebcamSource(args.camera, width, height, fps)
    
    return camera

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import mediapipe
    except ImportError:
        missing_deps.append("mediapipe")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
    
    if missing_deps:
        print("Error: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall missing dependencies with:")
        print(f"  pip install {' '.join(missing_deps)}")
        return False
    
    return True

def main():
    """Main entry point"""
    print("=" * 60)
    print("Just Dance - Skeleton Tracking PoC")
    print("Phase 1 & 2: Single Person Pose Detection and Matching")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Parse arguments
    args = parse_arguments()
    
    # Handle special commands
    if args.list_cameras:
        list_available_cameras()
        return
    
    # Create camera source
    try:
        camera_source = create_camera_source(args)
    except Exception as e:
        print(f"Error creating camera source: {e}")
        sys.exit(1)
    
    # Test camera availability
    if not camera_source.is_available():
        print(f"Error: Camera not available")
        if hasattr(camera_source, 'camera_id'):
            print(f"Camera index {camera_source.camera_id} not found")
            print("\nAvailable cameras:")
            list_available_cameras()
        sys.exit(1)
    
    print(f"Camera initialized: {camera_source.width}x{camera_source.height} @ {camera_source.fps}fps")
    
    # Run the application
    if args.no_gui:
        print("No-GUI mode not yet implemented")
        sys.exit(1)
    else:
        # Start GUI application
        try:
            display = OpenCVDisplay(camera_source)
            display.run()
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Application shutting down...")

if __name__ == "__main__":
    main()