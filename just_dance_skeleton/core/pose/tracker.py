from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from config.settings import (
    KEYPOINT_NAMES,
    POSE_INFERENCE_HEIGHT,
    POSE_INFERENCE_WIDTH,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
    TARGET_FPS,
)

from .models import Keypoint, Pose


class PoseTracker:
    """Pose tracking using MediaPipe Pose"""

    def __init__(
        self,
        min_detection_confidence: float = POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = POSE_MIN_TRACKING_CONFIDENCE,
        enable_segmentation: bool = False,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            # 0=Lite, 1=Full, 2=Heavy
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.inference_width = POSE_INFERENCE_WIDTH
        self.inference_height = POSE_INFERENCE_HEIGHT
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> List[Pose]:
        """
        Process a frame and extract poses
        Returns list of detected poses (currently limited to 1 person)
        """
        if frame is None:
            return []

        self.frame_count += 1
        original_height, original_width = frame.shape[:2]

        # My computer could die
        inference_frame = cv2.resize(
            frame, (self.inference_width, self.inference_height)
        )

        # BGR - > RGB for MP
        rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        poses = []
        if results.pose_landmarks:
            # MP 33 kps to Poses
            pose = self._landmarks_to_pose(
                results.pose_landmarks, original_width, original_height
            )
            if pose:
                poses.append(pose)

        return poses

    def _landmarks_to_pose(
        self, landmarks, original_width: int, original_height: int
    ) -> Optional[Pose]:
        """Convert MediaPipe landmarks to our Pose format"""
        keypoints = []
        total_confidence = 0.0
        valid_keypoints = 0

        for idx, landmark in enumerate(landmarks.landmark):
            # Scale coors to frame size
            x = landmark.x * original_width
            y = landmark.y * original_height
            confidence = landmark.visibility if hasattr(landmark, "visibility") else 1.0

            name = KEYPOINT_NAMES.get(idx, f"KEYPOINT_{idx}")
            keypoint = Keypoint(x=x, y=y, confidence=confidence, name=name)
            keypoints.append(keypoint)

            if confidence > 0.5:  # Count as valid if confidence > 0.5
                total_confidence += confidence
                valid_keypoints += 1

        if valid_keypoints == 0:
            return None

        # Calculate overall pose confidence
        pose_confidence = total_confidence / len(keypoints)

        return Pose(
            keypoints=keypoints,
            confidence=pose_confidence,
            # TODO - will need to track multiple people
            person_id=0,
            timestamp=self.frame_count / float(TARGET_FPS),
        )

    def get_pose_world_coordinates(self, frame: np.ndarray) -> Optional[Pose]:
        """
        Get pose in world coordinates (3D)
        Currently returns 2D coordinates but could be extended for 3D with depth info
        """
        poses = self.process_frame(frame)
        return poses[0] if poses else None

    def reset(self) -> None:
        """Reset the pose tracker state"""
        self.frame_count = 0

    def close(self) -> None:
        """Clean up resources"""
        if self.pose:
            self.pose.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
