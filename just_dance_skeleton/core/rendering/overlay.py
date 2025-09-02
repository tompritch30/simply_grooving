import cv2
import numpy as np
from typing import List, Optional, Tuple
from ..pose.models import Pose, PoseMatch
from config.settings import (
    SKELETON_COLOR, JOINT_COLOR, JOINT_RADIUS, SKELETON_THICKNESS,
    POSE_CONNECTIONS, GLOW_COLOR, GLOW_RADIUS, IMPORTANT_KEYPOINTS
)

class OverlayRenderer:
    """Renders pose overlays on video frames"""
    
    def __init__(self):
        self.skeleton_color = SKELETON_COLOR
        self.joint_color = JOINT_COLOR
        self.joint_radius = JOINT_RADIUS
        self.skeleton_thickness = SKELETON_THICKNESS
        self.glow_color = GLOW_COLOR
        self.glow_radius = GLOW_RADIUS
        
    def render_pose(self, frame: np.ndarray, pose: Pose, 
                   color_override: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Render a single pose on the frame"""
        if not pose or not pose.keypoints:
            return frame
        
        skeleton_color = color_override or self.skeleton_color
        
        # Draw skeleton connections
        frame = self._draw_skeleton(frame, pose, skeleton_color)
        
        # Draw joints
        frame = self._draw_joints(frame, pose, skeleton_color)
        
        # Highlight hands and feet with glow effect
        frame = self._draw_hand_foot_glow(frame, pose)
        
        return frame
    
    def render_pose_comparison(self, frame: np.ndarray, detected_pose: Pose, 
                             target_pose: Optional[Pose], pose_match: Optional[PoseMatch]) -> np.ndarray:
        """Render pose comparison with visual feedback"""
        if not detected_pose:
            return frame
        
        # Draw detected pose in green
        frame = self.render_pose(frame, detected_pose, (0, 255, 0))
        
        # Draw target pose overlay if available
        if target_pose:
            frame = self._draw_target_pose_overlay(frame, target_pose, detected_pose)
        
        # Add match feedback
        if pose_match:
            frame = self._draw_match_feedback(frame, pose_match)
        
        return frame
    
    def _draw_skeleton(self, frame: np.ndarray, pose: Pose, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw skeleton connections"""
        height, width = frame.shape[:2]
        
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            
            if start_idx < len(pose.keypoints) and end_idx < len(pose.keypoints):
                start_kp = pose.keypoints[start_idx]
                end_kp = pose.keypoints[end_idx]
                
                # Only draw if both keypoints are confident
                if start_kp.confidence > 0.5 and end_kp.confidence > 0.5:
                    start_point = start_kp.to_tuple()
                    end_point = end_kp.to_tuple()
                    
                    # Ensure points are within frame bounds
                    if (0 <= start_point[0] < width and 0 <= start_point[1] < height and
                        0 <= end_point[0] < width and 0 <= end_point[1] < height):
                        
                        cv2.line(frame, start_point, end_point, color, self.skeleton_thickness)
        
        return frame
    
    def _draw_joints(self, frame: np.ndarray, pose: Pose, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw joint circles"""
        height, width = frame.shape[:2]
        
        for i, keypoint in enumerate(pose.keypoints):
            if keypoint.confidence > 0.5:
                point = keypoint.to_tuple()
                
                # Ensure point is within frame bounds
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    # Use different colors for important keypoints
                    joint_color = (255, 255, 0) if i in IMPORTANT_KEYPOINTS else color
                    cv2.circle(frame, point, self.joint_radius, joint_color, -1)
                    
                    # Add a border
                    cv2.circle(frame, point, self.joint_radius + 1, (0, 0, 0), 1)
        
        return frame
    
    def _draw_hand_foot_glow(self, frame: np.ndarray, pose: Pose) -> np.ndarray:
        """Draw glowing effects around hands and feet"""
        # Hand and foot keypoint indices
        hand_foot_indices = [15, 16, 27, 28]  # Wrists and ankles
        
        height, width = frame.shape[:2]
        
        for idx in hand_foot_indices:
            if idx < len(pose.keypoints):
                keypoint = pose.keypoints[idx]
                if keypoint.confidence > 0.7:  # Higher confidence for glow
                    point = keypoint.to_tuple()
                    
                    if 0 <= point[0] < width and 0 <= point[1] < height:
                        # Create glow effect with multiple circles
                        for radius in range(self.glow_radius, 0, -2):
                            alpha = 0.1 * (self.glow_radius - radius) / self.glow_radius
                            overlay = frame.copy()
                            cv2.circle(overlay, point, radius, self.glow_color, -1)
                            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return frame
    
    def _draw_target_pose_overlay(self, frame: np.ndarray, target_pose: Pose, 
                                detected_pose: Pose) -> np.ndarray:
        """Draw target pose as a semi-transparent overlay"""
        height, width = frame.shape[:2]
        overlay = np.zeros_like(frame)
        
        # Scale target pose to match detected pose position/size
        if len(detected_pose.keypoints) >= 2 and len(target_pose.keypoints) >= 2:
            # Use shoulder positions for alignment
            det_left_shoulder = detected_pose.get_keypoint(11)  # LEFT_SHOULDER
            det_right_shoulder = detected_pose.get_keypoint(12)  # RIGHT_SHOULDER
            
            if det_left_shoulder and det_right_shoulder:
                # Calculate center and scale
                det_center_x = (det_left_shoulder.x + det_right_shoulder.x) / 2
                det_center_y = (det_left_shoulder.y + det_right_shoulder.y) / 2
                det_scale = abs(det_right_shoulder.x - det_left_shoulder.x)
                
                # Draw target pose skeleton in blue/cyan
                target_color = (255, 255, 0)  # Cyan
                
                # Draw connections for target pose
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    
                    if start_idx < len(target_pose.keypoints) and end_idx < len(target_pose.keypoints):
                        start_kp = target_pose.keypoints[start_idx]
                        end_kp = target_pose.keypoints[end_idx]
                        
                        if start_kp.confidence > 0.5 and end_kp.confidence > 0.5:
                            # Scale and position target keypoints
                            start_x = int((start_kp.x - 0.5) * det_scale * 2 + det_center_x)
                            start_y = int((start_kp.y - 0.5) * det_scale * 2 + det_center_y)
                            end_x = int((end_kp.x - 0.5) * det_scale * 2 + det_center_x)
                            end_y = int((end_kp.y - 0.5) * det_scale * 2 + det_center_y)
                            
                            if (0 <= start_x < width and 0 <= start_y < height and
                                0 <= end_x < width and 0 <= end_y < height):
                                cv2.line(overlay, (start_x, start_y), (end_x, end_y), 
                                       target_color, self.skeleton_thickness)
                
                # Blend overlay with original frame
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return frame
    
    def _draw_match_feedback(self, frame: np.ndarray, pose_match: PoseMatch) -> np.ndarray:
        """Draw pose matching feedback"""
        height, width = frame.shape[:2]
        
        # Choose color based on match quality
        if pose_match.is_perfect_match:
            feedback_color = (0, 255, 0)  # Green
            feedback_text = "PERFECT!"
        elif pose_match.is_good_match:
            feedback_color = (0, 255, 255)  # Yellow
            feedback_text = "GOOD"
        else:
            feedback_color = (0, 0, 255)  # Red
            feedback_text = "TRY AGAIN"
        
        # Draw feedback text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size for positioning
        (text_width, text_height), baseline = cv2.getTextSize(
            feedback_text, font, font_scale, thickness
        )
        
        # Position at top center
        x = (width - text_width) // 2
        y = text_height + 30
        
        # Draw text background
        cv2.rectangle(frame, (x - 10, y - text_height - 10), 
                     (x + text_width + 10, y + baseline + 10), 
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, feedback_text, (x, y), font, font_scale, 
                   feedback_color, thickness)
        
        # Draw pose name and similarity
        pose_info = f"{pose_match.pose_name}: {pose_match.match_percentage:.1f}%"
        info_y = y + 40
        
        cv2.putText(frame, pose_info, (10, info_y), font, 0.6, 
                   (255, 255, 255), 1)
        
        # Draw accuracy bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = info_y + 30
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Filled bar based on similarity
        fill_width = int(bar_width * pose_match.similarity)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     feedback_color, -1)
        
        return frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter"""
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        return frame
    
    def draw_instructions(self, frame: np.ndarray, instructions: str) -> np.ndarray:
        """Draw instruction text"""
        height, width = frame.shape[:2]
        
        # Split instructions into lines
        lines = instructions.split('\n')
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_spacing = 25
        
        start_y = height - (len(lines) * line_spacing + 20)
        
        for i, line in enumerate(lines):
            y_pos = start_y + (i * line_spacing)
            cv2.putText(frame, line, (10, y_pos), font, font_scale, 
                       (255, 255, 255), thickness)
        
        return frame