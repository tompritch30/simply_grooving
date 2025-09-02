import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import euclidean
from .models import Pose, DancePose, PoseMatch, Keypoint
from config.settings import (
    POSE_MATCH_THRESHOLD, MAX_POSE_DISTANCE, 
    IMPORTANT_KEYPOINTS, NORMALIZATION_KEYPOINTS
)

class DancePoseMatcher:
    """Matches detected poses against predefined dance poses"""
    
    def __init__(self, poses_file: str = "data/poses/dance_poses.json"):
        self.poses_file = Path(poses_file)
        self.dance_poses: Dict[str, DancePose] = {}
        self.load_dance_poses()
        
    def load_dance_poses(self) -> bool:
        """Load dance poses from JSON file"""
        try:
            if not self.poses_file.exists():
                # Create sample poses if file doesn't exist
                self._create_sample_poses()
                return True
                
            with open(self.poses_file, 'r') as f:
                data = json.load(f)
            
            self.dance_poses = {}
            for pose_data in data.get('poses', []):
                dance_pose = DancePose.from_dict(pose_data)
                self.dance_poses[dance_pose.name] = dance_pose
            
            print(f"Loaded {len(self.dance_poses)} dance poses")
            return True
            
        except Exception as e:
            print(f"Error loading dance poses: {e}")
            # Create sample poses as fallback
            self._create_sample_poses()
            return False
    
    def save_dance_poses(self) -> bool:
        """Save current dance poses to JSON file"""
        try:
            self.poses_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'poses': [pose.to_dict() for pose in self.dance_poses.values()],
                'version': '1.0'
            }
            
            with open(self.poses_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved {len(self.dance_poses)} dance poses")
            return True
            
        except Exception as e:
            print(f"Error saving dance poses: {e}")
            return False
    
    def match_pose(self, detected_pose: Pose, normalize: bool = True) -> Optional[PoseMatch]:
        """
        Match a detected pose against all dance poses
        Returns the best match or None if no good match found
        """
        if not detected_pose or not self.dance_poses:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        # Normalize the detected pose if requested
        if normalize:
            detected_pose = detected_pose.normalize_scale(NORMALIZATION_KEYPOINTS)
        
        for pose_name, dance_pose in self.dance_poses.items():
            # Calculate similarity
            similarity, distance, matched_kps = self._calculate_pose_similarity(
                detected_pose, dance_pose.to_pose(), normalize
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = PoseMatch(
                    pose_name=pose_name,
                    similarity=similarity,
                    distance=distance,
                    matched_keypoints=matched_kps,
                    total_keypoints=len(IMPORTANT_KEYPOINTS),
                    is_good_match=similarity >= POSE_MATCH_THRESHOLD * 0.7,
                    is_perfect_match=similarity >= POSE_MATCH_THRESHOLD
                )
        
        return best_match if best_match and best_match.similarity > 0.3 else None
    
    def _calculate_pose_similarity(self, pose1: Pose, pose2: Pose, normalize: bool = True) -> Tuple[float, float, int]:
        """
        Calculate similarity between two poses
        Returns (similarity, distance, matched_keypoints)
        """
        if normalize:
            pose2 = pose2.normalize_scale(NORMALIZATION_KEYPOINTS)
        
        total_distance = 0.0
        valid_comparisons = 0
        matched_keypoints = 0
        
        # Focus on important keypoints for dance poses
        for kp_idx in IMPORTANT_KEYPOINTS:
            kp1 = pose1.get_keypoint(kp_idx)
            kp2 = pose2.get_keypoint(kp_idx)
            
            if kp1 and kp2 and kp1.confidence > 0.5 and kp2.confidence > 0.5:
                distance = kp1.distance_to(kp2)
                total_distance += distance
                valid_comparisons += 1
                
                # Consider a keypoint "matched" if within reasonable distance
                if distance < MAX_POSE_DISTANCE * 0.3:
                    matched_keypoints += 1
        
        if valid_comparisons == 0:
            return 0.0, float('inf'), 0
        
        # Calculate average distance
        avg_distance = total_distance / valid_comparisons
        
        # Convert distance to similarity (0-1, where 1 is perfect match)
        # Using exponential decay function
        similarity = np.exp(-avg_distance / MAX_POSE_DISTANCE)
        
        return similarity, avg_distance, matched_keypoints
    
    def add_dance_pose(self, name: str, pose: Pose, difficulty: str = "medium", 
                      tags: List[str] = None, description: str = "") -> bool:
        """Add a new dance pose to the collection"""
        try:
            dance_pose = DancePose(
                name=name,
                keypoints=pose.keypoints.copy(),
                difficulty=difficulty,
                tags=tags or [],
                description=description
            )
            
            self.dance_poses[name] = dance_pose
            return True
            
        except Exception as e:
            print(f"Error adding dance pose '{name}': {e}")
            return False
    
    def remove_dance_pose(self, name: str) -> bool:
        """Remove a dance pose from the collection"""
        if name in self.dance_poses:
            del self.dance_poses[name]
            return True
        return False
    
    def get_pose_names(self) -> List[str]:
        """Get list of all available pose names"""
        return list(self.dance_poses.keys())
    
    def get_poses_by_difficulty(self, difficulty: str) -> List[str]:
        """Get pose names filtered by difficulty"""
        return [name for name, pose in self.dance_poses.items() 
                if pose.difficulty == difficulty]
    
    def _create_sample_poses(self) -> None:
        """Create sample dance poses for testing"""
        # Create the directory if it doesn't exist
        self.poses_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sample poses with normalized coordinates (0-1 range)
        sample_poses = {
            "T-Pose": {
                "name": "T-Pose",
                "keypoints": [
                    {"x": 0.5, "y": 0.2, "confidence": 1.0, "name": "NOSE"},
                    {"x": 0.5, "y": 0.25, "confidence": 1.0, "name": "LEFT_EYE"},
                    {"x": 0.5, "y": 0.25, "confidence": 1.0, "name": "RIGHT_EYE"},
                    {"x": 0.3, "y": 0.4, "confidence": 1.0, "name": "LEFT_SHOULDER"},
                    {"x": 0.7, "y": 0.4, "confidence": 1.0, "name": "RIGHT_SHOULDER"},
                    {"x": 0.1, "y": 0.4, "confidence": 1.0, "name": "LEFT_ELBOW"},
                    {"x": 0.9, "y": 0.4, "confidence": 1.0, "name": "RIGHT_ELBOW"},
                    {"x": 0.05, "y": 0.4, "confidence": 1.0, "name": "LEFT_WRIST"},
                    {"x": 0.95, "y": 0.4, "confidence": 1.0, "name": "RIGHT_WRIST"},
                    {"x": 0.4, "y": 0.6, "confidence": 1.0, "name": "LEFT_HIP"},
                    {"x": 0.6, "y": 0.6, "confidence": 1.0, "name": "RIGHT_HIP"},
                    {"x": 0.4, "y": 0.8, "confidence": 1.0, "name": "LEFT_KNEE"},
                    {"x": 0.6, "y": 0.8, "confidence": 1.0, "name": "RIGHT_KNEE"},
                    {"x": 0.4, "y": 1.0, "confidence": 1.0, "name": "LEFT_ANKLE"},
                    {"x": 0.6, "y": 1.0, "confidence": 1.0, "name": "RIGHT_ANKLE"}
                ],
                "difficulty": "easy",
                "tags": ["basic", "calibration"],
                "description": "Arms extended horizontally, basic pose"
            },
            "Victory_Pose": {
                "name": "Victory_Pose",
                "keypoints": [
                    {"x": 0.5, "y": 0.2, "confidence": 1.0, "name": "NOSE"},
                    {"x": 0.5, "y": 0.25, "confidence": 1.0, "name": "LEFT_EYE"},
                    {"x": 0.5, "y": 0.25, "confidence": 1.0, "name": "RIGHT_EYE"},
                    {"x": 0.3, "y": 0.4, "confidence": 1.0, "name": "LEFT_SHOULDER"},
                    {"x": 0.7, "y": 0.4, "confidence": 1.0, "name": "RIGHT_SHOULDER"},
                    {"x": 0.2, "y": 0.3, "confidence": 1.0, "name": "LEFT_ELBOW"},
                    {"x": 0.8, "y": 0.3, "confidence": 1.0, "name": "RIGHT_ELBOW"},
                    {"x": 0.15, "y": 0.15, "confidence": 1.0, "name": "LEFT_WRIST"},
                    {"x": 0.85, "y": 0.15, "confidence": 1.0, "name": "RIGHT_WRIST"},
                    {"x": 0.4, "y": 0.6, "confidence": 1.0, "name": "LEFT_HIP"},
                    {"x": 0.6, "y": 0.6, "confidence": 1.0, "name": "RIGHT_HIP"},
                    {"x": 0.4, "y": 0.8, "confidence": 1.0, "name": "LEFT_KNEE"},
                    {"x": 0.6, "y": 0.8, "confidence": 1.0, "name": "RIGHT_KNEE"},
                    {"x": 0.4, "y": 1.0, "confidence": 1.0, "name": "LEFT_ANKLE"},
                    {"x": 0.6, "y": 1.0, "confidence": 1.0, "name": "RIGHT_ANKLE"}
                ],
                "difficulty": "easy",
                "tags": ["celebration", "arms_up"],
                "description": "Both arms raised up in victory"
            },
            "Disco_Point": {
                "name": "Disco_Point",
                "keypoints": [
                    {"x": 0.5, "y": 0.2, "confidence": 1.0, "name": "NOSE"},
                    {"x": 0.5, "y": 0.25, "confidence": 1.0, "name": "LEFT_EYE"},
                    {"x": 0.5, "y": 0.25, "confidence": 1.0, "name": "RIGHT_EYE"},
                    {"x": 0.3, "y": 0.4, "confidence": 1.0, "name": "LEFT_SHOULDER"},
                    {"x": 0.7, "y": 0.4, "confidence": 1.0, "name": "RIGHT_SHOULDER"},
                    {"x": 0.25, "y": 0.35, "confidence": 1.0, "name": "LEFT_ELBOW"},
                    {"x": 0.9, "y": 0.25, "confidence": 1.0, "name": "RIGHT_ELBOW"},
                    {"x": 0.2, "y": 0.5, "confidence": 1.0, "name": "LEFT_WRIST"},
                    {"x": 1.0, "y": 0.1, "confidence": 1.0, "name": "RIGHT_WRIST"},
                    {"x": 0.4, "y": 0.6, "confidence": 1.0, "name": "LEFT_HIP"},
                    {"x": 0.6, "y": 0.6, "confidence": 1.0, "name": "RIGHT_HIP"},
                    {"x": 0.4, "y": 0.8, "confidence": 1.0, "name": "LEFT_KNEE"},
                    {"x": 0.6, "y": 0.8, "confidence": 1.0, "name": "RIGHT_KNEE"},
                    {"x": 0.4, "y": 1.0, "confidence": 1.0, "name": "LEFT_ANKLE"},
                    {"x": 0.6, "y": 1.0, "confidence": 1.0, "name": "RIGHT_ANKLE"}
                ],
                "difficulty": "medium",
                "tags": ["disco", "pointing", "dance"],
                "description": "Classic disco pointing pose"
            }
        }
        
        # Convert to our format and save
        for pose_name, pose_data in sample_poses.items():
            dance_pose = DancePose.from_dict(pose_data)
            self.dance_poses[pose_name] = dance_pose
        
        # Save to file
        self.save_dance_poses()
        print(f"Created {len(sample_poses)} sample dance poses")