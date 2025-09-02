from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class Keypoint:
    """Represents a single keypoint/joint in 2D space"""
    x: float
    y: float
    confidence: float
    name: Optional[str] = None
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to integer tuple for OpenCV drawing"""
        return (int(self.x), int(self.y))
    
    def distance_to(self, other: 'Keypoint') -> float:
        """Calculate Euclidean distance to another keypoint"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Pose:
    """Represents a complete pose with all keypoints"""
    keypoints: List[Keypoint]
    confidence: float
    person_id: Optional[int] = None
    timestamp: Optional[float] = None
    
    def get_keypoint(self, index: int) -> Optional[Keypoint]:
        """Get keypoint by index, return None if invalid"""
        if 0 <= index < len(self.keypoints):
            return self.keypoints[index]
        return None
    
    def get_keypoint_by_name(self, name: str) -> Optional[Keypoint]:
        """Get keypoint by name, return None if not found"""
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box (x, y, width, height) of all visible keypoints"""
        visible_kps = [kp for kp in self.keypoints if kp.confidence > 0.5]
        if not visible_kps:
            return (0, 0, 0, 0)
        
        xs = [kp.x for kp in visible_kps]
        ys = [kp.y for kp in visible_kps]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
    
    def normalize_scale(self, reference_keypoints: List[int]) -> 'Pose':
        """
        Normalize pose scale based on reference keypoints (e.g., shoulder width)
        Returns a new normalized pose
        """
        if len(reference_keypoints) < 2:
            return self
        
        # Calculate reference distance (e.g., shoulder width)
        kp1 = self.get_keypoint(reference_keypoints[0])
        kp2 = self.get_keypoint(reference_keypoints[1])
        
        if not kp1 or not kp2 or kp1.confidence < 0.5 or kp2.confidence < 0.5:
            return self
        
        ref_distance = kp1.distance_to(kp2)
        if ref_distance == 0:
            return self
        
        # Normalize all keypoints
        scale_factor = 100.0 / ref_distance  # Normalize to 100 units
        center_x = np.mean([kp.x for kp in self.keypoints if kp.confidence > 0.5])
        center_y = np.mean([kp.y for kp in self.keypoints if kp.confidence > 0.5])
        
        normalized_keypoints = []
        for kp in self.keypoints:
            new_x = (kp.x - center_x) * scale_factor + center_x
            new_y = (kp.y - center_y) * scale_factor + center_y
            normalized_keypoints.append(Keypoint(new_x, new_y, kp.confidence, kp.name))
        
        return Pose(normalized_keypoints, self.confidence, self.person_id, self.timestamp)

@dataclass
class PoseMatch:
    """Represents the result of matching a pose against a target"""
    pose_name: str
    similarity: float
    distance: float
    matched_keypoints: int
    total_keypoints: int
    is_good_match: bool = False
    is_perfect_match: bool = False
    
    @property
    def match_percentage(self) -> float:
        """Get match percentage (0-100)"""
        return self.similarity * 100
    
    @property
    def accuracy(self) -> float:
        """Get accuracy as ratio of matched keypoints"""
        if self.total_keypoints == 0:
            return 0.0
        return self.matched_keypoints / self.total_keypoints

@dataclass
class DancePose:
    """Represents a predefined dance pose"""
    name: str
    keypoints: List[Keypoint]
    difficulty: str = "medium"
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_pose(self) -> Pose:
        """Convert to regular Pose object"""
        return Pose(self.keypoints.copy(), 1.0)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DancePose':
        """Create DancePose from dictionary (for JSON loading)"""
        keypoints = []
        for kp_data in data.get('keypoints', []):
            keypoints.append(Keypoint(
                x=kp_data['x'],
                y=kp_data['y'],
                confidence=kp_data.get('confidence', 1.0),
                name=kp_data.get('name')
            ))
        
        return cls(
            name=data['name'],
            keypoints=keypoints,
            difficulty=data.get('difficulty', 'medium'),
            tags=data.get('tags', []),
            description=data.get('description', '')
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for JSON saving)"""
        return {
            'name': self.name,
            'keypoints': [
                {
                    'x': kp.x,
                    'y': kp.y,
                    'confidence': kp.confidence,
                    'name': kp.name
                } for kp in self.keypoints
            ],
            'difficulty': self.difficulty,
            'tags': self.tags,
            'description': self.description
        }

@dataclass
class GameState:
    """Represents the current game state"""
    current_pose: Optional[Pose] = None
    target_pose: Optional[DancePose] = None
    current_match: Optional[PoseMatch] = None
    score: float = 0.0
    combo_count: int = 0
    is_playing: bool = False
    frame_count: int = 0
    start_time: Optional[float] = None