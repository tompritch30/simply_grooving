import time
import random
from typing import Optional, List
from ..pose.models import Pose, GameState, PoseMatch
from ..pose.matcher import DancePoseMatcher
from config.settings import SCORE_DECAY_RATE, PERFECT_MATCH_BONUS, GOOD_MATCH_BONUS

class GameEngine:
    """Main game engine managing game state and logic"""
    
    def __init__(self, pose_matcher: DancePoseMatcher):
        self.pose_matcher = pose_matcher
        self.game_state = GameState()
        self.target_pose_sequence: List[str] = []
        self.current_pose_index = 0
        self.pose_hold_time = 2.0  # Seconds to hold each pose
        self.pose_start_time = None
        self.last_update_time = time.time()
        
    def start_game(self, pose_sequence: Optional[List[str]] = None) -> None:
        """Start a new game session"""
        self.game_state = GameState()
        self.game_state.is_playing = True
        self.game_state.start_time = time.time()
        self.last_update_time = time.time()
        
        # Set up pose sequence
        if pose_sequence:
            self.target_pose_sequence = pose_sequence
        else:
            # Create random sequence from available poses
            available_poses = self.pose_matcher.get_pose_names()
            if available_poses:
                self.target_pose_sequence = random.choices(available_poses, k=5)
            else:
                self.target_pose_sequence = []
        
        self.current_pose_index = 0
        self.pose_start_time = None
        
        if self.target_pose_sequence:
            self._set_current_target_pose()
    
    def stop_game(self) -> None:
        """Stop the current game session"""
        self.game_state.is_playing = False
        print(f"Game ended! Final score: {self.game_state.score:.1f}")
    
    def update(self, detected_pose: Optional[Pose]) -> None:
        """Update game state with new pose detection"""
        if not self.game_state.is_playing:
            return
        
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.game_state.frame_count += 1
        self.game_state.current_pose = detected_pose
        
        # Apply score decay over time
        self.game_state.score *= (SCORE_DECAY_RATE ** dt)
        
        if detected_pose and self.game_state.target_pose:
            # Match current pose against target
            pose_match = self.pose_matcher.match_pose(detected_pose)
            
            if pose_match and pose_match.pose_name == self.game_state.target_pose.name:
                self.game_state.current_match = pose_match
                self._handle_pose_match(pose_match, current_time)
            else:
                self.game_state.current_match = None
                # Reset pose timing if not matching
                if self.pose_start_time is not None:
                    self.pose_start_time = None
                    self.game_state.combo_count = 0
        else:
            self.game_state.current_match = None
    
    def _handle_pose_match(self, pose_match: PoseMatch, current_time: float) -> None:
        """Handle successful pose matching"""
        if pose_match.is_good_match:
            # Start timing if this is the first good match
            if self.pose_start_time is None:
                self.pose_start_time = current_time
            
            # Check if pose has been held long enough
            hold_duration = current_time - self.pose_start_time
            
            if hold_duration >= self.pose_hold_time:
                self._complete_pose(pose_match)
        else:
            # Reset timing for poor matches
            self.pose_start_time = None
            self.game_state.combo_count = 0
    
    def _complete_pose(self, pose_match: PoseMatch) -> None:
        """Complete current pose and move to next"""
        # Award points
        base_score = GOOD_MATCH_BONUS
        if pose_match.is_perfect_match:
            base_score = PERFECT_MATCH_BONUS
        
        # Apply combo multiplier
        combo_multiplier = 1.0 + (self.game_state.combo_count * 0.1)
        score_earned = base_score * combo_multiplier * pose_match.similarity
        
        self.game_state.score += score_earned
        self.game_state.combo_count += 1
        
        print(f"Pose completed! +{score_earned:.1f} points (combo x{combo_multiplier:.1f})")
        
        # Move to next pose
        self.current_pose_index += 1
        self.pose_start_time = None
        
        if self.current_pose_index >= len(self.target_pose_sequence):
            # Sequence complete, generate new one or end game
            self._generate_new_sequence()
        else:
            self._set_current_target_pose()
    
    def _set_current_target_pose(self) -> None:
        """Set the current target pose from sequence"""
        if (self.current_pose_index < len(self.target_pose_sequence) and 
            self.target_pose_sequence):
            
            pose_name = self.target_pose_sequence[self.current_pose_index]
            if pose_name in self.pose_matcher.dance_poses:
                self.game_state.target_pose = self.pose_matcher.dance_poses[pose_name]
            else:
                # Skip invalid pose
                self.current_pose_index += 1
                if self.current_pose_index < len(self.target_pose_sequence):
                    self._set_current_target_pose()
    
    def _generate_new_sequence(self) -> None:
        """Generate a new pose sequence"""
        available_poses = self.pose_matcher.get_pose_names()
        if available_poses:
            # Increase difficulty by adding more poses
            sequence_length = min(5 + (self.game_state.score // 100), 10)
            self.target_pose_sequence = random.choices(available_poses, k=sequence_length)
            self.current_pose_index = 0
            self._set_current_target_pose()
            print(f"New sequence started with {len(self.target_pose_sequence)} poses!")
        else:
            self.stop_game()
    
    def get_progress(self) -> float:
        """Get progress through current pose (0-1)"""
        if not self.pose_start_time or not self.game_state.is_playing:
            return 0.0
        
        current_time = time.time()
        hold_duration = current_time - self.pose_start_time
        return min(hold_duration / self.pose_hold_time, 1.0)
    
    def get_remaining_poses(self) -> int:
        """Get number of remaining poses in current sequence"""
        return max(0, len(self.target_pose_sequence) - self.current_pose_index)
    
    def get_current_pose_name(self) -> str:
        """Get name of current target pose"""
        if self.game_state.target_pose:
            return self.game_state.target_pose.name
        return "None"
    
    def get_game_time(self) -> float:
        """Get total game time in seconds"""
        if self.game_state.start_time:
            return time.time() - self.game_state.start_time
        return 0.0
    
    def is_pose_being_held(self) -> bool:
        """Check if a pose is currently being held correctly"""
        return (self.pose_start_time is not None and 
                self.game_state.current_match is not None and 
                self.game_state.current_match.is_good_match)
    
    def reset_combo(self) -> None:
        """Reset combo counter"""
        self.game_state.combo_count = 0
        self.pose_start_time = None