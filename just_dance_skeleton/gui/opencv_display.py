import cv2
import time
import numpy as np
from typing import Optional
from core.camera.base import CameraSource
from core.pose.tracker import PoseTracker
from core.pose.matcher import DancePoseMatcher
from core.rendering.overlay import OverlayRenderer
from core.game.engine import GameEngine
from config.settings import WINDOW_NAME, TARGET_FPS

class OpenCVDisplay:
    """OpenCV-based display for the Just Dance PoC"""
    
    def __init__(self, camera_source: CameraSource):
        self.camera_source = camera_source
        self.pose_tracker = PoseTracker()
        self.pose_matcher = DancePoseMatcher()
        self.overlay_renderer = OverlayRenderer()
        self.game_engine = GameEngine(self.pose_matcher)
        
        self.is_running = False
        self.show_target_overlay = True
        self.show_fps = True
        self.frame_times = []
        self.target_frame_time = 1.0 / TARGET_FPS
        
    def run(self) -> None:
        """Main display loop"""
        print("Starting Just Dance Skeleton Tracking...")
        print("Controls:")
        print("  SPACE - Start/Stop game")
        print("  T - Toggle target pose overlay")
        print("  F - Toggle FPS display")
        print("  Q/ESC - Quit")
        print()
        
        try:
            with self.camera_source as camera:
                self.is_running = True
                
                while self.is_running:
                    loop_start = time.time()
                    
                    # Read frame
                    ret, frame = camera.read_frame()
                    if not ret or frame is None:
                        print("Failed to read frame")
                        break
                    
                    # Process frame
                    self._process_frame(frame)
                    
                    # Handle input
                    if not self._handle_input():
                        break
                    
                    # Control frame rate
                    self._control_frame_rate(loop_start)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in display loop: {e}")
        finally:
            self.cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> None:
        """Process a single frame"""
        # Detect poses
        poses = self.pose_tracker.process_frame(frame)
        detected_pose = poses[0] if poses else None
        
        # Update game engine
        self.game_engine.update(detected_pose)
        
        # Render overlays
        display_frame = self._render_frame(frame, detected_pose)
        
        # Show frame
        cv2.imshow(WINDOW_NAME, display_frame)
        
        # Update frame timing
        self._update_fps()
    
    def _render_frame(self, frame: np.ndarray, detected_pose: Optional[object]) -> np.ndarray:
        """Render all overlays on the frame"""
        display_frame = frame.copy()
        
        if detected_pose:
            if self.game_engine.game_state.is_playing:
                # Game mode - show comparison with target pose
                target_pose = None
                if self.game_engine.game_state.target_pose and self.show_target_overlay:
                    target_pose = self.game_engine.game_state.target_pose.to_pose()
                
                display_frame = self.overlay_renderer.render_pose_comparison(
                    display_frame, detected_pose, target_pose, 
                    self.game_engine.game_state.current_match
                )
                
                # Draw game UI
                display_frame = self._draw_game_ui(display_frame)
                
            else:
                # Free play mode - just show skeleton
                display_frame = self.overlay_renderer.render_pose(display_frame, detected_pose)
                
                # Show available poses
                display_frame = self._draw_available_poses(display_frame)
        
        # Draw FPS if enabled
        if self.show_fps:
            fps = self._get_current_fps()
            display_frame = self.overlay_renderer.draw_fps(display_frame, fps)
        
        # Draw instructions
        instructions = self._get_instructions()
        display_frame = self.overlay_renderer.draw_instructions(display_frame, instructions)
        
        return display_frame
    
    def _draw_game_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw game-specific UI elements"""
        height, width = frame.shape[:2]
        
        # Draw score
        score_text = f"Score: {self.game_engine.game_state.score:.0f}"
        cv2.putText(frame, score_text, (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw combo counter
        if self.game_engine.game_state.combo_count > 0:
            combo_text = f"Combo: x{self.game_engine.game_state.combo_count}"
            cv2.putText(frame, combo_text, (width - 200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw current target pose name
        target_name = self.game_engine.get_current_pose_name()
        target_text = f"Target: {target_name}"
        cv2.putText(frame, target_text, (10, height - 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw progress bar for pose hold
        if self.game_engine.is_pose_being_held():
            progress = self.game_engine.get_progress()
            
            bar_width = 300
            bar_height = 20
            bar_x = (width - bar_width) // 2
            bar_y = 60
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Progress fill
            fill_width = int(bar_width * progress)
            color = (0, 255, 0) if progress >= 0.8 else (0, 255, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         color, -1)
            
            # Text
            cv2.putText(frame, "Hold pose!", (bar_x, bar_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw remaining poses count
        remaining = self.game_engine.get_remaining_poses()
        if remaining > 0:
            remaining_text = f"Remaining: {remaining}"
            cv2.putText(frame, remaining_text, (10, height - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw game time
        game_time = self.game_engine.get_game_time()
        time_text = f"Time: {game_time:.1f}s"
        cv2.putText(frame, time_text, (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _draw_available_poses(self, frame: np.ndarray) -> np.ndarray:
        """Draw list of available poses in free play mode"""
        available_poses = self.pose_matcher.get_pose_names()
        
        if available_poses:
            y_start = 100
            cv2.putText(frame, "Available Poses:", (10, y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for i, pose_name in enumerate(available_poses[:8]):  # Show max 8 poses
                y_pos = y_start + 30 + (i * 25)
                cv2.putText(frame, f"- {pose_name}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def _get_instructions(self) -> str:
        """Get instruction text based on current mode"""
        if self.game_engine.game_state.is_playing:
            return ("SPACE: Stop game | T: Toggle target overlay | F: Toggle FPS | Q: Quit")
        else:
            return ("SPACE: Start game | T: Toggle target overlay | F: Toggle FPS | Q: Quit")
    
    def _handle_input(self) -> bool:
        """Handle keyboard input. Returns False to quit."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            return False
        elif key == ord(' '):  # SPACE
            if self.game_engine.game_state.is_playing:
                self.game_engine.stop_game()
            else:
                self.game_engine.start_game()
        elif key == ord('t'):  # T
            self.show_target_overlay = not self.show_target_overlay
            print(f"Target overlay: {'ON' if self.show_target_overlay else 'OFF'}")
        elif key == ord('f'):  # F
            self.show_fps = not self.show_fps
            print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
        elif key == ord('r'):  # R - Reset combo (hidden feature)
            self.game_engine.reset_combo()
            print("Combo reset")
        
        return True
    
    def _update_fps(self) -> None:
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only last 30 frames for FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
    
    def _get_current_fps(self) -> float:
        """Get current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span > 0:
            return (len(self.frame_times) - 1) / time_span
        return 0.0
    
    def _control_frame_rate(self, loop_start: float) -> None:
        """Control frame rate to target FPS"""
        elapsed = time.time() - loop_start
        sleep_time = self.target_frame_time - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.is_running = False
        self.pose_tracker.close()
        cv2.destroyAllWindows()
        print("Display cleaned up")