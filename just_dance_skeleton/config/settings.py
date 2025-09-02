import cv2

# Camera Settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
WEBCAM_ID = 0

# Pose Detection Settings
POSE_INFERENCE_WIDTH = 480  # Downscaled for performance
POSE_INFERENCE_HEIGHT = 360
POSE_MIN_DETECTION_CONFIDENCE = 0.7
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Rendering Settings
GREEN_RGB = (0, 255, 0)
RED_RGB = (255, 0, 0)
YELLOW_RGB = (255, 255, 0)
SKELETON_COLOR = GREEN_RGB 
JOINT_COLOR = RED_RGB
JOINT_RADIUS = 5
SKELETON_THICKNESS = 2
GLOW_RADIUS = 15
GLOW_COLOR = YELLOW_RGB

### Pose Matching Settings
# Similarity threshold (0-1)
POSE_MATCH_THRESHOLD = 0.8    
# Maximum distance for pose matching
MAX_POSE_DISTANCE = 100       
# For scale normalization
NORMALIZATION_KEYPOINTS = ["LEFT_SHOULDER", "RIGHT_SHOULDER"] 

# Game Settings
TARGET_FPS = 30
SCORE_DECAY_RATE = 0.95
PERFECT_MATCH_BONUS = 10
GOOD_MATCH_BONUS = 5

# GUI Settings
WINDOW_NAME = "Just Dance - Skeleton Tracking"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2

# MediaPipe Pose Connections
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    # Shoulders and arms
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Legs
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

# Keypoint names mapping (MediaPipe Pose landmark indices)
KEYPOINT_NAMES = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "MOUTH_LEFT",
    10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX"
}

# Important keypoints for dance pose matching
IMPORTANT_KEYPOINTS = [
    11, 12,  # Shoulders
    13, 14,  # Elbows
    15, 16,  # Wrists
    23, 24,  # Hips
    25, 26,  # Knees
    27, 28   # Ankles
]