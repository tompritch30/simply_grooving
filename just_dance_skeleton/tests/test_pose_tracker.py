import unittest
from types import SimpleNamespace

from config.settings import TARGET_FPS

try:
    from core.pose.tracker import PoseTracker
except Exception:  # pragma: no cover - only hit when mediapipe is unavailable
    PoseTracker = None


@unittest.skipIf(PoseTracker is None, "MediaPipe is not available")
class TestPoseTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = PoseTracker()

    def tearDown(self):
        self.tracker.close()

    def test_process_frame_with_none_returns_empty_list(self):
        self.assertEqual(self.tracker.process_frame(None), [])

    def test_landmarks_to_pose_maps_coordinates_and_metadata(self):
        fake_landmarks = SimpleNamespace(
            landmark=[
                SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(33)
            ]
        )
        self.tracker.frame_count = 15

        pose = self.tracker._landmarks_to_pose(fake_landmarks, 640, 480)

        self.assertIsNotNone(pose)
        self.assertEqual(len(pose.keypoints), 33)
        self.assertEqual(pose.person_id, 0)
        self.assertAlmostEqual(pose.timestamp, 15 / float(TARGET_FPS))

    def test_landmarks_to_pose_returns_none_when_no_visible_keypoints(self):
        invisible_landmarks = SimpleNamespace(
            landmark=[
                SimpleNamespace(x=0.5, y=0.5, visibility=0.1) for _ in range(33)
            ]
        )

        pose = self.tracker._landmarks_to_pose(invisible_landmarks, 640, 480)

        self.assertIsNone(pose)


if __name__ == "__main__":
    unittest.main()
