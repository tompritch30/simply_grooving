import json
import tempfile
import unittest
from pathlib import Path

from core.pose.matcher import DancePoseMatcher
from core.pose.models import Keypoint, Pose


def _reference_keypoints():
    """Compact pose template covering all important joints."""
    points = [
        ("LEFT_SHOULDER", 0.40, 0.30),
        ("RIGHT_SHOULDER", 0.60, 0.30),
        ("LEFT_ELBOW", 0.32, 0.45),
        ("RIGHT_ELBOW", 0.68, 0.45),
        ("LEFT_WRIST", 0.25, 0.60),
        ("RIGHT_WRIST", 0.75, 0.60),
        ("LEFT_HIP", 0.45, 0.55),
        ("RIGHT_HIP", 0.55, 0.55),
        ("LEFT_KNEE", 0.45, 0.75),
        ("RIGHT_KNEE", 0.55, 0.75),
        ("LEFT_ANKLE", 0.45, 0.95),
        ("RIGHT_ANKLE", 0.55, 0.95),
    ]
    return [
        {"name": name, "x": x, "y": y, "confidence": 1.0}
        for name, x, y in points
    ]


class TestDancePoseMatcher(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.poses_path = Path(self.temp_dir.name) / "test_poses.json"

        pose_payload = {
            "poses": [
                {
                    "name": "Reference",
                    "difficulty": "easy",
                    "tags": ["test"],
                    "description": "Test reference pose",
                    "keypoints": _reference_keypoints(),
                }
            ],
            "version": "1.0",
        }

        self.poses_path.write_text(json.dumps(pose_payload), encoding="utf-8")
        self.matcher = DancePoseMatcher(str(self.poses_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loads_custom_pose_file(self):
        self.assertIn("Reference", self.matcher.get_pose_names())

    def test_exact_pose_returns_perfect_match(self):
        detected_pose = self.matcher.dance_poses["Reference"].to_pose()

        pose_match = self.matcher.match_pose(detected_pose, normalize=False)

        self.assertIsNotNone(pose_match)
        self.assertEqual(pose_match.pose_name, "Reference")
        self.assertTrue(pose_match.is_perfect_match)
        self.assertAlmostEqual(pose_match.similarity, 1.0)

    def test_low_confidence_pose_returns_none(self):
        template_pose = self.matcher.dance_poses["Reference"].to_pose()
        low_confidence_pose = Pose(
            [Keypoint(kp.x, kp.y, 0.0, kp.name) for kp in template_pose.keypoints],
            confidence=0.0,
        )

        pose_match = self.matcher.match_pose(low_confidence_pose, normalize=False)

        self.assertIsNone(pose_match)


if __name__ == "__main__":
    unittest.main()
