import unittest

from core.game.scoring import (
    JUDGEMENT_GOOD,
    JUDGEMENT_MISS,
    JUDGEMENT_PERFECT,
    ScoringEngine,
)


class TestScoringEngine(unittest.TestCase):
    def setUp(self):
        self.scoring_engine = ScoringEngine(
            good_match_bonus=5,
            perfect_match_bonus=10,
            combo_multiplier_step=0.1,
        )

    def test_perfect_match_score_uses_perfect_bonus(self):
        score = self.scoring_engine.calculate_pose_score(
            similarity=0.8,
            is_good_match=True,
            is_perfect_match=True,
            combo_count=2,
        )

        self.assertEqual(score.base_score, 10)
        self.assertEqual(score.judgement, JUDGEMENT_PERFECT)
        self.assertAlmostEqual(score.combo_multiplier, 1.2)
        self.assertAlmostEqual(score.total_score, 9.6)

    def test_good_match_score_uses_good_bonus(self):
        score = self.scoring_engine.calculate_pose_score(
            similarity=0.5,
            is_good_match=True,
            is_perfect_match=False,
            combo_count=1,
        )

        self.assertEqual(score.base_score, 5)
        self.assertEqual(score.judgement, JUDGEMENT_GOOD)
        self.assertAlmostEqual(score.combo_multiplier, 1.1)
        self.assertAlmostEqual(score.total_score, 2.75)

    def test_miss_returns_zero_score(self):
        score = self.scoring_engine.calculate_pose_score(
            similarity=0.9,
            is_good_match=False,
            is_perfect_match=False,
            combo_count=5,
        )

        self.assertEqual(score.base_score, 0.0)
        self.assertEqual(score.total_score, 0.0)
        self.assertEqual(score.judgement, JUDGEMENT_MISS)


if __name__ == "__main__":
    unittest.main()
