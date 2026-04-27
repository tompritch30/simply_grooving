from dataclasses import dataclass

from config.settings import COMBO_MULTIPLIER_STEP, GOOD_MATCH_BONUS, PERFECT_MATCH_BONUS

JUDGEMENT_MISS = "MISS"
JUDGEMENT_GOOD = "GOOD"
JUDGEMENT_PERFECT = "PERFECT"


@dataclass(frozen=True)
class ScoreResult:
    """Detailed score output for one completed pose."""

    base_score: float
    combo_multiplier: float
    similarity: float
    total_score: float
    judgement: str


class ScoringEngine:
    """Reusable score calculator for pose completion events."""

    def __init__(
        self,
        good_match_bonus: float = GOOD_MATCH_BONUS,
        perfect_match_bonus: float = PERFECT_MATCH_BONUS,
        combo_multiplier_step: float = COMBO_MULTIPLIER_STEP,
    ):
        self.good_match_bonus = good_match_bonus
        self.perfect_match_bonus = perfect_match_bonus
        self.combo_multiplier_step = combo_multiplier_step

    @staticmethod
    def _clamp_similarity(similarity: float) -> float:
        """Similarity should stay in a stable 0-1 range for scoring."""
        return max(0.0, min(1.0, similarity))

    def calculate_pose_score(
        self,
        *,
        similarity: float,
        is_good_match: bool,
        is_perfect_match: bool,
        combo_count: int,
    ) -> ScoreResult:
        """Calculate final score and judgement for a completed pose."""
        clamped_similarity = self._clamp_similarity(similarity)

        if not is_good_match:
            return ScoreResult(
                base_score=0.0,
                combo_multiplier=1.0,
                similarity=clamped_similarity,
                total_score=0.0,
                judgement=JUDGEMENT_MISS,
            )

        base_score = self.good_match_bonus
        judgement = JUDGEMENT_GOOD
        if is_perfect_match:
            base_score = self.perfect_match_bonus
            judgement = JUDGEMENT_PERFECT

        combo_multiplier = 1.0 + (max(0, combo_count) * self.combo_multiplier_step)
        total_score = base_score * combo_multiplier * clamped_similarity

        return ScoreResult(
            base_score=base_score,
            combo_multiplier=combo_multiplier,
            similarity=clamped_similarity,
            total_score=total_score,
            judgement=judgement,
        )
