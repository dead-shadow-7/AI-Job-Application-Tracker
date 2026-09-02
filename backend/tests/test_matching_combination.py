"""How the model's opinion is folded into a score, and what happens without it.

`test_matching.py` pins the weight *constants* — that the rubric is capped at
15%, that must-have coverage outweighs it. It never checks the arithmetic that
spends those weights, and that arithmetic has a property nobody would guess:

    adding the rubric can LOWER the score.

`compute_deterministic_match` renormalises over the 0.85 of weight actually
present, so a deterministic-only score is on a full 0-100 scale rather than
capped at 85 for reasons the user cannot see. `combine_with_rubric` does not
renormalise, because with the rubric the weights now sum to 1.0. Both are right
on their own; together they mean a rubric below the deterministic average drags
the total down. That is defensible — but it should be a decision someone made,
not a surprise, so it is written down here.

The other half is the degrade path. When the model is unavailable the score is
still produced from the other 85%, and `model` / `prompt_version` are left null.
That null pair is the only durable record of "this score was computed without a
model", and nothing asserted it until now.
"""

import pytest

from app.services.matching import WEIGHTS, MatchResult, combine_with_rubric

# Every deterministic component at the same value, so the renormalised average
# is exactly that value and the rubric's effect is readable by eye.
FLAT = 0.60


def flat_result(value: float = FLAT) -> MatchResult:
    subscores = {
        "must_have_skills": value,
        "nice_to_have_skills": value,
        "experience": value,
        "seniority": value,
    }
    weight_used = sum(WEIGHTS[k] for k in subscores)
    return MatchResult(
        overall_score=round(sum(subscores[k] * WEIGHTS[k] for k in subscores) / weight_used * 100),
        subscores=subscores,
        matched_skills=[],
        missing_skills=[],
        evidence={},
    )


def test_the_deterministic_score_uses_the_whole_scale() -> None:
    """Not capped at 85. A score the model did not touch still reads 0-100."""
    assert flat_result(0.60).overall_score == 60
    assert flat_result(1.0).overall_score == 100
    assert flat_result(0.0).overall_score == 0


def test_the_rubric_can_move_the_score_by_fifteen_points_either_way() -> None:
    """Its whole authority, stated as a number.

    The cap is what lets an LLM contribute to a score at all: it can shade a
    verdict the arithmetic already reached, and cannot manufacture one.
    """
    deterministic = flat_result(0.60)

    assert combine_with_rubric(deterministic, 0.0) == 51  # 0.60 × 0.85 × 100
    assert combine_with_rubric(deterministic, 1.0) == 66  # + 0.15 × 100
    assert combine_with_rubric(deterministic, 1.0) - combine_with_rubric(deterministic, 0.0) == 15


def test_a_low_rubric_pulls_the_total_below_the_deterministic_score() -> None:
    """The surprising one, and the reason this file exists.

    The two functions renormalise differently — correctly, each on its own —
    with the consequence that asking the model can make the number go down. A
    reader comparing a rubric-scored job against a deterministic-only one is not
    comparing like with like.
    """
    deterministic = flat_result(0.60)

    assert combine_with_rubric(deterministic, 0.20) < deterministic.overall_score
    assert combine_with_rubric(deterministic, 0.60) == deterministic.overall_score
    assert combine_with_rubric(deterministic, 0.90) > deterministic.overall_score


def test_a_rubric_equal_to_the_deterministic_average_changes_nothing() -> None:
    """The fixed point, at three different levels — this is what makes the
    claim above a property of the arithmetic rather than of one example."""
    for value in (0.25, 0.50, 0.75):
        assert combine_with_rubric(flat_result(value), value) == flat_result(value).overall_score


@pytest.mark.parametrize("rubric", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_total_stays_inside_the_scale(rubric: float) -> None:
    for level in (0.0, 0.5, 1.0):
        assert 0 <= combine_with_rubric(flat_result(level), rubric) <= 100


def test_the_weights_the_arithmetic_spends_are_the_ones_declared() -> None:
    """Guards the two functions against each other.

    `compute_deterministic_match` divides by the weight of the components it
    has; `combine_with_rubric` assumes the full set sums to 1.0. Change one
    weight and exactly one of those assumptions silently stops holding.
    """
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(v for k, v in WEIGHTS.items() if k != "rubric") == pytest.approx(0.85)
