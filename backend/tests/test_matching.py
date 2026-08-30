"""Scoring and resume chunking.

The scoring functions are pure and tested directly — that is the whole point of
keeping 85% of the score out of the model. If these are right, a score is
defensible even when the rubric is unavailable.
"""

import pytest

from app.services.matching import (
    WEIGHTS,
    SkillCoverage,
    score_experience,
    score_seniority,
    score_skills,
)
from app.services.resume_parser import chunk_resume, guess_years_experience


class FakeSkill:
    def __init__(self, slug: str, name: str) -> None:
        self.slug, self.name = slug, name


def test_weights_sum_to_one() -> None:
    """A drifting total would silently rescale every score in the database."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_the_model_can_shade_a_score_but_not_manufacture_one() -> None:
    """The rubric is capped so a confident model cannot override the arithmetic."""
    assert WEIGHTS["rubric"] <= 0.15
    assert WEIGHTS["must_have_skills"] > WEIGHTS["rubric"] * 2


# --- Skill coverage --------------------------------------------------------


def test_full_coverage_scores_one() -> None:
    required = [FakeSkill("python", "Python"), FakeSkill("kafka", "Kafka")]

    coverage = score_skills(required, {"python", "kafka", "docker"})

    assert coverage.score == 1.0
    assert coverage.missing == []


def test_partial_coverage_is_proportional() -> None:
    required = [FakeSkill("python", "Python"), FakeSkill("kafka", "Kafka")]

    coverage = score_skills(required, {"python"})

    assert coverage.score == 0.5
    assert coverage.missing == ["Kafka"]


def test_a_posting_stating_no_skills_is_not_a_perfect_match() -> None:
    """Returning 1.0 would let a vague posting outscore a detailed one the
    candidate genuinely fits — the worst possible ranking inversion here."""
    assert SkillCoverage().score == 0.5


# --- Experience ------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "years", "expected"),
    [
        (3, 6, 4, 1.0),  # inside the range
        (3, 6, 3, 1.0),  # exactly at the floor
        (5, None, 5, 1.0),  # open-ended, met
        (None, None, 4, 0.5),  # unstated requirement
        (3, 6, None, 0.5),  # unknown candidate years
    ],
)
def test_experience_within_range_scores_full(
    low: int | None, high: int | None, years: float | None, expected: float
) -> None:
    assert score_experience(low, high, years) == pytest.approx(expected)


def test_being_under_qualified_is_penalised_harder_than_being_over() -> None:
    """Three years short of a five-year requirement is usually a rejection.
    Three years beyond it rarely is, so the curves are deliberately asymmetric."""
    under = score_experience(5, 8, 2)
    over = score_experience(5, 8, 11)

    assert under < over
    assert over >= 0.5, "over-qualification should never crater a score"


def test_experience_floor_is_zero_not_negative() -> None:
    assert score_experience(10, 12, 0) >= 0.0


# --- Seniority -------------------------------------------------------------


def test_seniority_matches_implied_level() -> None:
    assert score_seniority("senior", 6) == pytest.approx(1.0)
    assert score_seniority("intern", 0.5) == pytest.approx(1.0)


def test_seniority_degrades_with_distance() -> None:
    close = score_seniority("senior", 4)
    far = score_seniority("principal", 0.5)

    assert close > far
    assert far >= 0.0


def test_unknown_seniority_is_neutral() -> None:
    assert score_seniority(None, 5) == 0.5
    assert score_seniority("senior", None) == 0.5
    assert score_seniority("nonsense", 5) == 0.5


# --- Chunking --------------------------------------------------------------

RESUME = """\
Aryan Jain
Backend Engineer | 4 years of experience

SUMMARY
Backend engineer building LLM-powered systems.

WORK EXPERIENCE
- Built RAG pipelines over enterprise documents using LangChain and Qdrant.
- Designed FastAPI microservices on AWS handling 2 million requests per day.
- Implemented Kafka event pipelines for asynchronous order processing.

SKILLS
Python, FastAPI, PostgreSQL, Kafka, Docker

EDUCATION
B.Tech in Computer Science, Pune Institute of Computer Technology
"""


def test_chunks_are_tagged_with_their_section() -> None:
    chunks = chunk_resume(RESUME)

    sections = {c.section for c in chunks}
    assert {"summary", "experience", "skills", "education"} <= sections


def test_each_bullet_becomes_its_own_chunk() -> None:
    """The unit that answers "does this person know Kafka?" is one
    accomplishment line, not a page."""
    chunks = chunk_resume(RESUME)

    experience = [c.content for c in chunks if c.section == "experience"]
    assert len(experience) == 3
    assert any("Kafka" in c for c in experience)
    assert any("RAG pipelines" in c for c in experience)


def test_bullets_are_never_split_mid_sentence() -> None:
    """Fixed-size chunking cuts bullets in half, so the fragment that should
    have evidenced a skill evidences neither half."""
    chunks = chunk_resume(RESUME)

    kafka = next(c for c in chunks if "Kafka" in c.content)
    assert kafka.content.endswith("processing.")


def test_ordinals_are_contiguous() -> None:
    chunks = chunk_resume(RESUME)

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_empty_resume_produces_no_chunks() -> None:
    assert chunk_resume("") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Backend Engineer | 4 years of experience", 4.0),
        ("6+ years of professional experience in Python", 6.0),
        ("Experience: 3 years", 3.0),
        ("No mention of duration here at all", None),
        ("99 years of experience", None),  # implausible, rejected
    ],
)
def test_years_of_experience_is_read_only_when_stated(text: str, expected: float | None) -> None:
    """Never inferred from employment dates: overlapping roles, internships and
    gaps make that unreliable, and a wrong number skews every score invisibly."""
    assert guess_years_experience(text) == expected
