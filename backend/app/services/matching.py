"""Resume-to-job scoring.

Deliberately **not** cosine similarity between a resume embedding and a job
embedding. Those are stylistically different documents, so that number lands
near the same value for every pair — it does not separate a great fit from a
poor one, and it produces a figure that cannot be explained to the person whose
job search depends on it.

Instead the score is a weighted sum of components that can each be shown and
argued with. Vectors do the job they are actually good at: retrieving which of
your resume bullets evidence a given requirement, so the narrative is grounded
in your real text rather than in a summary of it.

Every component is stored on the analysis, so the UI can answer "why 72?".
"""

import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import SENIORITY_RANK, Seniority
from app.models.job import Job, JobSkill
from app.models.resume import Resume, ResumeChunk
from app.models.skill import Skill
from app.services import embeddings
from app.services.resume_parser import seniority_from_title
from app.services.skills import extract_skills_from_text

logger = logging.getLogger(__name__)

# Weights sum to 1.0. Must-have coverage dominates because it is what a
# recruiter screens on first; the LLM rubric is capped at 15% deliberately, so
# a model having an opinion can shade a score but never manufacture one.
WEIGHTS = {
    "must_have_skills": 0.45,
    "nice_to_have_skills": 0.15,
    "experience": 0.15,
    "seniority": 0.10,
    "rubric": 0.15,
}

EVIDENCE_PER_REQUIREMENT = 3

# Multipliers on cosine *distance*, so above 1.0 is a penalty. Not all passages
# are equally good evidence even when they are equally on-topic: a skills list
# naming Kafka is a claim, an experience bullet describing a Kafka pipeline is
# proof, and a degree that mentions distributed systems is neither. Retrieval
# handing the rubric the skills line instead of the accomplishment is how a
# requirement the candidate genuinely meets gets judged as barely evidenced.
SECTION_PENALTY = {
    "experience": 1.0,
    "projects": 1.05,
    "publications": 1.15,
    "summary": 1.2,
    "skills": 1.25,
    "certifications": 1.3,
    "awards": 1.35,
    "education": 1.4,
}
UNSECTIONED_PENALTY = 1.1
"""Untagged text is usually experience the parser could not attribute, so it is
treated as slightly worse than experience rather than as worst."""

# Candidates pulled before re-ranking, as a multiple of what is returned. The
# vector index still does the search; weighting only reorders its shortlist, so
# a genuinely irrelevant experience bullet cannot outrank a relevant one merely
# for being in the right section.
EVIDENCE_OVERFETCH = 4


@dataclass
class SkillCoverage:
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = len(self.matched) + len(self.missing)
        # No stated requirements is not a perfect match, it is an absence of
        # evidence. Returning 1.0 would let a vague posting outscore a detailed
        # one the candidate genuinely fits.
        if total == 0:
            return 0.5
        return len(self.matched) / total


def score_skills(required: list[Skill], possessed: set[str]) -> SkillCoverage:
    coverage = SkillCoverage()
    for skill in required:
        if skill.slug in possessed:
            coverage.matched.append(skill.name)
        else:
            coverage.missing.append(skill.name)
    return coverage


def score_experience(
    required_min: int | None, required_max: int | None, candidate_years: float | None
) -> float:
    """How well the candidate's years fit the stated range.

    Being *over*-qualified is penalised far more gently than being under: three
    years short of a five-year requirement is usually a rejection, whereas three
    years beyond it rarely is.
    """
    if required_min is None and required_max is None:
        return 0.5  # unstated, same reasoning as SkillCoverage.score
    if candidate_years is None:
        return 0.5  # unknown on our side; neither reward nor punish

    low = required_min if required_min is not None else 0
    high = required_max if required_max is not None else low + 3

    if low <= candidate_years <= high:
        return 1.0

    if candidate_years < low:
        shortfall = low - candidate_years
        return max(0.0, 1.0 - (shortfall / max(low, 1)) * 0.8)

    excess = candidate_years - high
    return max(0.5, 1.0 - (excess / max(high, 1)) * 0.25)


def _level_from_years(candidate_years: float) -> int:
    """Seniority implied by tenure alone. Coarse on purpose — it is a proxy."""
    if candidate_years < 1:
        return 0
    if candidate_years < 2:
        return 1
    if candidate_years < 5:
        return 2
    if candidate_years < 8:
        return 3
    if candidate_years < 12:
        return 4
    return 5


def _level_from_titles(titles: list[str]) -> int | None:
    """The highest level any held title states outright.

    The most senior title reached, not the current one: a Staff Engineer now
    consulting is not junior again. Titles stating no level ("Software
    Engineer") contribute nothing rather than defaulting to mid — a resume of
    those should fall through to the years proxy, not be pinned at 2.
    """
    ranks = [SENIORITY_RANK[level] for title in titles if (level := seniority_from_title(title))]
    return max(ranks) if ranks else None


def score_seniority(
    job_seniority: str | None,
    candidate_years: float | None,
    candidate_titles: list[str] | None = None,
) -> float:
    """Ordinal distance between the posting's level and the candidate's.

    Held titles are preferred over tenure where the resume gives them. Years are
    only a proxy for level — they cannot tell a six-year engineer who was
    promoted to Staff from one who was not — whereas a title is the candidate's
    own statement of where they landed.
    """
    if not job_seniority:
        return 0.5
    try:
        target = SENIORITY_RANK[Seniority(job_seniority)]
    except (KeyError, ValueError):
        return 0.5

    implied = _level_from_titles(candidate_titles or [])
    if implied is None:
        if candidate_years is None:
            return 0.5
        implied = _level_from_years(candidate_years)

    distance = abs(target - implied)
    return max(0.0, 1.0 - distance * 0.25)


async def retrieve_evidence(
    session: AsyncSession,
    resume_id: uuid.UUID,
    query: str,
    limit: int = EVIDENCE_PER_REQUIREMENT,
) -> list[ResumeChunk]:
    """Resume passages most relevant to one requirement, best evidence first.

    Two steps, and the split is deliberate. Postgres does the vector search, so
    the HNSW index is used — pulling chunks into Python to compare them would
    make the index pointless. Section weighting is then applied to that
    shortlist in Python, because folding it into the ORDER BY would make the
    expression non-indexable and turn every requirement into a sequential scan.
    """
    vector = await embeddings.embedding_provider.embed_query(query)
    distance = ResumeChunk.embedding.cosine_distance(vector).label("distance")

    rows = (
        await session.execute(
            select(ResumeChunk, distance)
            .where(ResumeChunk.resume_id == resume_id)
            .order_by(distance)
            .limit(limit * EVIDENCE_OVERFETCH)
        )
    ).all()

    ranked = sorted(
        rows,
        key=lambda row: (
            row.distance * SECTION_PENALTY.get(row.ResumeChunk.section or "", UNSECTIONED_PENALTY)
        ),
    )
    return [row.ResumeChunk for row in ranked[:limit]]


@dataclass
class MatchResult:
    overall_score: int
    subscores: dict[str, float]
    matched_skills: list[str]
    missing_skills: list[str]
    evidence: dict[str, list[str]]


async def compute_deterministic_match(
    session: AsyncSession, resume: Resume, job: Job
) -> MatchResult:
    """The auditable part of the score, with no LLM involved.

    Kept separate from the rubric so it is testable in isolation and so a score
    still exists when the model is unavailable or rate-limited.
    """
    job_skills = (
        await session.execute(
            select(JobSkill, Skill)
            .join(Skill, Skill.id == JobSkill.skill_id)
            .where(JobSkill.job_id == job.id)
        )
    ).all()

    required = [skill for js, skill in job_skills if js.is_required]
    preferred = [skill for js, skill in job_skills if not js.is_required]

    resume_skills = await extract_skills_from_text(session, resume.parsed_text)
    possessed = {s.slug for s in resume_skills}

    must = score_skills(required, possessed)
    nice = score_skills(preferred, possessed)
    experience = score_experience(
        job.years_experience_min,
        job.years_experience_max,
        float(resume.years_experience) if resume.years_experience is not None else None,
    )
    seniority = score_seniority(
        job.seniority,
        float(resume.years_experience) if resume.years_experience is not None else None,
        [title for p in resume.positions if (title := p.get("title"))],
    )

    subscores = {
        "must_have_skills": must.score,
        "nice_to_have_skills": nice.score,
        "experience": experience,
        "seniority": seniority,
    }

    # Rubric excluded here; renormalise over the components actually present so
    # a deterministic-only score is still on a 0-100 scale rather than capped
    # at 85 for reasons the user cannot see.
    weight_used = sum(WEIGHTS[k] for k in subscores)
    weighted = sum(subscores[k] * WEIGHTS[k] for k in subscores) / weight_used

    return MatchResult(
        overall_score=round(weighted * 100),
        subscores=subscores,
        matched_skills=must.matched + nice.matched,
        missing_skills=must.missing,
        evidence={},
    )


def combine_with_rubric(deterministic: MatchResult, rubric_score: float) -> int:
    """Fold the LLM's judgment in at its capped weight."""
    subscores = {**deterministic.subscores, "rubric": rubric_score}
    weighted = sum(subscores[k] * WEIGHTS[k] for k in subscores)
    return round(weighted * 100)
