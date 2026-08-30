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
from app.services.embeddings import embedding_provider
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


def score_seniority(job_seniority: str | None, candidate_years: float | None) -> float:
    """Ordinal distance between the posting's level and the candidate's.

    Years are the only signal available for the candidate side, so the mapping
    is coarse on purpose — it is worth 10% of the total and should not pretend
    to more precision than it has.
    """
    if not job_seniority:
        return 0.5
    try:
        target = SENIORITY_RANK[Seniority(job_seniority)]
    except (KeyError, ValueError):
        return 0.5
    if candidate_years is None:
        return 0.5

    if candidate_years < 1:
        implied = 0
    elif candidate_years < 2:
        implied = 1
    elif candidate_years < 5:
        implied = 2
    elif candidate_years < 8:
        implied = 3
    elif candidate_years < 12:
        implied = 4
    else:
        implied = 5

    distance = abs(target - implied)
    return max(0.0, 1.0 - distance * 0.25)


async def retrieve_evidence(
    session: AsyncSession,
    resume_id: uuid.UUID,
    query: str,
    limit: int = EVIDENCE_PER_REQUIREMENT,
) -> list[ResumeChunk]:
    """Resume passages most relevant to one requirement.

    This is what pgvector is for here. The ordering is done in Postgres so the
    HNSW index is used; pulling chunks into Python to compare them would make
    the index pointless.
    """
    vector = await embedding_provider.embed_query(query)
    return list(
        (
            await session.execute(
                select(ResumeChunk)
                .where(ResumeChunk.resume_id == resume_id)
                .order_by(ResumeChunk.embedding.cosine_distance(vector))
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )


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
