"""Turning a finished ingestion run into a saveable draft.

Shared because there are now two ways to hand the tracker a job posting — the
paste screen and the assistant — and they must produce the same record. Written
twice they would drift, and the drift would be invisible: both paths would save
something plausible, one of them just quietly missing skills or salary.
"""

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from app.agent.validation import ValidationReport
from app.schemas.extraction import ExtractedJob
from app.schemas.ingest import JobDraft
from app.schemas.job import RequirementIn
from app.services.skills import SkillResolution

# Below this a message is a sentence, not a posting. Running extraction on "add
# the Amazon SDE role" burns a model call to learn what the sentence already
# says, and returns a confident record built from nothing.
MIN_POSTING_CHARS = 400

# Below this, the review screen opens with fields flagged rather than merely
# available. The model reports its own confidence; validation warnings override
# it, since a confident extraction with a discarded salary still needs eyes.
REVIEW_CONFIDENCE_THRESHOLD = 0.75


def needs_review(confidence: float, report: ValidationReport) -> bool:
    """Whether the review screen should open with fields flagged.

    Here rather than inline in the endpoint because the eval suite has to reach
    the same verdict from the same inputs. Measuring a copy of this expression
    would measure the copy: the two would agree until someone changed one, which
    is exactly the regression worth catching.
    """
    return (
        confidence < REVIEW_CONFIDENCE_THRESHOLD
        or bool(report.warnings)
        or bool(report.dropped_fields)
    )


def build_job_draft(
    state: Mapping[str, Any], *, url: str | None = None, source_platform: str | None = None
) -> JobDraft:
    """The extraction, shaped for the create endpoint.

    ``description`` is the cleaned source text rather than anything the model
    produced, so what gets stored is the posting as pasted.
    """
    extracted: ExtractedJob = state["extracted"]
    skills: SkillResolution = state.get("skills") or SkillResolution()

    return JobDraft(
        company_name=extracted.company_name,
        title=extracted.title,
        seniority=extracted.seniority,
        employment_type=extracted.employment_type,
        work_mode=extracted.work_mode,
        location=extracted.location,
        url=url,
        source_platform=source_platform,
        description=state["cleaned_text"],
        responsibilities=extracted.responsibilities,
        salary_min=_decimal(extracted.salary.min_amount),
        salary_max=_decimal(extracted.salary.max_amount),
        salary_currency=extracted.salary.currency,
        salary_period=extracted.salary.period,
        years_experience_min=extracted.years_experience_min,
        years_experience_max=extracted.years_experience_max,
        requirements=[RequirementIn(text=r.text, kind=r.kind) for r in extracted.requirements],
        skill_slugs=skills.slugs,
    )


def _decimal(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))
