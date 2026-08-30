from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import EmploymentType, SalaryPeriod, Seniority, WorkMode
from app.schemas.job import JobCreate, RequirementIn


class IngestRequest(BaseModel):
    """Paste a job description.

    Deliberately the same contract a browser extension would post, so adding one
    later is a new client rather than a new endpoint: it captures the rendered
    DOM as the logged-in user and sends it here unchanged.
    """

    raw_text: str = Field(min_length=1, max_length=200_000)
    url: str | None = None
    source_platform: str | None = Field(default=None, max_length=60)


class ExtractionWarning(BaseModel):
    message: str


class IngestPreview(BaseModel):
    """The extraction, for review before anything is saved.

    Nothing is written by the ingest call. The user confirms or corrects, then
    posts the result to the ordinary create endpoint — so a bad extraction costs
    an edit rather than a wrong row that has to be found and fixed later.
    """

    model_config = ConfigDict(from_attributes=True)

    job: JobCreate
    confidence: Decimal
    needs_review: bool
    warnings: list[str] = Field(default_factory=list)
    dropped_fields: list[str] = Field(default_factory=list)
    unmatched_skills: list[str] = Field(default_factory=list)

    model: str
    prompt_version: str
    tokens_used: int
    latency_ms: int

    duplicate_of: UUID | None = Field(
        default=None,
        description="An application you already track with an identical description.",
    )


class ParsedJobFields(BaseModel):
    """Flat view of the extracted fields, for the review form."""

    company_name: str
    title: str
    seniority: Seniority | None = None
    employment_type: EmploymentType | None = None
    work_mode: WorkMode | None = None
    location: str | None = None
    salary_min: Decimal | None = None
    salary_max: Decimal | None = None
    salary_currency: str | None = None
    salary_period: SalaryPeriod | None = None
    years_experience_min: int | None = None
    years_experience_max: int | None = None
    responsibilities: str | None = None
    requirements: list[RequirementIn] = Field(default_factory=list)
    skill_slugs: list[str] = Field(default_factory=list)
