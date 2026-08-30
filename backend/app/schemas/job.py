from datetime import date
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import (
    EmploymentType,
    RequirementKind,
    SalaryPeriod,
    Seniority,
    WorkMode,
)


class CompanyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    domain: str | None = None
    industry: str | None = None
    location: str | None = None


class RequirementIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    kind: RequirementKind = RequirementKind.MUST
    category: str | None = Field(default=None, max_length=60)


class RequirementRead(RequirementIn):
    model_config = ConfigDict(from_attributes=True)

    id: UUID


class SkillRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    slug: str
    category: str | None = None


class JobSkillRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    skill: SkillRead
    is_required: bool
    years_required: int | None = None


class JobBase(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    seniority: Seniority | None = None
    employment_type: EmploymentType | None = None
    work_mode: WorkMode | None = None
    location: str | None = Field(default=None, max_length=255)
    url: str | None = None
    source_platform: str | None = Field(default=None, max_length=60)
    posted_at: date | None = None
    description: str | None = None
    responsibilities: str | None = None
    salary_min: Decimal | None = Field(default=None, ge=0)
    salary_max: Decimal | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    salary_period: SalaryPeriod | None = None
    years_experience_min: int | None = Field(default=None, ge=0, le=60)
    years_experience_max: int | None = Field(default=None, ge=0, le=60)

    @model_validator(mode="after")
    def _ranges_ordered(self) -> "JobBase":
        """Mirror the database CHECK constraints so the caller gets a 422 naming
        the field, rather than a 500 from a constraint violation."""
        # Bound to locals so the None checks narrow the types. mypy does not
        # carry narrowing through an intermediate boolean variable.
        salary_min, salary_max = self.salary_min, self.salary_max
        years_min, years_max = self.years_experience_min, self.years_experience_max

        if salary_min is not None and salary_max is not None and salary_min > salary_max:
            raise ValueError("salary_min cannot exceed salary_max")
        if years_min is not None and years_max is not None and years_min > years_max:
            raise ValueError("years_experience_min cannot exceed years_experience_max")
        if (salary_min is not None or salary_max is not None) and not self.salary_currency:
            raise ValueError("salary_currency is required when a salary is given")
        return self


class JobCreate(JobBase):
    """A job entered by hand. Phase 2 fills the same shape from an LLM."""

    company_name: str = Field(min_length=1, max_length=300)
    company_domain: str | None = Field(default=None, max_length=255)
    requirements: list[RequirementIn] = Field(default_factory=list)
    skill_slugs: list[str] = Field(default_factory=list)


class JobUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=300)
    seniority: Seniority | None = None
    employment_type: EmploymentType | None = None
    work_mode: WorkMode | None = None
    location: str | None = None
    url: str | None = None
    source_platform: str | None = None
    posted_at: date | None = None
    description: str | None = None
    responsibilities: str | None = None
    salary_min: Decimal | None = Field(default=None, ge=0)
    salary_max: Decimal | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    salary_period: SalaryPeriod | None = None
    years_experience_min: int | None = Field(default=None, ge=0, le=60)
    years_experience_max: int | None = Field(default=None, ge=0, le=60)


class JobRead(JobBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    company: CompanyRead
    requirements: list[RequirementRead] = Field(default_factory=list)
    skills: list[JobSkillRead] = Field(default_factory=list)
    extraction_confidence: Decimal | None = None
    extraction_model: str | None = None


class JobSummary(BaseModel):
    """Trimmed shape for list rows — the full description would dominate the
    payload of a hundred-row table for no benefit."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    company: CompanyRead
    location: str | None = None
    work_mode: WorkMode | None = None
    seniority: Seniority | None = None
    salary_min: Decimal | None = None
    salary_max: Decimal | None = None
    salary_currency: str | None = None
    salary_period: SalaryPeriod | None = None
    url: str | None = None
    source_platform: str | None = None
