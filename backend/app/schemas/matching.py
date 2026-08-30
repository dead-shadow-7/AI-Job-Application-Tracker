from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RubricJudgment(BaseModel):
    """What the LLM returns. Strict-schema, like extraction."""

    score: float = Field(description="0.0-1.0 fit on evidenced requirements alone.")
    strengths: list[str] = Field(
        description="Concrete strengths, each traceable to a retrieved resume passage."
    )
    gaps: list[str] = Field(
        description="Specific unmet requirements. Name the missing thing, not a vague weakness."
    )
    narrative: str = Field(
        description=(
            "Two or three sentences addressed to the candidate: is this worth "
            "applying to, and what would most improve their odds?"
        )
    )


class MatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    resume_id: UUID
    job_id: UUID
    overall_score: int

    # Every component that produced overall_score, so the UI can answer
    # "why 72?" instead of asking the user to trust a number.
    subscores: dict[str, float] = Field(default_factory=dict)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    narrative: str | None = None

    model: str | None = None
    prompt_version: str | None = None
    created_at: datetime
