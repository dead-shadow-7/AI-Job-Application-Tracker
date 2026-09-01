"""Model registry.

Every model must be imported here so ``Base.metadata`` is fully populated
before Alembic autogenerate runs — otherwise it cheerfully generates a
migration that drops the tables it cannot see.
"""

from app.db.base import Base
from app.models.application import Application, ApplicationEvent, InterviewStage
from app.models.company import Company
from app.models.contact import Contact
from app.models.conversation import AgentMessage
from app.models.followup import FollowUpRule
from app.models.job import Job, JobRequirement, JobSkill
from app.models.resume import JobEmbedding, MatchAnalysis, Resume, ResumeChunk
from app.models.skill import Skill
from app.models.user import User

__all__ = [
    "AgentMessage",
    "Application",
    "ApplicationEvent",
    "Base",
    "Company",
    "Contact",
    "FollowUpRule",
    "InterviewStage",
    "Job",
    "JobEmbedding",
    "JobRequirement",
    "JobSkill",
    "MatchAnalysis",
    "Resume",
    "ResumeChunk",
    "Skill",
    "User",
]
