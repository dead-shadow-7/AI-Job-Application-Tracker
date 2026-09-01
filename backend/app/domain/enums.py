"""Domain vocabulary.

These are stored as ``varchar`` with a CHECK constraint rather than as native
Postgres enums. Native enums read better in psql, but ``ALTER TYPE ... ADD
VALUE`` cannot be rolled back and values can never be removed — and this
vocabulary is *known* to grow: Phase 4 adds follow-up events, Phase 5 adds more.
Widening a CHECK constraint is an ordinary, reversible migration.
"""

from enum import StrEnum


class ApplicationStatus(StrEnum):
    """Where an application currently stands.

    Derived from the event log, never set directly — see
    ``app.services.events.append_event``.
    """

    SAVED = "saved"
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEWING = "interviewing"
    OFFER = "offer"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    GHOSTED = "ghosted"


TERMINAL_STATUSES = frozenset(
    {
        ApplicationStatus.ACCEPTED,
        ApplicationStatus.REJECTED,
        ApplicationStatus.WITHDRAWN,
        ApplicationStatus.GHOSTED,
    }
)
"""Statuses the follow-up sweep skips — nothing is pending from your side."""


class EventType(StrEnum):
    """One entry in an application's timeline."""

    SAVED = "saved"
    APPLIED = "applied"
    ASSESSMENT_RECEIVED = "assessment_received"
    SCREENING_SCHEDULED = "screening_scheduled"
    SCREENING_DONE = "screening_done"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    INTERVIEW_DONE = "interview_done"
    OFFER_RECEIVED = "offer_received"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    MARKED_GHOSTED = "marked_ghosted"
    # Events that record contact without moving the application forward.
    RECRUITER_REPLY = "recruiter_reply"
    FOLLOW_UP_SENT = "follow_up_sent"
    NOTE_ADDED = "note_added"


STATUS_BY_EVENT: dict[EventType, ApplicationStatus] = {
    EventType.SAVED: ApplicationStatus.SAVED,
    EventType.APPLIED: ApplicationStatus.APPLIED,
    EventType.ASSESSMENT_RECEIVED: ApplicationStatus.SCREENING,
    EventType.SCREENING_SCHEDULED: ApplicationStatus.SCREENING,
    EventType.SCREENING_DONE: ApplicationStatus.SCREENING,
    EventType.INTERVIEW_SCHEDULED: ApplicationStatus.INTERVIEWING,
    EventType.INTERVIEW_DONE: ApplicationStatus.INTERVIEWING,
    EventType.OFFER_RECEIVED: ApplicationStatus.OFFER,
    EventType.ACCEPTED: ApplicationStatus.ACCEPTED,
    EventType.REJECTED: ApplicationStatus.REJECTED,
    EventType.WITHDRAWN: ApplicationStatus.WITHDRAWN,
    EventType.MARKED_GHOSTED: ApplicationStatus.GHOSTED,
}
"""Which events move the application, and where to.

Events absent from this map — ``recruiter_reply``, ``follow_up_sent``,
``note_added`` — are recorded on the timeline but leave the status alone. They
still refresh "last activity", which is what the Phase 4 follow-up rules
actually measure: a recruiter replying means the application is not stale, even
though it has not advanced.
"""


class EventSource(StrEnum):
    """Who created an event. Agent-written rows are attributable and reversible."""

    MANUAL = "manual"
    AGENT = "agent"
    SYSTEM = "system"


class WorkMode(StrEnum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"


class EmploymentType(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"


class Seniority(StrEnum):
    """Ordered — Phase 3 scores seniority fit by ordinal distance."""

    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    LEAD = "lead"


SENIORITY_RANK: dict[Seniority, int] = {
    Seniority.INTERN: 0,
    Seniority.JUNIOR: 1,
    Seniority.MID: 2,
    Seniority.SENIOR: 3,
    Seniority.STAFF: 4,
    Seniority.LEAD: 4,
    Seniority.PRINCIPAL: 5,
}


class RequirementKind(StrEnum):
    MUST = "must"
    NICE = "nice"


class InterviewStageType(StrEnum):
    HR_SCREEN = "hr_screen"
    RECRUITER_CALL = "recruiter_call"
    TECHNICAL = "technical"
    CODING = "coding"
    SYSTEM_DESIGN = "system_design"
    MANAGERIAL = "managerial"
    HIRING_MANAGER = "hiring_manager"
    CULTURE_FIT = "culture_fit"
    TAKE_HOME = "take_home"
    FINAL = "final"
    OTHER = "other"


class StageOutcome(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class FollowUpAction(StrEnum):
    """What a fired rule does.

    SUGGEST_FOLLOWUP surfaces the application for you to act on. MARK_GHOSTED
    is the long-stop that closes one nobody was ever going to answer, so the
    active list stays honest rather than accumulating dead entries forever.
    """

    SUGGEST_FOLLOWUP = "suggest_followup"
    MARK_GHOSTED = "mark_ghosted"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SalaryPeriod(StrEnum):
    YEAR = "year"
    MONTH = "month"
    HOUR = "hour"


def check_constraint(enum_cls: type[StrEnum]) -> str:
    """Render a SQL ``IN (...)`` list for a CHECK constraint.

    Keeps the migration and the Python enum from drifting: adding a member and
    forgetting the constraint is the failure mode this exists to prevent.
    """
    values = ", ".join(f"'{member.value}'" for member in enum_cls)
    return f"IN ({values})"
