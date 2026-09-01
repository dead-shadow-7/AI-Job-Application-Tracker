"""What the assistant may say, and what it may ask to do.

The agent has **no write tools**. Every ``propose_*`` tool returns an intention;
a separate confirmed endpoint performs the write. That is the whole safety
design in one sentence: a model that cannot write cannot write to the wrong row.

Proposals are polymorphic because the agent can now ask for four different
kinds of change, but they share one property that the design depends on:
``summary`` and ``details`` must describe the payload *completely*. The confirm
card is the only thing standing between a misread instruction and a real write,
and a card that hides a field the model filled in is not a check at all.
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from app.domain.enums import (
    EmploymentType,
    EventType,
    InterviewStageType,
    Priority,
    SalaryPeriod,
    Seniority,
    WorkMode,
)
from app.schemas.application import ApplicationRead
from app.schemas.job import JobCreate

# Long enough to paste a whole job posting in, which is what people actually do
# — 2,000 characters cut off mid-sentence on a real one. Not unbounded, because
# the message is replayed as history on every later turn, so an oversized one is
# paid for repeatedly rather than once. Pasting a posting for *extraction* still
# belongs in /jobs/ingest, which takes 200,000 and validates each field against
# the text; the assistant only reads what it is given.
MAX_MESSAGE_CHARS = 10_000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


ActionKind = Literal[
    "append_event",
    "create_application",
    "create_from_posting",
    "update_application",
    "edit_description",
    "schedule_interview",
    "delete_application",
]

# The one action that cannot be undone by appending a correcting event, which
# is how every other agent write is reversed. Flagged so the UI can style the
# card as destructive rather than leaving it to look like any other change.
DESTRUCTIVE: frozenset[str] = frozenset({"delete_application"})


class ActionPreview(BaseModel):
    """A proposal the user can accept or reject.

    ``details`` is rendered line by line on the confirm card and must name every
    value that would be written. ``payload`` is the confirm request body minus
    its ``kind``, so the client echoes it back rather than reconstructing it —
    which keeps the thing displayed and the thing executed from drifting apart.

    The server re-validates that payload against the typed request below. The
    client is not trusted; it is merely spared the mapping.
    """

    model_config = ConfigDict(from_attributes=True)

    kind: ActionKind
    summary: str = Field(description="One line: what will happen if confirmed.")
    details: list[str] = Field(
        default_factory=list, description="Every field that would be written, one per line."
    )
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_destructive(self) -> bool:
        return self.kind in DESTRUCTIVE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def destructive(self) -> bool:
        """Serialised so the client does not have to keep its own copy of which
        kinds are irreversible — a list that would silently fall out of date the
        next time one is added."""
        return self.is_destructive

    # Present when the action targets an application that already exists. A
    # creation has nothing to resolve, so these stay null — the card falls back
    # to `summary`, which is why that field is required rather than optional.
    application_id: UUID | None = None
    application_label: str | None = None
    confidence: float | None = None
    matched_on: str | None = None


class ChatAttachment(BaseModel):
    """A stored document shown alongside the reply, not inside it.

    Rendered by the client from the database text. It never passes through the
    model's output, which is the point: asked to relay a job description the
    model rewrites it, and sometimes announces it without reproducing anything
    at all. Nothing here is generated — a drafted follow-up belongs in `message`
    because the model genuinely wrote it.
    """

    kind: Literal["job_description"]
    title: str
    body: str


class ChatResponse(BaseModel):
    message: str
    pending_action: ActionPreview | None = None
    attachments: list[ChatAttachment] = Field(default_factory=list)

    model: str | None = None
    prompt_version: str | None = None
    tools_used: list[str] = Field(
        default_factory=list,
        description="Which tools ran, in order. Shown so an answer can be traced to its source.",
    )
    total_tokens: int = 0


# --- Confirmation ------------------------------------------------------------
#
# Typed per kind rather than an opaque dict. The agent's payload passes through
# the client on its way here, so it is untrusted input by the time it lands —
# and "the model proposed it" is not validation.


class ConfirmEvent(BaseModel):
    """Log something on an existing timeline."""

    kind: Literal["append_event"] = "append_event"
    application_id: UUID
    event_type: EventType
    occurred_at: datetime | None = None
    note: str | None = Field(default=None, max_length=2000)


class ConfirmCreateApplication(BaseModel):
    """Track a job the user described in conversation.

    Salary, requirements and skills are deliberately absent. Those come from the
    paste-a-description flow, which validates them against the posting text —
    salary in particular is dropped there unless it appears verbatim, because
    models invent plausible bands. Routing a number around that check through
    chat would defeat it. Anything the user wants recorded regardless goes in
    ``notes``, where it reads as a remark rather than as a verified field.
    """

    kind: Literal["create_application"]
    company_name: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    url: str | None = Field(default=None, max_length=1000)
    location: str | None = Field(default=None, max_length=200)
    work_mode: WorkMode | None = None
    source_platform: str | None = Field(default=None, max_length=60)
    notes: str | None = Field(default=None, max_length=4000)

    initial_event: Literal[EventType.SAVED, EventType.APPLIED] = EventType.SAVED
    # Days, not a date. The assistant is told it does not know today's date, and
    # asking it for one anyway is how you get an event stamped in 2024.
    occurred_days_ago: int = Field(default=0, ge=0, le=3650)


class ConfirmCreateFromPosting(JobCreate):
    """Track a posting the user pasted into the conversation, in full.

    Unlike the sibling above, this one *does* carry salary, requirements and
    skills — because they came out of the ingestion graph, which validated them
    against the pasted text rather than taking the model's word. The check that
    matters is not "did the model produce a number" but "does the number appear
    in the source", and that has already run by the time this is built.

    Extends JobCreate so the fields cannot drift from the ones the paste screen
    saves; both end up in the same `create_application` call.
    """

    kind: Literal["create_from_posting"]
    initial_event: Literal[EventType.SAVED, EventType.APPLIED] = EventType.APPLIED
    occurred_days_ago: int = Field(default=0, ge=0, le=3650)


CLEARABLE = frozenset(
    {
        "notes",
        "location",
        "work_mode",
        "seniority",
        "employment_type",
        "salary_min",
        "salary_max",
        "salary_currency",
        "salary_period",
        "years_experience_min",
        "years_experience_max",
        "source_platform",
        "url",
    }
)


class ConfirmUpdateApplication(BaseModel):
    """Correct any tracked detail.

    Two records behind one action: priority and notes are yours, the rest
    describe the posting and live on the shared job row. The assistant does not
    need to know which is which, and neither does the person confirming.

    Status is absent by design — it is derived from the event log, so moving an
    application means appending an event.

    `clear` is an explicit list rather than "a field sent as null", because
    optional tool parameters are declared nullable and models emit null for
    arguments they simply have nothing to say about. Reading those as "empty
    this column" would wipe a salary every time the model mentioned a priority.
    """

    kind: Literal["update_application"]
    application_id: UUID

    priority: Priority | None = None
    notes: str | None = Field(default=None, max_length=4000)

    title: str | None = Field(default=None, min_length=1, max_length=300)
    location: str | None = Field(default=None, max_length=200)
    work_mode: WorkMode | None = None
    seniority: Seniority | None = None
    employment_type: EmploymentType | None = None
    salary_min: Decimal | None = Field(default=None, ge=0)
    salary_max: Decimal | None = Field(default=None, ge=0)
    salary_currency: str | None = Field(default=None, min_length=3, max_length=3)
    salary_period: SalaryPeriod | None = None
    years_experience_min: int | None = Field(default=None, ge=0, le=60)
    years_experience_max: int | None = Field(default=None, ge=0, le=60)
    source_platform: str | None = Field(default=None, max_length=60)
    url: str | None = Field(default=None, max_length=1000)

    clear: list[str] = Field(default_factory=list)

    @field_validator("clear")
    @classmethod
    def _known_fields(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - CLEARABLE)
        if unknown:
            raise ValueError(f"Cannot clear: {', '.join(unknown)}")
        return value


class ConfirmEditDescription(BaseModel):
    """Replace a stored posting with a trimmed version of itself.

    The text was computed server-side by deleting whole lines the user quoted,
    never generated. A model handed a description and asked to correct it
    returns a rewrite — shorter, reflowed, quietly missing things — and here
    that would be written over the original.
    """

    kind: Literal["edit_description"]
    application_id: UUID
    description: str = Field(min_length=1, max_length=200_000)


class ConfirmScheduleInterview(BaseModel):
    """Add a round, and record that it was scheduled.

    Both, together: the stage carries the future date, and the event records
    that the scheduling happened *now*. An event stamped in the future would
    push ``last_activity_at`` forward and make a stalled application look fresh,
    silently disabling the follow-up detection this project exists for.
    """

    kind: Literal["schedule_interview"]
    application_id: UUID
    stage_type: InterviewStageType
    in_days: int = Field(default=0, ge=0, le=365, description="Days from now.")
    round_number: int | None = Field(default=None, ge=1, le=20)
    interviewer: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=4000)


class ConfirmDeleteApplication(BaseModel):
    """Remove an application and its whole history.

    The only agent action that is not reversible. Every other write is an event
    appended to a log, undone by appending a correction; this destroys the log.
    It exists because the alternative was worse — the assistant could create an
    application but not remove one, so its own mistake had to be cleaned up by
    hand. The proposal says how many events go with it, and offers `withdrawn`
    as the reversible alternative.
    """

    kind: Literal["delete_application"]
    application_id: UUID


ConfirmRequest = Annotated[
    ConfirmEvent
    | ConfirmCreateApplication
    | ConfirmCreateFromPosting
    | ConfirmUpdateApplication
    | ConfirmEditDescription
    | ConfirmScheduleInterview
    | ConfirmDeleteApplication,
    Field(discriminator="kind"),
]


class ConfirmResult(BaseModel):
    """What the confirmed action did.

    An envelope rather than the application itself, because a deletion has no
    application to return and handing back the row that no longer exists would
    be a lie the client has no way to detect.
    """

    kind: ActionKind
    summary: str = Field(description="Past tense: what was done.")
    application: ApplicationRead | None = None
