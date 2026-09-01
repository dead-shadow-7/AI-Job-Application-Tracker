"""What the assistant may say, and what it may ask to do.

The agent has **no write tools**. It returns a proposal, and a separate
confirmed endpoint performs the write. That is the whole safety design in one
sentence: a model that cannot write cannot write to the wrong row.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import EventType


class ProposedAction(BaseModel):
    """A change the assistant wants to make, pending confirmation."""

    kind: Literal["append_event", "none"] = Field(
        description="'append_event' to log something on a timeline, 'none' if answering only."
    )
    application_query: str | None = Field(
        description=(
            "How the user referred to the application — 'Amazon', 'the backend "
            "role'. Copy their words; do not invent an identifier. Null when "
            "kind is 'none'."
        )
    )
    event_type: EventType | None = Field(
        description="The timeline event to append. Null when kind is 'none'."
    )
    note: str | None = Field(description="Optional note to attach to the event.")


class AgentReply(BaseModel):
    """One turn of the assistant."""

    message: str = Field(
        description=(
            "Your reply to the user, in one or two sentences. Plain and direct. "
            "If you are proposing a change, say what you are about to do."
        )
    )
    action: ProposedAction


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ActionPreview(BaseModel):
    """A proposal the user can accept or reject.

    Carries the resolved target so the UI shows the actual application rather
    than the phrase the user typed — confirming "mark Amazon as rejected"
    without seeing *which* Amazon defeats the point of confirming.
    """

    model_config = ConfigDict(from_attributes=True)

    kind: Literal["append_event"]
    event_type: EventType
    note: str | None = None

    application_id: UUID
    application_label: str
    confidence: float
    matched_on: str


class ChatResponse(BaseModel):
    message: str

    # Exactly one of these is set when the assistant wants to act.
    # `pending_action` means resolution succeeded and the user need only
    # confirm; `disambiguation` means it did not, and the user must choose.
    pending_action: ActionPreview | None = None
    disambiguation: list[str] = Field(default_factory=list)

    model: str | None = None
    prompt_version: str | None = None


class ConfirmRequest(BaseModel):
    """Execute a previously proposed action.

    The application is named by id, not by the original phrase: re-resolving at
    confirmation time could land somewhere different if the data changed in
    between, and the user confirmed a specific row.
    """

    application_id: UUID
    event_type: EventType
    note: str | None = Field(default=None, max_length=2000)
