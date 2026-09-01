"""The assistant: ask questions, propose changes, confirm them separately.

The safety boundary lives here rather than in the model. `/chat` performs no
writes under any circumstances — it returns a proposal. `/confirm` performs the
write, and takes an application *id* that the user has already seen resolved.

That split is what makes a language model safe to point at someone's job
history. A misread instruction produces a wrong confirmation dialog, which the
user rejects, rather than a wrong row that nobody notices for weeks.
"""

import logging

from fastapi import APIRouter, status

from app.agent.assistant import run_assistant
from app.agent.llm_client import LLMError, llm_client
from app.agent.prompts.assistant import ASSISTANT_PROMPT_VERSION
from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import InvalidOperationError
from app.domain.enums import EventSource
from app.schemas.agent import (
    ActionPreview,
    ChatRequest,
    ChatResponse,
    ConfirmRequest,
)
from app.schemas.application import ApplicationRead
from app.services.events import append_event, get_application, reload_application
from app.services.followups import ensure_default_rules

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

# The whole picture is put in the prompt rather than fetched through tools: a
# job search is tens of applications, so it fits, and one call beats three
# round trips on a tight token budget. Bounded so a heavy user cannot silently
# blow the context window.


@router.post("/chat", response_model=ChatResponse, summary="Ask the assistant")
async def chat(payload: ChatRequest, user: CurrentUser, session: DbSession) -> ChatResponse:
    """Answer, using tools as needed. Never writes.

    Every tool the loop can reach is read-only; `propose_event` records an
    intention and returns it here for confirmation. So a misread instruction
    produces a dialog the user rejects, not a changed row.
    """
    if not llm_client.is_configured:
        raise InvalidOperationError(
            "No LLM is configured, so the assistant is unavailable. "
            "Set GROQ_API_KEY (or switch LLM_PROVIDER) and restart the API."
        )

    await ensure_default_rules(session, user.id)

    try:
        result = await run_assistant(session, user.id, payload.message)
    except LLMError as exc:
        raise InvalidOperationError(str(exc)) from exc

    response = ChatResponse(
        message=result.message,
        model=settings.extraction_model,
        prompt_version=ASSISTANT_PROMPT_VERSION,
    )

    if result.proposal:
        response.pending_action = ActionPreview(**result.proposal)

    return response


@router.post(
    "/confirm",
    response_model=ApplicationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Carry out a proposed action",
)
async def confirm(
    payload: ConfirmRequest, user: CurrentUser, session: DbSession
) -> ApplicationRead:
    """Append the confirmed event.

    Takes an application id, not the original phrase: re-resolving now could
    land somewhere different if the data changed in between, and the user
    confirmed one specific row.

    Written with source='agent', so it appears on the timeline marked as such
    and is undone by appending a correction rather than by editing history.
    """
    application = await get_application(session, payload.application_id, user.id)

    await append_event(
        session,
        application=application,
        event_type=payload.event_type,
        source=EventSource.AGENT,
        note=payload.note,
    )
    return ApplicationRead.model_validate(
        await reload_application(session, application.id, user.id)
    )
