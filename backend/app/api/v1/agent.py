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

from app.agent.llm_client import LLMError, llm_client
from app.agent.prompts.assistant import ASSISTANT_SYSTEM_PROMPT, build_assistant_prompt
from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import InvalidOperationError
from app.domain.enums import EventSource
from app.models.application import Application
from app.schemas.agent import (
    ASSISTANT_PROMPT_VERSION,
    ActionPreview,
    AgentReply,
    ChatRequest,
    ChatResponse,
    ConfirmRequest,
)
from app.schemas.application import ApplicationRead
from app.services.applications import list_applications
from app.services.events import append_event, get_application, reload_application
from app.services.followups import ensure_default_rules, find_stale_applications
from app.services.resolver import resolve_application

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

# The whole picture is put in the prompt rather than fetched through tools: a
# job search is tens of applications, so it fits, and one call beats three
# round trips on a tight token budget. Bounded so a heavy user cannot silently
# blow the context window.
MAX_APPLICATIONS_IN_CONTEXT = 60


@router.post("/chat", response_model=ChatResponse, summary="Ask the assistant")
async def chat(payload: ChatRequest, user: CurrentUser, session: DbSession) -> ChatResponse:
    """Answer, or propose an action. Never writes."""
    if not llm_client.is_configured:
        raise InvalidOperationError(
            "No LLM is configured, so the assistant is unavailable. "
            "Set GROQ_API_KEY (or switch LLM_PROVIDER) and restart the API."
        )

    await ensure_default_rules(session, user.id)

    rows, _ = await list_applications(session, user_id=user.id, limit=MAX_APPLICATIONS_IN_CONTEXT)
    stale = await find_stale_applications(session, user.id)

    try:
        result = await llm_client.extract(
            schema=AgentReply,
            system=ASSISTANT_SYSTEM_PROMPT,
            user=build_assistant_prompt(
                message=payload.message,
                applications=[_describe(a) for a in rows],
                stale=[item.reason for item in stale],
            ),
        )
    except LLMError as exc:
        raise InvalidOperationError(str(exc)) from exc

    reply = result.data
    response = ChatResponse(
        message=reply.message,
        model=settings.extraction_model,
        prompt_version=ASSISTANT_PROMPT_VERSION,
    )

    if reply.action.kind != "append_event" or not reply.action.event_type:
        return response

    if not reply.action.application_query:
        response.message += " Which application do you mean?"
        return response

    # The model proposed a target by name; the tracker decides which row that
    # is. The model never sees or supplies an id, so it cannot aim at a row it
    # was not shown.
    resolution = await resolve_application(session, user.id, reply.action.application_query)

    if resolution.best is None:
        # Ambiguous or unmatched. Surfacing the options is the correct outcome:
        # guessing here is the failure mode this whole design exists to avoid.
        response.disambiguation = [c.label for c in resolution.candidates]
        response.message = resolution.describe()
        return response

    candidate = resolution.best
    response.pending_action = ActionPreview(
        kind="append_event",
        event_type=reply.action.event_type,
        note=reply.action.note,
        application_id=candidate.application.id,
        application_label=candidate.label,
        confidence=candidate.score,
        matched_on=candidate.matched_on,
    )
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


def _describe(application: Application) -> str:
    job = application.job
    parts = [f"{job.title} at {job.company.name}", f"status {application.current_status}"]
    if application.match_score is not None:
        parts.append(f"match {application.match_score}/100")
    return " | ".join(parts)
