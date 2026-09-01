"""The assistant: ask questions, propose changes, confirm them separately.

The safety boundary lives here rather than in the model. `/chat` performs no
writes under any circumstances — it returns a proposal. `/confirm` performs the
write, from a typed request the user has already seen described in full.

That split is what makes a language model safe to point at someone's job
history. A misread instruction produces a wrong confirmation dialog, which the
user rejects, rather than a wrong row that nobody notices for weeks.

The proposal travels out through the client and back, so by the time it lands
here it is untrusted input like any other — "the model proposed it" is not
validation. Every branch below re-validates through a typed schema and goes
through the same services the manual UI uses, with `source='agent'` on the
resulting events so an agent-made change is visible on the timeline and
reversible by appending a correction.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.assistant import run_assistant, save_turn
from app.agent.llm_client import LLMError, llm_client
from app.agent.prompts.assistant import ASSISTANT_PROMPT_VERSION
from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import InvalidOperationError
from app.domain.enums import EventSource, EventType, MessageRole
from app.models.application import Application, InterviewStage
from app.schemas.agent import (
    ActionPreview,
    ChatAttachment,
    ChatRequest,
    ChatResponse,
    ConfirmCreateApplication,
    ConfirmDeleteApplication,
    ConfirmEvent,
    ConfirmRequest,
    ConfirmResult,
    ConfirmScheduleInterview,
    ConfirmUpdateApplication,
)
from app.schemas.application import detail
from app.schemas.job import JobCreate
from app.services.applications import create_application
from app.services.events import append_event, get_application, reload_application
from app.services.followups import ensure_default_rules

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=ChatResponse, summary="Ask the assistant")
async def chat(payload: ChatRequest, user: CurrentUser, session: DbSession) -> ChatResponse:
    """Answer, using tools as needed. Never writes.

    Every tool the loop can reach is read-only; the `propose_*` tools record an
    intention and return it here for confirmation. So a misread instruction
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

    return ChatResponse(
        message=result.message,
        pending_action=ActionPreview(**result.proposal) if result.proposal else None,
        attachments=[ChatAttachment(**a) for a in result.attachments],
        model=settings.extraction_model,
        prompt_version=ASSISTANT_PROMPT_VERSION,
        tools_used=result.tools_used,
        total_tokens=result.total_tokens,
    )


@router.post("/confirm", response_model=ConfirmResult, summary="Carry out a proposed action")
async def confirm(
    payload: Annotated[ConfirmRequest, Body()], user: CurrentUser, session: DbSession
) -> ConfirmResult:
    """Perform the confirmed change and say what it did.

    Existing applications are named by id, not by the original phrase:
    re-resolving now could land somewhere different if the data changed in
    between, and the user confirmed one specific row.

    The outcome is written into the conversation before returning. Without that
    the assistant never learns its proposal was accepted: confirmation happens
    on a different endpoint, so the transcript still ended at "I'm about to
    record…" and the next question got answered as though nothing had happened —
    "I haven't created that yet, so there's nothing to delete", about a row
    sitting in the table.
    """
    # Read off the ORM object before anything writes. Appending an event expires
    # application rows in the identity map, and a `user` that has to reload
    # itself mid-request cannot do so lazily in async context.
    user_id = user.id

    if isinstance(payload, ConfirmDeleteApplication):
        result = ConfirmResult(kind=payload.kind, summary=await _delete(session, user_id, payload))
    else:
        if isinstance(payload, ConfirmCreateApplication):
            application = await _create(session, user_id, payload)
            summary = f"started tracking {payload.title} at {payload.company_name}"
        elif isinstance(payload, ConfirmEvent):
            application = await _append(session, user_id, payload)
            summary = f"logged {payload.event_type.value}"
        elif isinstance(payload, ConfirmUpdateApplication):
            application = await _update(session, user_id, payload)
            summary = "updated"
        else:
            application = await _schedule(session, user_id, payload)
            summary = f"scheduled a {payload.stage_type.value} round"

        read = detail(await reload_application(session, application.id, user_id))
        result = ConfirmResult(
            kind=payload.kind,
            summary=f"{summary} on {read.job.title} at {read.job.company.name}"
            if not isinstance(payload, ConfirmCreateApplication)
            else summary,
            application=read,
        )

    await save_turn(
        session,
        user_id,
        MessageRole.ASSISTANT,
        f"Done — I {result.summary}. That is saved; it is no longer pending.",
    )
    return result


def _backdated(days: int) -> datetime | None:
    """Days-ago into a timestamp. Returns None for today so `append_event` uses
    its own clock rather than one we computed a few milliseconds earlier."""
    return None if days <= 0 else datetime.now(UTC) - timedelta(days=days)


async def _append(session: AsyncSession, user_id: UUID, payload: ConfirmEvent) -> Application:
    application = await get_application(session, payload.application_id, user_id)
    await append_event(
        session,
        application=application,
        event_type=payload.event_type,
        occurred_at=payload.occurred_at,
        source=EventSource.AGENT,
        note=payload.note,
    )
    return application


async def _create(
    session: AsyncSession, user_id: UUID, payload: ConfirmCreateApplication
) -> Application:
    """Track a job described in conversation.

    Goes through the ordinary create service, so the company is deduplicated and
    the job embedded exactly as it would be from the manual form. The record is
    thin by design — no salary, requirements or skills — and pasting the
    description later fills those in against the posting text.
    """
    return await create_application(
        session,
        user_id=user_id,
        job_id=None,
        job_payload=JobCreate(
            company_name=payload.company_name,
            title=payload.title,
            url=payload.url,
            location=payload.location,
            work_mode=payload.work_mode,
            source_platform=payload.source_platform,
        ),
        priority="medium",
        notes=payload.notes,
        initial_event=payload.initial_event,
        occurred_at=_backdated(payload.occurred_days_ago),
    )


async def _update(
    session: AsyncSession, user_id: UUID, payload: ConfirmUpdateApplication
) -> Application:
    application = await get_application(session, payload.application_id, user_id)
    if payload.priority is not None:
        application.priority = payload.priority.value
    if payload.notes is not None:
        application.notes = payload.notes
    await session.flush()
    return application


async def _schedule(
    session: AsyncSession, user_id: UUID, payload: ConfirmScheduleInterview
) -> Application:
    """Add the round and record that scheduling happened.

    The future date goes on the stage; the event is stamped now. An event dated
    in the future would push `last_activity_at` forward and make a stalled
    application look fresh, silently disabling the follow-up detection.
    """
    application = await get_application(session, payload.application_id, user_id)

    round_number = payload.round_number
    if round_number is None:
        highest = (
            await session.execute(
                select(func.max(InterviewStage.round_number)).where(
                    InterviewStage.application_id == application.id
                )
            )
        ).scalar_one_or_none()
        round_number = (highest or 0) + 1

    session.add(
        InterviewStage(
            application_id=application.id,
            user_id=user_id,
            round_number=round_number,
            stage_type=payload.stage_type.value,
            scheduled_at=datetime.now(UTC) + timedelta(days=payload.in_days),
            interviewer=payload.interviewer,
            notes=payload.notes,
        )
    )
    await session.flush()

    await append_event(
        session,
        application=application,
        event_type=EventType.INTERVIEW_SCHEDULED,
        source=EventSource.AGENT,
        note=payload.notes,
    )
    return application


async def _delete(session: AsyncSession, user_id: UUID, payload: ConfirmDeleteApplication) -> str:
    """Remove an application and everything hanging off it.

    Returns the label rather than the row, because after this there is no row.
    Ownership is re-checked here and not taken from the proposal: `/confirm`
    accepts a raw id, and an id that arrived through the client is not proof of
    anything.
    """
    application = await get_application(session, payload.application_id, user_id)
    label = f"{application.job.title} at {application.job.company.name}"
    await session.delete(application)
    await session.flush()
    return f"permanently deleted {label} and its history"
