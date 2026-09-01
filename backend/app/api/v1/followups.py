from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.deps import CurrentUser, DbSession
from app.core.exceptions import ConflictError, NotFoundError
from app.domain.enums import ApplicationStatus, FollowUpAction
from app.models.followup import FollowUpRule
from app.schemas.job import JobSummary
from app.services.followups import apply_ghosting, ensure_default_rules, find_stale_applications

router = APIRouter(tags=["follow-ups"])


class RuleRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    applies_to_status: ApplicationStatus
    days_threshold: int
    action: FollowUpAction
    enabled: bool


class RuleCreate(BaseModel):
    applies_to_status: ApplicationStatus
    days_threshold: int = Field(ge=1, le=365)
    action: FollowUpAction = FollowUpAction.SUGGEST_FOLLOWUP
    enabled: bool = True


class RuleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days_threshold: int | None = Field(default=None, ge=1, le=365)
    enabled: bool | None = None


class AttentionItem(BaseModel):
    """One application a rule has fired on."""

    application_id: UUID
    job: JobSummary
    current_status: ApplicationStatus
    days_idle: int
    last_activity_at: datetime

    # Which rule fired, so the dashboard can say *why* rather than presenting a
    # number the user cannot trace back to anything.
    rule_id: UUID
    rule_threshold: int
    rule_action: FollowUpAction
    reason: str


@router.get("/follow-up-rules", response_model=list[RuleRead], summary="Your follow-up rules")
async def list_rules(user: CurrentUser, session: DbSession) -> list[RuleRead]:
    """Seeds sensible defaults on first read.

    Lazily rather than at signup, so accounts created before this feature
    existed get them too.
    """
    await ensure_default_rules(session, user.id)
    rows = (
        (
            await session.execute(
                select(FollowUpRule)
                .where(FollowUpRule.user_id == user.id)
                .order_by(FollowUpRule.applies_to_status, FollowUpRule.days_threshold)
            )
        )
        .scalars()
        .all()
    )
    return [RuleRead.model_validate(r) for r in rows]


@router.post(
    "/follow-up-rules",
    response_model=RuleRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a rule",
)
async def create_rule(payload: RuleCreate, user: CurrentUser, session: DbSession) -> RuleRead:
    rule = FollowUpRule(
        user_id=user.id,
        applies_to_status=payload.applies_to_status.value,
        days_threshold=payload.days_threshold,
        action=payload.action.value,
        enabled=payload.enabled,
    )
    session.add(rule)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise ConflictError(
            f"A {payload.action.value} rule already exists for "
            f"{payload.applies_to_status.value}. Edit that one instead."
        ) from exc
    return RuleRead.model_validate(rule)


@router.patch("/follow-up-rules/{rule_id}", response_model=RuleRead, summary="Edit a rule")
async def update_rule(
    rule_id: UUID, payload: RuleUpdate, user: CurrentUser, session: DbSession
) -> RuleRead:
    rule = await _owned_rule(session, rule_id, user.id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(rule, field, value)
    await session.flush()
    return RuleRead.model_validate(rule)


@router.delete(
    "/follow-up-rules/{rule_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a rule",
)
async def delete_rule(rule_id: UUID, user: CurrentUser, session: DbSession) -> None:
    await session.delete(await _owned_rule(session, rule_id, user.id))


@router.get(
    "/needs-attention",
    response_model=list[AttentionItem],
    summary="Applications that have gone quiet",
)
async def needs_attention(user: CurrentUser, session: DbSession) -> list[AttentionItem]:
    """The dashboard panel.

    Computed on read rather than from a stored view: it is one indexed query
    over the caller's own rows, and a cached answer would go stale the moment
    an event was logged — which is exactly when the user is looking.
    """
    await ensure_default_rules(session, user.id)
    stale = await find_stale_applications(session, user.id)

    return [
        AttentionItem(
            application_id=item.application.id,
            job=JobSummary.model_validate(item.application.job),
            current_status=ApplicationStatus(item.application.current_status),
            days_idle=item.days_idle,
            last_activity_at=item.application.last_activity_at,
            rule_id=item.rule.id,
            rule_threshold=item.rule.days_threshold,
            rule_action=FollowUpAction(item.rule.action),
            reason=item.reason,
        )
        for item in stale
    ]


@router.post(
    "/needs-attention/close-ghosted",
    summary="Close applications past their give-up threshold",
)
async def close_ghosted(user: CurrentUser, session: DbSession) -> dict[str, int]:
    """Applies the mark_ghosted rules.

    Explicit rather than automatic on read: closing an application is a real
    decision, and a GET that quietly changes status would be a nasty surprise.
    The scheduled sweep calls the same function.
    """
    return {"closed": await apply_ghosting(session, user.id)}


async def _owned_rule(session: DbSession, rule_id: UUID, user_id: UUID) -> FollowUpRule:
    rule = (
        await session.execute(
            select(FollowUpRule).where(FollowUpRule.id == rule_id, FollowUpRule.user_id == user_id)
        )
    ).scalar_one_or_none()
    if rule is None:
        raise NotFoundError(f"Follow-up rule {rule_id} not found")
    return rule
