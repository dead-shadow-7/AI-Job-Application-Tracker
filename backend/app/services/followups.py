"""Which applications have gone quiet, and which rule says so.

This is a SQL query. Deciding that something has been silent for seven days is
a date subtraction; giving that to a model would make a deterministic rule
non-deterministic, spend tokens on every sweep, and remove the ability to test
the one part of this feature that must not be wrong.

The agent's work starts *after* this: it takes the candidate set and explains
each case in context — how far along the process was, whether you already
followed up, whether the recruiter replied — and drafts the message.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import (
    TERMINAL_STATUSES,
    ApplicationStatus,
    EventSource,
    EventType,
    FollowUpAction,
)
from app.models.application import Application
from app.models.followup import FollowUpRule

logger = logging.getLogger(__name__)

# Seeded for a new user. Chosen from how hiring actually behaves rather than
# round numbers: a week of quiet after applying is normal, the same silence
# after a final interview is not, and three weeks with no reply at all means
# nobody is coming.
DEFAULT_RULES: list[tuple[ApplicationStatus, int, FollowUpAction]] = [
    (ApplicationStatus.APPLIED, 7, FollowUpAction.SUGGEST_FOLLOWUP),
    (ApplicationStatus.SCREENING, 5, FollowUpAction.SUGGEST_FOLLOWUP),
    (ApplicationStatus.INTERVIEWING, 7, FollowUpAction.SUGGEST_FOLLOWUP),
    (ApplicationStatus.OFFER, 3, FollowUpAction.SUGGEST_FOLLOWUP),
    (ApplicationStatus.APPLIED, 21, FollowUpAction.MARK_GHOSTED),
]


@dataclass(frozen=True, slots=True)
class StaleApplication:
    """One application a rule has fired on."""

    application: Application
    rule: FollowUpRule
    days_idle: int

    @property
    def reason(self) -> str:
        """The sentence the dashboard shows, and the agent starts from."""
        company = self.application.job.company.name
        status = self.application.current_status
        return (
            f"{company} has had no activity for {self.days_idle} days since it moved to {status}."
        )


async def ensure_default_rules(session: AsyncSession, user_id: uuid.UUID) -> int:
    """Seed a user's rules if they have none.

    Called lazily rather than at signup, so users created before this feature
    existed get them too, and so a user who deletes every rule deliberately is
    not silently re-seeded on their next request — the check is "none at all",
    which only holds for someone who has never had any.
    """
    existing = (
        await session.execute(
            select(FollowUpRule.id).where(FollowUpRule.user_id == user_id).limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return 0

    for status, days, action in DEFAULT_RULES:
        session.add(
            FollowUpRule(
                user_id=user_id,
                applies_to_status=status.value,
                days_threshold=days,
                action=action.value,
                enabled=True,
            )
        )
    await session.flush()
    return len(DEFAULT_RULES)


async def find_stale_applications(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    now: datetime | None = None,
) -> list[StaleApplication]:
    """Applications whose silence has crossed a rule's threshold.

    Measured against ``last_activity_at``, not ``current_status_at``. The
    distinction is the point: a recruiter replying means the application is not
    stale even though it has not advanced, so the clock resets on contact
    rather than only on progress.

    Terminal statuses are skipped — nothing is pending from your side on a
    rejection.
    """
    moment = now or datetime.now(UTC)

    rules = list(
        (
            await session.execute(
                select(FollowUpRule).where(
                    FollowUpRule.user_id == user_id, FollowUpRule.enabled.is_(True)
                )
            )
        )
        .scalars()
        .all()
    )
    if not rules:
        return []

    by_status: dict[str, list[FollowUpRule]] = {}
    for rule in rules:
        by_status.setdefault(rule.applies_to_status, []).append(rule)

    terminal = {s.value for s in TERMINAL_STATUSES}
    candidates = list(
        (
            await session.execute(
                select(Application)
                .where(
                    Application.user_id == user_id,
                    Application.current_status.in_(list(by_status)),
                    Application.current_status.notin_(list(terminal)),
                )
                .order_by(Application.last_activity_at)
            )
        )
        .scalars()
        .all()
    )

    stale: list[StaleApplication] = []
    for application in candidates:
        days_idle = (moment - application.last_activity_at).days
        applicable = by_status.get(application.current_status, [])

        # Where several rules fire, the longest threshold wins: `mark_ghosted`
        # at 21 days is a stronger statement than `suggest_followup` at 7, and
        # reporting both would just be noise.
        fired = [r for r in applicable if days_idle >= r.days_threshold]
        if not fired:
            continue

        rule = max(fired, key=lambda r: r.days_threshold)
        stale.append(StaleApplication(application=application, rule=rule, days_idle=days_idle))

    return stale


async def apply_ghosting(
    session: AsyncSession, user_id: uuid.UUID, *, now: datetime | None = None
) -> int:
    """Close applications a ``mark_ghosted`` rule has caught.

    The only rule action that writes. It goes through ``append_event`` like
    everything else, with ``source='system'``, so it lands on the timeline and
    can be undone by appending a correction rather than by editing history.
    """
    from app.services.events import append_event

    closed = 0
    for candidate in await find_stale_applications(session, user_id, now=now):
        if candidate.rule.action != FollowUpAction.MARK_GHOSTED.value:
            continue

        await append_event(
            session,
            application=candidate.application,
            event_type=EventType.MARKED_GHOSTED,
            source=EventSource.SYSTEM,
            note=(
                f"No response for {candidate.days_idle} days "
                f"(rule: {candidate.rule.days_threshold}-day threshold)."
            ),
        )
        closed += 1

    return closed
