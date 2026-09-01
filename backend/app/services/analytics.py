"""What the application history actually says.

Every figure here carries the count it was computed from, and the API says
plainly when that count is too small to mean anything. A response rate of "0%"
over two applications is not a finding, it is noise — and a dashboard that
presents it as a finding will be believed, then acted on.

This is why the plan gated rejection analysis, skill trends and recommendations
on roughly fifty applications: they are data-gated, not engineering-gated, and
building them now would produce confident nonsense.
"""

import uuid
from dataclasses import dataclass, field
from statistics import median

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import ApplicationStatus, EventType
from app.models.application import Application, ApplicationEvent
from app.models.job import Job

# Below this, proportions are reported but flagged as unreliable. Ten is not a
# statistical threshold, it is the point at which a percentage stops being one
# application swinging the number by fifty points.
MEANINGFUL_SAMPLE = 10

# Events that mean somebody on the other side actually responded. A follow-up
# you sent is not a response, which is the distinction that makes this figure
# worth anything.
RESPONSE_EVENTS = {
    EventType.SCREENING_SCHEDULED,
    EventType.SCREENING_DONE,
    EventType.ASSESSMENT_RECEIVED,
    EventType.INTERVIEW_SCHEDULED,
    EventType.INTERVIEW_DONE,
    EventType.OFFER_RECEIVED,
    EventType.RECRUITER_REPLY,
    EventType.REJECTED,
}


@dataclass
class FunnelStage:
    status: str
    count: int


@dataclass
class PlatformStats:
    platform: str
    applications: int
    responses: int

    @property
    def response_rate(self) -> float | None:
        if self.applications == 0:
            return None
        return round(self.responses / self.applications, 3)


@dataclass
class Analytics:
    total: int
    funnel: list[FunnelStage] = field(default_factory=list)
    by_platform: list[PlatformStats] = field(default_factory=list)

    # The denominator behind response_rate, and not the same as `total` —
    # saved-but-never-applied rows are excluded. Reported so the UI can render
    # "2 of 7 replied" rather than a bare percentage whose base is a guess.
    submitted: int = 0
    responses: int = 0
    response_rate: float | None = None
    median_days_to_response: float | None = None

    # True when there is simply not enough history for the proportions to say
    # anything. Surfaced rather than hidden so the UI can caveat instead of
    # quietly implying significance.
    sample_is_small: bool = True


async def compute_analytics(session: AsyncSession, user_id: uuid.UUID) -> Analytics:
    applications = list(
        (await session.execute(select(Application).where(Application.user_id == user_id)))
        .unique()
        .scalars()
        .all()
    )

    result = Analytics(total=len(applications))
    if not applications:
        return result

    counts: dict[str, int] = {}
    for application in applications:
        counts[application.current_status] = counts.get(application.current_status, 0) + 1
    result.funnel = [
        FunnelStage(status=status.value, count=counts.get(status.value, 0))
        for status in ApplicationStatus
    ]

    # Only applications actually sent can be said to have gone unanswered.
    # Counting saved-but-never-applied rows against the response rate would
    # punish the user for keeping a shortlist.
    submitted = [a for a in applications if a.applied_at is not None]
    result.submitted = len(submitted)
    result.sample_is_small = len(submitted) < MEANINGFUL_SAMPLE

    if not submitted:
        return result

    events = list(
        (
            await session.execute(
                select(ApplicationEvent)
                .where(ApplicationEvent.user_id == user_id)
                .order_by(ApplicationEvent.occurred_at)
            )
        )
        .scalars()
        .all()
    )

    response_types = {e.value for e in RESPONSE_EVENTS}
    first_response: dict[uuid.UUID, ApplicationEvent] = {}
    for event in events:
        if event.event_type in response_types and event.application_id not in first_response:
            first_response[event.application_id] = event

    lags: list[float] = []
    for application in submitted:
        responded: ApplicationEvent | None = first_response.get(application.id)
        if responded is None or application.applied_at is None:
            continue
        event = responded
        days = (event.occurred_at - application.applied_at).total_seconds() / 86_400
        # A response recorded before the application date is a data-entry
        # artefact of backfilling, not a negative wait.
        if days >= 0:
            lags.append(days)

    result.responses = sum(1 for a in submitted if a.id in first_response)
    result.response_rate = round(result.responses / len(submitted), 3)
    # Median, not mean: one company that replied after four months would drag
    # a mean far away from what you can actually expect.
    result.median_days_to_response = round(median(lags), 1) if lags else None

    platforms: dict[str, PlatformStats] = {}
    platform_rows = (
        await session.execute(
            select(Job.id, Job.source_platform).where(Job.id.in_([a.job_id for a in submitted]))
        )
    ).all()
    job_platforms: dict[uuid.UUID, str | None] = {row[0]: row[1] for row in platform_rows}
    for application in submitted:
        name = job_platforms.get(application.job_id) or "unknown"
        entry = platforms.setdefault(
            name, PlatformStats(platform=name, applications=0, responses=0)
        )
        entry.applications += 1
        if application.id in first_response:
            entry.responses += 1

    result.by_platform = sorted(platforms.values(), key=lambda p: (-p.applications, p.platform))
    return result
