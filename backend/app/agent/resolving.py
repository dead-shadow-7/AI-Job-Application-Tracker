"""Shared resolve step for every tool that names an application.

Lives on its own because both the read tools and the proposal tools need it,
and because the rule it enforces is worth stating once: the tool that resolves
is never the tool that acts. A query that does not land on exactly one
application comes back as a *question* for the model to pass on, not as a best
guess it can proceed with.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.application import Application
from app.services.resolver import Candidate, resolve_application


async def resolve_one(
    session: AsyncSession, user_id: uuid.UUID, query: str
) -> tuple[Candidate | None, str | None]:
    """Returns (candidate, problem). Exactly one of the two is set.

    ``problem`` is already phrased for the model to relay — ambiguity is a
    question to ask, not an error to work around.
    """
    if not query.strip():
        return None, "Which application? I need the name they used."

    resolution = await resolve_application(session, user_id, query)
    if resolution.best is None:
        return None, resolution.describe()
    return resolution.best, None


async def resolve_application_only(
    session: AsyncSession, user_id: uuid.UUID, query: str
) -> tuple[Application | None, str | None]:
    candidate, problem = await resolve_one(session, user_id, query)
    return (candidate.application if candidate else None), problem
