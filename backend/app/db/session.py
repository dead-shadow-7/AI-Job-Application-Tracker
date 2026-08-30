from collections.abc import AsyncIterator
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=settings.db_echo,
    pool_pre_ping=True,
)

SessionFactory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def open_session() -> AsyncIterator[AsyncSession]:
    """An unscoped session. Use only where no user context exists (health checks,
    migrations, the scheduled sweep before it fans out per user)."""
    async with SessionFactory() as session:
        yield session


async def open_user_session(user_id: UUID) -> AsyncIterator[AsyncSession]:
    """A session scoped to one user for the lifetime of one request.

    Sets ``app.user_id`` as a *transaction-local* GUC, which every RLS policy
    reads via ``current_setting('app.user_id', true)``. That means a query which
    forgets its ``WHERE user_id = ...`` filter returns zero rows rather than
    another user's data — the database enforces isolation, not our diligence.

    This is deliberately one transaction per request. ``set_config(..., true)``
    is scoped to the current transaction, so a mid-request ``commit()`` would
    silently drop the setting and the *next* statement would run unscoped. Routes
    must therefore not commit; this dependency commits once, at the end.
    """
    async with SessionFactory() as session:
        try:
            await session.execute(
                text("SELECT set_config('app.user_id', :uid, true)"),
                {"uid": str(user_id)},
            )
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
