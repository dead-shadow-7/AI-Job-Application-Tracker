from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import open_session

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe — checks DB and pgvector")
async def readiness(
    session: Annotated[AsyncSession, Depends(open_session)],
) -> dict[str, Any]:
    await session.execute(text("SELECT 1"))
    has_vector = await session.scalar(
        text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
    )
    return {"status": "ok", "database": "ok", "pgvector": bool(has_vector)}
