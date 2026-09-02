import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.http_client import close_http_client
from app.agent.tracing import configure_tracing
from app.api.v1 import health
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import (
    ConflictError,
    DomainError,
    InvalidOperationError,
    NotFoundError,
)
from app.core.logging import configure_logging
from app.db.session import engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    tracing = await configure_tracing()
    logger.info(
        "Starting AI Job Tracker API (env=%s, model=%s, tracing=%s)",
        settings.environment,
        settings.extraction_model,
        "on" if tracing else "off",
    )
    yield
    await close_http_client()
    await engine.dispose()
    logger.info("Shutdown complete")


app = FastAPI(
    title="AI Job Tracker API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Domain errors carry no HTTP knowledge, so they are translated here — once,
# at the edge — rather than each service reaching for HTTPException. That keeps
# the same services callable from the Phase 4 agent tools and scheduled sweep.
DOMAIN_ERROR_STATUS: list[tuple[type[DomainError], int]] = [
    (NotFoundError, status.HTTP_404_NOT_FOUND),
    (ConflictError, status.HTTP_409_CONFLICT),
    (InvalidOperationError, status.HTTP_422_UNPROCESSABLE_CONTENT),
]


def _register_domain_handler(exc_type: type[DomainError], status_code: int) -> None:
    @app.exception_handler(exc_type)
    async def handler(_: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})


for error_type, code in DOMAIN_ERROR_STATUS:
    _register_domain_handler(error_type, code)


app.include_router(health.router)
app.include_router(api_router, prefix=settings.api_v1_prefix)
