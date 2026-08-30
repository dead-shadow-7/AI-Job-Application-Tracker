"""LangSmith tracing.

Off unless ``LANGSMITH_TRACING=true`` and a key is present, and configured by
setting the environment variables the LangSmith SDK reads. Doing it here rather
than expecting them in the shell keeps a single source of truth — the same .env
that configures everything else — and means a missing key degrades to "no
traces" rather than to a crash on first extraction.

What the traces are actually for: token cost per ingestion and extraction
latency, both of which need to be watched once real postings start flowing
through, and neither of which is visible from the API response alone.
"""

import logging
import os

from app.core.config import settings

logger = logging.getLogger(__name__)


def configure_tracing() -> bool:
    """Returns whether tracing ended up enabled."""
    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty; tracing disabled.")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    return True


def run_metadata(
    user_id: str | None = None, **extra: str | int | float | None
) -> dict[str, object]:
    """Tags for one graph run.

    ``user_id`` is included so a trace can be tied back to whose ingestion it
    was when debugging a specific complaint. No posting text and no email — the
    trace is enough to find the run, not to reconstruct its contents.
    """
    metadata: dict[str, object] = {
        "prompt_version": None,
        "model": settings.groq_extraction_model,
        "environment": settings.environment,
    }
    if user_id:
        metadata["user_id"] = user_id
    metadata.update({k: v for k, v in extra.items() if v is not None})
    return metadata
