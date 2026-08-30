import logging
import sys

from pythonjsonlogger import json as jsonlogger

from app.core.config import settings


def configure_logging() -> None:
    """JSON logs in deployed environments, human-readable locally."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(settings.log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    if settings.environment == "local":
        handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s : %(message)s"))
    else:
        handler.setFormatter(
            jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"asctime": "timestamp", "levelname": "level"},
            )
        )
    root.addHandler(handler)

    # uvicorn installs its own handlers; make them defer to ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
