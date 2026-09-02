"""What a run is, so two runs can be compared later.

The header is the point. A number without the prompt version, the model and the
tool count beside it cannot be compared to anything — which is how the one
hand-run eval this project already did ended up recorded three times with
contradictory figures, one saying a 20-tool schema and another 22. Whatever
produced a metric is written down next to it, and the comparison script refuses
to compare runs whose headers disagree in ways that matter.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.loader import RUNS
from evals.scoring import CaseScore, aggregate


class BudgetExceeded(RuntimeError):
    """The stop before a case file that points somewhere expensive drains a key."""


def _git_sha() -> str:
    """Which commit produced these numbers.

    Read from the environment first because the usual way to run this is inside
    the container, where only `backend/` is mounted — the repository, and so
    `git`, is on the host. Without the override every run records "unknown",
    which quietly removes the one field that makes two runs attributable:

        docker compose exec -e LLM_EVAL=1 -e EVAL_GIT_SHA=$(git rev-parse --short HEAD) ...
    """
    if sha := os.environ.get("EVAL_GIT_SHA", "").strip():
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def provenance() -> dict[str, Any]:
    """Everything that would change a number, read at run time.

    Imported from the constants rather than read from the database, deliberately
    — the eval measures the code in front of it, not what some historical row
    was labelled with.
    """
    from app.agent.prompts.assistant import ASSISTANT_PROMPT_VERSION
    from app.agent.prompts.rubric import RUBRIC_PROMPT_VERSION
    from app.agent.tools import TOOL_SCHEMAS
    from app.core.config import settings
    from app.schemas.extraction import EXTRACTION_PROMPT_VERSION

    return {
        "git_sha": _git_sha(),
        "provider": settings.llm_provider,
        "model": settings.extraction_model,
        "tool_count": len(TOOL_SCHEMAS),
        "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
        "assistant_prompt_version": ASSISTANT_PROMPT_VERSION,
        "rubric_prompt_version": RUBRIC_PROMPT_VERSION,
    }


@dataclass
class RunRecorder:
    """Collects each surface's scores and writes one JSON at the end."""

    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    surfaces: dict[str, list[CaseScore]] = field(default_factory=dict)
    budget: int = field(default_factory=lambda: int(os.environ.get("EVAL_TOKEN_BUDGET", "500000")))

    def spend(self, tokens: int) -> None:
        """Called per model answer; raises rather than quietly running on.

        Insurance against the mistake that actually happens — a case pointing at
        a sixty-thousand-character posting, or a loop that does not terminate.
        """
        if self.tokens + tokens > self.budget:
            raise BudgetExceeded(
                f"eval run would spend {self.tokens + tokens} tokens, over the "
                f"{self.budget} budget. Raise EVAL_TOKEN_BUDGET if this is expected."
            )
        self._spent += tokens

    _spent: int = 0

    @property
    def tokens(self) -> int:
        return self._spent

    def record(self, surface: str, scores: list[CaseScore]) -> None:
        self.surfaces[surface] = scores

    def metrics(self) -> dict[str, float]:
        return {
            f"{surface}.{metric}": value
            for surface, scores in self.surfaces.items()
            for metric, value in aggregate(scores).items()
        }

    def write(self) -> Path | None:
        if not self.surfaces:
            return None

        RUNS.mkdir(parents=True, exist_ok=True)
        header = provenance()
        stamp = self.started_at.replace(":", "").replace("-", "")
        path = RUNS / f"{stamp}-{header['provider']}-{header['model'].replace('/', '_')}.json"
        path.write_text(
            json.dumps(
                {
                    "run": {
                        "started_at": self.started_at,
                        "total_tokens": self.tokens,
                        **header,
                    },
                    "metrics": self.metrics(),
                    "cases": [
                        {
                            "id": s.case_id,
                            "surface": surface,
                            "hard_failed": s.hard_failed,
                            "metrics": s.metrics,
                            "failures": s.failures,
                            "error": s.error,
                            "tokens": s.tokens,
                            "latency_ms": s.latency_ms,
                        }
                        for surface, scores in self.surfaces.items()
                        for s in scores
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path
