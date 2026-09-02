"""Reading cases off disk, and the knobs for running fewer of them.

Cases are JSONL with `.txt` sidecars for anything long. One case is one line, so
one case is one diff hunk and one review comment; a posting body is a file, so
it reads as prose rather than as an escaped string. Neither is a small point —
the corpus is reviewed far more often than it is written, and a format nobody
can review is a corpus nobody trusts.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "cases"
FIXTURES = ROOT / "fixtures"
RUNS = ROOT / "runs"

Surface = Literal["extraction", "assistant", "rubric"]


@dataclass(frozen=True, slots=True)
class Case:
    """One thing a prompt claims, and how to check it.

    ``rule`` names the numbered prompt rule this case holds down, so a failure
    points at the sentence that stopped being true rather than at a case id.
    ``gate`` separates safety from quality: a *hard* case failing is a bug on
    its own, where a *scored* one only contributes to an average.
    """

    id: str
    surface: Surface
    gate: Literal["hard", "scored"] = "scored"
    rule: str | None = None
    body: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.body.get(key, default)


class CaseError(Exception):
    """A case file that cannot be read is a broken test, not a failed one."""


def load_cases(surface: Surface) -> list[Case]:
    """Every case for a surface, honouring the subsetting variables.

    ``EVAL_CASES`` takes a comma-separated list of ids — for when you are
    working on one failure and do not want to pay for the other twenty-three.
    ``EVAL_SAMPLE`` takes the first N, which is the smoke run.
    """
    path = CASES / f"{surface}.jsonl"
    if not path.exists():
        raise CaseError(f"No case file at {path}")

    cases: list[Case] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaseError(f"{path.name}:{number} is not valid JSON — {exc}") from exc

        case_id = raw.pop("id", None)
        if not case_id:
            raise CaseError(f"{path.name}:{number} has no id")
        if case_id in seen:
            # Ids name failures in the report and select them on the command
            # line; two cases answering to one name makes both unreachable.
            raise CaseError(f"{path.name}:{number} repeats the id {case_id!r}")
        seen.add(case_id)

        cases.append(
            Case(
                id=case_id,
                surface=surface,
                gate=raw.pop("gate", "scored"),
                rule=raw.pop("rule", None),
                body=raw,
            )
        )

    return _subset(cases)


def _subset(cases: list[Case]) -> list[Case]:
    wanted = os.environ.get("EVAL_CASES", "").strip()
    if wanted:
        ids = {name.strip() for name in wanted.split(",") if name.strip()}
        chosen = [c for c in cases if c.id in ids]
        missing = ids - {c.id for c in chosen}
        if missing:
            # Silently running nothing would look like a pass.
            raise CaseError(f"EVAL_CASES names cases that do not exist: {sorted(missing)}")
        return chosen

    sample = os.environ.get("EVAL_SAMPLE", "").strip()
    if sample:
        return cases[: int(sample)]

    return cases


def load_fixture(kind: str, name: str) -> str:
    """A posting or resume body, by filename."""
    path = FIXTURES / kind / name
    if not path.exists():
        raise CaseError(f"No fixture at {path}")
    return path.read_text(encoding="utf-8")


def thresholds() -> dict[str, dict[str, float]]:
    """The gates, kept in a file so raising one is a reviewable diff.

    In JSON rather than in Python because the argument for moving a threshold
    belongs in a commit message, and a config change makes that argument
    unavoidable in a way that editing a constant does not.
    """
    return json.loads((ROOT / "thresholds.json").read_text(encoding="utf-8"))
