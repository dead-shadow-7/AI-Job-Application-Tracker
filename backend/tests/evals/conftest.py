"""The gates, and the shared run record.

Two independent gates, because they fail differently. The `eval` marker stops
these being *collected* — it is what keeps `pytest -q` and CI free. `LLM_EVAL=1`
stops them *spending* — it is what stops someone who typed `-m eval` to see what
the tests look like from finding out via an invoice. Either alone would be one
mistake away from a live call; both means the mistake has to be made twice.

Everything under `tests/` inherits the root conftest, which is why these live
here rather than in a script: the disposable database, the truncation between
cases, the RLS-scoped session factory and the Supabase tripwire are all already
built, and an eval that seeds applications needs every one of them.
"""

import os
from collections.abc import Iterator

import pytest

from evals.recorder import RunRecorder


def pytest_configure(config: pytest.Config) -> None:
    config._eval_recorder = RunRecorder()  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def eval_run(request: pytest.FixtureRequest) -> RunRecorder:
    return request.config._eval_recorder  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _spending_gate() -> Iterator[None]:
    """The second gate. Skips rather than fails: not having a key configured is
    an ordinary state for a checkout, not a broken one."""
    if os.environ.get("LLM_EVAL") != "1":
        pytest.skip("evals spend real tokens; set LLM_EVAL=1 to run them")

    from app.agent.llm_client import llm_client

    if not llm_client.is_configured:
        pytest.skip("no LLM key configured for this provider")
    yield


def pytest_terminal_summary(config: pytest.Config, terminalreporter: object) -> None:
    """The table you actually read, plus the run file for the one you compare."""
    recorder: RunRecorder = config._eval_recorder  # type: ignore[attr-defined]
    if not recorder.surfaces:
        return

    from evals.loader import thresholds

    gates = thresholds()
    write = terminalreporter.write_line  # type: ignore[attr-defined]

    write("")
    write("=" * 72)
    header = recorder.metrics()
    provenance = {k: v for k, v in _header(recorder).items()}
    write(
        f"eval — {provenance['provider']}/{provenance['model']} · "
        f"{provenance['tool_count']} tools · {recorder.tokens} tokens"
    )
    write("=" * 72)

    for name in sorted(header):
        surface, metric = name.split(".", 1)
        floor = gates.get(surface, {}).get(metric)
        value = header[name]
        if floor is None:
            write(f"  {name:<44} {value:>6.3f}   (reported)")
        else:
            mark = "ok  " if value >= floor else "FAIL"
            write(f"  {name:<44} {value:>6.3f} >= {floor:<5} {mark}")

    hard = [s for scores in recorder.surfaces.values() for s in scores if s.hard_failed]
    if hard:
        write("")
        write(f"  {len(hard)} hard-gated case(s) failed — these are bugs, not scores:")
        for score in hard:
            for failure in score.failures:
                write(f"    {score.case_id}: {failure}")

    if (path := recorder.write()) is not None:
        write("")
        write(f"  run written to {path}")


def _header(recorder: RunRecorder) -> dict[str, object]:
    from evals.recorder import provenance

    return provenance()
