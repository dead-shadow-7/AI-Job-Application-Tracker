"""Does the assistant reach for the right tool, with the user's own words?

Runs the real loop against a real database. Nothing is applied — every write
tool only prepares a proposal — so a case can ask for a deletion safely, which
is the point of the design being tested.

Cases run one at a time rather than concurrently. The rest of the suite shares
one event loop and truncates between tests, and each case here needs its own
seeded world; running them in parallel would have them scribbling over each
other's applications. Slower, and the alternative is a flaky eval, which is
worse than a slow one.
"""

from time import perf_counter

import pytest

from app.agent.assistant import run_assistant
from app.db.session import open_user_session
from evals.loader import Case, load_cases, thresholds
from evals.recorder import RunRecorder
from evals.scoring import CaseScore, aggregate, format_failures, score_assistant
from tests.evals.seeds import build
from tests.factories import Session

pytestmark = pytest.mark.eval


@pytest.fixture
def recorded_tool_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Every tool call with its arguments, which the result does not carry.

    ``AssistantResult.tools_used`` is names only, and rule 1 is about the
    ``query`` *argument* — whether the user's own phrase reached the resolver or
    the model substituted its own idea of the title. Wrapping the dispatch is
    the only place both are visible, and it also catches a forbidden call that
    produced no proposal.
    """
    from app.agent import tools as tools_module

    real = tools_module.run_tool
    seen: list[dict] = []

    async def recording(name, arguments, session, user_id, *, message=""):
        seen.append({"name": name, "arguments": arguments})
        return await real(name, arguments, session, user_id, message=message)

    monkeypatch.setattr("app.agent.assistant.run_tool", recording)
    return seen


async def run_one(case: Case, client, calls: list[dict], run: RunRecorder) -> CaseScore:
    calls.clear()
    user = await Session(client).start()
    await build(case.get("seed", "empty"), user)

    # Timed from here, after seeding: a turn's cost to the user is the model
    # round trips, not the fixture setup this eval does and production does not.
    started = perf_counter()
    try:
        async for session in open_user_session(user.user_id):
            result = await run_assistant(session, user.user_id, case.get("message"))
            break
    except Exception as exc:  # noqa: BLE001 - one bad case must not lose the run
        failed = CaseScore(case_id=case.id, error=f"{type(exc).__name__}: {exc}")
        failed.hard_failed = case.gate == "hard"
        return failed

    run.spend(result.total_tokens)
    score = score_assistant(case, result.message, list(calls))
    score.tokens = result.total_tokens
    score.latency_ms = int((perf_counter() - started) * 1000)

    # Declared separately from the tool metrics because it is about the *turn*
    # rather than about a call: an ambiguous reference must produce a question,
    # not a card the user is invited to confirm.
    if case.get("expect_no_proposal"):
        score.check(
            "no_proposal_on_ambiguity",
            result.proposal is None,
            f"prepared {result.proposal.get('kind') if result.proposal else None} "
            "for a reference that matches more than one application",
        )
        score.hard_failed = case.gate == "hard" and bool(score.failures)

    return score


async def test_the_assistant_obeys_its_prompt(
    client, recorded_tool_calls: list[dict], eval_run: RunRecorder
) -> None:
    scores = [
        await run_one(case, client, recorded_tool_calls, eval_run)
        for case in load_cases("assistant")
    ]
    eval_run.record("assistant", scores)

    hard = [s for s in scores if s.hard_failed]
    assert not hard, (
        f"{len(hard)} hard-gated case(s) failed — each is a bug, not a score:\n"
        + format_failures(hard)
    )

    metrics = aggregate(scores)
    floors = thresholds()["assistant"]
    below = {
        name: (metrics[name], floor)
        for name, floor in floors.items()
        if name in metrics and metrics[name] < floor
    }
    assert not below, (
        "metrics below their floor:\n"
        + "\n".join(f"  {name}: {value:.3f} < {floor}" for name, (value, floor) in below.items())
        + "\n\nper-case detail:\n"
        + format_failures([s for s in scores if s.failures])
    )
