"""Does the rubric judge on the evidence it was given, and only that?

Evidence is supplied by hand rather than retrieved. That is the whole design:
under rule 1 a requirement whose retrieval missed the right passage *should*
score as unmet, so an eval that ran retrieval could not tell a judgement failure
from an index failure. Feeding the passages directly means a regression here is
about the model and nothing else.

No database and no embeddings, so these are the cheapest cases in the suite —
one call each, straight to the prompt builder the endpoint uses.
"""

import asyncio
import os
from time import perf_counter

import pytest

from app.agent.llm_client import llm_client
from app.agent.prompts.rubric import RUBRIC_SYSTEM_PROMPT, build_rubric_prompt
from app.schemas.matching import RubricJudgment
from evals.loader import Case, load_cases, thresholds
from evals.recorder import RunRecorder
from evals.scoring import CaseScore, aggregate, format_failures, score_rubric

pytestmark = pytest.mark.eval


async def run_case(case: Case, gate: asyncio.Semaphore, run: RunRecorder) -> CaseScore:
    started = perf_counter()
    prompt = build_rubric_prompt(
        title=case.get("title"),
        company=case.get("company"),
        requirements_with_evidence=[(req, ev) for req, ev in case.get("requirements")],
        matched_skills=case.get("matched_skills") or [],
        missing_skills=case.get("missing_skills") or [],
    )

    async with gate:
        try:
            result = await llm_client.extract(
                schema=RubricJudgment, system=RUBRIC_SYSTEM_PROMPT, user=prompt
            )
        except Exception as exc:  # noqa: BLE001 - one bad case must not lose the run
            failed = CaseScore(case_id=case.id, error=f"{type(exc).__name__}: {exc}")
            failed.hard_failed = case.gate == "hard"
            return failed

    run.spend(result.usage.total_tokens)
    score = score_rubric(case, result.data)
    score.tokens = result.usage.total_tokens
    score.latency_ms = int((perf_counter() - started) * 1000)
    return score


async def test_the_rubric_judges_on_evidence(eval_run: RunRecorder) -> None:
    cases = load_cases("rubric")
    gate = asyncio.Semaphore(int(os.environ.get("EVAL_CONCURRENCY", "4")))

    scores = await asyncio.gather(*(run_case(c, gate, eval_run) for c in cases))
    eval_run.record("rubric", list(scores))

    hard = [s for s in scores if s.hard_failed]
    assert not hard, (
        f"{len(hard)} hard-gated case(s) failed — each is a bug, not a score:\n"
        + format_failures(hard)
    )

    metrics = aggregate(list(scores))
    floors = thresholds()["rubric"]
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
