"""Does the model obey the seven rules it was given?

Calls `extract` directly and runs `validate_extraction` in process, rather than
going through `POST /jobs/ingest`. The endpoint returns a post-validation draft,
and validation has already stripped any invented skill — which would destroy the
strictest metric in the suite. One model call yields both the raw fields and the
pipeline's `needs_review` verdict.

One test rather than one per case. A stochastic system graded case by case shows
red on a good day and teaches everyone to stop reading it; the aggregate is what
gates, and the failure message carries the per-case detail.
"""

import asyncio
import os
from time import perf_counter

import pytest

from app.agent.graphs.ingestion import _clean
from app.agent.llm_client import llm_client
from app.agent.prompts.extraction import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from app.agent.validation import validate_extraction
from app.schemas.extraction import ExtractedJob
from app.services.job_drafts import needs_review
from evals.loader import Case, load_cases, load_fixture, thresholds
from evals.recorder import RunRecorder
from evals.scoring import CaseScore, aggregate, format_failures, score_extraction

pytestmark = pytest.mark.eval


async def run_case(case: Case, gate: asyncio.Semaphore, run: RunRecorder) -> CaseScore:
    posting = load_fixture("postings", case.get("posting"))
    cleaned = _clean(posting)
    started = perf_counter()

    async with gate:
        try:
            result = await llm_client.extract(
                schema=ExtractedJob,
                system=EXTRACTION_SYSTEM_PROMPT,
                user=build_extraction_user_prompt(posting),
            )
        except Exception as exc:  # noqa: BLE001 - one bad case must not lose the run
            failed = CaseScore(case_id=case.id, error=f"{type(exc).__name__}: {exc}")
            failed.hard_failed = case.gate == "hard"
            return failed

    run.spend(result.usage.total_tokens)

    # In process and in the same order production runs them: validation mutates
    # the extraction, and `needs_review` reads what it left behind.
    report = validate_extraction(result.data, cleaned)
    score = score_extraction(
        case, result.data, report, needs_review(result.data.confidence, report), cleaned
    )
    score.tokens = result.usage.total_tokens
    score.latency_ms = int((perf_counter() - started) * 1000)
    return score


async def test_extraction_obeys_its_prompt(eval_run: RunRecorder) -> None:
    cases = load_cases("extraction")
    gate = asyncio.Semaphore(int(os.environ.get("EVAL_CONCURRENCY", "4")))

    scores = await asyncio.gather(*(run_case(c, gate, eval_run) for c in cases))
    eval_run.record("extraction", list(scores))

    hard = [s for s in scores if s.hard_failed]
    assert not hard, (
        f"{len(hard)} hard-gated case(s) failed — each is a bug, not a score:\n"
        + format_failures(hard)
    )

    metrics = aggregate(list(scores))
    floors = thresholds()["extraction"]
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
