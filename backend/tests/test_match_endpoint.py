"""Scoring a job against a resume, with and without the model.

The endpoint had no tests at all, which left two things unguarded that only
show up as a wrong number rather than as an error.

**The degrade path.** When the model is unavailable the score is still produced
from the deterministic 85% — a partial score you can read beats an error you
cannot act on. The durable record that this happened is that `model` and
`prompt_version` come back null. Nothing asserted it, so a change that started
stamping them regardless would make every deterministic-only score look
LLM-scored, retroactively and silently.

**The clamp.** `RubricJudgment.score` is documented 0.0-1.0 and nothing enforces
it. The total used to be computed from a clamped value while the stored
subscore kept the raw one, so a model answering 1.4 produced a card whose parts
did not add up to its whole — visible only to someone checking the arithmetic
by hand.

No tokens spent: the model is stubbed, as everywhere else in this suite.
"""

from typing import Any

import pytest
from httpx import AsyncClient

from app.agent.llm_client import LLMError, LLMUsage, StructuredResult
from app.schemas.matching import RubricJudgment
from tests.conftest import StubEmbeddings
from tests.factories import Session

RESUME_TEXT = """\
EXPERIENCE
Senior Backend Engineer, Fintech Co (2019-2024)
- Built Python services on FastAPI handling payment flows.
- Owned PostgreSQL schema design for a payments ledger.

SKILLS
Python, FastAPI, PostgreSQL
"""


class StubRubric:
    """Answers the rubric call with a fixed judgment, or refuses to answer."""

    def __init__(self, judgment: RubricJudgment | None = None, *, fail: bool = False) -> None:
        self.judgment = judgment
        self.fail = fail
        self.calls = 0

    @property
    def is_configured(self) -> bool:
        return True

    async def extract(self, **_: Any) -> StructuredResult[RubricJudgment]:
        self.calls += 1
        if self.fail:
            raise LLMError("Groq rate limit reached.")
        assert self.judgment is not None
        return StructuredResult(data=self.judgment, usage=LLMUsage(model="stub", total_tokens=10))


class UnconfiguredLLM:
    is_configured = False


def judgment(score: float) -> RubricJudgment:
    return RubricJudgment(
        score=score,
        strengths=["Owned a payments ledger schema in PostgreSQL"],
        gaps=["No Kafka or streaming experience shown"],
        narrative="Worth applying to; the streaming gap is the thing to close.",
    )


async def scored(user: Session) -> tuple[str, dict[str, Any]]:
    """A resume, a tracked job, and the match the endpoint computes for them."""
    await user.post("/api/v1/resumes/text", {"label": "Main", "text": RESUME_TEXT})
    application = await user.create_application(company_name="Razorpay", title="Backend Engineer")
    response = await user.post(f"/api/v1/applications/{application['id']}/match", {})
    assert response.status_code == 201, response.text
    return application["id"], response.json()


# --- The degrade path ------------------------------------------------------


async def test_a_score_is_still_produced_when_the_model_is_unavailable(
    client: AsyncClient, embeddings: StubEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rate-limited mid-search is the case this exists for: an error here would
    lose the 85% of the answer that needed no model at all."""
    stub = StubRubric(fail=True)
    monkeypatch.setattr("app.api.v1.matching.llm_client", stub)
    user = await Session(client).start()

    _, match = await scored(user)

    assert stub.calls == 1, "the rubric was attempted"
    assert 0 <= match["overall_score"] <= 100
    assert "rubric" not in match["subscores"], "no rubric weight was spent"


async def test_an_unscored_match_says_so_by_leaving_its_provenance_null(
    client: AsyncClient, embeddings: StubEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only durable marker distinguishing a deterministic-only score from a
    model-scored one. Stamping these regardless would rewrite history."""
    monkeypatch.setattr("app.api.v1.matching.llm_client", StubRubric(fail=True))
    user = await Session(client).start()

    _, match = await scored(user)

    assert match["model"] is None
    assert match["prompt_version"] is None
    assert match["narrative"] is None
    assert match["strengths"] == []
    assert match["gaps"] == []


async def test_no_key_configured_skips_the_rubric_without_attempting_it(
    client: AsyncClient, embeddings: StubEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.v1.matching.llm_client", UnconfiguredLLM())
    user = await Session(client).start()

    _, match = await scored(user)

    assert match["model"] is None
    assert "rubric" not in match["subscores"]


# --- With the model --------------------------------------------------------


async def test_a_model_scored_match_records_what_scored_it(
    client: AsyncClient, embeddings: StubEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.v1.matching.llm_client", StubRubric(judgment(0.8)))
    user = await Session(client).start()

    _, match = await scored(user)

    assert match["model"] is not None
    assert match["prompt_version"] is not None
    assert match["subscores"]["rubric"] == pytest.approx(0.8)
    assert match["gaps"] == ["No Kafka or streaming experience shown"]


async def test_the_breakdown_adds_up_to_the_total(
    client: AsyncClient, embeddings: StubEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The card shows both, so they have to agree."""
    from app.services.matching import WEIGHTS

    monkeypatch.setattr("app.api.v1.matching.llm_client", StubRubric(judgment(0.8)))
    user = await Session(client).start()

    _, match = await scored(user)

    recomputed = round(sum(match["subscores"][k] * WEIGHTS[k] for k in match["subscores"]) * 100)
    assert recomputed == match["overall_score"]


@pytest.mark.parametrize("raw", [1.4, -0.3])
async def test_a_score_outside_the_scale_is_clamped_before_it_is_stored(
    client: AsyncClient, embeddings: StubEmbeddings, monkeypatch: pytest.MonkeyPatch, raw: float
) -> None:
    """The schema says 0.0-1.0 and nothing makes the model obey it.

    Clamping only the total left the stored subscore raw, so the displayed
    breakdown could not reproduce the displayed total — the failure is a card
    that quietly does not add up, which nobody reads closely enough to catch.
    """
    from app.services.matching import WEIGHTS

    monkeypatch.setattr("app.api.v1.matching.llm_client", StubRubric(judgment(raw)))
    user = await Session(client).start()

    _, match = await scored(user)

    assert 0.0 <= match["subscores"]["rubric"] <= 1.0
    recomputed = round(sum(match["subscores"][k] * WEIGHTS[k] for k in match["subscores"]) * 100)
    assert recomputed == match["overall_score"]


async def test_the_score_is_denormalised_onto_the_application(
    client: AsyncClient, embeddings: StubEmbeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dashboard sorts by it without joining, so a drift here shows up as a
    list ordered by a number that is not the one on the card."""
    monkeypatch.setattr("app.api.v1.matching.llm_client", StubRubric(judgment(0.8)))
    user = await Session(client).start()

    application_id, match = await scored(user)
    application = (await user.get(f"/api/v1/applications/{application_id}")).json()

    assert application["match_score"] == match["overall_score"]
