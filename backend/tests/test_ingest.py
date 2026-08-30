"""The ingest endpoint, driven by a stubbed model.

CI must not depend on a third-party API: it would make the suite flaky when
Groq is slow, fail on a rate limit, and spend tokens on every push. The stub
returns whatever the test asks for, so the pipeline around the model — cleaning,
validation, skill resolution, duplicate detection — is what actually gets
exercised. Accuracy of the model itself is the eval suite's job, and that one is
opt-in.
"""

from typing import Any

import pytest
from httpx import AsyncClient

from app.agent.groq_client import LLMError, LLMUsage, StructuredResult
from app.schemas.extraction import ExtractedJob, ExtractedRequirement, ExtractedSalary
from tests.factories import Session

POSTING = """\
Senior Backend Engineer
Razorpay - Bangalore, India (Hybrid)

We are looking for a Senior Backend Engineer to join the Payments team.
You will design and operate high-throughput services processing millions of
transactions daily, own systems end to end, and mentor junior engineers.

Requirements
- 5+ years of backend engineering experience
- Strong proficiency in Python and SQL
- Deep experience with PostgreSQL at scale
- Hands-on experience with Kafka

Nice to have
- Experience with Kubernetes and Docker

Compensation: 45-60 LPA depending on experience.
"""


def extraction(**overrides: Any) -> ExtractedJob:
    defaults: dict[str, Any] = dict(
        company_name="Razorpay",
        title="Backend Engineer",
        seniority="senior",
        employment_type="full_time",
        work_mode="hybrid",
        location="Bangalore, India",
        salary=ExtractedSalary(
            raw_text="45-60 LPA",
            min_amount=4_500_000,
            max_amount=6_000_000,
            currency="INR",
            period="year",
        ),
        years_experience_min=5,
        years_experience_max=None,
        responsibilities="Design and operate high-throughput payment services.",
        requirements=[
            ExtractedRequirement(text="5+ years of backend engineering", kind="must"),
            ExtractedRequirement(text="Experience with Kubernetes", kind="nice"),
        ],
        skills=["Python", "SQL", "PostgreSQL", "Kafka", "Kubernetes", "Docker"],
        confidence=0.95,
    )
    defaults.update(overrides)
    return ExtractedJob(**defaults)


class StubGroq:
    def __init__(self, result: ExtractedJob | None = None, error: Exception | None = None) -> None:
        self._result = result if result is not None else extraction()
        self._error = error
        self.calls = 0
        self.is_configured = True

    async def extract(self, **_: Any) -> StructuredResult:
        self.calls += 1
        if self._error:
            raise self._error
        return StructuredResult(
            data=self._result,
            usage=LLMUsage(
                model="stub",
                prompt_tokens=900,
                completion_tokens=400,
                total_tokens=1300,
                latency_ms=1200,
            ),
        )


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubGroq:
    """Patched where it is looked up — the graph resolves the module global at
    call time, and the endpoint checks its own import for configuration."""
    client = StubGroq()
    monkeypatch.setattr("app.agent.graphs.ingestion.groq_client", client)
    monkeypatch.setattr("app.api.v1.ingest.groq_client", client)
    return client


async def test_ingest_returns_a_reviewable_preview(client: AsyncClient, stub: StubGroq) -> None:
    user = await Session(client).start()

    response = await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})

    assert response.status_code == 200
    body = response.json()
    assert body["job"]["company_name"] == "Razorpay"
    assert body["job"]["title"] == "Backend Engineer"
    assert body["job"]["salary_min"] == "4500000.0"
    assert body["confidence"] == "0.95"
    assert body["needs_review"] is False


async def test_ingest_writes_nothing(client: AsyncClient, stub: StubGroq) -> None:
    """Extraction is good, not perfect. A wrong row saved silently costs far
    more to find later than an edit made now."""
    user = await Session(client).start()

    await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})

    assert (await user.get("/api/v1/applications")).json()["total"] == 0


async def test_extracted_skills_resolve_to_canonical_slugs(
    client: AsyncClient, stub: StubGroq
) -> None:
    user = await Session(client).start()

    body = (await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})).json()

    assert set(body["job"]["skill_slugs"]) == {
        "python",
        "sql",
        "postgresql",
        "kafka",
        "kubernetes",
        "docker",
    }


async def test_alias_spellings_resolve(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'Golang' and 'ReactJS' must land on the same rows as 'Go' and 'React',
    or the Phase 3 match score counts one skill as two."""
    stub = StubGroq(extraction(skills=["Golang", "ReactJS", "Postgres", "K8s"]))
    monkeypatch.setattr("app.agent.graphs.ingestion.groq_client", stub)
    monkeypatch.setattr("app.api.v1.ingest.groq_client", stub)
    user = await Session(client).start()

    posting = POSTING + "\nAlso Golang, ReactJS, Postgres and K8s.\n"
    body = (await user.post("/api/v1/jobs/ingest", {"raw_text": posting})).json()

    assert set(body["job"]["skill_slugs"]) == {"go", "react", "postgresql", "kubernetes"}


async def test_unknown_skills_are_reported_not_invented(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A taxonomy that grows on every typo stops being a taxonomy."""
    stub = StubGroq(extraction(skills=["Python", "Blorptron"]))
    monkeypatch.setattr("app.agent.graphs.ingestion.groq_client", stub)
    monkeypatch.setattr("app.api.v1.ingest.groq_client", stub)
    user = await Session(client).start()

    body = (
        await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING + "\nBlorptron required.\n"})
    ).json()

    assert body["job"]["skill_slugs"] == ["python"]
    assert body["unmatched_skills"] == ["Blorptron"]


async def test_hallucinated_salary_is_stripped_and_flagged(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubGroq(
        extraction(
            salary=ExtractedSalary(
                raw_text="₹80,00,000 per annum",
                min_amount=8_000_000,
                max_amount=8_000_000,
                currency="INR",
                period="year",
            )
        )
    )
    monkeypatch.setattr("app.agent.graphs.ingestion.groq_client", stub)
    monkeypatch.setattr("app.api.v1.ingest.groq_client", stub)
    user = await Session(client).start()

    body = (await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})).json()

    assert body["job"]["salary_min"] is None
    assert body["needs_review"] is True, "a discarded salary must force review"
    assert "salary" in body["dropped_fields"]


async def test_low_confidence_forces_review(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubGroq(extraction(confidence=0.4))
    monkeypatch.setattr("app.agent.graphs.ingestion.groq_client", stub)
    monkeypatch.setattr("app.api.v1.ingest.groq_client", stub)
    user = await Session(client).start()

    body = (await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})).json()

    assert body["needs_review"] is True


async def test_too_short_input_is_rejected_before_spending_tokens(
    client: AsyncClient, stub: StubGroq
) -> None:
    user = await Session(client).start()

    response = await user.post("/api/v1/jobs/ingest", {"raw_text": "Backend engineer wanted"})

    assert response.status_code == 422
    assert "too short" in response.json()["detail"].lower()
    assert stub.calls == 0, "the model should not be called for obviously unusable input"


async def test_model_failure_surfaces_as_a_clear_error(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = StubGroq(error=LLMError("Groq rate limit reached. Limit 8000 TPM."))
    monkeypatch.setattr("app.agent.graphs.ingestion.groq_client", stub)
    monkeypatch.setattr("app.api.v1.ingest.groq_client", stub)
    user = await Session(client).start()

    response = await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})

    assert response.status_code == 422
    assert "rate limit" in response.json()["detail"].lower()


async def test_unusable_extraction_is_retried_once(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank company or title means the model produced nothing usable, which
    one more attempt at temperature 0 may fix. Dropped salary or skills are not
    retried — validation already removed them, and asking again would most
    likely reproduce the same invention at double the cost."""
    stub = StubGroq(extraction(company_name="  ", title="  "))
    monkeypatch.setattr("app.agent.graphs.ingestion.groq_client", stub)
    monkeypatch.setattr("app.api.v1.ingest.groq_client", stub)
    user = await Session(client).start()

    response = await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})

    assert stub.calls == 2
    assert response.status_code == 422


async def test_duplicate_posting_is_detected(client: AsyncClient, stub: StubGroq) -> None:
    """Catches pasting the same posting twice, or finding one job on two
    boards, before it becomes a second timeline."""
    user = await Session(client).start()
    preview = (await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})).json()
    created = await user.post("/api/v1/applications", {"job": preview["job"]})
    assert created.status_code == 201

    again = (await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})).json()

    assert again["duplicate_of"] == created.json()["id"]


async def test_preview_can_be_saved_directly(client: AsyncClient, stub: StubGroq) -> None:
    """The preview's `job` is exactly the create endpoint's payload, so review
    and save is an edit rather than a translation."""
    user = await Session(client).start()
    preview = (await user.post("/api/v1/jobs/ingest", {"raw_text": POSTING})).json()

    created = await user.post(
        "/api/v1/applications",
        {"job": preview["job"], "initial_event": "applied"},
    )

    assert created.status_code == 201
    body = created.json()
    assert body["job"]["company"]["name"] == "Razorpay"
    assert body["current_status"] == "applied"
    assert {s["skill"]["slug"] for s in body["job"]["skills"]} >= {"python", "kafka"}


async def test_ingest_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/jobs/ingest", json={"raw_text": POSTING})
    assert response.status_code == 401
