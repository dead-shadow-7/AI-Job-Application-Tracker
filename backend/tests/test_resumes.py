"""Resume upload, embedding, and isolation.

Embeddings are stubbed. The real provider downloads a ~130 MB model on first
use and runs CPU inference; making CI do that on every push would add minutes
and a network dependency to prove something the embedding library already
guarantees. What is worth testing here is the pipeline around it — parsing,
chunking, defaults, and above all that one user's resume is invisible to
another.
"""

from typing import Any

from httpx import AsyncClient
from sqlalchemy import text

from app.db.session import SessionFactory, open_user_session
from tests.conftest import StubEmbeddings
from tests.factories import Session

RESUME_TEXT = """\
Aryan Jain
Backend Engineer | 4 years of experience

SUMMARY
Backend engineer building LLM-powered systems and scalable Python services.

WORK EXPERIENCE
- Built RAG pipelines over enterprise documents using LangChain and Qdrant.
- Designed FastAPI microservices on AWS handling 2 million requests per day.
- Implemented Kafka event pipelines for asynchronous order processing.
- Owned PostgreSQL schema design for a payments ledger.

SKILLS
Python, FastAPI, PostgreSQL, Kafka, Docker, Kubernetes, AWS, LangChain

EDUCATION
B.Tech in Computer Science, Pune Institute of Computer Technology
"""


async def upload_text(user: Session, label: str = "Main", body: str = RESUME_TEXT) -> Any:
    return await user.post("/api/v1/resumes/text", {"label": label, "text": body})


async def test_pasted_resume_is_chunked_and_embedded(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    user = await Session(client).start()

    response = await upload_text(user)

    assert response.status_code == 201
    body = response.json()
    assert body["label"] == "Main"
    assert body["chunk_count"] > 4, "each bullet should be its own chunk"
    assert embeddings.documents_embedded == body["chunk_count"]


async def test_years_of_experience_is_detected_on_upload(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    user = await Session(client).start()

    body = (await upload_text(user)).json()

    assert body["years_experience"] == "4.0"
    assert body["years_experience_source"] == "stated"


DATED_RESUME = """\
Aryan Jain
Backend Engineer

WORK EXPERIENCE

Senior Backend Engineer, Fintech Co (Jan 2021 - Present)
- Designed the double-entry ledger that reconciles every payout.
- Implemented Kafka event pipelines for asynchronous order processing.

Backend Engineer, Acme Technologies | March 2018 - December 2020
- Designed FastAPI microservices on AWS handling 2 million requests per day.

SKILLS
Python, FastAPI, PostgreSQL, Kafka, Docker, Kubernetes, AWS

EDUCATION
B.Tech in Computer Science, Pune Institute of Technology, 2014 - 2018
"""


async def test_roles_are_extracted_from_the_experience_section(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    user = await Session(client).start()

    body = (await upload_text(user, body=DATED_RESUME)).json()

    assert [(p["title"], p["company"]) for p in body["positions"]] == [
        ("Senior Backend Engineer", "Fintech Co"),
        ("Backend Engineer", "Acme Technologies"),
    ]
    assert body["positions"][0]["end"] is None, "the current role has no end date"


async def test_years_are_inferred_when_the_resume_only_gives_dates(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    """This is the common case. Leaving it null is not neutral: experience and
    seniority both fall back to 0.5, so a quarter of every score goes constant."""
    user = await Session(client).start()

    body = (await upload_text(user, body=DATED_RESUME)).json()

    assert body["years_experience_source"] == "dates"
    assert float(body["years_experience"]) > 5


# --- Re-parsing ------------------------------------------------------------


async def test_reparsing_rebuilds_chunks_rather_than_duplicating_them(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    """The point of keeping parsed_text is to benefit from a better parser
    without re-uploading. Appending instead of replacing would double every
    passage and hand the rubric the same evidence twice."""
    user = await Session(client).start()
    resume = (await upload_text(user, body=DATED_RESUME)).json()

    reparsed = (await user.post(f"/api/v1/resumes/{resume['id']}/reparse", {})).json()

    assert reparsed["chunk_count"] == resume["chunk_count"]
    assert reparsed["positions"] == resume["positions"]
    async for session in open_user_session(user.user_id):
        stored = (await session.execute(text("SELECT count(*) FROM resume_chunks"))).scalar_one()
    assert stored == resume["chunk_count"]


async def test_reparsing_discards_scores_computed_from_the_old_parse(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    """A cached score was produced from passages that no longer exist. Keeping
    it is worse than having none — a stale score reads as a current one."""
    user = await Session(client).start()
    resume = (await upload_text(user, body=DATED_RESUME)).json()
    application = await user.create_application(company_name="Razorpay", title="Backend Engineer")
    scored = await user.post(f"/api/v1/applications/{application['id']}/match", {})
    assert scored.status_code == 201, scored.text

    await user.post(f"/api/v1/resumes/{resume['id']}/reparse", {})

    assert (await user.get(f"/api/v1/applications/{application['id']}/match")).json() is None
    refreshed = (await user.get(f"/api/v1/applications/{application['id']}")).json()
    assert refreshed["match_score"] is None, "the denormalized copy must go too"


async def test_another_users_resume_cannot_be_reparsed(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    resume = (await upload_text(alice)).json()

    assert (await bob.post(f"/api/v1/resumes/{resume['id']}/reparse", {})).status_code == 404


async def test_first_upload_becomes_the_default(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    user = await Session(client).start()

    body = (await upload_text(user)).json()

    assert body["is_default"] is True


async def test_exactly_one_resume_is_default(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    """Two defaults would make "score against my resume" ambiguous."""
    user = await Session(client).start()
    await upload_text(user, label="Old")
    await upload_text(user, label="New")

    resumes = (await user.get("/api/v1/resumes")).json()

    assert sum(1 for r in resumes if r["is_default"]) == 1
    assert resumes[0]["label"] == "New", "the default should sort first"


async def test_default_can_be_switched(client: AsyncClient, embeddings: StubEmbeddings) -> None:
    user = await Session(client).start()
    first = (await upload_text(user, label="First")).json()
    await upload_text(user, label="Second")

    switched = await user.post(f"/api/v1/resumes/{first['id']}/default")

    assert switched.status_code == 200
    assert switched.json()["is_default"] is True
    resumes = (await user.get("/api/v1/resumes")).json()
    assert sum(1 for r in resumes if r["is_default"]) == 1


async def test_too_short_text_is_rejected(client: AsyncClient, embeddings: StubEmbeddings) -> None:
    user = await Session(client).start()

    response = await user.post("/api/v1/resumes/text", {"label": "Tiny", "text": "Hi"})

    assert response.status_code == 422
    assert embeddings.documents_embedded == 0


async def test_deleting_a_resume_removes_its_chunks(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    user = await Session(client).start()
    resume = (await upload_text(user)).json()

    deleted = await user.delete(f"/api/v1/resumes/{resume['id']}")

    assert deleted.status_code == 204
    async for session in open_user_session(user.user_id):
        remaining = (await session.execute(text("SELECT count(*) FROM resume_chunks"))).scalar_one()
    assert remaining == 0


# --- Isolation -------------------------------------------------------------


async def test_resumes_are_invisible_across_users(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    await upload_text(alice)

    assert (await bob.get("/api/v1/resumes")).json() == []


async def test_another_users_resume_is_not_reachable(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    resume = (await upload_text(alice)).json()

    assert (await bob.delete(f"/api/v1/resumes/{resume['id']}")).status_code == 404
    assert (await bob.post(f"/api/v1/resumes/{resume['id']}/default")).status_code == 404


async def test_database_refuses_cross_tenant_chunk_reads(
    client: AsyncClient, embeddings: StubEmbeddings
) -> None:
    """Below the API. Resume chunks are the most personal rows in the system, so
    the policy is asserted directly rather than only through the endpoints."""
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    await upload_text(alice)

    async for session in open_user_session(bob.user_id):
        rows = (await session.execute(text("SELECT id FROM resume_chunks"))).all()
    assert rows == []

    async with SessionFactory() as session:
        unscoped = (await session.execute(text("SELECT id FROM resume_chunks"))).all()
    assert unscoped == [], "an unscoped session must see nothing"
