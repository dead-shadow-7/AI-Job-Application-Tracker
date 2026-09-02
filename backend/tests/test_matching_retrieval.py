"""Does pgvector hand the rubric the passage that answers the requirement?

Separate from the rubric eval on purpose, and the separation is the point. Under
rule 1 of the rubric prompt a requirement whose evidence retrieval missed is
*correctly* scored as unmet — so if the two were measured together, a failing
index and a failing model would look identical. The rubric eval supplies its
evidence by hand; this one measures the step that supplies it in production.

Free, and no model call — `retrieve_evidence` embeds locally. Deselected by
default only because the real embedding model is a ~130 MB download on first
use, which is not something a routine `pytest -q` should trigger.

    docker compose exec backend pytest -m embeddings_real -q

The stubbed vectors every other test uses are hash-derived and carry no meaning,
so they cannot answer this question at all — which is what the note on
`StubEmbeddings` means by "nothing here asserts on retrieval quality".
"""

import pytest
from httpx import AsyncClient

from app.db.session import open_user_session
from app.models.resume import Resume
from app.services.matching import retrieve_evidence
from tests.factories import Session

pytestmark = pytest.mark.embeddings_real

# Each bullet is a chunk, and each names a different technology, so "which
# passage answers this requirement" has exactly one defensible answer. Written
# so the right chunk shares few words with the question — retrieval that only
# matched on a repeated keyword would be doing string search, not semantics.
RESUME = """\
EXPERIENCE

Senior Backend Engineer, Fintech Co (2019-2024)
- Designed the double-entry ledger that reconciles every payout, in PostgreSQL.
- Built the asynchronous order pipeline on Kafka, processing four million events a day.
- Ran the service mesh and deployments on Kubernetes across three regions.
- Wrote the customer-facing dashboard in React and TypeScript.
- Owned the on-call rotation and cut paging volume by two thirds in a year.

SKILLS
Python, PostgreSQL, Kafka, Kubernetes, React, TypeScript
"""

# (requirement as a job posting would phrase it, the word that must appear in a
# retrieved passage). Deliberately not the same wording as the resume.
REQUIREMENTS = [
    ("Experience with event streaming at scale", "Kafka"),
    ("Relational database schema design", "ledger"),
    ("Container orchestration in production", "Kubernetes"),
    ("Front-end development", "dashboard"),
]


@pytest.fixture
def embeddings():  # type: ignore[no-untyped-def]
    """Override the autouse stub — this is the one test that needs real vectors."""
    from app.services import embeddings as module

    return module.embedding_provider


async def stored_resume(client: AsyncClient) -> tuple[Session, Resume]:
    user = await Session(client).start()
    await user.post("/api/v1/resumes/text", {"label": "Main", "text": RESUME})

    async for session in open_user_session(user.user_id):
        from sqlalchemy import select

        resume = (await session.execute(select(Resume))).scalars().one()
        return user, resume
    raise AssertionError("no session")


@pytest.mark.parametrize(("requirement", "expected"), REQUIREMENTS)
async def test_the_passage_that_answers_a_requirement_is_retrieved(
    client: AsyncClient, requirement: str, expected: str
) -> None:
    """Recall@3, which is what the rubric actually receives.

    Three passages per requirement is what production retrieves, so a hit at
    rank three is a hit: the model sees all of them. What this rules out is the
    index returning three passages none of which bear on the question, which is
    the failure that makes a match score meaningless while looking fine.
    """
    user, resume = await stored_resume(client)

    async for session in open_user_session(user.user_id):
        chunks = await retrieve_evidence(session, resume.id, requirement)
        break

    assert chunks, "retrieval returned nothing at all"
    retrieved = " ".join(c.content for c in chunks).lower()
    assert expected.lower() in retrieved, (
        f"{requirement!r} retrieved passages that do not mention {expected!r}:\n"
        + "\n".join(f"  - {c.content}" for c in chunks)
    )


async def test_an_accomplishment_outranks_the_skills_list_naming_the_same_tool(
    client: AsyncClient,
) -> None:
    """The skills line names Kafka in six words; the bullet describes shipping a
    Kafka pipeline. Both are on-topic, and the shorter one often embeds closer
    to a short requirement precisely because it is short — which hands the
    rubric a keyword list as its evidence and gets a met requirement judged as
    barely evidenced. Section weighting is what stops that.
    """
    user, resume = await stored_resume(client)

    async for session in open_user_session(user.user_id):
        chunks = await retrieve_evidence(session, resume.id, "Experience with Apache Kafka")
        break

    assert chunks[0].section == "experience", (
        "the strongest evidence should be the accomplishment, not the claim:\n"
        + "\n".join(f"  [{c.section}] {c.content}" for c in chunks)
    )


async def test_retrieval_is_scoped_to_one_resume(client: AsyncClient) -> None:
    """The evidence for my match must not come from someone else's resume.

    RLS covers the query, but retrieval also filters by resume_id — a candidate
    with two resumes must be scored against the one they chose, not against
    whichever passage happened to embed closest.
    """
    user, mine = await stored_resume(client)
    await user.post(
        "/api/v1/resumes/text",
        {"label": "Other", "text": "EXPERIENCE\n\n- Managed a team of pastry chefs in Lyon.\n"},
    )

    async for session in open_user_session(user.user_id):
        chunks = await retrieve_evidence(session, mine.id, "kitchen management")
        break

    assert all(c.resume_id == mine.id for c in chunks)
    assert "pastry" not in " ".join(c.content for c in chunks).lower()
