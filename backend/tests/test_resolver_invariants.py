"""The property that ties retrieval to scoring, which shipped broken once.

`resolve_application` is two halves. SQL decides which rows are *candidates*;
`score_candidate` decides how well each one answers the query. They are written
apart, and nothing made them agree — so a phrasing the scorer rates 1.0 could be
one the SQL never retrieves, and the scorer would never be asked.

That is not hypothetical. Asked to reject "Software Developer at IQVIA" the
resolver offered "Gen AI - Engineer at Iris Software" — a different company and
a different role — because no column contained the whole phrase, leaving trigram
similarity on the company as the only predicate that could surface anything. It
is a ratio, so "IQVIA" scored 0.214 inside a twenty-seven character query while
"Iris Software" reached 0.3125 by sharing the word "Software". The right row was
never a candidate, and the wrong one was the only one.

The invariant, stated as a test:

    every query `score_candidate` rates above zero must be retrievable.

It found a second instance of the same bug on its first run. Widening the filter
fixed the phrasing above and left the next one: trigrams score a transposition
("Amzaon") at 0.27 while the scorer's SequenceMatcher rates it 0.6 and would
happily ask "did you mean Amazon?". Two similarity measures, still disagreeing.
The fix was to stop having two — `resolve_application` now fetches the user's
rows and lets the scorer decide, so retrieval cannot be narrower than scoring
because there is no longer anything to be narrower than.

So these tests now guard a property that holds by construction rather than by
care. That is the point: they will fail the day someone reintroduces a filter as
an optimisation, which is exactly when the reasoning above needs re-reading.
They run against the real database rather than a Python mirror of any predicate,
for the same reason — a mirror is a second description, and drift between two
descriptions of one rule is the whole failure.

These tests spend no tokens. The resolver is deterministic — this is the half of
the agent that can be pinned for free, and it is the half that aims every write.
"""

import pytest
from httpx import AsyncClient

from app.db.session import open_user_session
from app.services.resolver import resolve_application as resolve_in_db
from app.services.resolver import score_candidate
from tests.factories import Session

# Shapes chosen because each one broke, or nearly broke, the retrieval half.
#
# IQVIA is the reported failure: a short all-caps name inside a longer phrase.
# Iris Software is its accomplice — it shares a word with the *other* row's
# title, which is how it won a query that was not about it. Setoo is a short
# name that is also a substring of nothing; Set is at the `length > 2` boundary
# guarding the "name appears inside the phrase" predicate, below which a name
# would match nearly every query and drag the whole table in.
SEEDED = [
    ("IQVIA", "Software Developer"),
    ("Iris Software", "Gen AI - Engineer"),
    ("Amazon", "Backend Engineer"),
    ("Setoo", "AI/ML Intern"),
    ("Set", "Platform Engineer"),
    ("Razorpay", "Senior Data Engineer"),
]


def phrasings(company: str, title: str) -> list[str]:
    """Every way of naming this row that `score_candidate` rewards.

    Drawn from its branches rather than invented: exact company, exact title,
    the three orderings it scores 1.0, a prefix (partial company), a colloquial
    wrapper (word overlap), and a transposition (approximate spelling). If the
    scorer grows a branch, this list should grow with it.
    """
    transposed = (
        company if len(company) < 4 else company[:2] + company[3] + company[2] + company[4:]
    )
    return [
        company,
        title,
        f"{title} at {company}",
        f"{company} {title}",
        f"{title} {company}",
        company[:4],
        f"the {company} one",
        transposed,
    ]


async def seed(user: Session) -> None:
    for company, title in SEEDED:
        await user.create_application(company_name=company, title=title)


async def candidates_for(user: Session, query: str) -> list[tuple[str, str]]:
    async for session in open_user_session(user.user_id):
        resolution = await resolve_in_db(session, user.user_id, query)
        return [
            (c.application.job.company.name, c.application.job.title) for c in resolution.candidates
        ]
    raise AssertionError("no session")


async def test_anything_the_scorer_would_rate_can_be_retrieved(client: AsyncClient) -> None:
    """The invariant, over every seeded row and every phrasing it rewards.

    One test rather than a parametrised sweep because the seeding is the
    expensive part and the failure message needs the whole picture: which
    phrasings went missing tells you which SQL predicate is absent, where a
    single red case only tells you something is.
    """
    user = await Session(client).start()
    await seed(user)

    missing: list[str] = []
    for company, title in SEEDED:
        for query in phrasings(company, title):
            if score_candidate(query, company, title, True)[0] <= 0:
                continue  # the scorer would reject it too; retrieval need not find it
            if (company, title) not in await candidates_for(user, query):
                score, basis = score_candidate(query, company, title, True)
                missing.append(f"  {query!r} -> {company} / {title} (scores {score}, {basis})")

    assert not missing, (
        "retrieval is narrower than scoring — these queries would score above "
        "zero but never reach the scorer:\n" + "\n".join(missing)
    )


async def test_naming_the_role_and_company_beats_a_row_that_shares_a_word(
    client: AsyncClient,
) -> None:
    """The reported failure, as the user typed it.

    Retrievability alone is not enough here: the point is that the right row
    also *wins*. Iris Software is still a legitimate candidate for this query —
    it does share a word — and the test is that sharing a word loses to naming
    both the role and the company.
    """
    user = await Session(client).start()
    await seed(user)

    async for session in open_user_session(user.user_id):
        resolution = await resolve_in_db(session, user.user_id, "Software Developer at IQVIA")
        break

    assert resolution.is_confident, resolution.describe()
    assert resolution.best is not None
    assert resolution.best.application.job.company.name == "IQVIA"
    assert resolution.best.matched_on == "company and role"


async def test_a_short_name_is_not_buried_by_a_long_sentence(client: AsyncClient) -> None:
    """Trigram similarity is a ratio, so it degrades as the query grows.

    Whether a row can be found must not depend on how much else was said around
    its name. This is the cause underneath the reported failure, stated on its
    own so a fix that only patched the symptom would still fail here.
    """
    user = await Session(client).start()
    await seed(user)

    found = await candidates_for(
        user, "the Software Developer role I applied for at IQVIA back in July"
    )

    assert ("IQVIA", "Software Developer") in found


async def test_a_two_character_name_does_not_drag_in_the_whole_table(
    client: AsyncClient,
) -> None:
    """The other direction: permissive retrieval must not mean permissive results.

    A one- or two-character company name is a substring of almost everything, so
    any rule matching names inside a phrase has to exclude it. Now that the
    scorer alone decides, the guard is `score_candidate` returning 0 — but the
    hazard is unchanged, and it is worth a test because its symptom is not an
    error: asking about one application returns every application, which reads
    to the user as ambiguity rather than as a bug.
    """
    user = await Session(client).start()
    await user.create_application(company_name="Ai", title="Research Engineer")
    await user.create_application(company_name="Amazon", title="Backend Engineer")
    await user.create_application(company_name="Razorpay", title="Data Engineer")

    found = await candidates_for(user, "what did the Amazon one ask for")

    assert ("Ai", "Research Engineer") not in found, "a 2-char name matched an unrelated query"
    assert ("Amazon", "Backend Engineer") in found


@pytest.mark.parametrize("query", ["", "   ", "Spotify", "zzzzzzzz"])
async def test_retrieval_stays_narrow_where_the_scorer_would_reject(
    client: AsyncClient, query: str
) -> None:
    """The invariant only runs one way.

    Retrieval must be at least as permissive as scoring; it must not be
    unboundedly more so. A predicate that matched everything would satisfy the
    property above and destroy the resolver — every query would return five
    candidates and the agent would ask which one you meant, always.
    """
    user = await Session(client).start()
    await seed(user)

    found = await candidates_for(user, query)

    assert found == [], f"{query!r} should match nothing, got {found}"
