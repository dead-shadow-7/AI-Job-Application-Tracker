"""Semantic search, duplicate detection, and analytics.

Search and dedup run against the stubbed embedder, which produces vectors from
a hash of the text. That is fine for the plumbing — scoping, thresholds,
ordering — but it means nothing here asserts on retrieval *quality*. Whether
"RAG roles" actually finds an ML posting is a property of bge-small, verified
by hand against the real model rather than pinned in CI where it would only
prove the stub is deterministic.

The analytics tests carry more weight, because the risk there is a number that
is quietly wrong rather than obviously absent.
"""

from datetime import UTC, datetime, timedelta

from httpx import AsyncClient

from tests.factories import Session


def iso(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


# --- Semantic search -------------------------------------------------------


async def test_search_is_scoped_to_the_caller(client: AsyncClient) -> None:
    """job_embeddings is shared reference data with no row policy, so the
    tenant filter has to come from the join through applications. Getting this
    wrong would expose one user's tracked roles to another."""
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    await alice.create_application(company_name="Amazon", title="Backend Engineer")

    hits = (await bob.get("/api/v1/search?q=backend engineer")).json()

    assert hits == []


async def test_search_returns_the_callers_own_applications(client: AsyncClient) -> None:
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon", title="Backend Engineer")

    hits = (await user.get("/api/v1/search?q=backend engineer")).json()

    assert any(h["application_id"] == application["id"] for h in hits)
    assert all(0.0 <= h["similarity"] <= 1.0 for h in hits)


async def test_a_short_query_is_rejected(client: AsyncClient) -> None:
    """One character would match everything and mean nothing."""
    user = await Session(client).start()

    assert (await user.get("/api/v1/search?q=a")).status_code == 422


# --- Analytics -------------------------------------------------------------


async def test_analytics_on_an_empty_tracker_says_nothing_confidently(
    client: AsyncClient,
) -> None:
    user = await Session(client).start()

    body = (await user.get("/api/v1/analytics")).json()

    assert body["total"] == 0
    assert body["response_rate"] is None
    assert body["sample_is_small"] is True
    assert body["caveat"] is not None


async def test_a_small_sample_is_flagged_rather_than_presented_as_a_finding(
    client: AsyncClient,
) -> None:
    """A response rate over two applications is noise. Presenting it without
    saying so invites acting on it."""
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")
    await user.create_application(company_name="Razorpay", initial_event="applied")

    body = (await user.get("/api/v1/analytics")).json()

    assert body["sample_is_small"] is True
    assert "not yet meaningful" in body["caveat"]


async def test_saved_but_never_applied_rows_do_not_count_against_the_response_rate(
    client: AsyncClient,
) -> None:
    """Otherwise keeping a shortlist would look like being ignored."""
    user = await Session(client).start()
    applied = await user.create_application(company_name="Amazon", initial_event="applied")
    await user.create_application(company_name="Razorpay", initial_event="saved")
    await user.add_event(applied["id"], "recruiter_reply")

    body = (await user.get("/api/v1/analytics")).json()

    assert body["total"] == 2
    assert body["submitted"] == 1, "the shortlisted row is tracked but was never sent"
    assert body["response_rate"] == 1.0, "one submitted application, one response"


async def test_the_response_rate_reports_its_own_denominator(client: AsyncClient) -> None:
    """A percentage without its base cannot be read. `total` is the wrong base
    and would understate the rate, so the count it was actually divided by is
    returned alongside it."""
    user = await Session(client).start()
    applied = await user.create_application(company_name="Amazon", initial_event="applied")
    await user.create_application(company_name="Razorpay", initial_event="applied")
    await user.create_application(company_name="Zerodha", initial_event="saved")
    await user.add_event(applied["id"], "recruiter_reply")

    body = (await user.get("/api/v1/analytics")).json()

    assert (body["responses"], body["submitted"]) == (1, 2)
    assert body["response_rate"] == round(1 / 2, 3)


async def test_a_follow_up_you_sent_is_not_a_response(client: AsyncClient) -> None:
    """The distinction that makes the figure worth anything: chasing someone is
    not the same as hearing back."""
    user = await Session(client).start()
    application = await user.create_application(
        company_name="Amazon", initial_event="applied", occurred_at=iso(20)
    )
    await user.add_event(application["id"], "follow_up_sent", iso(5))

    body = (await user.get("/api/v1/analytics")).json()

    assert body["responses"] == 0
    assert body["response_rate"] == 0.0


async def test_time_to_response_is_measured_from_the_application_date(
    client: AsyncClient,
) -> None:
    user = await Session(client).start()
    application = await user.create_application(
        company_name="Amazon", initial_event="applied", occurred_at=iso(20)
    )
    await user.add_event(application["id"], "recruiter_reply", iso(14))

    body = (await user.get("/api/v1/analytics")).json()

    assert body["median_days_to_response"] == 6.0


async def test_the_funnel_lists_every_status_including_empty_ones(
    client: AsyncClient,
) -> None:
    """A funnel with stages missing reads as data loss rather than a zero."""
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    funnel = (await user.get("/api/v1/analytics")).json()["funnel"]

    statuses = {stage["status"] for stage in funnel}
    assert {"saved", "applied", "screening", "offer", "rejected"} <= statuses
    assert next(s for s in funnel if s["status"] == "applied")["count"] == 1


async def test_platform_breakdown_groups_by_source(client: AsyncClient) -> None:
    user = await Session(client).start()
    a = await user.create_application(
        company_name="Amazon", source_platform="LinkedIn", initial_event="applied"
    )
    await user.create_application(
        company_name="Razorpay", source_platform="LinkedIn", initial_event="applied"
    )
    await user.create_application(
        company_name="Zerodha", source_platform="Naukri", initial_event="applied"
    )
    await user.add_event(a["id"], "recruiter_reply")

    platforms = {
        p["platform"]: p for p in (await user.get("/api/v1/analytics")).json()["by_platform"]
    }

    assert platforms["LinkedIn"]["applications"] == 2
    assert platforms["LinkedIn"]["response_rate"] == 0.5
    assert platforms["Naukri"]["response_rate"] == 0.0


async def test_analytics_are_per_user(client: AsyncClient) -> None:
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    await alice.create_application(company_name="Amazon", initial_event="applied")

    assert (await bob.get("/api/v1/analytics")).json()["total"] == 0
