"""Resolving "the Amazon application" to one row — or refusing to.

The most safety-critical code in the agent. Every agent write is aimed by this,
and picking the wrong application corrupts a timeline in a way nobody notices
until they are reading history that is quietly false.

The behaviour under test is therefore mostly about *declining* to answer.
"""

import pytest
from httpx import AsyncClient

from app.db.session import open_user_session
from app.services.resolver import resolve_application, score_candidate
from tests.factories import Session


async def resolve(user: Session, query: str):
    async for session in open_user_session(user.user_id):
        return await resolve_application(session, user.user_id, query)
    raise AssertionError("no session")


# --- Scoring, in isolation -------------------------------------------------


def test_company_outranks_title() -> None:
    """People say "the Amazon one", not "the backend engineer one"."""
    company, _ = score_candidate("Amazon", "Amazon", "Backend Engineer", True)
    title, _ = score_candidate("Backend Engineer", "Amazon", "Backend Engineer", True)

    assert company > title


def test_naming_both_scores_highest() -> None:
    both, basis = score_candidate("Backend Engineer at Amazon", "Amazon", "Backend Engineer", True)

    assert both == 1.0
    assert "company and role" in basis


def test_closed_applications_are_demoted_not_excluded() -> None:
    """Rarely what someone means, but they may be correcting a mistaken
    rejection — so it stays reachable."""
    live, _ = score_candidate("Amazon", "Amazon", "Backend Engineer", True)
    closed, basis = score_candidate("Amazon", "Amazon", "Backend Engineer", False)

    assert 0 < closed < live
    assert "closed" in basis


def test_an_unrelated_query_scores_zero() -> None:
    score, _ = score_candidate("Spotify", "Amazon", "Backend Engineer", True)

    assert score == 0.0


# --- Resolution against real rows ------------------------------------------


async def test_an_unambiguous_reference_resolves(client: AsyncClient) -> None:
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")
    await user.create_application(company_name="Razorpay", title="ML Engineer")

    resolution = await resolve(user, "Amazon")

    assert resolution.is_confident
    assert resolution.best is not None
    assert resolution.best.application.job.company.name == "Amazon"


async def test_two_roles_at_one_company_is_ambiguous(client: AsyncClient) -> None:
    """The case the whole design exists for. Guessing here would silently write
    to the wrong timeline, and the user would not find out for weeks."""
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")
    await user.create_application(company_name="Amazon", title="Data Engineer")

    resolution = await resolve(user, "Amazon")

    assert not resolution.is_confident, "must not pick one of two equal matches"
    assert resolution.best is None
    assert len(resolution.candidates) == 2


async def test_the_ambiguous_message_lists_the_options(client: AsyncClient) -> None:
    """Refusing is only useful if the user can then choose."""
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")
    await user.create_application(company_name="Amazon", title="Data Engineer")

    message = (await resolve(user, "Amazon")).describe()

    assert "Backend Engineer" in message
    assert "Data Engineer" in message
    assert "which one" in message.lower()


async def test_adding_the_role_disambiguates(client: AsyncClient) -> None:
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")
    await user.create_application(company_name="Amazon", title="Data Engineer")

    resolution = await resolve(user, "Data Engineer at Amazon")

    assert resolution.is_confident
    assert resolution.best.application.job.title == "Data Engineer"


async def test_nothing_matching_returns_nothing(client: AsyncClient) -> None:
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    resolution = await resolve(user, "Spotify")

    assert resolution.candidates == []
    assert resolution.best is None
    assert "Nothing matches" in resolution.describe()


async def test_a_misspelling_still_surfaces_the_row(client: AsyncClient) -> None:
    """Returning nothing for "Amazn" reads as "no such application", which
    sends the user looking for a bug rather than a typo."""
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")

    resolution = await resolve(user, "Amazn")

    assert resolution.candidates, "trigram similarity should still find it"


async def test_an_empty_query_resolves_to_nothing(client: AsyncClient) -> None:
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    assert (await resolve(user, "   ")).candidates == []


@pytest.mark.parametrize("query", ["amazon", "AMAZON", "  Amazon  "])
async def test_matching_ignores_case_and_padding(client: AsyncClient, query: str) -> None:
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")

    assert (await resolve(user, query)).is_confident


# --- Isolation -------------------------------------------------------------


async def test_the_resolver_cannot_see_another_users_applications(
    client: AsyncClient,
) -> None:
    """An agent acting for one user must not be able to aim at another's rows,
    even by naming them exactly."""
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    await alice.create_application(company_name="Amazon", title="Backend Engineer")

    resolution = await resolve(bob, "Amazon")

    assert resolution.candidates == []
