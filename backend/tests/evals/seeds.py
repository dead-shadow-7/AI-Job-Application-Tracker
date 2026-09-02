"""The worlds an assistant case is asked a question about.

Built through the real API rather than by inserting rows, for the same reason
tests/factories.py does: a world assembled by hand can reach states the
application cannot, and an eval measuring behaviour in an impossible state
measures nothing. Status here is derived from events, RLS scoping is real, and
the search vectors exist — all of which the assistant's tools depend on.

Each world is named for the question it makes answerable. `two_amazon_roles`
exists so "the Amazon one" is genuinely ambiguous and the resolver has to refuse;
`one_iqvia_application` exists so it is genuinely not.
"""

from typing import Any

from tests.factories import Session

LONG_DESCRIPTION = """\
About Foundry Commerce

Foundry Commerce runs checkout infrastructure for mid-market retailers, handling
several million orders a month across fourteen countries.

The role

You will own our order service end to end: the API other teams build against,
the state machine behind it, and the reconciliation jobs that keep it honest.
This is a senior individual contributor role with a large amount of autonomy.

What we are looking for
- Six or more years of backend engineering
- Strong Python, and comfort reading code you did not write
- Experience with PostgreSQL under real load
- Familiarity with Redis or a similar cache

Equal opportunity employer statement follows.
Foundry Commerce is an equal opportunity employer and does not discriminate on
any protected basis.
"""


async def empty(user: Session) -> dict[str, Any]:
    """Nothing tracked. For questions about adding something."""
    return {}


async def two_amazon_roles(user: Session) -> dict[str, Any]:
    """Two roles at one company, so a company name alone cannot resolve.

    The case the whole resolver design exists for. Guessing here writes to the
    wrong timeline and nobody notices for weeks.
    """
    backend = await user.create_application(
        company_name="Amazon", title="Backend Engineer", initial_event="applied"
    )
    data = await user.create_application(
        company_name="Amazon", title="Data Engineer", initial_event="applied"
    )
    return {"backend": backend, "data": data}


async def one_iqvia_application(user: Session) -> dict[str, Any]:
    """One application, named the way the reported bug named it.

    A short all-caps company beside a role title that shares a common word with
    nothing else here — the shape that broke resolution in production.
    """
    iqvia = await user.create_application(
        company_name="IQVIA", title="Software Developer", initial_event="applied"
    )
    await user.create_application(
        company_name="Iris Software", title="Gen AI - Engineer", initial_event="applied"
    )
    return {"iqvia": iqvia}


async def stale_applied_nine_days(user: Session) -> dict[str, Any]:
    """Applied and silent, long enough to be worth chasing.

    Nine days rather than a round ten so an answer quoting the number has to
    have read it rather than guessed it.
    """
    razorpay = await user.create_application(
        company_name="Razorpay",
        title="Senior Backend Engineer",
        initial_event="applied",
        occurred_at=_days_ago(9),
    )
    return {"razorpay": razorpay}


async def long_description_job(user: Session) -> dict[str, Any]:
    """A stored posting long enough that relaying it would be rewriting it.

    Carries a line worth removing, so a description edit has something real to
    aim at, and so reaching for a note instead is visibly the wrong tool.
    """
    foundry = await user.create_application(
        company_name="Foundry Commerce",
        title="Senior Backend Engineer",
        description=LONG_DESCRIPTION,
        initial_event="applied",
    )
    return {"foundry": foundry}


def _days_ago(days: int) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


WORLDS = {
    "empty": empty,
    "two_amazon_roles": two_amazon_roles,
    "one_iqvia_application": one_iqvia_application,
    "stale_applied_nine_days": stale_applied_nine_days,
    "long_description_job": long_description_job,
}


async def build(name: str, user: Session) -> dict[str, Any]:
    if name not in WORLDS:
        raise KeyError(f"No seed named {name!r}; have {sorted(WORLDS)}")
    return await WORLDS[name](user)
