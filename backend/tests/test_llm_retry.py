"""The retry policy, which had no test at all and is not a generic one.

Three of its rules are specific to this deployment and would be lost to any
library's defaults:

* **413 is retryable.** It normally means "payload too large", but Groq also
  returns it for tokens-per-minute exhaustion, with ``rate_limit_exceeded`` in
  the body. Treating it as a client error fails a request that would have
  succeeded a second later, and sends whoever is debugging off to shrink a
  prompt that was never the problem.
* **Jitter is added to ``Retry-After``, never subtracted.** Groq sends that
  header on every 429, so under exactly the contention worth spreading out, an
  un-jittered sleep re-aligns every caller on the same instant.
* **A bare ``RuntimeError`` is retried.** That is "Event loop is closed" from a
  pooled connection outliving the loop that opened it — transient, and
  indistinguishable from a transport error from here.

The last test is about the deadline rather than the retries: the two settings
have to stay in the relationship the config comment assumes.
"""

import asyncio
import json
from datetime import timedelta
from typing import Any

import httpx
import pytest

from app.agent.llm_client import LLMClient, LLMError
from app.core.config import settings
from app.schemas.extraction import ExtractedJob

HTTP = "app.agent.llm_client.get_http_client"
BASE_URL = "https://example.invalid/v1"

OK_CONTENT = {
    "company_name": "Acme",
    "title": "Backend Engineer",
    "seniority": None,
    "employment_type": None,
    "work_mode": None,
    "location": None,
    "salary": {
        "raw_text": None,
        "min_amount": None,
        "max_amount": None,
        "currency": None,
        "period": None,
    },
    "years_experience_min": None,
    "years_experience_max": None,
    "responsibilities": None,
    "requirements": [],
    "skills": [],
    "confidence": 0.5,
}


class Attempts:
    """Replays a scripted sequence of replies, one per attempt.

    A callable in the list is invoked instead of returned, which is how a
    transport-level failure is scripted alongside HTTP ones. The last entry
    repeats, so a test only has to script as far as it cares about.
    """

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        reply = self.replies[min(self.count, len(self.replies) - 1)]
        self.count += 1
        if callable(reply):
            raise reply()
        return reply


class Sleeps:
    """Stands in for ``asyncio.sleep`` and records what it was asked to wait."""

    def __init__(self) -> None:
        self.seconds: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.seconds.append(delay)


def install(monkeypatch: pytest.MonkeyPatch, *replies: Any) -> tuple[LLMClient, Attempts, Sleeps]:
    attempts = Attempts(*replies)
    stub = httpx.AsyncClient(transport=httpx.MockTransport(attempts))
    monkeypatch.setattr(HTTP, lambda: stub)

    sleeps = Sleeps()
    monkeypatch.setattr(asyncio, "sleep", sleeps)
    # The low end of every jitter range, so the arithmetic below is exact rather
    # than a tolerance. The ranges themselves are asserted by their endpoints.
    monkeypatch.setattr("app.agent.llm_client.random.uniform", lambda low, _high: low)

    return LLMClient(api_key="test-key", base_url=BASE_URL), attempts, sleeps


def error(status: int, message: str, code: str | None = None, **headers: str) -> httpx.Response:
    body: dict[str, Any] = {"error": {"message": message}}
    if code:
        body["error"]["code"] = code
    return httpx.Response(status, json=body, headers=headers)


def ok() -> httpx.Response:
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(OK_CONTENT)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )
    # See test_llm_payload.py: a body handed to MockTransport whole never
    # streams, so httpx leaves `elapsed` unset and the non-streamed path raises.
    response.elapsed = timedelta(milliseconds=5)
    return response


async def extract(client: LLMClient) -> Any:
    return await client.extract(schema=ExtractedJob, system="s", user="u")


# --- Which statuses are retried -------------------------------------------


async def test_a_413_is_retried_because_groq_means_rate_limit_by_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, attempts, _ = install(
        monkeypatch,
        error(413, "Request too large for gpt-oss-120b: 9000 > 8000 TPM", "rate_limit_exceeded"),
        ok(),
    )

    result = await extract(client)

    assert attempts.count == 2
    assert result.data.company_name == "Acme"


async def test_a_rate_limit_body_is_reported_as_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The status alone would read as "payload too large" and mislead."""
    client, _, _ = install(monkeypatch, error(413, "9000 tokens > 8000 TPM", "rate_limit_exceeded"))

    with pytest.raises(LLMError, match="Groq rate limit reached. 9000 tokens"):
        await extract(client)


async def test_a_non_retryable_status_fails_on_the_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 will say the same thing three times; the body is the useful part."""
    client, attempts, sleeps = install(monkeypatch, error(400, "unknown field 'stream_options'"))

    with pytest.raises(LLMError, match="Groq returned 400: unknown field"):
        await extract(client)

    assert attempts.count == 1
    assert sleeps.seconds == []


async def test_a_closed_event_loop_is_treated_as_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pooled connection outliving its loop raises this from inside httpcore.

    Not an httpx error, so it needs naming explicitly or the first caller on a
    second loop — a script, or the scheduled sweep — gets a hard failure.
    """
    client, attempts, _ = install(monkeypatch, lambda: RuntimeError("Event loop is closed"), ok())

    await extract(client)

    assert attempts.count == 2


async def test_a_transport_failure_surfaces_as_an_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing httpx-shaped may escape: LLMError is what the routes catch.

    ``LLMError`` is absent from the domain-error status table, so anything else
    reaching a route is a 500 — losing /chat's 422, the in-stream error frame on
    /chat/stream, and the graceful score-from-85% degrade in matching.
    """
    client, attempts, _ = install(monkeypatch, lambda: httpx.ConnectError("no route to host"))

    with pytest.raises(LLMError, match="unreachable after 3 attempts"):
        await extract(client)

    assert attempts.count == settings.llm_max_retries


# --- How long it waits ----------------------------------------------------


async def test_retry_after_is_honoured_and_only_ever_lengthened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, sleeps = install(
        monkeypatch, error(429, "slow down", "rate_limit_exceeded", **{"retry-after": "5"}), ok()
    )

    await extract(client)

    # The jitter multiplier is drawn from [1.0, 1.3] — the floor is the header's
    # own value, so a server-supplied wait is never undercut.
    assert sleeps.seconds == [5.0]


async def test_an_absurd_retry_after_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Waiting ten minutes inside a 100-second deadline helps nobody."""
    client, _, sleeps = install(
        monkeypatch, error(429, "slow down", **{"retry-after": "600"}), ok()
    )

    await extract(client)

    assert sleeps.seconds == [30.0]


async def test_an_unparseable_retry_after_falls_back_to_the_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header may be an HTTP date, or nonsense. Neither should raise."""
    client, _, sleeps = install(
        monkeypatch,
        error(429, "slow down", **{"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        ok(),
    )

    await extract(client)

    assert sleeps.seconds == [1.0]


async def test_the_backoff_doubles_and_never_sleeps_after_the_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sleeping after the final attempt delays only the exception."""
    client, attempts, sleeps = install(monkeypatch, error(503, "upstream unavailable"))

    with pytest.raises(LLMError, match="unreachable after 3 attempts"):
        await extract(client)

    assert attempts.count == settings.llm_max_retries
    # 2**0 and 2**1, each times the 0.5 floor of the [0.5, 1.5] jitter range.
    assert sleeps.seconds == [0.5, 1.0]


# --- How the retries sit inside the request deadline ----------------------


def test_the_deadline_is_what_actually_bounds_a_request() -> None:
    """The retry budget is deliberately larger than the deadline.

    Three attempts at a 90-second timeout is 270 seconds before backoff, and the
    ingestion graph retries extraction twice on top of that. The deadline is the
    only thing standing between a slow provider and an RLS transaction holding
    one of ten pooled connections for minutes. Raising it above the retry budget
    would silently retire that guarantee, so the relationship is asserted rather
    than left to the comment in config.py.
    """
    budget = settings.llm_max_retries * settings.llm_timeout_seconds

    assert settings.ingest_deadline_seconds < budget
    assert settings.assistant_deadline_seconds < budget
