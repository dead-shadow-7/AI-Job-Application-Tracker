"""The HTTP connection pool every model call shares.

Its own module because two things now need it — the LLM client and the chat-model
factory — and neither owns it. It is a process resource, like the database
engine, and it is closed from the application lifespan for the same reason.

Both clients are shared deliberately, for different reasons. The async one
carries the connection pool that matters: building an ``AsyncClient`` per call
discarded it along with the client, so every round paid a fresh TCP and TLS
handshake to a remote host — six of them in an assistant turn, for nothing. The
sync one is never used to send anything, but the OpenAI SDK builds one whether or
not it is asked to, and a chat model constructed per call would otherwise leave a
new one behind on every request.
"""

import asyncio

import httpx

from app.core.config import settings

# One pooled async client per event loop, and one sync client for the process.
_async_client: httpx.AsyncClient | None = None
_async_loop: asyncio.AbstractEventLoop | None = None
_sync_client: httpx.Client | None = None


def _timeout() -> httpx.Timeout:
    """Split rather than one scalar.

    A connect that hangs is a dead host and should fail fast, where a 90-second
    read is just a long generation. One number for both made them
    indistinguishable.
    """
    return httpx.Timeout(settings.llm_timeout_seconds, connect=10.0)


def _limits() -> httpx.Limits:
    return httpx.Limits(max_connections=20, max_keepalive_connections=10)


def get_http_client() -> httpx.AsyncClient:
    """The shared async client, rebuilt if the running loop has changed.

    Keyed on the loop rather than on ``is_closed``: a pooled keep-alive
    connection is bound to the loop that opened it, and after that loop closes
    the client still reports ``is_closed == False``. Reusing it then raises
    ``RuntimeError: Event loop is closed`` from deep inside httpcore. This is
    the same hazard the test suite disposes the DB engine for — see the
    docstring in tests/conftest.py — and it would otherwise bite the first
    caller to run on a second loop (a script, or the scheduled sweep).
    """
    global _async_client, _async_loop
    loop = asyncio.get_running_loop()
    if _async_client is None or _async_client.is_closed or _async_loop is not loop:
        _async_client = httpx.AsyncClient(timeout=_timeout(), limits=_limits())
        _async_loop = loop
    return _async_client


def get_sync_http_client() -> httpx.Client:
    """A client the OpenAI SDK builds whether or not anything sends through it.

    Nothing in this application makes a synchronous model call. The SDK still
    constructs a sync client eagerly, so handing it one keeps a chat model built
    per request from leaving a fresh pool behind each time.
    """
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        _sync_client = httpx.Client(timeout=_timeout(), limits=_limits())
    return _sync_client


async def close_http_client() -> None:
    global _async_client, _async_loop, _sync_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None
        _async_loop = None
    if _sync_client is not None:
        # Closing a sync client is blocking. It holds no connections today —
        # nothing ever sends through it — but running it off the loop keeps that
        # a fact about this application rather than a requirement of shutdown.
        await asyncio.to_thread(_sync_client.close)
        _sync_client = None
