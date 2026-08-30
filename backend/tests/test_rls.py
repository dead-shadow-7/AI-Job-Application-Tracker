"""Multi-tenancy isolation.

These assert the *database* refuses cross-tenant reads, not that our queries
remember to filter. A ``WHERE user_id = ...`` that someone forgets to write in
Phase 2 is exactly the bug this is here to catch, so these run on every commit.
"""

import uuid

from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import SessionFactory, open_user_session
from tests.conftest import auth_headers, make_token


async def _seed_user(session: AsyncSession, user_id: uuid.UUID, email: str) -> None:
    """Insert bypassing RLS. TRUNCATE/INSERT as owner still respects FORCE RLS,
    so scope the insert to the user being created."""
    await session.execute(
        text("SELECT set_config('app.user_id', :uid, true)"), {"uid": str(user_id)}
    )
    await session.execute(
        text("INSERT INTO users (id, email) VALUES (:id, :email)"),
        {"id": user_id, "email": email},
    )
    await session.commit()


async def test_scoped_session_sees_only_its_own_row() -> None:
    alice, bob = uuid.uuid4(), uuid.uuid4()
    async with SessionFactory() as s:
        await _seed_user(s, alice, "alice@example.com")
    async with SessionFactory() as s:
        await _seed_user(s, bob, "bob@example.com")

    async for session in open_user_session(bob):
        rows = (await session.execute(text("SELECT id, email FROM users"))).all()

    # No WHERE clause at all — the policy is the only thing filtering.
    assert len(rows) == 1
    assert rows[0].id == bob
    assert rows[0].email == "bob@example.com"


async def test_unscoped_session_sees_nothing() -> None:
    """Fail closed: if app.user_id was never set, current_setting returns NULL
    and every row comparison is NULL, so the result set is empty rather than
    complete."""
    alice = uuid.uuid4()
    async with SessionFactory() as s:
        await _seed_user(s, alice, "alice@example.com")

    async with SessionFactory() as session:
        rows = (await session.execute(text("SELECT id FROM users"))).all()

    assert rows == []


async def test_scoped_session_cannot_read_another_user_by_id() -> None:
    """Even naming the row explicitly returns nothing."""
    alice, bob = uuid.uuid4(), uuid.uuid4()
    async with SessionFactory() as s:
        await _seed_user(s, alice, "alice@example.com")
    async with SessionFactory() as s:
        await _seed_user(s, bob, "bob@example.com")

    async for session in open_user_session(bob):
        result = await session.execute(text("SELECT id FROM users WHERE id = :id"), {"id": alice})
        assert result.first() is None


async def test_scoped_session_cannot_write_another_users_row() -> None:
    """WITH CHECK blocks the write path, not just reads."""
    alice, bob = uuid.uuid4(), uuid.uuid4()
    async with SessionFactory() as s:
        await _seed_user(s, alice, "alice@example.com")

    async for session in open_user_session(bob):
        result = await session.execute(
            text("UPDATE users SET email = 'hijacked@example.com' WHERE id = :id"),
            {"id": alice},
        )
        assert result.rowcount == 0

    async for session in open_user_session(alice):
        email = await session.scalar(text("SELECT email FROM users WHERE id = :id"), {"id": alice})
        assert email == "alice@example.com"


async def test_api_isolates_two_users(client: AsyncClient) -> None:
    """End-to-end through the real dependency chain."""
    alice_id, alice_token = make_token(email="alice@example.com")
    bob_id, bob_token = make_token(email="bob@example.com")

    await client.get("/api/v1/me", headers=auth_headers(alice_token))
    await client.patch(
        "/api/v1/me",
        headers=auth_headers(alice_token),
        json={"display_name": "Alice"},
    )

    bob_response = await client.get("/api/v1/me", headers=auth_headers(bob_token))

    assert bob_response.status_code == 200
    assert bob_response.json()["id"] == str(bob_id)
    assert bob_response.json()["display_name"] is None
    assert bob_id != alice_id
