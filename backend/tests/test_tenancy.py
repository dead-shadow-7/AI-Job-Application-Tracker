"""Tenant isolation across the Phase 1 tables.

test_rls.py proves the mechanism on ``users``. This proves it holds on the
tables that actually carry your job search — and that the shared job/company
rows are not reachable by someone who does not track them.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from app.db.session import SessionFactory, open_user_session
from tests.factories import Session


async def test_applications_are_invisible_across_users(client: AsyncClient) -> None:
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    await alice.create_application(company_name="Setoo")

    bob_list = (await bob.get("/api/v1/applications")).json()

    assert bob_list["total"] == 0


async def test_another_users_application_is_404_not_403(client: AsyncClient) -> None:
    """404 rather than 403 on purpose: confirming a row exists but is not yours
    still leaks the existence of another user's data."""
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    application = await alice.create_application()

    response = await bob.get(f"/api/v1/applications/{application['id']}")

    assert response.status_code == 404


async def test_cannot_append_events_to_another_users_application(client: AsyncClient) -> None:
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    application = await alice.create_application()

    response = await bob.add_event(application["id"], "rejected")
    still_mine = (await alice.get(f"/api/v1/applications/{application['id']}")).json()

    assert response.status_code == 404
    assert still_mine["current_status"] == "saved"


async def test_cannot_delete_another_users_application(client: AsyncClient) -> None:
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    application = await alice.create_application()

    response = await bob.delete(f"/api/v1/applications/{application['id']}")

    assert response.status_code == 404
    assert (await alice.get(f"/api/v1/applications/{application['id']}")).status_code == 200


async def test_a_job_you_do_not_track_is_not_readable(client: AsyncClient) -> None:
    """``jobs`` is shared reference data with no row policy, so reachability is
    gated in the router instead. Without that gate any job id would be readable
    by any user."""
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    application = await alice.create_application()
    job_id = application["job"]["id"]

    assert (await bob.get(f"/api/v1/jobs/{job_id}")).status_code == 404
    assert (await alice.get(f"/api/v1/jobs/{job_id}")).status_code == 200


async def test_contacts_are_isolated(client: AsyncClient) -> None:
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    created = await alice.post("/api/v1/contacts", {"name": "Meera", "role": "Recruiter"})

    bob_contacts = (await bob.get("/api/v1/contacts")).json()
    bob_edit = await bob.patch(f"/api/v1/contacts/{created.json()['id']}", {"name": "Hijacked"})

    assert bob_contacts == []
    assert bob_edit.status_code == 404


async def test_database_refuses_cross_tenant_event_reads(client: AsyncClient) -> None:
    """Below the API: even raw SQL with no WHERE clause sees only one tenant."""
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    await alice.create_application(company_name="Setoo")
    await bob.create_application(company_name="Razorpay")

    async for session in open_user_session(bob.user_id):
        rows = (await session.execute(text("SELECT user_id FROM application_events"))).all()

    assert rows, "bob should see his own events"
    assert {r.user_id for r in rows} == {bob.user_id}


async def test_unscoped_session_sees_no_applications(client: AsyncClient) -> None:
    alice = await Session(client, "alice@example.com").start()
    await alice.create_application()

    async with SessionFactory() as session:
        rows = (await session.execute(text("SELECT id FROM applications"))).all()

    assert rows == []


async def test_cannot_forge_an_event_for_another_user(client: AsyncClient) -> None:
    """WITH CHECK blocks writes, not just reads.

    Note the asymmetry: an UPDATE against someone else's row matches nothing and
    reports zero rows, because USING filters what is visible. An INSERT of a row
    that fails WITH CHECK raises 42501 outright — there is nothing to filter, so
    Postgres refuses loudly. Both fail closed; only one is silent.
    """
    alice = await Session(client, "alice@example.com").start()
    application = await alice.create_application()
    intruder = uuid.uuid4()

    with pytest.raises(ProgrammingError, match="row-level security"):
        async for session in open_user_session(intruder):
            await session.execute(
                text(
                    "INSERT INTO application_events (application_id, user_id, event_type) "
                    "VALUES (:app, :uid, 'rejected')"
                ),
                {"app": uuid.UUID(application["id"]), "uid": alice.user_id},
            )

    # And the forged event never landed.
    timeline = (await alice.get(f"/api/v1/applications/{application['id']}/events")).json()
    assert [e["event_type"] for e in timeline] == ["saved"]
