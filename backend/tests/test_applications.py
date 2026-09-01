from datetime import UTC, datetime, timedelta

from httpx import AsyncClient

from tests.factories import Session, job_payload


def iso(days_ago: int = 0) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


async def test_create_with_inline_job_resolves_company_and_skills(client: AsyncClient) -> None:
    user = await Session(client).start()

    application = await user.create_application()

    job = application["job"]
    assert job["company"]["name"] == "Amazon"
    assert job["title"] == "Backend Engineer"
    assert {r["kind"] for r in job["requirements"]} == {"must", "nice"}
    assert {s["skill"]["slug"] for s in job["skills"]} == {"python", "postgresql"}


async def test_unknown_skill_slug_is_rejected_not_silently_dropped(client: AsyncClient) -> None:
    """A typo'd slug that vanishes quietly produces a job with incomplete
    skills, which then scores wrongly in Phase 3."""
    user = await Session(client).start()

    response = await user.post(
        "/api/v1/applications",
        {"job": job_payload(skill_slugs=["python", "pyhton"])},
    )

    assert response.status_code == 422
    assert "pyhton" in response.json()["detail"]


async def test_cannot_track_the_same_job_twice(client: AsyncClient) -> None:
    user = await Session(client).start()
    first = await user.create_application()

    response = await user.post("/api/v1/applications", {"job_id": first["job"]["id"]})

    assert response.status_code == 409
    assert "already tracking" in response.json()["detail"].lower()


async def test_job_id_and_inline_job_are_mutually_exclusive(client: AsyncClient) -> None:
    user = await Session(client).start()
    existing = await user.create_application()

    both = await user.post(
        "/api/v1/applications",
        {"job_id": existing["job"]["id"], "job": job_payload()},
    )
    neither = await user.post("/api/v1/applications", {})

    assert both.status_code == 422
    assert neither.status_code == 422


async def test_salary_range_must_be_ordered(client: AsyncClient) -> None:
    user = await Session(client).start()

    response = await user.post(
        "/api/v1/applications",
        {"job": job_payload(salary_min="2400000", salary_max="1800000")},
    )

    assert response.status_code == 422


async def test_salary_requires_a_currency(client: AsyncClient) -> None:
    user = await Session(client).start()
    payload = job_payload()
    del payload["salary_currency"]

    response = await user.post("/api/v1/applications", {"job": payload})

    assert response.status_code == 422


async def test_status_cannot_be_set_directly(client: AsyncClient) -> None:
    """Status is derived from the log. Allowing a direct write would let the
    cache diverge from its own source of truth."""
    user = await Session(client).start()
    application = await user.create_application()

    response = await user.patch(
        f"/api/v1/applications/{application['id']}", {"current_status": "offer"}
    )

    assert response.status_code == 422


async def test_priority_and_notes_are_editable(client: AsyncClient) -> None:
    user = await Session(client).start()
    application = await user.create_application()

    response = await user.patch(
        f"/api/v1/applications/{application['id']}",
        {"priority": "high", "notes": "Referred by Meera"},
    )

    assert response.status_code == 200
    assert response.json()["priority"] == "high"
    assert response.json()["notes"] == "Referred by Meera"


async def test_list_filters_by_status_and_search(client: AsyncClient) -> None:
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")
    await user.create_application(
        company_name="Razorpay", title="ML Engineer", initial_event="applied"
    )
    await user.create_application(company_name="Zerodha", title="Data Engineer")

    by_status = (await user.get("/api/v1/applications?status=applied")).json()
    by_company = (await user.get("/api/v1/applications?search=razor")).json()
    by_title = (await user.get("/api/v1/applications?search=Engineer")).json()

    assert by_status["total"] == 1
    assert by_status["items"][0]["job"]["company"]["name"] == "Razorpay"
    assert by_company["total"] == 1
    assert by_title["total"] == 3


async def test_active_only_excludes_terminal_statuses(client: AsyncClient) -> None:
    user = await Session(client).start()
    live = await user.create_application(company_name="Amazon", initial_event="applied")
    dead = await user.create_application(company_name="Zerodha", initial_event="applied")
    await user.add_event(dead["id"], "rejected")

    active = (await user.get("/api/v1/applications?active_only=true")).json()

    assert [i["id"] for i in active["items"]] == [live["id"]]


async def test_list_sorts_and_paginates(client: AsyncClient) -> None:
    user = await Session(client).start()
    for name in ["Alpha", "Bravo", "Charlie"]:
        await user.create_application(company_name=name)

    page = (await user.get("/api/v1/applications?sort=company&order=asc&limit=2&offset=0")).json()

    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert [i["job"]["company"]["name"] for i in page["items"]] == ["Alpha", "Bravo"]


async def test_invalid_sort_key_is_rejected(client: AsyncClient) -> None:
    user = await Session(client).start()

    response = await user.get("/api/v1/applications?sort=; DROP TABLE users")

    assert response.status_code == 422


async def test_days_since_activity_is_computed(client: AsyncClient) -> None:
    user = await Session(client).start()
    await user.create_application(initial_event="applied", occurred_at=iso(days_ago=9))

    page = (await user.get("/api/v1/applications")).json()

    assert page["items"][0]["days_since_activity"] == 9


async def test_stats_counts_active_and_stale(client: AsyncClient) -> None:
    user = await Session(client).start()
    await user.create_application(
        company_name="Amazon", initial_event="applied", occurred_at=iso(days_ago=14)
    )
    await user.create_application(
        company_name="Razorpay", initial_event="applied", occurred_at=iso(days_ago=1)
    )
    closed = await user.create_application(company_name="Zerodha", initial_event="applied")
    await user.add_event(closed["id"], "rejected")

    stats = (await user.get("/api/v1/applications/stats")).json()

    assert stats["total"] == 3
    assert stats["active"] == 2
    assert stats["needs_attention"] == 1, "only the 14-day-old open application is stale"
    assert {s["status"]: s["count"] for s in stats["by_status"]}["rejected"] == 1


async def test_delete_removes_the_application_and_its_timeline(client: AsyncClient) -> None:
    user = await Session(client).start()
    application = await user.create_application()

    deleted = await user.delete(f"/api/v1/applications/{application['id']}")
    fetched = await user.get(f"/api/v1/applications/{application['id']}")

    assert deleted.status_code == 204
    assert fetched.status_code == 404


async def test_interview_stages_round_trip(client: AsyncClient) -> None:
    user = await Session(client).start()
    application = await user.create_application(initial_event="applied")

    created = await user.post(
        f"/api/v1/applications/{application['id']}/stages",
        {"round_number": 1, "stage_type": "hr_screen", "interviewer": "Meera"},
    )
    stage_id = created.json()["id"]
    updated = await user.patch(
        f"/api/v1/applications/{application['id']}/stages/{stage_id}",
        {"outcome": "passed", "notes": "Went well"},
    )

    assert created.status_code == 201
    assert updated.status_code == 200
    assert updated.json()["outcome"] == "passed"
    assert updated.json()["interviewer"] == "Meera"


async def test_missing_application_is_404(client: AsyncClient) -> None:
    user = await Session(client).start()

    response = await user.get("/api/v1/applications/00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
