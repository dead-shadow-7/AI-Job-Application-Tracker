"""Correcting a job by hand.

Extraction is good and not perfect, and it is not the only way a record gets
created — jobs added through the assistant or typed in by hand start with
almost nothing. Every gap the pipeline leaves has to be closable by the person
applying, or the dash on the detail page is a dead end.

The interesting cases are not "does PATCH write the field". They are the ones
where editing something has a consequence elsewhere: the search vector, the
match score, and a list that must be replaced rather than merged.
"""

from httpx import AsyncClient

from tests.factories import Session


async def test_a_missing_field_can_be_filled_in(client: AsyncClient) -> None:
    """A posting that never stated a salary, or a job the assistant created
    from a sentence, leaves blanks only the applicant can fill."""
    user = await Session(client).start()
    application = await user.create_application(
        company_name="Amazon", salary_min=None, salary_max=None, seniority=None
    )
    job_id = application["job"]["id"]

    response = await user.patch(
        f"/api/v1/jobs/{job_id}",
        {
            "salary_min": "1800000",
            "salary_max": "2400000",
            "salary_currency": "INR",
            "salary_period": "year",
            "seniority": "senior",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert float(body["salary_min"]) == 1_800_000
    assert body["seniority"] == "senior"


async def test_a_field_can_be_cleared(client: AsyncClient) -> None:
    """Clearing is a real edit. An explicit null has to reach the column rather
    than being filtered out as "nothing to change"."""
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon")
    job_id = application["job"]["id"]

    await user.patch(f"/api/v1/jobs/{job_id}", {"location": None, "salary_min": None})

    body = (await user.get(f"/api/v1/jobs/{job_id}")).json()
    assert body["location"] is None
    assert body["salary_min"] is None


async def test_requirements_are_replaced_not_merged(client: AsyncClient) -> None:
    """Editing a requirement list is a rewrite. Merging by text would resurrect
    a line the moment you fixed a typo in it — extraction splits bulleted lists
    badly, so fixing them is the common case, not the rare one."""
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon")
    job_id = application["job"]["id"]
    assert len(application["job"]["requirements"]) == 2

    body = (
        await user.patch(
            f"/api/v1/jobs/{job_id}",
            {"requirements": [{"text": "5 years of Python", "kind": "must"}]},
        )
    ).json()

    assert [r["text"] for r in body["requirements"]] == ["5 years of Python"]


async def test_skills_can_be_corrected(client: AsyncClient) -> None:
    """Skills drive the match score, so a wrongly extracted one is not cosmetic
    — it changes what the candidate is measured against."""
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon")
    job_id = application["job"]["id"]

    body = (await user.patch(f"/api/v1/jobs/{job_id}", {"skill_slugs": ["python"]})).json()

    assert [s["skill"]["slug"] for s in body["skills"]] == ["python"]


async def test_an_unknown_skill_slug_is_refused(client: AsyncClient) -> None:
    """Silently dropping it would leave a job with quietly incomplete skills,
    which then scores wrongly and gives no hint why."""
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon")

    response = await user.patch(
        f"/api/v1/jobs/{application['job']['id']}", {"skill_slugs": ["pyhton"]}
    )

    assert response.status_code == 422
    assert "pyhton" in response.json()["detail"]


async def test_editing_the_title_refreshes_the_search_vector(client: AsyncClient) -> None:
    """The stored embedding is built from title, seniority, location and the
    requirements. Editing one without re-embedding leaves semantic search
    answering from the old text — quietly, and until someone re-saves the job."""
    from sqlalchemy import text as sql

    from app.db.session import open_user_session

    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon", title="Backend Engineer")
    job_id = application["job"]["id"]

    await user.patch(f"/api/v1/jobs/{job_id}", {"title": "Site Reliability Engineer"})

    async for session in open_user_session(user.user_id):
        stored = (
            await session.execute(
                sql("SELECT content FROM job_embeddings WHERE job_id = :j"), {"j": job_id}
            )
        ).scalar_one()
    assert "Site Reliability Engineer" in stored
    assert "Backend Engineer" not in stored


async def test_replacing_requirements_refreshes_the_search_vector(client: AsyncClient) -> None:
    from sqlalchemy import text as sql

    from app.db.session import open_user_session

    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon")
    job_id = application["job"]["id"]

    await user.patch(
        f"/api/v1/jobs/{job_id}",
        {"requirements": [{"text": "Kubernetes at scale", "kind": "must"}]},
    )

    async for session in open_user_session(user.user_id):
        stored = (
            await session.execute(
                sql("SELECT content FROM job_embeddings WHERE job_id = :j"), {"j": job_id}
            )
        ).scalar_one()
    assert "Kubernetes at scale" in stored


async def test_you_cannot_edit_a_job_you_do_not_track(client: AsyncClient) -> None:
    """`jobs` is shared reference data with no row policy, so reachability is
    gated by having an application against it. Without that check any job id
    would be editable by anyone."""
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    application = await alice.create_application(company_name="Amazon")

    response = await bob.patch(f"/api/v1/jobs/{application['job']['id']}", {"title": "Vandalised"})

    assert response.status_code == 404


async def test_priority_and_notes_live_on_the_application_not_the_job(
    client: AsyncClient,
) -> None:
    """Two people can track the same posting. Priority is a judgement about
    your own search, so it must not travel with the shared job row."""
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon")

    body = (
        await user.patch(
            f"/api/v1/applications/{application['id']}",
            {"priority": "high", "notes": "Referral from Priya"},
        )
    ).json()

    assert body["priority"] == "high"
    assert body["notes"] == "Referral from Priya"
