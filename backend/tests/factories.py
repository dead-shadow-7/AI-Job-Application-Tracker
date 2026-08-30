"""Helpers for building fixture data through the real API.

Going through HTTP rather than inserting rows directly means the fixtures
exercise the same validation, RLS scoping and event bookkeeping as production
traffic — a factory that bypassed them could set up states the app cannot
actually reach.
"""

from typing import Any

from httpx import AsyncClient

from tests.conftest import auth_headers, make_token


def job_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "company_name": "Setoo",
        "title": "Backend Engineer",
        "work_mode": "remote",
        "seniority": "mid",
        "location": "Pune, India",
        "salary_min": "1800000",
        "salary_max": "2400000",
        "salary_currency": "INR",
        "salary_period": "year",
        "description": "Build and operate Python services.",
        "requirements": [
            {"text": "3+ years with Python", "kind": "must"},
            {"text": "Exposure to Kubernetes", "kind": "nice"},
        ],
        "skill_slugs": ["python", "postgresql"],
    }
    payload.update(overrides)
    return payload


class Session:
    """An authenticated user for the duration of a test."""

    def __init__(self, client: AsyncClient, email: str = "candidate@example.com") -> None:
        self.client = client
        self.user_id, self.token = make_token(email=email)
        self.headers = auth_headers(self.token)

    async def start(self) -> "Session":
        """Provision the profile row. First authenticated request does this."""
        await self.client.get("/api/v1/me", headers=self.headers)
        return self

    async def create_application(
        self,
        *,
        initial_event: str | None = None,
        occurred_at: str | None = None,
        **job_overrides: Any,
    ) -> dict[str, Any]:
        """Track a job.

        ``initial_event`` / ``occurred_at`` mirror how backfilling works: an
        application imported from a spreadsheet is created directly as
        ``applied`` on its real date, rather than saved-now-then-backdated.
        """
        body: dict[str, Any] = {"job": job_payload(**job_overrides)}
        if initial_event:
            body["initial_event"] = initial_event
        if occurred_at:
            body["occurred_at"] = occurred_at

        response = await self.client.post("/api/v1/applications", headers=self.headers, json=body)
        assert response.status_code == 201, response.text
        return response.json()

    async def add_event(
        self, application_id: str, event_type: str, occurred_at: str | None = None, **extra: Any
    ):
        body: dict[str, Any] = {"event_type": event_type, **extra}
        if occurred_at:
            body["occurred_at"] = occurred_at
        return await self.client.post(
            f"/api/v1/applications/{application_id}/events", headers=self.headers, json=body
        )

    async def get(self, path: str, **kwargs: Any):
        return await self.client.get(path, headers=self.headers, **kwargs)

    async def post(self, path: str, json: Any = None, **kwargs: Any):
        return await self.client.post(path, headers=self.headers, json=json, **kwargs)

    async def patch(self, path: str, json: Any = None, **kwargs: Any):
        return await self.client.patch(path, headers=self.headers, json=json, **kwargs)

    async def delete(self, path: str, **kwargs: Any):
        return await self.client.delete(path, headers=self.headers, **kwargs)
