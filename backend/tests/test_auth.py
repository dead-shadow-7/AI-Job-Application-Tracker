from datetime import timedelta

import pytest
from httpx import AsyncClient

from tests.conftest import auth_headers, make_token


async def test_me_requires_a_token(client: AsyncClient) -> None:
    response = await client.get("/api/v1/me")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("expired", {"expires_in": timedelta(seconds=-30)}),
        ("wrong audience", {"audience": "anon"}),
        ("wrong signing key", {"secret": "an-attackers-secret-padded-past-the-hs256-minimum"}),
    ],
)
async def test_invalid_tokens_are_rejected(client: AsyncClient, label: str, kwargs: dict) -> None:
    _, token = make_token(**kwargs)
    response = await client.get("/api/v1/me", headers=auth_headers(token))
    assert response.status_code == 401, f"{label} token was accepted"


async def test_malformed_bearer_is_rejected(client: AsyncClient) -> None:
    response = await client.get("/api/v1/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


async def test_first_request_provisions_the_profile(client: AsyncClient) -> None:
    """A user exists in Supabase before they exist here; the first authenticated
    request is what materialises the profile row."""
    user_id, token = make_token(email="new.candidate@example.com")

    response = await client.get("/api/v1/me", headers=auth_headers(token))

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(user_id)
    assert body["email"] == "new.candidate@example.com"
    assert body["preferences"] == {}


async def test_provisioning_is_idempotent_and_syncs_email(client: AsyncClient) -> None:
    user_id, token = make_token(email="old@example.com")
    await client.get("/api/v1/me", headers=auth_headers(token))

    _, second_token = make_token(user_id=user_id, email="changed@example.com")
    response = await client.get("/api/v1/me", headers=auth_headers(second_token))

    assert response.status_code == 200
    assert response.json()["id"] == str(user_id)
    assert response.json()["email"] == "changed@example.com"


async def test_profile_update_round_trips(client: AsyncClient) -> None:
    _, token = make_token()
    await client.get("/api/v1/me", headers=auth_headers(token))

    response = await client.patch(
        "/api/v1/me",
        headers=auth_headers(token),
        json={
            "display_name": "Aryan",
            "preferences": {"work_mode": "remote", "min_salary_lpa": 24},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == "Aryan"
    assert body["preferences"]["work_mode"] == "remote"
