from httpx import AsyncClient


async def test_liveness(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_reports_pgvector(client: AsyncClient) -> None:
    """pgvector is load-bearing from Phase 3 on; catch a missing extension at
    boot rather than at the first embedding write."""
    response = await client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "ok"
    assert body["pgvector"] is True
