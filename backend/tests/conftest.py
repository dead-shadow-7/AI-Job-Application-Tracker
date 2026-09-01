"""Test harness.

Environment is rewritten *before* any ``app.*`` import so the settings singleton
binds to the throwaway test database and to symmetric JWT verification. Import
order matters here; keep the os.environ block at the top.
"""

import os
from urllib.parse import urlsplit, urlunsplit

TEST_DB_NAME = "jobtracker_test"
TEST_JWT_SECRET = "test-only-secret-not-used-anywhere-real"


def _swap_database(url: str, name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{name}"))


# Bound to the local container, and deliberately NOT read from DATABASE_URL.
#
# This suite drops and recreates its database on every run and truncates tables
# between tests. Inheriting the application's connection string would aim that
# at whatever the app is pointed at — which, once deployment starts, is real
# data. The override is a separate TEST_* variable so pointing the app
# elsewhere can never redirect the tests.
#
# The two-role split is preserved: the app connects as the NOBYPASSRLS runtime
# role, only the owner creates/migrates/truncates. Running the suite as the
# owner would make every isolation test pass vacuously.
_OWNER_URL = os.environ.get(
    "TEST_MIGRATION_DATABASE_URL", "postgresql+asyncpg://jobtracker:jobtracker@db:5432/jobtracker"
)
_APP_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://app_user:app_password@db:5432/jobtracker"
)

if "supabase" in _OWNER_URL or "supabase" in _APP_URL:
    raise RuntimeError(
        "The test database points at Supabase. This suite drops databases and "
        "truncates tables; it must only ever run against a disposable local "
        "Postgres. Unset TEST_DATABASE_URL / TEST_MIGRATION_DATABASE_URL."
    )
OWNER_TEST_URL = _swap_database(_OWNER_URL, TEST_DB_NAME)

os.environ["DATABASE_URL"] = _swap_database(_APP_URL, TEST_DB_NAME)
os.environ["MIGRATION_DATABASE_URL"] = OWNER_TEST_URL
os.environ["SUPABASE_JWT_SECRET"] = TEST_JWT_SECRET
os.environ["SUPABASE_JWT_AUDIENCE"] = "authenticated"
os.environ["ENVIRONMENT"] = "ci"

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import uuid  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402

import asyncpg  # noqa: E402
import jwt  # noqa: E402
import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402


def _asyncpg_dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


async def _recreate_database() -> None:
    admin = await asyncpg.connect(_asyncpg_dsn(_swap_database(_OWNER_URL, "postgres")))
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    finally:
        await admin.close()

    # Extensions and per-database grants are applied by the migration; only the
    # CONNECT privilege has to exist before it runs.
    owner = await asyncpg.connect(_asyncpg_dsn(OWNER_TEST_URL))
    try:
        await owner.execute(f'GRANT CONNECT ON DATABASE "{TEST_DB_NAME}" TO app_user')
    finally:
        await owner.close()


@pytest.fixture(scope="session", autouse=True)
def migrated_database() -> None:
    """Drop, recreate, and migrate the test database once per session.

    Migrations are run rather than ``create_all`` on purpose: the RLS policies
    live in the migration, and a schema built without them would let the
    isolation tests below pass while production stayed wide open.
    """
    asyncio.run(_recreate_database())

    from alembic.config import Config

    from alembic import command

    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Unscoped session — no ``app.user_id`` set. Used to seed fixtures and to
    prove that unscoped access sees nothing."""
    from app.db.session import SessionFactory

    async with SessionFactory() as s:
        yield s
        await s.rollback()


@pytest.fixture(autouse=True)
async def clean_tables() -> AsyncIterator[None]:
    """Truncate between tests, then drop every pooled connection.

    The engine is a module-level singleton, but pytest-asyncio gives each test a
    fresh event loop. A pooled asyncpg connection is bound to the loop that
    opened it, so reusing one in the next test raises "attached to a different
    loop". Disposing here forces each test to open its own.
    """
    yield
    from app.db.session import engine

    # Truncate as the owner: the runtime role holds no TRUNCATE privilege, and a
    # DELETE would be filtered by the very policies under test.
    #
    # `skills` is deliberately excluded — it is seeded by migration 0002 and is
    # reference data, not test fixture data. Truncating it would leave every
    # subsequent test unable to attach skills to a job.
    owner = await asyncpg.connect(_asyncpg_dsn(OWNER_TEST_URL))
    try:
        await owner.execute("TRUNCATE TABLE users, companies, jobs CASCADE")
    finally:
        await owner.close()
    await engine.dispose()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def make_token(
    user_id: uuid.UUID | None = None,
    email: str = "candidate@example.com",
    *,
    expires_in: timedelta = timedelta(hours=1),
    audience: str = "authenticated",
    secret: str = TEST_JWT_SECRET,
) -> tuple[uuid.UUID, str]:
    """Mint a Supabase-shaped HS256 access token."""
    uid = user_id or uuid.uuid4()
    now = datetime.now(UTC)
    payload = {
        "sub": str(uid),
        "email": email,
        "aud": audience,
        "role": "authenticated",
        "iat": now,
        "exp": now + expires_in,
    }
    return uid, jwt.encode(payload, secret, algorithm="HS256")


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class StubEmbeddings:
    """Deterministic pseudo-embeddings, used by every test.

    Autouse below, because job creation now embeds too — so without a global
    stub any test that creates an application would download a ~130 MB model
    and run CPU inference. Vectors are derived from a hash of the text, so
    identical input yields an identical vector, which is the only property the
    code under test relies on. They carry no semantic meaning, which is fine:
    nothing here asserts on retrieval quality.

    Tests that genuinely need real embeddings can override this fixture.
    """

    def __init__(self) -> None:
        from app.core.config import settings as _settings

        self.dimension = _settings.embedding_dim
        self.documents_embedded = 0
        self.queries_embedded = 0

    def _vector(self, value: str) -> list[float]:
        digest = hashlib.sha256(value.encode()).digest()
        raw = [(digest[i % len(digest)] / 255.0) - 0.5 for i in range(self.dimension)]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents_embedded += len(texts)
        return [self._vector(t) for t in texts]

    async def embed_query(self, text_: str) -> list[float]:
        self.queries_embedded += 1
        return self._vector(text_)


@pytest.fixture(autouse=True)
def embeddings(monkeypatch: pytest.MonkeyPatch) -> StubEmbeddings:
    """Patched on the module, so every consumer resolving
    ``embeddings.embedding_provider`` at call time picks up the stub."""
    stub = StubEmbeddings()
    monkeypatch.setattr("app.services.embeddings.embedding_provider", stub)
    return stub
