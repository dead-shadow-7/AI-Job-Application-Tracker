from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["local", "ci", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    # --- Database ---------------------------------------------------------
    # Two roles on purpose. The runtime role must be NOSUPERUSER/NOBYPASSRLS or
    # row level security is silently skipped; the migration role owns the schema
    # and needs DDL rights the runtime role must never have.
    database_url: str = "postgresql+asyncpg://app_user:app_password@localhost:5432/jobtracker"
    migration_database_url: str = (
        "postgresql+asyncpg://jobtracker:jobtracker@localhost:5432/jobtracker"
    )
    db_echo: bool = False

    # --- Supabase Auth ----------------------------------------------------
    supabase_url: str = ""
    supabase_anon_key: str = ""
    # Blank => verify via JWKS (asymmetric). Set => verify via HS256.
    supabase_jwt_secret: str = ""
    supabase_jwt_audience: str = "authenticated"

    # --- CORS -------------------------------------------------------------
    # NoDecode: without it pydantic-settings JSON-decodes env values for complex
    # types *before* validators run, so a plain comma-separated string raises
    # rather than reaching the validator below.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    # --- LLM (Phase 2) ----------------------------------------------------
    groq_api_key: str = ""
    groq_extraction_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"

    # --- Embeddings (Phase 3) --------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        """Accept a comma-separated string as well as a real list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def jwks_url(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    @property
    def use_symmetric_jwt(self) -> bool:
        """Legacy Supabase projects sign with a shared HS256 secret."""
        return bool(self.supabase_jwt_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
