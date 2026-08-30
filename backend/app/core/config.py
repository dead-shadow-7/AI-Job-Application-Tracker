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
    # Provider-neutral because the free-tier budgets differ by two orders of
    # magnitude: Groq allows 8000 tokens per minute, which a bulk import of a
    # spreadsheet will exhaust, while Gemini's free tier is capped on requests
    # rather than tokens. Both speak the OpenAI dialect, so switching is a base
    # URL, a key, and a model.
    #
    # The constraint that rules out most alternatives is strict `json_schema`
    # response_format — schema adherence by construction rather than by parsing
    # and hoping. Extraction is worthless without it.
    llm_provider: Literal["groq", "gemini"] = "groq"

    # Verified against Groq's live model list, not the docs: the Llama chat
    # models are gone (only prompt-guard moderation variants remain), and
    # strict json_schema is supported only by the gpt-oss and qwen families.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_extraction_model: str = "openai/gpt-oss-120b"
    groq_fast_model: str = "openai/gpt-oss-20b"

    # Gemini's OpenAI-compatible endpoint. Slower than Groq (~4-8s vs ~2-4s)
    # but the free tier is token-generous enough that the practical limit
    # becomes requests per minute, which a human pasting jobs will never reach.
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_extraction_model: str = "gemini-2.5-flash"
    gemini_fast_model: str = "gemini-2.5-flash-lite"

    @property
    def llm_api_key(self) -> str:
        return self.gemini_api_key if self.llm_provider == "gemini" else self.groq_api_key

    @property
    def llm_base_url(self) -> str:
        return self.gemini_base_url if self.llm_provider == "gemini" else self.groq_base_url

    @property
    def extraction_model(self) -> str:
        if self.llm_provider == "gemini":
            return self.gemini_extraction_model
        return self.groq_extraction_model

    @property
    def fast_model(self) -> str:
        return self.gemini_fast_model if self.llm_provider == "gemini" else self.groq_fast_model

    # Provider-independent request settings.
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 3
    # Counted against a tokens-per-minute budget *before* generation starts, so
    # this is a real cost knob rather than just a safety ceiling. Groq's free
    # tier allows 8000 TPM; a full extraction fits comfortably in 3000.
    llm_max_output_tokens: int = 3000

    # --- Observability ----------------------------------------------------
    # The endpoint is regional and is not optional. A key issued in one region
    # is refused by another with a bare 403 that mentions nothing about
    # geography, which reads exactly like an invalid key. Copy the endpoint from
    # the same LangSmith setup screen as the key.
    #   US (default) https://api.smith.langchain.com
    #   EU           https://eu.api.smith.langchain.com
    #   APAC         https://apac.api.smith.langchain.com
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "ai-job-tracker"

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
