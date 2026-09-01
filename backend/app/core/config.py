from functools import lru_cache
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
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
    # Sized to the server, not to demand. Supabase's session pooler allows this
    # role 15 clients (verified: connection 16 fails with EMAXCONNSESSION, a
    # hard 500 rather than a wait — SQLAlchemy only queues when its *own* pool
    # is full, so an oversized pool never blocks, it just gets refused).
    # 14 leaves one spare for migrations. Raise the dashboard's Pool Size first
    # if these ever need to go up.
    db_pool_size: int = 10
    db_max_overflow: int = 4
    db_pool_timeout: float = 30.0

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
    llm_provider: Literal["groq", "gemini", "aicredits"] = "groq"

    # Verified against Groq's live model list, not the docs: the Llama chat
    # models are gone (only prompt-guard moderation variants remain), and
    # strict json_schema is supported only by the gpt-oss and qwen families.
    #
    # Free tier: 8,000 tokens/minute AND 200,000 tokens/day. The daily cap is
    # the one that bites — backing off does not help, you are simply done until
    # tomorrow. Measured against Groq's own billed prompt_tokens, the fixed
    # prefix (22 tool schemas + system prompt) is ~3,320 per round, so a
    # two-round turn runs ~7,200-8,500 and roughly 20-25 agent messages exhaust
    # a day. Note the per-minute limiter debits prompt + max_completion_tokens
    # together, which is why llm_chat_output_tokens is a real cost knob.
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

    # AI Credits — a paid INR-billed gateway, chosen to escape the free-tier
    # daily cap. There is no tokens-per-minute or per-day ceiling; the limits
    # are requests per minute and the wallet balance, so a long session costs
    # money instead of stopping dead until tomorrow.
    #
    # Model ids are provider-prefixed and must come from GET /v1/models — the
    # id in their own documentation example does not exist in the catalogue,
    # and the catalogue's capability metadata is wrong in places (tts models
    # listed as chat, o1-mini's context length off by 16x). Treat it as "what
    # the API accepts as a model string", not as a capability sheet.
    #
    # gpt-4o-mini from measurement, not taste. Four candidates, scored on a
    # 10-case tool-selection eval against the real 20-tool schema and on the
    # salary field twice — salary being the one most likely to be wrong and
    # least likely to be re-read once it is in the table. Monthly cost assumes
    # 300 assistant messages and 40 pasted postings, priced from the platform's
    # own catalogue:
    #
    #   gpt-oss-120b   9/10   salary WRONG    Rs 3.62
    #   gpt-4.1-nano   6/10   salary ok       Rs 8.08
    #   gpt-4o-mini   10/10   salary ok       Rs 12.34
    #   gpt-5-nano     9/10   salary ok       Rs 16.77
    #
    # Two results worth keeping. gpt-oss-120b read "45-60 LPA" as 45 to 60
    # rupees, twice — a hundred-thousand-fold error, from the same model id that
    # reads it correctly on Groq. Same name, different serving stack; this
    # gateway documents a fallback to an "aggregated provider pool" and does not
    # say when it fires. And gpt-5-nano costs *more* than gpt-4o-mini despite a
    # lower headline price, because it spends 735 completion tokens per
    # assistant turn against gpt-4o-mini's 23 — reasoning tokens are billed as
    # output. Those tokens also count against llm_max_output_tokens, so a long
    # posting can hit the ceiling and fail outright.
    #
    # The whole spread is about Rs 13 a month. Nothing here is worth trading a
    # wrong salary for.
    #
    # gpt-4o-mini also echoes back the exact id requested, where gpt-5-mini is
    # served as a dated snapshot — on a gateway that reserves the right to
    # reroute, an id that comes back verbatim is the cheapest evidence you have
    # of what actually answered.
    # Both spellings accepted. `extra="ignore"` means a near-miss env var name
    # is not an error — it is silently dropped, the key reads as empty, and the
    # app reports "no LLM configured" while the key sits in .env looking correct.
    aicredits_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("AI_CREDITS_API_KEY", "AICREDITS_API_KEY"),
    )
    aicredits_base_url: str = "https://api.aicredits.in/v1"
    aicredits_extraction_model: str = "openai/gpt-4o-mini"
    aicredits_fast_model: str = "openai/gpt-4o-mini"

    @property
    def _llm(self) -> tuple[str, str, str, str]:
        """(key, base URL, extraction model, fast model) for the active provider.

        A table rather than three parallel if/else chains. Adding a provider
        used to mean editing every accessor and there was nothing to catch a
        missed one — the result would be the right key sent to the wrong host.
        """
        table = {
            "groq": (
                self.groq_api_key,
                self.groq_base_url,
                self.groq_extraction_model,
                self.groq_fast_model,
            ),
            "gemini": (
                self.gemini_api_key,
                self.gemini_base_url,
                self.gemini_extraction_model,
                self.gemini_fast_model,
            ),
            "aicredits": (
                self.aicredits_api_key,
                self.aicredits_base_url,
                self.aicredits_extraction_model,
                self.aicredits_fast_model,
            ),
        }
        return table[self.llm_provider]

    @property
    def llm_api_key(self) -> str:
        return self._llm[0]

    @property
    def llm_base_url(self) -> str:
        return self._llm[1]

    @property
    def extraction_model(self) -> str:
        return self._llm[2]

    @property
    def fast_model(self) -> str:
        return self._llm[3]

    # Provider-independent request settings.
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 3
    # Counted against a tokens-per-minute budget *before* generation starts, so
    # this is a real cost knob rather than just a safety ceiling. Groq's free
    # tier allows 8000 TPM; a full extraction fits comfortably in 3000.
    llm_max_output_tokens: int = 3000
    # Separate ceiling for the assistant loop. The reply is one or two sentences
    # or a tool call — measured at 23 completion tokens per turn — so reserving
    # the extraction ceiling there spent 37% of Groq's 8000 TPM budget on output
    # that never arrived, on every one of up to six rounds.
    llm_chat_output_tokens: int = 1024

    # Whole-request deadlines. Without one the nesting multiplies: the ingestion
    # graph retries extraction twice, each _post retries three times at a 90s
    # timeout, so a single request could hold its RLS transaction — and one of
    # db_pool_size connections — for roughly nine minutes. The user has long
    # since given up by then; the connection had not.
    assistant_deadline_seconds: float = 120.0
    ingest_deadline_seconds: float = 100.0

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
