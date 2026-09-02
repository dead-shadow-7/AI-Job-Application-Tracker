"""The provider switch.

Three providers now speak to the same OpenAI-dialect client, selected by one
environment variable. The failure this guards against is quiet: a provider added
to the enum but missed in one of the accessors sends a real key to the wrong
host, and the only symptom is an authentication error that looks like a bad key.
"""

import pytest

from app.core.config import Settings

PROVIDERS = ["groq", "gemini", "aicredits"]


def settings_for(provider: str, **overrides: str) -> Settings:
    """A Settings instance that ignores the developer's own .env.

    Without `_env_file=None` this reads the real file, and the assertions below
    would pass or fail depending on whose machine ran them.
    """
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider=provider,
        groq_api_key="groq-key",
        gemini_api_key="gemini-key",
        aicredits_api_key="aicredits-key",
        **overrides,
    )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_every_provider_resolves_a_complete_profile(provider: str) -> None:
    """Key, host and both models. A provider missing from one accessor would
    otherwise surface as someone else's default."""
    settings = settings_for(provider)

    assert settings.llm_api_key == f"{provider}-key"
    assert settings.llm_base_url.startswith("https://")
    assert settings.extraction_model
    assert settings.fast_model


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_key_and_the_host_belong_to_the_same_provider(provider: str) -> None:
    """The specific mix-up worth preventing: one provider's credential sent to
    another's endpoint, which reads as an invalid key rather than as a config
    error and sends you to the wrong dashboard."""
    settings = settings_for(provider)
    host = settings.llm_base_url

    expected = {"groq": "groq.com", "gemini": "googleapis.com", "aicredits": "aicredits.in"}
    assert expected[provider] in host
    for other, marker in expected.items():
        if other != provider:
            assert marker not in host


def test_the_provider_table_covers_the_whole_enum() -> None:
    """The enum and the lookup table are declared separately, so they can drift.
    A provider accepted by validation but absent from the table raises KeyError
    at request time rather than at startup."""
    from typing import get_args

    declared = get_args(Settings.model_fields["llm_provider"].annotation)
    assert set(declared) == set(PROVIDERS), "update this test when adding a provider"

    for provider in declared:
        assert settings_for(provider).llm_api_key, "unmapped provider raises KeyError here"


def test_the_ai_credits_key_is_read_under_either_spelling() -> None:
    """The env var is AI_CREDITS_API_KEY in the shipped example, while the field
    name implies AICREDITS_API_KEY. `extra="ignore"` makes the near-miss silent:
    the key reads as empty and the app reports no LLM configured while the value
    sits in .env looking perfectly correct."""
    underscored = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="aicredits",
        AI_CREDITS_API_KEY="from-underscored",  # type: ignore[call-arg]
    )
    joined = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="aicredits",
        AICREDITS_API_KEY="from-joined",  # type: ignore[call-arg]
    )

    assert underscored.llm_api_key == "from-underscored"
    assert joined.llm_api_key == "from-joined"


def test_the_suite_does_not_write_traces() -> None:
    """Tests used to emit into the real LangSmith project — the run list filled
    with rows whose input was `<test_ingest.StubLLM object at 0x...>`, spending
    retention on runs nobody would read and burying real traffic among them.

    Guarding the environment variable rather than the behaviour because that is
    where the mistake would be: the SDK reads it directly, at import.

    Both spellings of each name, because they are aliases and the SDK honours
    whichever it sees. Now that langchain-core is in the tree it starts a tracer
    of its own on import, so a single unguarded alias is enough to leak.
    """
    import os

    for flag in (
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
    ):
        assert os.environ.get(flag) == "false", flag
    for key in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"):
        assert not os.environ.get(key), key


def test_models_are_overridable_per_provider() -> None:
    """Pinning an exact model id matters more on a gateway than on a first-party
    API: the catalogue changes without notice and the ids are not stable."""
    settings = settings_for("aicredits", aicredits_extraction_model="openai/gpt-4.1-mini")

    assert settings.extraction_model == "openai/gpt-4.1-mini"
    assert settings.fast_model == "openai/gpt-4o-mini", "unset models keep their default"
