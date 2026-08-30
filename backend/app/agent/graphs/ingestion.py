"""The ingestion pipeline.

A LangGraph ``StateGraph`` with one conditional retry edge — not a ReAct agent.
There is no decision here for a model to make about what to do next: the steps
are fixed and ordered, and handing that control to an LLM would only add
latency, cost, and new ways to fail. LangGraph earns its place by making the
retry edge and the per-node state explicit, and by giving Phase 4's genuinely
agentic work the same runtime.

    normalize -> extract -> validate --(unusable, <2 tries)--> extract
                               |
                               v
                    resolve_company -> resolve_skills -> assemble
"""

import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.llm_client import LLMClient, LLMError, LLMUsage, llm_client
from app.agent.prompts.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_prompt,
)
from app.agent.tracing import run_metadata
from app.agent.validation import ValidationReport, validate_extraction
from app.core.config import settings
from app.schemas.extraction import EXTRACTION_PROMPT_VERSION, ExtractedJob
from app.services.companies import normalize_company_name
from app.services.skills import SkillResolution, resolve_skills

logger = logging.getLogger(__name__)

MAX_EXTRACTION_ATTEMPTS = 2
MIN_TEXT_LENGTH = 120
MAX_TEXT_LENGTH = 60_000


class IngestionState(TypedDict, total=False):
    # Inputs
    raw_text: str
    url: str | None
    source_platform: str | None
    session: Any
    client: Any

    # Working state
    cleaned_text: str
    extracted: ExtractedJob | None
    report: ValidationReport
    attempts: int
    skills: SkillResolution
    company_normalized: str

    # Outputs
    usage: Annotated[list[LLMUsage], lambda a, b: a + b]
    error: str | None
    prompt_version: str


def _clean(text: str) -> str:
    """Strip the noise that survives copy-paste from a job board.

    Zero-width characters and non-breaking spaces come through invisibly and
    would otherwise defeat the verbatim salary check, which compares the
    model's quote against this text.
    """
    for junk in ("​", "‌", "‍", "﻿"):
        text = text.replace(junk, "")
    text = text.replace("\xa0", " ").replace(" ", " ")
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]

    # Collapse runs of blank lines; job boards emit many.
    cleaned: list[str] = []
    blanks = 0
    for line in lines:
        if line.strip():
            blanks = 0
            cleaned.append(line)
        else:
            blanks += 1
            if blanks <= 1:
                cleaned.append("")
    return "\n".join(cleaned).strip()


async def normalize_node(state: IngestionState) -> dict[str, Any]:
    cleaned = _clean(state["raw_text"])

    if len(cleaned) < MIN_TEXT_LENGTH:
        return {
            "cleaned_text": cleaned,
            "error": (
                "That looks too short to be a job description. Paste the full posting — "
                "extraction needs the requirements and responsibilities, not just the title."
            ),
        }

    # Truncating rather than rejecting: a very long paste is usually a whole
    # careers page, and the posting itself is near the top. The window is 131k
    # tokens, so this bound is about cost and latency, not capacity.
    if len(cleaned) > MAX_TEXT_LENGTH:
        cleaned = cleaned[:MAX_TEXT_LENGTH]
        logger.info("Truncated pasted text to %d characters", MAX_TEXT_LENGTH)

    return {"cleaned_text": cleaned, "attempts": 0}


async def extract_node(state: IngestionState) -> dict[str, Any]:
    client: LLMClient = state["client"]
    attempts = state.get("attempts", 0)

    try:
        result = await client.extract(
            schema=ExtractedJob,
            system=EXTRACTION_SYSTEM_PROMPT,
            user=build_extraction_user_prompt(state["cleaned_text"], state.get("url")),
            model=settings.extraction_model,
        )
    except LLMError as exc:
        return {"attempts": attempts + 1, "extracted": None, "error": str(exc)}

    return {
        "attempts": attempts + 1,
        "extracted": result.data,
        "usage": [result.usage],
        "error": None,
    }


async def validate_node(state: IngestionState) -> dict[str, Any]:
    extracted = state.get("extracted")
    if extracted is None:
        return {"report": ValidationReport()}

    report = validate_extraction(extracted, state["cleaned_text"])
    return {"report": report}


def should_retry(state: IngestionState) -> str:
    """Retry only what a retry can fix.

    A transport failure or a genuinely unusable extraction is worth one more
    attempt at temperature 0. Dropped salary or skills are *not* — validation
    already removed them, and asking again would most likely reproduce the same
    invention while doubling the cost.
    """
    extracted = state.get("extracted")
    attempts = state.get("attempts", 0)

    # A missing company or title is *not* unusable. The posting genuinely may
    # not name them, the prompt forbids inventing, and the review screen asks
    # for what is missing. Retrying would spend a second call to be told the
    # same true thing — and on the free tier's 8000 TPM budget, that second call
    # is what turns a working ingestion into a rate-limit failure.
    unusable = extracted is None

    if unusable and attempts < MAX_EXTRACTION_ATTEMPTS:
        logger.info("Retrying extraction (attempt %d)", attempts + 1)
        return "retry"
    if unusable:
        return "failed"
    return "continue"


async def resolve_company_node(state: IngestionState) -> dict[str, Any]:
    extracted = state["extracted"]
    assert extracted is not None
    # May be absent: plenty of postings never name the employer, and the model
    # is told to return null rather than guess.
    if not extracted.company_name:
        return {"company_normalized": ""}
    return {"company_normalized": normalize_company_name(extracted.company_name)}


async def resolve_skills_node(state: IngestionState) -> dict[str, Any]:
    extracted = state["extracted"]
    assert extracted is not None
    session: AsyncSession = state["session"]

    resolution = await resolve_skills(session, extracted.skills)

    if resolution.unmatched:
        # Reported, never auto-created. The review screen shows these so the
        # taxonomy grows by decision rather than by typo.
        logger.info("Unmatched skills: %s", ", ".join(resolution.unmatched))

    return {"skills": resolution}


async def fail_node(state: IngestionState) -> dict[str, Any]:
    return {
        "error": state.get("error")
        or "Could not read a job posting from that text. Check it is a full job description."
    }


def build_ingestion_graph() -> Any:
    graph = StateGraph(IngestionState)

    graph.add_node("normalize", normalize_node)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("resolve_company", resolve_company_node)
    graph.add_node("resolve_skills", resolve_skills_node)
    graph.add_node("failed", fail_node)

    graph.set_entry_point("normalize")
    graph.add_conditional_edges(
        "normalize",
        lambda s: "failed" if s.get("error") else "extract",
        {"failed": "failed", "extract": "extract"},
    )
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {"retry": "extract", "continue": "resolve_company", "failed": "failed"},
    )
    graph.add_edge("resolve_company", "resolve_skills")
    graph.add_edge("resolve_skills", END)
    graph.add_edge("failed", END)

    return graph.compile()


_graph = None


def get_ingestion_graph() -> Any:
    """Compiled once; compilation is not free and the graph is stateless."""
    global _graph
    if _graph is None:
        _graph = build_ingestion_graph()
    return _graph


async def run_ingestion(
    *,
    session: AsyncSession,
    raw_text: str,
    url: str | None = None,
    source_platform: str | None = None,
    client: LLMClient | None = None,
    user_id: str | None = None,
) -> IngestionState:
    result = await get_ingestion_graph().ainvoke(
        {
            "raw_text": raw_text,
            "url": url,
            "source_platform": source_platform,
            "session": session,
            "client": client or llm_client,
            "usage": [],
        },
        # Tagged so a run can be found in LangSmith by user or platform when
        # chasing a specific bad extraction, and so cost can be grouped.
        config={
            "run_name": "job_ingestion",
            "metadata": run_metadata(
                user_id=user_id,
                source_platform=source_platform,
                prompt_version=EXTRACTION_PROMPT_VERSION,
            ),
        },
    )
    result["prompt_version"] = EXTRACTION_PROMPT_VERSION
    return result
