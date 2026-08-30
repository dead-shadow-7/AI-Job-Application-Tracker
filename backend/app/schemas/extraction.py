"""The shape the LLM must return, and the strict-JSON-Schema transform.

Every field is declared required with an explicit ``| None``. That is not
stylistic: Groq's strict mode demands `required` list every property and
`additionalProperties: false` on every object, with optionality expressed as a
union with null. A field carrying a Python default would be omitted from
`required` and the request would be rejected.

Requiring the model to emit `null` explicitly is also better extraction
behaviour — an absent key is ambiguous between "not present in the posting" and
"I forgot", whereas an explicit null is a decision.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Bump when the prompt or this schema changes in a way that alters output.
# Stored on every job so historical extractions stay interpretable and the eval
# suite can attribute a regression to a specific version.
EXTRACTION_PROMPT_VERSION = "2026-08-30.2"


class ExtractedSalary(BaseModel):
    """Salary, with the source text kept alongside the parsed numbers.

    ``raw_text`` exists because of a failure this design was built around: given
    "45-60 LPA", one model returned 45/60 and another 4500000/6000000. Both are
    defensible readings; storing either without provenance silently corrupts
    every salary comparison. The validator checks ``raw_text`` appears verbatim
    in the posting and discards the whole block if it does not, so a fabricated
    band cannot survive.
    """

    raw_text: str | None = Field(
        description=(
            "The salary text copied VERBATIM from the posting, e.g. '45-60 LPA' "
            "or '$120,000 - $150,000 per year'. Null if the posting states no "
            "salary. Never paraphrase or reconstruct this."
        )
    )
    min_amount: float | None = Field(
        description=(
            "Lower bound as an absolute number in the currency's major unit. "
            "Indian postings quote lakhs: '45 LPA' is 4500000, not 45. "
            "'12 LPA' is 1200000. A crore is 10000000."
        )
    )
    max_amount: float | None = Field(description="Upper bound, same units as min_amount.")
    currency: str | None = Field(description="ISO 4217 code: INR, USD, EUR, GBP. Null if unclear.")
    period: Literal["year", "month", "hour"] | None = Field(
        description="Pay period. Indian LPA and US annual figures are 'year'."
    )


class ExtractedRequirement(BaseModel):
    text: str = Field(description="One requirement, as a single clear sentence.")
    kind: Literal["must", "nice"] = Field(
        description=(
            "'must' for stated hard requirements, 'nice' for preferred, bonus, "
            "or 'plus' items. When the posting does not distinguish, use 'must'."
        )
    )


class ExtractedJob(BaseModel):
    """One job posting, fully structured."""

    # Nullable, despite both being required to save a job.
    #
    # A LinkedIn "About the job" body frequently never names the employer — it
    # lives in the page chrome, not the description — and plenty of postings
    # open with "join our team". The prompt forbids inventing, so the model
    # correctly returns null; declaring these non-nullable made strict mode
    # reject that correct answer with a 400 and retry into the rate limit.
    #
    # An extraction that admits it could not find the company is useful. The
    # review screen asks for it, which is a far better outcome than a
    # confidently wrong employer or a failed ingestion.
    company_name: str | None = Field(
        description=(
            "The hiring company, not the recruiting agency. Null if the posting "
            "text never names it — do not guess from the role or the writing style."
        )
    )
    title: str | None = Field(
        description=(
            "The role title, without seniority padding or location. Null if the "
            "posting text contains no title."
        )
    )

    seniority: Literal["intern", "junior", "mid", "senior", "staff", "lead", "principal"] | None = (
        Field(description="Inferred from title and required years. Null if genuinely unclear.")
    )
    employment_type: Literal["full_time", "part_time", "contract", "internship"] | None = Field(
        description="Null unless the posting says."
    )
    work_mode: Literal["onsite", "hybrid", "remote"] | None = Field(
        description="Null unless the posting says. Do not infer remote from a city being absent."
    )
    location: str | None = Field(description="City and country as written, e.g. 'Pune, India'.")

    salary: ExtractedSalary
    years_experience_min: int | None = Field(
        description="Minimum years required. Null if unstated."
    )
    years_experience_max: int | None = Field(description="Maximum years, if a range is given.")

    responsibilities: str | None = Field(
        description="What the person will do, as prose. Null if the posting omits it."
    )
    requirements: list[ExtractedRequirement] = Field(
        description="Each distinct requirement as its own entry. Empty list if none stated."
    )
    skills: list[str] = Field(
        description=(
            "Concrete technologies, languages, and tools named in the posting — "
            "'Python', 'PostgreSQL', 'Kubernetes'. Not soft skills, not "
            "responsibilities. Empty list if none named."
        )
    )

    confidence: float = Field(
        description=(
            "0.0-1.0. How complete and unambiguous the posting was. Below 0.5 "
            "when the text is truncated, is not a job posting, or forced heavy "
            "guessing."
        )
    )


def to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Render a Pydantic model as a Groq/OpenAI strict-mode JSON Schema.

    Pydantic emits a schema that is nearly right; strict mode additionally
    demands that every object be closed (``additionalProperties: false``) and
    list every property in ``required``. Both are applied recursively, including
    into ``$defs``, so nested models are covered.
    """
    schema = model.model_json_schema()
    _tighten(schema)
    for definition in schema.get("$defs", {}).values():
        _tighten(definition)
    return schema


def _tighten(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _tighten(value)
    elif isinstance(node, list):
        for item in node:
            _tighten(item)
