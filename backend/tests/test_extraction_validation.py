"""The anti-hallucination layer.

Strict mode guarantees the response is schema-shaped. It guarantees nothing
about whether the contents are true, and these are the checks that decide what
survives. They run without touching the network.
"""

import pytest

from app.agent.validation import validate_extraction
from app.schemas.extraction import ExtractedJob, ExtractedRequirement, ExtractedSalary

POSTING = """\
Senior Backend Engineer at Razorpay, Bangalore (Hybrid).
We need 5+ years of Python, strong PostgreSQL, and Kafka experience.
Compensation: 45-60 LPA depending on experience.
Kubernetes is a plus.
"""


def build(**overrides) -> ExtractedJob:
    defaults = dict(
        company_name="Razorpay",
        title="Backend Engineer",
        seniority="senior",
        employment_type=None,
        work_mode="hybrid",
        location="Bangalore",
        salary=ExtractedSalary(
            raw_text="45-60 LPA",
            min_amount=4_500_000,
            max_amount=6_000_000,
            currency="INR",
            period="year",
        ),
        years_experience_min=5,
        years_experience_max=None,
        responsibilities=None,
        requirements=[ExtractedRequirement(text="5+ years Python", kind="must")],
        skills=["Python", "PostgreSQL", "Kafka", "Kubernetes"],
        confidence=0.9,
    )
    defaults.update(overrides)
    return ExtractedJob(**defaults)


def test_a_faithful_extraction_passes_untouched() -> None:
    extracted = build()

    report = validate_extraction(extracted, POSTING)

    assert report.is_clean
    assert extracted.salary.min_amount == 4_500_000
    assert extracted.skills == ["Python", "PostgreSQL", "Kafka", "Kubernetes"]


def test_invented_salary_is_discarded() -> None:
    """The single most valuable check here.

    Models produce plausible salary bands readily, and a fabricated figure in a
    tracker is worse than a blank one — it gets compared against real offers and
    used to decide where to spend effort.
    """
    extracted = build(
        salary=ExtractedSalary(
            raw_text="₹32,00,000 - ₹40,00,000 per annum",
            min_amount=3_200_000,
            max_amount=4_000_000,
            currency="INR",
            period="year",
        )
    )

    report = validate_extraction(extracted, POSTING)

    assert extracted.salary.raw_text is None
    assert extracted.salary.min_amount is None
    assert extracted.salary.max_amount is None
    assert "salary" in report.dropped_fields


def test_salary_numbers_without_source_text_are_discarded() -> None:
    """Numbers with no quote to support them cannot be checked, so they go."""
    extracted = build(
        salary=ExtractedSalary(
            raw_text=None,
            min_amount=5_000_000,
            max_amount=7_000_000,
            currency="INR",
            period="year",
        )
    )

    report = validate_extraction(extracted, POSTING)

    assert extracted.salary.min_amount is None
    assert "salary" in report.dropped_fields


def test_verbatim_check_tolerates_typography() -> None:
    """Postings wrap lines and use en-dashes; rejecting a correct quote for
    typographic reasons would make the check useless in practice."""
    posting = "Compensation:\n  45 – 60   LPA depending on experience."
    extracted = build(
        salary=ExtractedSalary(
            raw_text="45 - 60 LPA",
            min_amount=4_500_000,
            max_amount=6_000_000,
            currency="INR",
            period="year",
        )
    )

    report = validate_extraction(extracted, posting)

    assert extracted.salary.raw_text == "45 - 60 LPA"
    assert "salary" not in report.dropped_fields


def test_unconverted_lakh_figures_are_flagged() -> None:
    """The exact disagreement observed between two models on one posting: one
    returned 45/60, the other 4500000/6000000."""
    extracted = build(
        salary=ExtractedSalary(
            raw_text="45-60 LPA",
            min_amount=45,
            max_amount=60,
            currency="INR",
            period="year",
        )
    )

    report = validate_extraction(extracted, POSTING)

    assert any("unconverted" in w for w in report.warnings)


def test_inverted_salary_range_is_swapped() -> None:
    extracted = build(
        salary=ExtractedSalary(
            raw_text="45-60 LPA",
            min_amount=6_000_000,
            max_amount=4_500_000,
            currency="INR",
            period="year",
        )
    )

    report = validate_extraction(extracted, POSTING)

    assert extracted.salary.min_amount == 4_500_000
    assert extracted.salary.max_amount == 6_000_000
    assert any("swapped" in w for w in report.warnings)


def test_skills_the_posting_never_names_are_dropped() -> None:
    """Models pattern-complete a stack. A posting naming Django invites
    'Celery' and 'Redis' whether or not they appear, and those phantom skills
    would produce a fictitious gap when scored against a resume."""
    extracted = build(skills=["Python", "PostgreSQL", "Redis", "Celery", "GraphQL"])

    report = validate_extraction(extracted, POSTING)

    assert extracted.skills == ["Python", "PostgreSQL"]
    assert "skills" in report.dropped_fields
    assert "Redis" in report.warnings[0]


def test_company_absent_from_posting_is_warned_not_cleared() -> None:
    """Company is required, so a wrong name the user can correct beats an empty
    record they cannot identify."""
    extracted = build(company_name="Stripe")

    report = validate_extraction(extracted, POSTING)

    assert extracted.company_name == "Stripe"
    assert any("Stripe" in w for w in report.warnings)


@pytest.mark.parametrize("years", [-3, 99])
def test_implausible_experience_is_cleared(years: int) -> None:
    extracted = build(years_experience_min=years)

    report = validate_extraction(extracted, POSTING)

    assert extracted.years_experience_min is None
    assert any("implausible" in w for w in report.warnings)


def test_inverted_experience_range_is_swapped() -> None:
    extracted = build(years_experience_min=8, years_experience_max=5)

    validate_extraction(extracted, POSTING)

    assert (extracted.years_experience_min, extracted.years_experience_max) == (5, 8)


def test_non_iso_currency_is_cleared() -> None:
    extracted = build(
        salary=ExtractedSalary(
            raw_text="45-60 LPA",
            min_amount=4_500_000,
            max_amount=6_000_000,
            currency="Rupees",
            period="year",
        )
    )

    report = validate_extraction(extracted, POSTING)

    assert extracted.salary.currency is None
    assert any("ISO" in w for w in report.warnings)


# --- Strict-schema generation ----------------------------------------------


def test_generated_schemas_contain_no_refs() -> None:
    """Groq's strict validator does not resolve $ref.

    An optional enum field becomes anyOf: [{$ref: ...}, {type: null}], and
    because the validator cannot see inside the ref it cannot tell the branches
    apart — it rejects the whole request with "anyOf branches must be
    disambiguated". That failure only appears against the live API, so this
    asserts the property directly rather than waiting to be surprised by it.
    """
    import json

    from app.schemas.agent import AgentReply
    from app.schemas.extraction import ExtractedJob, to_strict_json_schema
    from app.schemas.matching import RubricJudgment

    for model in (ExtractedJob, AgentReply, RubricJudgment):
        rendered = json.dumps(to_strict_json_schema(model))
        assert "$ref" not in rendered, f"{model.__name__} still contains a $ref"
        assert "$defs" not in rendered, f"{model.__name__} still contains $defs"


def test_every_object_in_a_schema_is_closed_and_fully_required() -> None:
    """Strict mode demands both, on nested objects as well as the root."""
    from app.schemas.agent import AgentReply
    from app.schemas.extraction import ExtractedJob, to_strict_json_schema

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(to_strict_json_schema(ExtractedJob))
    walk(to_strict_json_schema(AgentReply))
