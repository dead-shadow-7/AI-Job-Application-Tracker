"""Company deduplication.

Left unnormalised, one employer becomes four rows and the Phase 5 question
"which companies reply fastest?" quietly returns nonsense.
"""

import pytest
from httpx import AsyncClient

from app.services.companies import normalize_company_name
from tests.factories import Session


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Google", "google"),
        ("Google LLC", "google"),
        ("Google, Inc.", "google"),
        ("  GOOGLE   Inc  ", "google"),
        ("Infosys Limited", "infosys"),
        ("Tata Consultancy Services Pvt Ltd", "tata consultancy"),
        ("Zoho Corporation", "zoho"),
        ("Freshworks Technologies Pvt. Ltd.", "freshworks"),
        ("Café Coffee Day", "cafe coffee day"),
        ("Amazon", "amazon"),
    ],
)
def test_normalization_collapses_legal_suffixes(raw: str, expected: str) -> None:
    assert normalize_company_name(raw) == expected


def test_leading_suffix_words_are_preserved() -> None:
    """Suffixes are stripped only when trailing — 'Limited Brands' is a company,
    not 'Brands'."""
    assert normalize_company_name("Limited Brands") == "limited brands"


def test_a_name_that_is_only_a_suffix_survives() -> None:
    """Peeling must not reduce a name to nothing, which would collapse every
    such row onto one key."""
    assert normalize_company_name("Ltd") == "ltd"


async def test_spelling_variants_resolve_to_one_company(client: AsyncClient) -> None:
    user = await Session(client).start()

    first = await user.create_application(company_name="Razorpay", title="Backend Engineer")
    second = await user.create_application(
        company_name="Razorpay Software Pvt Ltd", title="ML Engineer"
    )

    assert first["job"]["company"]["id"] == second["job"]["company"]["id"]


async def test_the_first_spelling_entered_wins(client: AsyncClient) -> None:
    """A later shouty variant must not rewrite a carefully typed display name
    everywhere it already appears."""
    user = await Session(client).start()
    await user.create_application(company_name="Razorpay", title="Backend Engineer")

    second = await user.create_application(company_name="RAZORPAY LLC", title="ML Engineer")

    assert second["job"]["company"]["name"] == "Razorpay"


async def test_distinct_companies_stay_distinct(client: AsyncClient) -> None:
    user = await Session(client).start()

    a = await user.create_application(company_name="Swiggy", title="Backend Engineer")
    b = await user.create_application(company_name="Zomato", title="Backend Engineer")

    assert a["job"]["company"]["id"] != b["job"]["company"]["id"]
