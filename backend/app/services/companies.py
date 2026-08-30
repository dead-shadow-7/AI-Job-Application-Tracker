"""Company resolution and deduplication.

"Google", "Google LLC", "google india pvt ltd" and "Google  Inc." are one
employer. Left unnormalised they become four rows, and the Phase 5 question
"which companies reply fastest?" quietly returns nonsense.

Phase 2 layers embedding similarity on top of this for the cases string
normalisation cannot reach ("Meta" vs "Facebook"). This deterministic pass runs
first because it is free and cannot hallucinate.
"""

import re
import unicodedata

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import Company

# Stripped only when trailing — "Limited Brands" must not become "Brands".
LEGAL_SUFFIXES = (
    "private limited",
    "pvt ltd",
    "pvt. ltd.",
    "pvt",
    "limited",
    "ltd",
    "llc",
    "l.l.c",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "gmbh",
    "s.a",
    "b.v",
    "plc",
    "llp",
    "technologies",
    "technology",
    "labs",
    "software",
    "solutions",
    "systems",
    "services",
    "group",
    "holdings",
    "india",
    "usa",
    "global",
)

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize_company_name(name: str) -> str:
    """Reduce a company name to a stable deduplication key.

    Lowercase, strip accents, drop punctuation, then peel trailing legal and
    generic suffixes repeatedly ("Foo Technologies Pvt Ltd" needs three passes).

    Returns the punctuation-stripped lowercase form if peeling would leave
    nothing — "Ltd." as a company name is absurd, but returning "" would
    collapse every such row onto one key.
    """
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    cleaned = _PUNCTUATION.sub(" ", folded.lower())
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()

    if not cleaned:
        return name.lower().strip()

    changed = True
    while changed:
        changed = False
        for suffix in LEGAL_SUFFIXES:
            if cleaned.endswith(f" {suffix}"):
                candidate = cleaned[: -(len(suffix) + 1)].strip()
                if candidate:
                    cleaned = candidate
                    changed = True
                    break

    return cleaned or name.lower().strip()


async def resolve_company(
    session: AsyncSession,
    name: str,
    *,
    domain: str | None = None,
    location: str | None = None,
) -> Company:
    """Find the company by normalized name, or create it.

    Not an upsert-and-forget: an existing row keeps its original display name.
    The first spelling entered wins, so a later "GOOGLE LLC" does not rewrite a
    carefully typed "Google" everywhere it already appears.
    """
    normalized = normalize_company_name(name)

    existing = (
        await session.execute(select(Company).where(Company.normalized_name == normalized))
    ).scalar_one_or_none()

    if existing is not None:
        # Backfill only what was missing; never overwrite.
        if domain and not existing.domain:
            existing.domain = domain
        if location and not existing.location:
            existing.location = location
        return existing

    company = Company(
        name=name.strip(),
        normalized_name=normalized,
        domain=domain,
        location=location,
    )
    session.add(company)
    await session.flush()
    return company
