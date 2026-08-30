"""Mapping free-text skills onto the canonical taxonomy.

Deterministic, and deliberately so. An LLM asked to normalise "ReactJS" will
usually say "React", but "usually" compounds badly: every variant that slips
through becomes a separate row, and the Phase 3 match score then counts one
skill as two — inflating the gap against a resume that actually has it.

Unrecognised terms are reported, not invented. The plan is explicit that new
canonical skills are flagged for review rather than auto-created, because a
taxonomy that grows on every typo stops being a taxonomy.
"""

import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.skill import Skill

_PUNCT = re.compile(r"[^\w+#.\s]")
_SPACE = re.compile(r"\s+")


def normalize_token(text: str) -> str:
    """Fold a skill mention to a comparison key.

    ``+``, ``#`` and ``.`` survive because they are load-bearing in this
    domain — dropping them would collide C++ with C, and C# with C.
    """
    folded = _PUNCT.sub(" ", text.lower())
    return _SPACE.sub(" ", folded).strip()


@dataclass
class SkillResolution:
    matched: dict[str, Skill] = field(default_factory=dict)
    """Original mention -> canonical skill."""

    unmatched: list[str] = field(default_factory=list)
    """Mentions with no taxonomy entry. Surfaced for review, never auto-created."""

    @property
    def slugs(self) -> list[str]:
        seen: dict[str, None] = {}
        for skill in self.matched.values():
            seen.setdefault(skill.slug, None)
        return list(seen)


async def resolve_skills(session: AsyncSession, mentions: list[str]) -> SkillResolution:
    """Resolve mentions against canonical names and aliases in one pass.

    The whole taxonomy is loaded rather than queried per mention: it is ~100
    rows, so one round trip beats N, and alias matching needs the full set in
    memory anyway.
    """
    resolution = SkillResolution()
    if not mentions:
        return resolution

    skills = (await session.execute(select(Skill))).scalars().all()

    lookup: dict[str, Skill] = {}
    for entry in skills:
        lookup[normalize_token(entry.name)] = entry
        lookup[normalize_token(entry.slug)] = entry
        # setdefault, not assignment: a canonical name always wins over another
        # skill's alias claiming the same token.
        for alias in entry.aliases:
            lookup.setdefault(normalize_token(alias), entry)

    for mention in mentions:
        key = normalize_token(mention)
        if not key:
            continue

        skill: Skill | None = lookup.get(key)

        # "React.js developer" or "experience with Kafka" — the posting names a
        # real skill inside a phrase. Only tried after exact matching, and only
        # for multi-word mentions, so "Java" cannot swallow "JavaScript".
        if skill is None and " " in key:
            for candidate_key, candidate in lookup.items():
                if len(candidate_key) >= 3 and re.search(
                    rf"(?<!\w){re.escape(candidate_key)}(?!\w)", key
                ):
                    skill = candidate
                    break

        if skill is not None:
            resolution.matched[mention] = skill
        else:
            resolution.unmatched.append(mention)

    return resolution
