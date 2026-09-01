"""Turning "the Amazon application" into one specific row — or refusing to.

This is the single most safety-critical piece of the agent. Every write the
agent makes is aimed by this function, and an agent that confidently picks the
wrong application corrupts a timeline in a way nobody notices until they are
reading history that is quietly false.

So it does not guess. It returns *ranked candidates* with a confidence, and the
caller decides. It never writes anything. The tool that resolves is deliberately
not the tool that acts.
"""

import re
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import TERMINAL_STATUSES
from app.models.application import Application
from app.models.company import Company
from app.models.job import Job

# Above this, one candidate is clear enough to act on after confirmation.
# Below it, the user is asked to choose. Set deliberately high: the cost of
# asking an unnecessary question is one click, the cost of a wrong write is a
# corrupted history you may never notice.
CONFIDENT = 0.75

# A second candidate this close to the first means the query genuinely did not
# distinguish them, whatever the leader's absolute score.
AMBIGUOUS_MARGIN = 0.15

MAX_CANDIDATES = 5


@dataclass(frozen=True, slots=True)
class Candidate:
    application: Application
    score: float
    matched_on: str

    @property
    def label(self) -> str:
        job = self.application.job
        return f"{job.title} at {job.company.name} ({self.application.current_status})"


@dataclass(frozen=True, slots=True)
class Resolution:
    candidates: list[Candidate]
    query: str

    @property
    def is_confident(self) -> bool:
        """One clear winner, and nothing close behind it."""
        if len(self.candidates) != 1:
            if not self.candidates or self.candidates[0].score < CONFIDENT:
                return False
            runner_up = self.candidates[1].score
            return self.candidates[0].score - runner_up >= AMBIGUOUS_MARGIN
        return self.candidates[0].score >= CONFIDENT

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.is_confident else None

    def describe(self) -> str:
        """What the agent says when it cannot proceed alone.

        A single weak candidate is not ambiguity, and saying "matches more than
        one application" above a list of exactly one reads as a bug. The two
        cases need different questions: one asks "did you mean this?", the other
        asks "which of these?".
        """
        if not self.candidates:
            return f"Nothing matches {self.query!r}."

        if len(self.candidates) == 1:
            only = self.candidates[0]
            return (
                f"{self.query!r} probably means {only.label}, but not closely enough "
                f"to act on — matched on {only.matched_on}. Ask them to confirm that "
                "is the one, and mention its status if it looks closed."
            )

        lines = [f"{self.query!r} matches more than one application:"]
        lines.extend(f"  {i}. {c.label}" for i, c in enumerate(self.candidates, start=1))
        lines.append("Say which one you mean.")
        return "\n".join(lines)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def score_candidate(query: str, company: str, title: str, is_active: bool) -> tuple[float, str]:
    """How well one application answers the query, and on what basis.

    Company name is weighted above job title because that is how people refer
    to applications out loud — "the Amazon one", not "the backend engineer one".
    Someone tracking two roles at one company will get an ambiguous result, and
    being asked is the correct outcome there.
    """
    q = _normalize(query)
    c = _normalize(company)
    t = _normalize(title)

    if not q:
        return 0.0, "empty query"

    if q == c:
        score, basis = 0.95, "company name"
    elif q == t:
        score, basis = 0.90, "role title"
    elif q == f"{t} at {c}" or q == f"{c} {t}" or q == f"{t} {c}":
        score, basis = 1.0, "company and role"
    elif c and (c in q or q in c):
        score, basis = 0.80, "partial company name"
    elif t and (t in q or q in t):
        score, basis = 0.70, "partial role title"
    else:
        # Word overlap, for "backend role at amazon" against "Backend Engineer"
        # + "Amazon". Weak by construction; it should rarely clear CONFIDENT on
        # its own.
        query_words = set(q.split())
        target_words = set(c.split()) | set(t.split())
        overlap = (
            len(query_words & target_words) / len(query_words)
            if query_words and target_words
            else 0.0
        )

        if overlap > 0:
            score, basis = 0.4 + 0.3 * overlap, "word overlap"
        else:
            # Character-level similarity, so a typo surfaces the row instead of
            # returning nothing — "no such application" sends the user hunting
            # for a bug when they made a spelling mistake.
            #
            # The SQL query already tolerates this via trigram similarity; if
            # the scorer did not, retrieval and ranking would disagree and the
            # row would be found and then silently discarded.
            #
            # Scored low on purpose. It stays below CONFIDENT, so a fuzzy match
            # asks "did you mean…?" rather than acting on a guess.
            closeness = max(
                SequenceMatcher(None, q, c).ratio() if c else 0.0,
                SequenceMatcher(None, q, t).ratio() if t else 0.0,
            )
            if closeness < 0.7:
                return 0.0, "no match"
            score, basis = round(0.35 + 0.3 * closeness, 3), "approximate spelling"

    # A closed application is rarely what someone means by "the Amazon one",
    # but it is not impossible — they may be correcting a mistaken rejection, or
    # asking why it ended. Demoted rather than excluded.
    #
    # The factor is tuned, not arbitrary. At 0.7 an exact company match on a
    # closed application scored 0.665, below CONFIDENT, so a user whose only
    # Amazon application was rejected got "which one do you mean?" followed by a
    # list of one — a dead end. At 0.8 it clears the threshold alone (0.76) while
    # an active application still beats it outright (0.95 - 0.76 = 0.19, wider
    # than AMBIGUOUS_MARGIN), which is the behaviour wanted in both cases.
    if not is_active:
        score *= 0.8
        basis += " (closed)"

    return round(min(score, 1.0), 3), basis


async def resolve_application(session: AsyncSession, user_id: uuid.UUID, query: str) -> Resolution:
    """Rank the caller's applications against a free-text reference.

    Deliberately scoped to one user's rows and never writes. A caller that
    wants to act must take ``best``, which is None unless one candidate is both
    strong and clearly ahead of the rest.
    """
    cleaned = query.strip()
    if not cleaned:
        return Resolution(candidates=[], query=query)

    pattern = f"%{cleaned}%"
    rows = list(
        (
            await session.execute(
                select(Application)
                .join(Job, Job.id == Application.job_id)
                .join(Company, Company.id == Job.company_id)
                .where(
                    Application.user_id == user_id,
                    or_(
                        Company.name.ilike(pattern),
                        Job.title.ilike(pattern),
                        # Fall back to trigram similarity so a misspelling
                        # ("Amazn") still surfaces the row rather than silently
                        # returning nothing, which reads as "no such application".
                        func.similarity(Company.name, cleaned) > 0.3,
                    ),
                )
            )
        )
        .unique()
        .scalars()
        .all()
    )

    terminal = {s.value for s in TERMINAL_STATUSES}
    scored: list[Candidate] = []
    for application in rows:
        score, basis = score_candidate(
            cleaned,
            application.job.company.name,
            application.job.title,
            application.current_status not in terminal,
        )
        if score > 0:
            scored.append(Candidate(application=application, score=score, matched_on=basis))

    scored.sort(key=lambda c: (-c.score, c.application.job.company.name))
    return Resolution(candidates=scored[:MAX_CANDIDATES], query=cleaned)
