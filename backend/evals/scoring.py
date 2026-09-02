"""Turning a model's answer into numbers, deterministically.

The metrics here are chosen so that a failing one names its own cause. Two
examples carry most of the design:

**Salary is two metrics, not one.** `salary_verbatim` asks whether the quoted
text appears in the posting; `salary_units` asks whether the numbers are right.
A single "salary correct" metric would collapse "invented a figure" and "read
LPA as rupees" into one number, and those have completely different fixes — one
is a grounding failure, the other an arithmetic one.

**Skills are precision and recall, separately, and precision needs no ground
truth.** "Every skill returned appears in the posting" is checkable against the
posting alone, which makes it the cheapest and strictest metric in the suite —
and it is the safety one, because an invented skill becomes a fictitious gap in
a match score later. Recall needs labels and is graded gently.

Where a metric is a proxy it says so. `brevity` measures characters because
"one or two sentences" has no honest deterministic form; it is reported as what
it is rather than dressed up.
"""

import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from app.agent.validation import ValidationReport, _normalize
from app.schemas.extraction import ExtractedJob
from app.schemas.matching import RubricJudgment
from evals.loader import Case

# An ISO date in a tool argument means the model computed one. It cannot: it
# does not know today's date, which is why every schema offers days-ago instead.
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Rule 7 asks for one or two sentences. Measured in characters because that is
# what can be counted honestly; the number is generous so it catches an essay
# rather than adjudicating a long sentence.
BRIEF_CHARS = 400


@dataclass(slots=True)
class CaseScore:
    """One case's numbers, plus why it failed in words a human can act on."""

    case_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    hard_failed: bool = False
    tokens: int = 0
    latency_ms: int = 0
    error: str | None = None

    def check(self, metric: str, passed: bool, detail: str) -> None:
        self.metrics[metric] = 1.0 if passed else 0.0
        if not passed:
            self.failures.append(f"{metric}: {detail}")

    def observe(self, metric: str, value: float, *, failed_detail: str = "") -> None:
        """A ratio rather than a pass — recorded, and only narrated if poor."""
        self.metrics[metric] = value
        if failed_detail and value < 1.0:
            self.failures.append(f"{metric}={value:.2f}: {failed_detail}")


def _text(value: Any) -> str:
    return _normalize(str(value)) if value is not None else ""


def _matches(actual: Any, expected: Any) -> bool:
    """Declared expectations, in the three forms a case may use.

    ``{"any_of": [...]}`` exists for the rules that are genuinely judgements —
    rule 6's title stripping has more than one defensible answer, and a case
    that pretended otherwise would be measuring taste.
    """
    if isinstance(expected, dict) and "any_of" in expected:
        return any(_matches(actual, option) for option in expected["any_of"])
    if isinstance(expected, str):
        return _text(actual) == _text(expected)
    return actual == expected


def _dig(obj: Any, path: str) -> Any:
    """`salary.raw_text` on a nested model."""
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# --- Extraction ------------------------------------------------------------


def score_extraction(
    case: Case,
    extracted: ExtractedJob,
    report: ValidationReport,
    needs_review: bool,
    posting: str,
) -> CaseScore:
    score = CaseScore(case_id=case.id)
    source = _normalize(posting)

    for path, expected in (case.get("expect") or {}).items():
        score.check(
            f"field:{path}",
            _matches(_dig(extracted, path), expected),
            f"expected {expected!r}, got {_dig(extracted, path)!r}",
        )

    # Rule 1, never invent — expressed as fields that must have stayed null.
    for path in case.get("expect_absent") or []:
        score.check(
            f"absent:{path}",
            _dig(extracted, path) is None,
            f"should be null, got {_dig(extracted, path)!r}",
        )

    _score_salary(case, extracted, source, score)
    _score_skills(extracted, source, score, case)
    _score_requirements(case, extracted, score)

    if (band := case.get("confidence")) is not None:
        _score_confidence(extracted.confidence, band, score)

    if (expected_review := case.get("needs_review")) is not None:
        score.check(
            "needs_review",
            needs_review is expected_review,
            f"expected {expected_review}, got {needs_review} "
            f"(confidence {extracted.confidence}, warnings {len(report.warnings)}, "
            f"dropped {report.dropped_fields})",
        )

    score.hard_failed = case.gate == "hard" and bool(score.failures)
    return score


def _score_salary(case: Case, extracted: ExtractedJob, source: str, score: CaseScore) -> None:
    """Provenance and arithmetic, kept apart.

    The verbatim check is the one the validator also enforces by discarding an
    unsupported block, so a failure here means the model fabricated a quote that
    would have been thrown away in production — a silent loss rather than a
    wrong number, and worth its own metric.
    """
    quoted = extracted.salary.raw_text
    if quoted is not None:
        score.check(
            "salary_verbatim",
            _normalize(quoted) in source,
            f"{quoted!r} does not appear in the posting",
        )

    if "salary_min" not in case.body:
        return

    # The numbers, and that the quote covers the figure they came from.
    #
    # Containment rather than equality on the quote: "1.2 Cr per annum" and
    # "1.2 Cr" are both verbatim and both correct, and a case cannot know which
    # surrounding words the model will take. Demanding an exact string measured
    # the case author's guess rather than the model — the first run failed on
    # exactly that, and the model was right.
    expected_quote = case.get("salary_quote")
    covers = expected_quote is None or _text(expected_quote) in _text(quoted)
    # `_matches` so a case can declare `any_of` on the numbers too. A single
    # stated figure has two defensible readings — max null, or max equal to min
    # — and the corpus should not be pinning a preference it cannot justify.
    numbers_right = _matches(extracted.salary.min_amount, case.get("salary_min")) and _matches(
        extracted.salary.max_amount, case.get("salary_max")
    )

    detail = []
    if not covers:
        detail.append(f"quote {quoted!r} does not contain {expected_quote!r}")
    if not numbers_right:
        detail.append(
            f"read {extracted.salary.min_amount}-{extracted.salary.max_amount}, "
            f"expected {case.get('salary_min')}-{case.get('salary_max')}"
        )
    score.check("salary_units", covers and numbers_right, "; ".join(detail))


def _score_skills(extracted: ExtractedJob, source: str, score: CaseScore, case: Case) -> None:
    """Precision against the posting; recall against declared labels.

    Precision is the safety half and needs no ground truth at all — a returned
    skill either occurs in the source or the model supplied it. Models
    pattern-complete stacks (Django, therefore PostgreSQL and Celery), and a
    phantom skill becomes a fictitious gap in a match score months later.
    """
    if extracted.skills:
        grounded = [s for s in extracted.skills if _normalize(s) in source]
        invented = [s for s in extracted.skills if _normalize(s) not in source]
        score.observe(
            "skill_precision",
            len(grounded) / len(extracted.skills),
            failed_detail=f"not in the posting: {invented}",
        )
    else:
        score.metrics["skill_precision"] = 1.0

    wanted = case.get("skills")
    if wanted:
        found = {_normalize(s) for s in extracted.skills}
        hit = [s for s in wanted if _normalize(s) in found]
        score.observe(
            "skill_recall",
            len(hit) / len(wanted),
            # What was returned, not only what was missed. A miss is usually the
            # model naming the same thing differently — "Golang" for "Go" — and
            # that is a fact about the corpus rather than about the model, which
            # a list of absences alone cannot tell you.
            failed_detail=(
                f"missed {[s for s in wanted if _normalize(s) not in found]}; "
                f"returned {extracted.skills}"
            ),
        )


def _score_requirements(case: Case, extracted: ExtractedJob, score: CaseScore) -> None:
    """Anchored `kind` accuracy, not list similarity.

    The model rephrases requirements freely and correctly, so comparing whole
    lists would measure paraphrase style. Rule 3 is only about the must/nice
    split, so each anchor names a phrase and the classification it should carry
    — which is the rule itself, checked directly.
    """
    anchors = case.get("requirement_kinds") or {}
    if anchors:
        correct = 0
        for phrase, expected_kind in anchors.items():
            found = next(
                (r for r in extracted.requirements if _normalize(phrase) in _normalize(r.text)),
                None,
            )
            if found is not None and found.kind == expected_kind:
                correct += 1
            else:
                got = found.kind if found else "absent"
                score.failures.append(
                    f"requirement {phrase!r}: expected {expected_kind}, got {got}"
                )
        score.metrics["requirement_kind"] = correct / len(anchors)

    if (expected_count := case.get("requirement_count")) is not None:
        # A window, because "one entry per requirement" is a judgement about
        # where a bullet list stops being one bullet.
        actual = len(extracted.requirements)
        score.check(
            "requirement_count",
            abs(actual - expected_count) <= 2,
            f"{actual} requirements, expected about {expected_count}",
        )


def _score_confidence(actual: float, band: dict[str, float], score: CaseScore) -> None:
    """Direction, never a point value.

    Confidence only changes anything at the 0.75 review threshold with an
    otherwise clean report, so grading it more finely than "above or below"
    would be grading noise.
    """
    if "below" in band:
        score.check(
            "confidence_direction",
            actual < band["below"],
            f"{actual} should be below {band['below']} for a posting this poor",
        )
    if "at_least" in band:
        score.check(
            "confidence_direction",
            actual >= band["at_least"],
            f"{actual} should be at least {band['at_least']} for a clean posting",
        )


# --- Assistant -------------------------------------------------------------


def score_assistant(case: Case, message: str, calls: list[dict[str, Any]]) -> CaseScore:
    score = CaseScore(case_id=case.id)
    names = [c["name"] for c in calls]

    if (expected := case.get("expect_tool")) is not None:
        allowed = expected["any_of"] if isinstance(expected, dict) else [expected]
        score.check(
            "tool_selection",
            bool(names) and names[0] in allowed,
            f"called {names[0] if names else 'nothing'}, expected one of {allowed}",
        )

    # Hard: reaching for the wrong tool is how an approximation gets recorded as
    # the thing that was asked for.
    forbidden = [n for n in (case.get("expect_not_tool") or []) if n in names]
    if case.get("expect_not_tool"):
        score.check("forbidden_tool", not forbidden, f"called {forbidden}, which it must not")

    _score_query_fidelity(case, calls, score)

    dated = [
        f"{c['name']}.{key}={value!r}"
        for c in calls
        for key, value in (c.get("arguments") or {}).items()
        if isinstance(value, str) and ISO_DATE.search(value)
    ]
    score.check("no_invented_dates", not dated, f"passed a computed date: {dated}")

    if case.get("expect_brief"):
        score.check(
            "brevity",
            len(message) <= BRIEF_CHARS,
            f"{len(message)} characters; rule 7 asks for one or two sentences",
        )

    score.metrics["rounds"] = float(len(calls))
    score.hard_failed = case.gate == "hard" and bool(score.failures)
    return score


def _score_query_fidelity(case: Case, calls: list[dict[str, Any]], score: CaseScore) -> None:
    """Rule 1, and the resolver's only blind spot.

    The resolver is deterministic and refuses to guess — but it can only refuse
    on the string it was handed. A model that substitutes its own idea of the
    title for the user's words aims a correct mechanism at the wrong row, and
    nothing downstream can tell.
    """
    phrase = case.get("expect_query_contains")
    if phrase is None:
        return

    queries = [
        str((c.get("arguments") or {}).get("query", ""))
        for c in calls
        if "query" in (c.get("arguments") or {})
    ]
    score.check(
        "query_fidelity",
        any(phrase.lower() in q.lower() for q in queries),
        f"no query contained {phrase!r}; sent {queries}",
    )


# --- Rubric ----------------------------------------------------------------


def score_rubric(case: Case, judgment: RubricJudgment) -> CaseScore:
    score = CaseScore(case_id=case.id)

    if (band := case.get("score_band")) is not None:
        low, high = band
        score.check(
            "score_band",
            low <= judgment.score <= high,
            f"{judgment.score} outside [{low}, {high}]",
        )

    _score_evidence_only(case, judgment, score)

    evidence = " ".join(
        chunk for _, chunks in (case.get("requirements") or []) for chunk in chunks
    ).lower()
    if judgment.strengths and evidence:
        grounded = sum(1 for s in judgment.strengths if _shares_a_word(s, evidence))
        score.observe(
            "strength_groundedness",
            grounded / len(judgment.strengths),
            failed_detail="a strength points at nothing in the evidence",
        )

    wanted = " ".join(req for req, _ in (case.get("requirements") or [])).lower()
    if judgment.gaps and wanted:
        specific = sum(1 for g in judgment.gaps if _shares_a_word(g, wanted))
        score.observe(
            "gap_specificity",
            specific / len(judgment.gaps),
            failed_detail="a gap names nothing from the requirements",
        )

    score.hard_failed = case.gate == "hard" and bool(score.failures)
    return score


def _score_evidence_only(case: Case, judgment: RubricJudgment, score: CaseScore) -> None:
    """Rule 1 of the rubric prompt: absence of evidence is absence.

    Tested through its negative, which is the only form that means anything. A
    requirement supplied with no evidence must be reported as a gap and must not
    appear as a strength — the failure mode is a model reasoning that a Python
    developer surely knows Django, and scoring the candidate on it.
    """
    unevidenced = case.get("unevidenced")
    if not unevidenced:
        return

    in_gaps = any(unevidenced.lower() in g.lower() for g in judgment.gaps)
    in_strengths = any(unevidenced.lower() in s.lower() for s in judgment.strengths)
    score.check(
        "evidence_only",
        in_gaps and not in_strengths,
        f"{unevidenced!r} had no supporting passage; "
        f"in gaps={in_gaps}, claimed as a strength={in_strengths}",
    )


def _shares_a_word(claim: str, source: str) -> bool:
    """Crude on purpose: it catches a sentence invented wholesale, nothing subtler."""
    words = {w for w in re.findall(r"[a-z]{5,}", claim.lower())}
    return any(w in source for w in words) if words else False


# --- Aggregation -----------------------------------------------------------


def aggregate(scores: list[CaseScore]) -> dict[str, float]:
    """Mean per metric across the cases that reported it.

    Averaged over reporters rather than over all cases, because a case that says
    nothing about salary should not be counted as having got it right — that
    would let the corpus improve a metric by growing in an unrelated direction.
    """
    collected: dict[str, list[float]] = {}
    for score in scores:
        for metric, value in score.metrics.items():
            collected.setdefault(metric, []).append(value)
    return {metric: mean(values) for metric, values in sorted(collected.items())}


def format_failures(scores: list[CaseScore]) -> str:
    lines: list[str] = []
    for score in scores:
        if score.error:
            lines.append(f"  {score.case_id}: ERRORED — {score.error}")
        for failure in score.failures:
            lines.append(f"  {score.case_id}: {failure}")
    return "\n".join(lines)
