"""Scoring and resume chunking.

The scoring functions are pure and tested directly — that is the whole point of
keeping 85% of the score out of the model. If these are right, a score is
defensible even when the rubric is unavailable.
"""

from datetime import date

import pytest

from app.services.matching import (
    SECTION_PENALTY,
    UNSECTIONED_PENALTY,
    WEIGHTS,
    SkillCoverage,
    score_experience,
    score_seniority,
    score_skills,
)
from app.services.resume_parser import (
    chunk_resume,
    estimate_years_experience,
    parse_positions,
    seniority_from_title,
    stated_years_experience,
)

TODAY = date(2026, 9, 1)
"""Pinned. "Present" resolves against the current date, so a real one would
make these assertions drift by a month at a time."""


class FakeSkill:
    def __init__(self, slug: str, name: str) -> None:
        self.slug, self.name = slug, name


def test_weights_sum_to_one() -> None:
    """A drifting total would silently rescale every score in the database."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_the_model_can_shade_a_score_but_not_manufacture_one() -> None:
    """The rubric is capped so a confident model cannot override the arithmetic."""
    assert WEIGHTS["rubric"] <= 0.15
    assert WEIGHTS["must_have_skills"] > WEIGHTS["rubric"] * 2


# --- Skill coverage --------------------------------------------------------


def test_full_coverage_scores_one() -> None:
    required = [FakeSkill("python", "Python"), FakeSkill("kafka", "Kafka")]

    coverage = score_skills(required, {"python", "kafka", "docker"})

    assert coverage.score == 1.0
    assert coverage.missing == []


def test_partial_coverage_is_proportional() -> None:
    required = [FakeSkill("python", "Python"), FakeSkill("kafka", "Kafka")]

    coverage = score_skills(required, {"python"})

    assert coverage.score == 0.5
    assert coverage.missing == ["Kafka"]


def test_a_posting_stating_no_skills_is_not_a_perfect_match() -> None:
    """Returning 1.0 would let a vague posting outscore a detailed one the
    candidate genuinely fits — the worst possible ranking inversion here."""
    assert SkillCoverage().score == 0.5


# --- Experience ------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "years", "expected"),
    [
        (3, 6, 4, 1.0),  # inside the range
        (3, 6, 3, 1.0),  # exactly at the floor
        (5, None, 5, 1.0),  # open-ended, met
        (None, None, 4, 0.5),  # unstated requirement
        (3, 6, None, 0.5),  # unknown candidate years
    ],
)
def test_experience_within_range_scores_full(
    low: int | None, high: int | None, years: float | None, expected: float
) -> None:
    assert score_experience(low, high, years) == pytest.approx(expected)


def test_being_under_qualified_is_penalised_harder_than_being_over() -> None:
    """Three years short of a five-year requirement is usually a rejection.
    Three years beyond it rarely is, so the curves are deliberately asymmetric."""
    under = score_experience(5, 8, 2)
    over = score_experience(5, 8, 11)

    assert under < over
    assert over >= 0.5, "over-qualification should never crater a score"


def test_experience_floor_is_zero_not_negative() -> None:
    assert score_experience(10, 12, 0) >= 0.0


# --- Seniority -------------------------------------------------------------


def test_seniority_matches_implied_level() -> None:
    assert score_seniority("senior", 6) == pytest.approx(1.0)
    assert score_seniority("intern", 0.5) == pytest.approx(1.0)


def test_seniority_degrades_with_distance() -> None:
    close = score_seniority("senior", 4)
    far = score_seniority("principal", 0.5)

    assert close > far
    assert far >= 0.0


def test_unknown_seniority_is_neutral() -> None:
    assert score_seniority(None, 5) == 0.5
    assert score_seniority("senior", None) == 0.5
    assert score_seniority("nonsense", 5) == 0.5


def test_a_held_title_beats_the_years_proxy() -> None:
    """Six years says "senior" by tenure, but this candidate reached Staff.
    Years cannot tell the promoted engineer from the un-promoted one."""
    by_years = score_seniority("staff", 6)
    by_title = score_seniority("staff", 6, ["Staff Software Engineer"])

    assert by_title == pytest.approx(1.0)
    assert by_title > by_years


def test_the_most_senior_title_held_is_the_one_that_counts() -> None:
    """A Staff Engineer now consulting is not a junior again."""
    titles = ["Junior Developer", "Staff Software Engineer", "Consultant"]

    assert score_seniority("staff", None, titles) == pytest.approx(1.0)


def test_titles_stating_no_level_fall_through_to_years() -> None:
    """A bare "Software Engineer" claims no level. Reading it as mid would pin
    every such resume at one level regardless of a decade of tenure."""
    titles = ["Software Engineer", "Software Developer"]

    assert score_seniority("principal", 15, titles) == score_seniority("principal", 15)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Senior Backend Engineer", "senior"),
        ("Data Science Intern", "intern"),
        ("Principal Architect", "principal"),
        ("Senior Engineering Manager", "lead"),  # a manager, not a senior IC
        ("Software Engineer", None),
    ],
)
def test_seniority_is_read_from_the_title(title: str, expected: str | None) -> None:
    level = seniority_from_title(title)

    assert (level.value if level else None) == expected


# --- Evidence weighting ----------------------------------------------------


def test_experience_is_the_least_penalised_evidence() -> None:
    """A skills line naming Kafka is a claim; an experience bullet describing a
    Kafka pipeline is proof. Retrieval must prefer the second."""
    assert SECTION_PENALTY["experience"] == 1.0
    assert SECTION_PENALTY["skills"] > SECTION_PENALTY["experience"]
    assert SECTION_PENALTY["education"] == max(SECTION_PENALTY.values())


def test_weighting_only_ever_penalises() -> None:
    """Every multiplier is applied to a cosine distance. One below 1.0 would
    make a passage rank ahead of a strictly closer one."""
    assert all(penalty >= 1.0 for penalty in SECTION_PENALTY.values())
    assert UNSECTIONED_PENALTY >= 1.0


# --- Chunking --------------------------------------------------------------

RESUME = """\
Aryan Jain
Backend Engineer | 4 years of experience

SUMMARY
Backend engineer building LLM-powered systems.

WORK EXPERIENCE
- Built RAG pipelines over enterprise documents using LangChain and Qdrant.
- Designed FastAPI microservices on AWS handling 2 million requests per day.
- Implemented Kafka event pipelines for asynchronous order processing.

SKILLS
Python, FastAPI, PostgreSQL, Kafka, Docker

EDUCATION
B.Tech in Computer Science, Pune Institute of Computer Technology
"""


def test_chunks_are_tagged_with_their_section() -> None:
    chunks = chunk_resume(RESUME)

    sections = {c.section for c in chunks}
    assert {"summary", "experience", "skills", "education"} <= sections


def test_each_bullet_becomes_its_own_chunk() -> None:
    """The unit that answers "does this person know Kafka?" is one
    accomplishment line, not a page."""
    chunks = chunk_resume(RESUME)

    experience = [c.content for c in chunks if c.section == "experience"]
    assert len(experience) == 3
    assert any("Kafka" in c for c in experience)
    assert any("RAG pipelines" in c for c in experience)


def test_bullets_are_never_split_mid_sentence() -> None:
    """Fixed-size chunking cuts bullets in half, so the fragment that should
    have evidenced a skill evidences neither half."""
    chunks = chunk_resume(RESUME)

    kafka = next(c for c in chunks if "Kafka" in c.content)
    assert kafka.content.endswith("processing.")


def test_ordinals_are_contiguous() -> None:
    chunks = chunk_resume(RESUME)

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_empty_resume_produces_no_chunks() -> None:
    assert chunk_resume("") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Backend Engineer | 4 years of experience", 4.0),
        ("6+ years of professional experience in Python", 6.0),
        ("Experience: 3 years", 3.0),
        ("No mention of duration here at all", None),
        ("99 years of experience", None),  # implausible, rejected
    ],
)
def test_a_stated_number_is_read_as_stated(text: str, expected: float | None) -> None:
    assert stated_years_experience(text) == expected


# --- Employment history ----------------------------------------------------


def estimate(text: str) -> tuple[float | None, str | None]:
    result = estimate_years_experience(text, parse_positions(text, today=TODAY))
    return result.years, result.source


DATED = """\
WORK EXPERIENCE

Senior Backend Engineer, Fintech Co (Jan 2021 - Present)
- Designed the double-entry ledger that reconciles every payout.

Backend Engineer, Acme Technologies | March 2018 – December 2020
- Built internal tooling.

EDUCATION
B.Tech in Computer Science, Pune Institute of Technology, 2013 - 2017
"""


def test_years_are_summed_from_dates_when_none_is_stated() -> None:
    """Most resumes never write the number; they write the dates and leave the
    arithmetic to the reader. Leaving it unknown is not neutral — experience and
    seniority both fall back to 0.5, making a quarter of the score a constant."""
    years, source = estimate(DATED)

    assert source == "dates"
    assert years == pytest.approx(8.6, abs=0.1)  # 2018-03 to 2026-09, no gap


def test_a_stated_number_wins_over_the_dates_below_it() -> None:
    """It is what a recruiter reads and what the candidate stands behind, and it
    already accounts for what dates cannot see — a break, unlisted early roles."""
    years, source = estimate(f"Backend Engineer | 4 years of experience\n{DATED}")

    assert (years, source) == (4.0, "stated")


def test_education_dates_are_not_employment() -> None:
    """The degree in DATED spans four more years. Counting them would hand a
    graduate four years of experience they do not have."""
    years, _ = estimate(DATED)

    assert years is not None and years < 10


def test_concurrent_roles_are_counted_once() -> None:
    """A promotion listed as two entries, or a contract held alongside a staff
    job, would otherwise be added together into a career nobody has had."""
    text = """\
EXPERIENCE
Lead Engineer, Acme Ltd (Jan 2020 - Dec 2023)
Consultant, Freelance (Jun 2021 - Dec 2023)
"""

    years, _ = estimate(text)

    assert years == pytest.approx(4.0, abs=0.1), "the second role sits inside the first"


def test_dates_inside_a_bullet_are_not_a_job() -> None:
    """A bullet reading "cut deploy time between 2010 and 2015" describes the
    work, not a period of employment. Counting those is how a total runs away."""
    text = """\
EXPERIENCE
Backend Engineer, Acme (Jan 2023 - Jan 2024)
- Cut deploy time between 2010 and 2015 by half.
"""

    years, _ = estimate(text)

    assert years == pytest.approx(1.1, abs=0.1)


def test_a_gap_between_roles_is_not_counted() -> None:
    text = """\
EXPERIENCE
Engineer, Acme (Jan 2016 - Dec 2017)
Engineer, Globex (Jan 2022 - Dec 2023)
"""

    years, _ = estimate(text)

    assert years == pytest.approx(4.0, abs=0.1), "four years worked, four years out"


def test_no_dates_and_no_statement_stays_unknown() -> None:
    """Guessing here would be worse than admitting ignorance: the scorer treats
    unknown as neutral, but a wrong number skews every score invisibly."""
    assert estimate("SKILLS\nPython, Kafka, Docker\n") == (None, None)


# --- Roles -----------------------------------------------------------------


def test_title_and_company_are_read_from_a_role_header() -> None:
    positions = parse_positions(DATED, today=TODAY)

    assert [(p.title, p.company) for p in positions] == [
        ("Senior Backend Engineer", "Fintech Co"),
        ("Backend Engineer", "Acme Technologies"),
    ]


def test_a_role_still_held_is_marked_current() -> None:
    positions = parse_positions(DATED, today=TODAY)

    assert positions[0].is_current
    assert positions[0].as_dict()["end"] is None
    assert not positions[1].is_current


@pytest.mark.parametrize(
    "header",
    [
        "Acme Labs — Senior Backend Engineer | Jan 2021 – Present",
        "Senior Backend Engineer, Acme Labs (Jan 2021 - Present)",
        "Senior Backend Engineer at Acme Labs, 01/2021 - Present",
    ],
)
def test_either_ordering_of_title_and_company_is_read(header: str) -> None:
    """Both are about equally common, so position on the line says nothing —
    the words do. One part names a role; the other usually does not."""
    position = parse_positions(f"EXPERIENCE\n{header}\n", today=TODAY)[0]

    assert position.title == "Senior Backend Engineer"
    assert position.company == "Acme Labs"


def test_a_title_on_its_own_line_is_still_found() -> None:
    """Plenty of resumes put the role above and the company with the dates."""
    text = """\
PROFESSIONAL EXPERIENCE
Machine Learning Engineer
Zeta Systems, Bangalore | 03/2022 - 08/2025
"""

    position = parse_positions(text, today=TODAY)[0]

    assert position.title == "Machine Learning Engineer"
    assert position.company == "Zeta Systems"


THREE_LINE_HEADERS = """\
EXPERIENCE
Senior GenAI Engineer
Synconic Technologies
Jun 2023 - Present
- Built RAG pipelines over enterprise documents.
- Shipped an agent that books meetings.
Backend Engineer
Acme Labs
Jan 2021 - May 2023
- Owned the payments ledger.
"""


def test_a_header_split_over_three_lines_is_read_whole() -> None:
    """What a PDF does with a right-aligned date: it lands on its own line, so
    the line identifying the role carries nothing but the dates."""
    positions = parse_positions(THREE_LINE_HEADERS, today=TODAY)

    assert [(p.title, p.company) for p in positions] == [
        ("Senior GenAI Engineer", "Synconic Technologies"),
        ("Backend Engineer", "Acme Labs"),
    ]


def test_a_consumed_header_is_not_reused_by_the_next_role() -> None:
    """Without clearing it, the second role inherits the first one's title —
    every position on the resume then reports the same job."""
    titles = [p.title for p in parse_positions(THREE_LINE_HEADERS, today=TODAY)]

    assert len(set(titles)) == len(titles)


def test_a_bullet_is_never_offered_as_a_job_title() -> None:
    """The lookback reaches over the previous role's bullets. "Led the
    migration" is an accomplishment, not a position."""
    text = """\
EXPERIENCE
- Led the migration of every service onto Kubernetes.
Acme Technologies
Jan 2021 - May 2023
"""

    position = parse_positions(text, today=TODAY)[0]

    assert position.title is None
    assert position.company == "Acme Technologies"


def test_a_bare_year_range_is_read_as_mid_year() -> None:
    """A range written "2019 - 2024" is five years to whoever reads the resume.
    Anchoring both ends to January would make it six."""
    position = parse_positions("EXPERIENCE\nEngineer, Acme (2019 - 2024)\n", today=TODAY)[0]

    assert position.months == pytest.approx(61, abs=1)
    assert position.as_dict()["start"] == "2019-06"


def test_a_resume_with_no_headings_still_yields_roles() -> None:
    """Where a resume marks no experience section, unsectioned lines are read
    instead — otherwise it produces nothing at all."""
    text = """\
Aryan Jain
Software Developer at Initech (2020 - Present)
B.Tech, Some University, 2016 - 2020
"""

    positions = parse_positions(text, today=TODAY)

    assert [p.company for p in positions] == ["Initech"]


# --- Job embedding ---------------------------------------------------------


async def test_creating_a_job_embeds_it(client, embeddings) -> None:
    """Without this, semantic search and near-duplicate detection are inert:
    the table and index exist but nothing ever populates them."""
    from sqlalchemy import text as sql

    from app.db.session import open_user_session
    from tests.factories import Session as UserSession

    user = await UserSession(client).start()
    before = embeddings.documents_embedded

    await user.create_application(company_name="Razorpay", title="Backend Engineer")

    assert embeddings.documents_embedded > before
    async for session in open_user_session(user.user_id):
        count = (await session.execute(sql("SELECT count(*) FROM job_embeddings"))).scalar_one()
    assert count == 1


async def test_embedding_text_omits_boilerplate() -> None:
    """Postings pad themselves with culture and benefits prose that is nearly
    identical across companies. Including it drags every job toward the same
    point in the space, which is precisely what makes semantic search useless."""
    from app.services.applications import job_embedding_text

    content = job_embedding_text(
        title="Backend Engineer",
        company_name="Razorpay",
        seniority="senior",
        location="Pune",
        requirements=["5+ years Python", "Kafka"],
        responsibilities="Build services.",
    )

    assert "Backend Engineer" in content
    assert "Razorpay" in content
    assert "5+ years Python" in content
    assert "Kafka" in content


async def test_a_job_that_fails_to_embed_is_still_saved(client, monkeypatch) -> None:
    """An un-embedded job is merely absent from semantic search. Refusing to
    save it would lose the user's actual work over a secondary feature."""

    class BrokenEmbeddings:
        dimension = 384

        async def embed_documents(self, texts):
            raise RuntimeError("model unavailable")

        async def embed_query(self, text_):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr("app.services.embeddings.embedding_provider", BrokenEmbeddings())
    from tests.factories import Session as UserSession

    user = await UserSession(client).start()
    application = await user.create_application(company_name="Zerodha")

    assert application["job"]["company"]["name"] == "Zerodha"
