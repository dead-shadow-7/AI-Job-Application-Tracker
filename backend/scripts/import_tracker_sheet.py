"""Backfill the tracker from the hand-maintained "Internship & Job Tracker" sheet.

This is the migration the app was built to replace. It reads the spreadsheet,
resolves each row to a company and a job, and replays it as an event timeline
rather than writing statuses directly -- ``services.events.append_event`` stays
the only writer of ``applications.current_status``, exactly as in the request
path, so the cached columns are derived here the same way they are everywhere
else.

Two things in the source need judgement, and both are recorded rather than
guessed silently:

*Dates.* The sheet was typed ``DD/MM/YYYY`` but opened in an ``M/D/Y`` locale,
so every cell whose first component was <= 12 was silently converted to a real
date with its day and month transposed, while the rest survived as text. The
tell is that every surviving string has a first component > 12 -- never a valid
month. So strings parse day-first and datetimes get swapped back. Two
independent checks confirm it: the reconstructed sequence is monotonic across
156 of 157 rows, and reading the datetimes as-is would scatter August 2026
applications into December.

*Missing event dates.* The sheet records one date per row -- when you applied.
A rejection therefore has no date of its own. Rather than invent a gap or stamp
the import time (which would make sixty-one dead applications look like they
were all rejected today), the closing event is dated to the application date and
says so in its note. The clock times below keep the ordering unambiguous.

Requires ``openpyxl``, which is not a runtime dependency:

    pip install openpyxl
    python scripts/import_tracker_sheet.py --user <email> --file <sheet.xlsx>
    python scripts/import_tracker_sheet.py --user <email> --file <sheet.xlsx> --commit
"""

import argparse
import asyncio
import re
import sys
import uuid
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy import types as sqltypes
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.config import settings
from app.db.session import SessionFactory
from app.domain.enums import EmploymentType, EventSource, EventType, Seniority
from app.models.application import Application, ApplicationEvent
from app.models.company import Company
from app.models.job import Job
from app.models.user import User
from app.schemas.job import JobCreate
from app.services.applications import create_application
from app.services.companies import normalize_company_name
from app.services.events import append_event

BATCH_ID = "internship-job-tracker-xlsx"
"""Stamped into every event payload, so a second run can refuse to double-import."""

# Events land at fixed UTC times rather than midnight for two reasons. The fold
# in _refresh_cached_state orders by (occurred_at, created_at), and created_at
# defaults to now() -- which in Postgres is the *transaction* timestamp, identical
# for every row written here. Same-day events would therefore have no stable
# order at all. Spacing them by hours makes the ordering come from the data.
# The window is also chosen so that 09:00-15:00 UTC renders as the same calendar
# date in IST (14:30-20:30), where these applications were actually made.
APPLIED_AT = time(9, 0, tzinfo=UTC)
SECOND_EVENT_AT = time(12, 0, tzinfo=UTC)
THIRD_EVENT_AT = time(15, 0, tzinfo=UTC)

UNDATED_NOTE = "Exact date not recorded in the source sheet; dated to the application date."
UNSPECIFIED_ROLE = "Unspecified role"

# --- Sheet vocabulary -> domain vocabulary ---------------------------------
# The sheet's four statuses do not map one-to-one onto the event model, because
# a status is a position and an event is a thing that happened. Each sheet
# status expands into the sequence of events that would have produced it.
STATUS_EVENTS: dict[str, tuple[EventType, ...]] = {
    "Submitted": (EventType.APPLIED,),
    "Rejected": (EventType.APPLIED, EventType.REJECTED),
    "Accepted": (EventType.APPLIED, EventType.ACCEPTED),
    # "In Progress" is refined per row below -- the Details column says which
    # kind of progress, and the two rows differ.
    "In Progress": (EventType.APPLIED,),
}

# Keyed by sheet row. Both "In Progress" rows describe a concrete stage in their
# Details column; using it beats collapsing them onto one generic event.
IN_PROGRESS_EVENTS: dict[int, EventType] = {
    66: EventType.ASSESSMENT_RECEIVED,  # IBM -- "Gave the coding assesment"
    149: EventType.SCREENING_SCHEDULED,  # Setoo -- "Got a call for phone screening round"
}

# Free text in the "Applicant Portal" column, folded onto stable labels so the
# Insights group-by is not split three ways between "Career Portal", "Official
# career pages" and a bare company URL. Anything unrecognised is kept verbatim.
PLATFORM_ALIASES = {
    "on campus": "On Campus",
    "linkedin": "LinkedIn",
    "indeed": "Indeed",
    "unstop": "Unstop",
    "glassdoor": "Glassdoor",
    "career portal": "Company Site",
    "official career pages": "Company Site",
    "email": "Email",
    "emailed the hr": "Email",
    "mailed hr": "Email",
    "referral": "Referral",
    "well found": "Wellfound",
    "turbohire": "TurboHire",
}

# Applied only when the cell holds a bare URL with no label of its own.
PLATFORM_BY_HOST = (
    ("linkedin.com", "LinkedIn"),
    ("indeed.com", "Indeed"),
    ("glassdoor.", "Glassdoor"),
    ("naukri.com", "Naukri"),
    ("greenhouse.io", "Greenhouse"),
    ("myworkdayjobs.com", "Workday"),
    ("ashbyhq.com", "Ashby"),
    ("lever.co", "Lever"),
    ("icims.com", "iCIMS"),
    ("oraclecloud.com", "Oracle Recruiting"),
    ("successfactors", "SuccessFactors"),
    ("eightfold.ai", "Eightfold"),
    ("keka.com", "Keka"),
    ("bamboohr.com", "BambooHR"),
    ("turbohire.co", "TurboHire"),
    ("avature.net", "Avature"),
    ("peoplestrong.com", "PeopleStrong"),
    ("docs.google.com", "Google Form"),
    ("tally.so", "Tally Form"),
    ("fabrichq.ai", "FabricHQ"),
    ("ycombinator.com", "Y Combinator"),
)

_INTERN = re.compile(r"\bintern(ship)?\b", re.I)
_JUNIOR = re.compile(r"\b(junior|jr\.?|associate|trainee|fresher|entry[ -]level)\b", re.I)
_URL = re.compile(r"https?://\S+")


# --- Parsing ---------------------------------------------------------------


def parse_applied_date(cell: Any, row: int) -> date | None:
    """Undo the locale transposition described in the module docstring."""
    if cell is None:
        return None
    if isinstance(cell, datetime):
        if cell.day > 12:
            # Would mean the locale read a day as a month, which cannot happen.
            raise SystemExit(f"row {row}: datetime {cell!r} cannot be transposed back")
        return date(cell.year, cell.day, cell.month)

    raw = str(cell).strip()
    if raw in {"", "-"}:
        return None
    parts = raw.replace("-", "/").split("/")
    if len(parts) != 3:
        raise SystemExit(f"row {row}: cannot parse date {raw!r}")
    day, month, year = (int(p) for p in parts)
    return date(year + 2000 if year < 100 else year, month, day)


def split_portal(cell: Any) -> tuple[str | None, str | None]:
    """Separate the "Applicant Portal" cell into a URL and a platform label.

    The column holds whichever the user had to hand: sometimes a deep link into
    an ATS, sometimes just "LinkedIn", occasionally both.
    """
    if not cell:
        return None, None
    raw = str(cell).strip()

    match = _URL.search(raw)
    url = match.group(0) if match else None
    label = (raw[: match.start()] + raw[match.end() :]).strip() if match else raw

    if label:
        platform = PLATFORM_ALIASES.get(label.lower(), label)
    elif url:
        host = url.lower()
        platform = next((name for frag, name in PLATFORM_BY_HOST if frag in host), "Company Site")
    else:
        platform = None

    return url, platform[:60] if platform else None


def infer_role_shape(title: str) -> tuple[Seniority | None, EmploymentType | None]:
    """Read seniority and employment type out of the job title, or leave them unset.

    Only what the title actually says. The sheet has no columns for these, and a
    blank field is honest where a guessed ``full_time`` would be indistinguishable
    from something the user confirmed.
    """
    if _INTERN.search(title):
        return Seniority.INTERN, EmploymentType.INTERNSHIP
    if _JUNIOR.search(title):
        return Seniority.JUNIOR, None
    return None, None


def read_sheet(path: Path) -> list[dict[str, Any]]:
    import openpyxl  # not a runtime dependency; see the module docstring

    worksheet = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = list(worksheet.iter_rows(values_only=True))

    header = next(i for i, r in enumerate(rows) if r and r[0] == "Company")

    records = []
    for offset, row in enumerate(rows[header + 1 :]):
        number = header + 2 + offset  # 1-indexed, as the spreadsheet shows it
        if not row or not row[0] or not str(row[0]).strip():
            continue  # trailing rows carrying only a stray date
        company, position, raw_date, status, details, portal = (list(row) + [None] * 6)[:6]

        if str(status).strip() not in STATUS_EVENTS:
            raise SystemExit(f"row {number}: unknown status {status!r}")

        url, platform = split_portal(portal)
        records.append(
            {
                "row": number,
                "company": str(company).strip(),
                "title": str(position).strip() if position else UNSPECIFIED_ROLE,
                "applied": parse_applied_date(raw_date, number),
                "status": str(status).strip(),
                "details": str(details).strip() if details else None,
                "url": url,
                "platform": platform,
            }
        )
    return records


def fill_missing_dates(records: list[dict[str, Any]]) -> None:
    """Carry the preceding row's date into any row that has none.

    The sheet is kept in application order, so a blank date sits between two
    known ones -- the neighbour is the closest defensible value. There is
    exactly one such row (an on-campus drive recorded as "-"), and it is
    labelled in its own note rather than left to look precise.
    """
    previous: date | None = None
    for record in records:
        if record["applied"] is None:
            record["applied"] = previous
            record["date_missing"] = True
            if previous is None:
                raise SystemExit(f"row {record['row']}: no date and no earlier row to borrow from")
        else:
            previous = record["applied"]
            record["date_missing"] = False


def plan_events(record: dict[str, Any]) -> list[tuple[EventType, datetime, str | None]]:
    """Expand one sheet row into its timeline."""
    day: date = record["applied"]
    types = list(STATUS_EVENTS[record["status"]])
    if record["status"] == "In Progress":
        types.append(IN_PROGRESS_EVENTS[record["row"]])

    clocks = (APPLIED_AT, SECOND_EVENT_AT, THIRD_EVENT_AT)
    events = []
    for index, event_type in enumerate(types):
        occurred = datetime.combine(day, clocks[index])

        caveats = []
        if event_type is EventType.APPLIED and record["date_missing"]:
            caveats.append("Date not recorded in the source sheet; taken from the preceding row.")
        elif event_type is not EventType.APPLIED:
            caveats.append(UNDATED_NOTE)

        parts = [record["details"]] if index and record["details"] else []
        parts += caveats
        events.append((event_type, occurred, " ".join(parts) or None))
    return events


# --- Writing ---------------------------------------------------------------


async def resolve_user_id(email: str) -> uuid.UUID | None:
    """Look the user up with the migration role.

    ``users`` is tenant-scoped, and its policy compares against ``app.user_id``
    -- which is the very thing being looked up. The runtime role therefore
    cannot resolve an email to an id by construction, so this one query runs as
    the schema owner. Everything that follows runs scoped, under the same
    policies as a request.
    """
    engine = create_async_engine(settings.migration_database_url)
    try:
        async with engine.connect() as connection:
            row = (
                await connection.execute(
                    text("SELECT id FROM users WHERE email = :email"), {"email": email}
                )
            ).first()
            return row[0] if row else None
    finally:
        await engine.dispose()


async def already_imported(session: AsyncSession, user_id: uuid.UUID) -> int:
    return (
        await session.execute(
            select(func.count())
            .select_from(ApplicationEvent)
            .where(
                ApplicationEvent.user_id == user_id,
                ApplicationEvent.payload["import_batch"].astext == BATCH_ID,
            )
        )
    ).scalar_one()


async def find_reusable_job(
    session: AsyncSession, record: dict[str, Any], claimed: set[uuid.UUID]
) -> Job | None:
    """Match a row onto a job that already existed before this import.

    Jobs are shared reference data, so a posting entered by hand earlier should
    not be duplicated by the backfill. Restricted to pre-existing rows and each
    reused at most once, so two sheet rows can never collapse onto one job and
    trip the (user_id, job_id) uniqueness rule.
    """
    normalized = normalize_company_name(record["company"])
    candidates = (
        (
            await session.execute(
                select(Job)
                .join(Company, Company.id == Job.company_id)
                .where(
                    Company.normalized_name == normalized,
                    func.lower(Job.title) == record["title"].lower(),
                )
            )
        )
        .scalars()
        .all()
    )
    for job in candidates:
        if job.id in claimed:
            continue
        if record["url"] and job.url and record["url"] == job.url:
            claimed.add(job.id)
            return job
    return None


async def import_records(
    session: AsyncSession, user_id: uuid.UUID, records: list[dict[str, Any]]
) -> dict[str, int]:
    tally = {"applications": 0, "events": 0, "jobs_created": 0, "jobs_reused": 0}
    claimed: set[uuid.UUID] = set()

    for record in records:
        seniority, employment_type = infer_role_shape(record["title"])
        existing = await find_reusable_job(session, record, claimed)

        timeline = plan_events(record)
        first_type, first_at, first_note = timeline[0]

        application = await create_application(
            session,
            user_id=user_id,
            job_id=existing.id if existing else None,
            job_payload=None
            if existing
            else JobCreate(
                company_name=record["company"],
                title=record["title"],
                seniority=seniority,
                employment_type=employment_type,
                url=record["url"],
                source_platform=record["platform"],
            ),
            priority="medium",
            notes=record["details"],
            initial_event=first_type,
            occurred_at=first_at,
        )

        # create_application appends the opening event without a note or
        # payload, so stamp them on afterwards -- the batch marker is what makes
        # a re-run detectable, and it must be on every row.
        opening = next(e for e in application.events if e.event_type == first_type.value)
        opening.note = first_note
        opening.payload = {"import_batch": BATCH_ID, "sheet_row": record["row"]}

        for event_type, occurred, note in timeline[1:]:
            await append_event(
                session,
                application=application,
                event_type=event_type,
                occurred_at=occurred,
                source=EventSource.MANUAL,
                note=note,
                payload={"import_batch": BATCH_ID, "sheet_row": record["row"]},
            )

        tally["applications"] += 1
        tally["events"] += len(timeline)
        tally["jobs_reused" if existing else "jobs_created"] += 1

    return tally


EXPECTED_STATUS = {
    "Submitted": "applied",
    "Rejected": "rejected",
    "Accepted": "accepted",
    "In Progress": "screening",
}


async def verify(session: AsyncSession, user_id: uuid.UUID, records: list[dict[str, Any]]) -> bool:
    """Check the derived state against the sheet, before anything is committed.

    Statuses are never written directly here -- they are folded out of the
    events by ``append_event``. That indirection is the right design but it
    means an error in the event plan surfaces as a wrong status rather than a
    crash, so the two are compared explicitly while the transaction can still
    be rolled back.
    """
    rows = (
        await session.execute(
            select(
                ApplicationEvent.payload["sheet_row"].astext.cast(sqltypes.Integer),
                Application.current_status,
                Application.applied_at,
            )
            .select_from(Application)
            .join(ApplicationEvent, ApplicationEvent.application_id == Application.id)
            .where(
                Application.user_id == user_id,
                ApplicationEvent.payload["import_batch"].astext == BATCH_ID,
                ApplicationEvent.event_type == EventType.APPLIED.value,
            )
        )
    ).all()

    by_row = {row: (status, applied_at) for row, status, applied_at in rows}
    problems = []
    for record in records:
        found = by_row.get(record["row"])
        if found is None:
            problems.append(f"row {record['row']}: not imported")
            continue
        status, applied_at = found
        expected = EXPECTED_STATUS[record["status"]]
        if status != expected:
            problems.append(f"row {record['row']}: status {status!r}, expected {expected!r}")
        if applied_at.date() != record["applied"]:
            problems.append(
                f"row {record['row']}: applied_at {applied_at.date()}, expected {record['applied']}"
            )

    if len(by_row) != len(records):
        problems.append(f"imported {len(by_row)} rows, sheet had {len(records)}")

    for problem in problems[:20]:
        print(f"  MISMATCH {problem}", file=sys.stderr)
    print(f"verified {len(records) - len(problems)}/{len(records)} rows against the sheet")
    return not problems


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="email of the owning user")
    parser.add_argument("--file", required=True, type=Path, help="path to the .xlsx")
    parser.add_argument(
        "--commit", action="store_true", help="write; without it the import is rolled back"
    )
    args = parser.parse_args()

    records = read_sheet(args.file)
    fill_missing_dates(records)

    statuses: dict[str, int] = {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    print(f"read {len(records)} rows: {statuses}")
    print(f"dates {min(r['applied'] for r in records)} -> {max(r['applied'] for r in records)}")

    user_id = await resolve_user_id(args.user)
    if user_id is None:
        print(f"no user with email {args.user}", file=sys.stderr)
        return 1

    async with SessionFactory() as session:
        # Scope the transaction the same way a request would, so every insert is
        # written through the same RLS policies the API runs under.
        await session.execute(
            text("SELECT set_config('app.user_id', :uid, true)"), {"uid": str(user_id)}
        )

        # Reading the profile back through RLS confirms the scope actually took
        # effect. If it had not, this returns nothing rather than importing 157
        # applications onto an unscoped connection.
        user = (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if user is None:
            print(
                f"user {user_id} is not visible under RLS; scope was not applied", file=sys.stderr
            )
            return 1

        seen = await already_imported(session, user_id)
        if seen:
            print(f"refusing: {seen} events from batch {BATCH_ID!r} are already present")
            return 1

        tally = await import_records(session, user_id, records)
        print(f"{tally} for {user.email} ({user.id})")

        if not await verify(session, user_id, records):
            await session.rollback()
            print("verification failed -- nothing written", file=sys.stderr)
            return 1

        if args.commit:
            await session.commit()
            print("committed")
        else:
            await session.rollback()
            print("dry run -- rolled back; re-run with --commit to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
