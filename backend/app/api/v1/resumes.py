from datetime import datetime
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import CurrentUser, DbSession
from app.core.exceptions import InvalidOperationError
from app.models.resume import Resume
from app.services.resume_parser import extract_text
from app.services.resumes import (
    count_chunks,
    create_resume,
    get_resume,
    list_resumes,
    set_default,
)

router = APIRouter(prefix="/resumes", tags=["resumes"])


class ResumeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    label: str
    filename: str | None = None
    is_default: bool
    years_experience: Decimal | None = None
    created_at: datetime
    chunk_count: int = 0


def _read(resume: Resume, chunk_count: int = 0) -> ResumeRead:
    payload = ResumeRead.model_validate(resume)
    payload.chunk_count = chunk_count
    return payload


async def _read_one(session: DbSession, resume: Resume) -> ResumeRead:
    counts = await count_chunks(session, [resume.id])
    return _read(resume, counts.get(resume.id, 0))


@router.get("", response_model=list[ResumeRead], summary="Your uploaded resumes")
async def list_all(user: CurrentUser, session: DbSession) -> list[ResumeRead]:
    resumes = await list_resumes(session, user.id)
    counts = await count_chunks(session, [r.id for r in resumes])
    return [_read(r, counts.get(r.id, 0)) for r in resumes]


@router.post(
    "",
    response_model=ResumeRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a resume (PDF, DOCX, or text)",
)
async def upload(
    user: CurrentUser,
    session: DbSession,
    file: UploadFile = File(...),
    label: str = Form(default=""),
) -> ResumeRead:
    """Parse, chunk, and embed a resume.

    The uploaded file itself is **not** stored — only the extracted text. It is
    never needed again after parsing, and not keeping it is the simplest way not
    to leak the most personal document in the system. Embedding runs locally, so
    the resume never leaves this machine either.
    """
    data = await file.read()
    if not data:
        raise InvalidOperationError("That file is empty.")

    text = extract_text(file.filename or "resume.pdf", data)

    resume = await create_resume(
        session,
        user_id=user.id,
        label=label.strip() or (file.filename or "Resume"),
        filename=file.filename,
        text=text,
    )
    return await _read_one(session, resume)


@router.post("/{resume_id}/default", response_model=ResumeRead, summary="Make this the default")
async def make_default(resume_id: UUID, user: CurrentUser, session: DbSession) -> ResumeRead:
    resume = await get_resume(session, resume_id, user.id)
    await set_default(session, user_id=user.id, resume_id=resume.id)
    await session.flush()
    # Refresh this row only. set_default issues bulk UPDATEs that bypass the
    # identity map, so the in-memory object still shows the old flag —
    # but expire_all() would also expire `user`, and reading user.id afterwards
    # would then need a lazy round trip that async cannot make.
    await session.refresh(resume)
    return await _read_one(session, resume)


@router.delete("/{resume_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a resume")
async def delete(resume_id: UUID, user: CurrentUser, session: DbSession) -> None:
    await session.delete(await get_resume(session, resume_id, user.id))


class ResumeTextIn(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=200)


@router.post(
    "/text",
    response_model=ResumeRead,
    status_code=status.HTTP_201_CREATED,
    summary="Paste resume text instead of uploading a file",
)
async def upload_text(payload: ResumeTextIn, user: CurrentUser, session: DbSession) -> ResumeRead:
    """The fallback for scanned PDFs, whose words are an image rather than text
    and which no parser can read."""
    resume = await create_resume(
        session,
        user_id=user.id,
        label=payload.label,
        filename=None,
        text=payload.text,
    )
    return await _read_one(session, resume)
