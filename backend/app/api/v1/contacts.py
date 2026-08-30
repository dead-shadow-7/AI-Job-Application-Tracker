from uuid import UUID

from fastapi import APIRouter, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DbSession
from app.core.exceptions import NotFoundError
from app.models.contact import Contact

router = APIRouter(prefix="/contacts", tags=["contacts"])


class ContactBase(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    role: str | None = Field(default=None, max_length=120)
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=40)
    linkedin_url: str | None = None
    notes: str | None = None
    company_id: UUID | None = None
    application_id: UUID | None = None


class ContactRead(ContactBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID


class ContactUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    role: str | None = None
    email: str | None = None
    phone: str | None = None
    linkedin_url: str | None = None
    notes: str | None = None
    company_id: UUID | None = None
    application_id: UUID | None = None


@router.get("", response_model=list[ContactRead], summary="Your recruiter contacts")
async def list_contacts(
    user: CurrentUser, session: DbSession, application_id: UUID | None = None
) -> list[ContactRead]:
    stmt = select(Contact).where(Contact.user_id == user.id).order_by(Contact.name)
    if application_id:
        stmt = stmt.where(Contact.application_id == application_id)
    rows = (await session.execute(stmt)).scalars().all()
    return [ContactRead.model_validate(c) for c in rows]


@router.post(
    "", response_model=ContactRead, status_code=status.HTTP_201_CREATED, summary="Add a contact"
)
async def create_contact(
    payload: ContactBase, user: CurrentUser, session: DbSession
) -> ContactRead:
    contact = Contact(user_id=user.id, **payload.model_dump())
    session.add(contact)
    await session.flush()
    return ContactRead.model_validate(contact)


@router.patch("/{contact_id}", response_model=ContactRead, summary="Edit a contact")
async def update_contact(
    contact_id: UUID, payload: ContactUpdate, user: CurrentUser, session: DbSession
) -> ContactRead:
    contact = await _owned(session, contact_id, user.id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(contact, field, value)
    await session.flush()
    return ContactRead.model_validate(contact)


@router.delete("/{contact_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a contact")
async def delete_contact(contact_id: UUID, user: CurrentUser, session: DbSession) -> None:
    await session.delete(await _owned(session, contact_id, user.id))


async def _owned(session: AsyncSession, contact_id: UUID, user_id: UUID) -> Contact:
    contact = (
        await session.execute(
            select(Contact).where(Contact.id == contact_id, Contact.user_id == user_id)
        )
    ).scalar_one_or_none()
    if contact is None:
        raise NotFoundError(f"Contact {contact_id} not found")
    return contact
