from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession
from app.schemas.user import UserRead, UserUpdate

router = APIRouter(prefix="/me", tags=["me"])


@router.get("", response_model=UserRead, summary="Current user's profile")
async def read_me(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user)


@router.patch("", response_model=UserRead, summary="Update the current user's profile")
async def update_me(payload: UserUpdate, user: CurrentUser, session: DbSession) -> UserRead:
    data = payload.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(user, field, value)
    await session.flush()
    await session.refresh(user)
    return UserRead.model_validate(user)
