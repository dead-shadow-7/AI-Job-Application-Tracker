from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import TokenClaims, get_current_claims
from app.db.session import open_user_session
from app.models.user import User


async def get_db(
    claims: Annotated[TokenClaims, Depends(get_current_claims)],
) -> AsyncIterator[AsyncSession]:
    """RLS-scoped session for the authenticated caller.

    Routes must not call ``commit()`` — see ``open_user_session`` for why.
    """
    async for session in open_user_session(claims.user_id):
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db)]
Claims = Annotated[TokenClaims, Depends(get_current_claims)]


async def get_current_user(claims: Claims, session: DbSession) -> User:
    """Fetch the caller's profile, provisioning it on first login.

    Supabase owns identity; this table owns everything we add to it. A user
    exists in ``auth.users`` the moment they click the magic link, so the first
    authenticated request is the natural place to materialise the profile.

    Written as an upsert rather than get-then-insert: two requests arriving
    together on first login would otherwise race and one would fail on the
    primary key. The conflict branch also keeps ``email`` in sync when it
    changes upstream.
    """
    if not claims.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Auth provider did not supply an email address.",
        )

    stmt = (
        insert(User)
        .values(id=claims.user_id, email=claims.email)
        .on_conflict_do_update(index_elements=[User.id], set_={"email": claims.email})
        .returning(User)
    )
    result = await session.execute(stmt)
    return result.scalar_one()


CurrentUser = Annotated[User, Depends(get_current_user)]
