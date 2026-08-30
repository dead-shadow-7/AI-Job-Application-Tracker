"""Supabase JWT verification.

Supabase projects sign access tokens one of two ways depending on vintage:

* **Asymmetric (current)** — ES256/RS256, public keys published at the project's
  JWKS endpoint. Preferred: the backend never holds a signing secret.
* **Symmetric (legacy)** — HS256 with the shared project JWT secret.

Which path is used is decided by whether ``SUPABASE_JWT_SECRET`` is set, so a
project of either vintage works without a code change.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from app.core.config import settings

logger = logging.getLogger(__name__)

# auto_error=False so we can raise our own 401 with a WWW-Authenticate header
# rather than FastAPI's bare 403.
bearer_scheme = HTTPBearer(auto_error=False)

ASYMMETRIC_ALGORITHMS = ["ES256", "RS256"]


@dataclass(frozen=True, slots=True)
class TokenClaims:
    user_id: UUID
    email: str | None
    role: str | None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


@lru_cache
def _jwk_client() -> jwt.PyJWKClient:
    if not settings.supabase_url:
        raise RuntimeError(
            "SUPABASE_URL must be set when SUPABASE_JWT_SECRET is blank "
            "(asymmetric JWT verification needs the JWKS endpoint)."
        )
    # PyJWKClient caches fetched keys in-process; cache_keys handles rotation.
    return jwt.PyJWKClient(settings.jwks_url, cache_keys=True)


def _decode_sync(token: str) -> dict:
    """Blocking verification. Called via a threadpool — the JWKS fetch is network I/O.

    ``require`` is set so a token missing ``exp`` or ``sub`` is rejected rather
    than silently treated as non-expiring or anonymous. The literal is repeated
    at both call sites because PyJWT types the parameter as a TypedDict, which a
    shared ``dict`` variable does not narrow to.
    """
    if settings.use_symmetric_jwt:
        return jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience=settings.supabase_jwt_audience,
            options={"require": ["exp", "sub"]},
        )
    signing_key = _jwk_client().get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=ASYMMETRIC_ALGORITHMS,
        audience=settings.supabase_jwt_audience,
        options={"require": ["exp", "sub"]},
    )


async def verify_token(token: str) -> TokenClaims:
    try:
        payload = await run_in_threadpool(_decode_sync, token)
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("Token has expired") from exc
    except jwt.InvalidAudienceError as exc:
        raise _unauthorized("Token audience mismatch") from exc
    except jwt.PyJWTError as exc:
        # Deliberately vague to the caller; the detail goes to logs only.
        logger.warning("JWT rejected: %s", exc)
        raise _unauthorized("Could not validate credentials") from exc

    try:
        user_id = UUID(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise _unauthorized("Token subject is not a valid user id") from exc

    return TokenClaims(
        user_id=user_id,
        email=payload.get("email"),
        role=payload.get("role"),
    )


async def get_current_claims(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> TokenClaims:
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Not authenticated")
    return await verify_token(credentials.credentials)
