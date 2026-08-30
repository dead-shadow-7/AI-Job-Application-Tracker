"""Domain errors.

Services raise these rather than HTTPException so the business rules stay
usable from the Phase 4 agent tools and the scheduled sweep, neither of which
is an HTTP request. ``app.main`` maps them onto status codes at the edge.
"""


class DomainError(Exception):
    """Base for anything the caller did wrong, as opposed to a bug."""


class NotFoundError(DomainError):
    """The requested row does not exist, or is not visible to this user.

    Deliberately not distinguished from "forbidden": telling a caller that a row
    exists but is not theirs confirms the existence of another user's data.
    """


class ConflictError(DomainError):
    """The request contradicts current state — a duplicate, or a lost update."""


class InvalidOperationError(DomainError):
    """Well-formed input that the domain rules refuse."""
