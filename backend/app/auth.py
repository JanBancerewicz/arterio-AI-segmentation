"""Authentication endpoints and shared session dependencies."""
from datetime import datetime, timezone
from math import ceil
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.security import APIKeyCookie

from . import config, sessions
from .errors import ApiError
from .schemas import ChangePasswordInput, ErrorResponse, LoginInput, PublicUser
from .limits import login_succeeded, reserve_login
from .users import validate_password

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"], responses={
    422: {"model": ErrorResponse, "description": "VALIDATION_ERROR"},
})
cookie = APIKeyCookie(name=config.SESSION_COOKIE_NAME, auto_error=False)


def same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    allowed = origin == config.APP_ORIGIN or origin in config.APP_EXTRA_ORIGINS
    if request.headers.get("sec-fetch-site") == "cross-site" or (origin is not None and not allowed):
        raise ApiError(403, "FORBIDDEN_ORIGIN", "Żądanie pochodzi z niedozwolonego źródła.")


def set_session_cookie(response: Response, session: sessions.Session) -> None:
    remaining = max(0, ceil((session.idle_expires_at - datetime.now(timezone.utc)).total_seconds()))
    response.set_cookie(
        config.SESSION_COOKIE_NAME, session.token, max_age=remaining,
        path=config.SESSION_COOKIE_PATH, httponly=True, samesite="lax",
        secure=config.SESSION_COOKIE_SECURE,
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        config.SESSION_COOKIE_NAME, path=config.SESSION_COOKIE_PATH,
        httponly=True, samesite="lax", secure=config.SESSION_COOKIE_SECURE,
    )


def require_session(request: Request, token: Annotated[str | None, Depends(cookie)]) -> sessions.Session:
    """Authenticate and let the HTTP middleware refresh every response type."""
    session = sessions.current_session(token)
    if not session:
        raise ApiError(401, "UNAUTHENTICATED", "Sesja wygasła lub użytkownik nie jest zalogowany.")
    request.state.auth_session = session
    return session


def require_user(session: Annotated[sessions.Session, Depends(require_session)]) -> PublicUser:
    if session.user.must_change_password:
        raise ApiError(403, "PASSWORD_CHANGE_REQUIRED", "Przed dalszą pracą zmień hasło tymczasowe.")
    return session.user


@router.post("/login", response_model=PublicUser, dependencies=[Depends(same_origin)], responses={
    401: {"model": ErrorResponse, "description": "INVALID_CREDENTIALS"},
    403: {"model": ErrorResponse, "description": "FORBIDDEN_ORIGIN"},
    429: {"model": ErrorResponse, "description": "RATE_LIMITED"},
})
def login(data: LoginInput, request: Request, response: Response):
    reserve_login(data.username, request.client.host if request.client else "unknown")
    session = sessions.login(data.username, data.password, request.cookies.get(config.SESSION_COOKIE_NAME))
    if not session:
        raise ApiError(401, "INVALID_CREDENTIALS", "Nieprawidłowy identyfikator lub hasło.")
    login_succeeded(data.username)
    set_session_cookie(response, session)
    return session.user


@router.get("/me", response_model=PublicUser, responses={
    401: {"model": ErrorResponse, "description": "UNAUTHENTICATED"},
})
def me(session: Annotated[sessions.Session, Depends(require_session)]):
    return session.user


@router.post("/password", response_model=PublicUser, dependencies=[Depends(same_origin)], responses={
    401: {"model": ErrorResponse, "description": "UNAUTHENTICATED"},
    403: {"model": ErrorResponse, "description": "FORBIDDEN_ORIGIN or INVALID_CURRENT_PASSWORD"},
    429: {"model": ErrorResponse, "description": "RATE_LIMITED"},
})
def change_password(data: ChangePasswordInput, request: Request,
                    session: Annotated[sessions.Session, Depends(require_session)]):
    try:
        validate_password(data.new_password)
    except ValueError as exc:
        raise ApiError(422, "INVALID_NEW_PASSWORD", str(exc)) from None
    if data.new_password == data.current_password:
        raise ApiError(422, "INVALID_NEW_PASSWORD", "Nowe hasło musi różnić się od obecnego.")
    reserve_login(session.user.username, request.client.host if request.client else "unknown")
    user = sessions.change_password(session, data.current_password, data.new_password)
    if not user:
        raise ApiError(403, "INVALID_CURRENT_PASSWORD", "Obecne hasło jest nieprawidłowe.")
    login_succeeded(session.user.username)
    return user


@router.post("/logout", status_code=204, dependencies=[Depends(same_origin)], responses={
    403: {"model": ErrorResponse, "description": "FORBIDDEN_ORIGIN"},
})
def logout(request: Request):
    sessions.logout(request.cookies.get(config.SESSION_COOKIE_NAME))
    response = Response(status_code=204)
    clear_session_cookie(response)
    return response
