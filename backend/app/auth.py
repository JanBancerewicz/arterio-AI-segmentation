"""Authentication endpoints and shared session dependencies."""
from datetime import datetime, timezone
from math import ceil
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.security import APIKeyCookie

from . import config, sessions
from .errors import ApiError
from .schemas import ErrorResponse, LoginInput, PublicUser

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"], responses={
    422: {"model": ErrorResponse, "description": "VALIDATION_ERROR"},
})
cookie = APIKeyCookie(name=config.SESSION_COOKIE_NAME, auto_error=False)


def same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if request.headers.get("sec-fetch-site") == "cross-site" or (origin is not None and origin != config.APP_ORIGIN):
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


def require_user(request: Request, token: Annotated[str | None, Depends(cookie)]) -> PublicUser:
    """Authenticate and let the HTTP middleware refresh every response type."""
    session = sessions.current_session(token)
    if not session:
        raise ApiError(401, "UNAUTHENTICATED", "Sesja wygasła lub użytkownik nie jest zalogowany.")
    request.state.auth_session = session
    return session.user


@router.post("/login", response_model=PublicUser, dependencies=[Depends(same_origin)], responses={
    401: {"model": ErrorResponse, "description": "INVALID_CREDENTIALS"},
    403: {"model": ErrorResponse, "description": "FORBIDDEN_ORIGIN"},
})
def login(data: LoginInput, request: Request, response: Response):
    session = sessions.login(data.username, data.password, request.cookies.get(config.SESSION_COOKIE_NAME))
    if not session:
        raise ApiError(401, "INVALID_CREDENTIALS", "Nieprawidłowy identyfikator lub hasło.")
    set_session_cookie(response, session)
    return session.user


@router.get("/me", response_model=PublicUser, responses={
    401: {"model": ErrorResponse, "description": "UNAUTHENTICATED"},
})
def me(user: Annotated[PublicUser, Depends(require_user)]):
    return user


@router.post("/logout", status_code=204, dependencies=[Depends(same_origin)], responses={
    403: {"model": ErrorResponse, "description": "FORBIDDEN_ORIGIN"},
})
def logout(request: Request):
    sessions.logout(request.cookies.get(config.SESSION_COOKIE_NAME))
    response = Response(status_code=204)
    clear_session_cookie(response)
    return response
