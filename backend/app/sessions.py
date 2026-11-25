"""Opaque browser tokens; only SHA-256 digests are stored in PostgreSQL."""
import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import datetime

from argon2.exceptions import InvalidHashError, VerificationError

from .config import SESSION_IDLE_SECONDS, SESSION_MAX_SECONDS
from .db import connect
from .schemas import PublicUser
from .users import normalize_username, password_hasher

# Unknown users still incur password verification, reducing username timing leaks.
_dummy_hash = password_hasher.hash(secrets.token_urlsafe(32))


@dataclass(frozen=True)
class Session:
    user: PublicUser
    token: str
    idle_expires_at: datetime


def token_digest(token: str | None) -> str | None:
    if not token or not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
        return None
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def login(username: str, password: str, previous_token: str | None = None) -> Session | None:
    try:
        normalized = normalize_username(username)
    except ValueError:
        normalized = ""  # Cannot match a valid username in the database.
    with connect() as conn:
        row = conn.execute(
            "SELECT id, username, display_name, must_change_password, password_hash FROM users WHERE username = %s",
            (normalized,),
        ).fetchone()
    try:
        password_hasher.verify(row["password_hash"] if row else _dummy_hash, password)
    except (VerificationError, InvalidHashError):
        return None
    if not row:
        return None
    replacement_hash = None
    if password_hasher.check_needs_rehash(row["password_hash"]):
        replacement_hash = password_hasher.hash(password)
    token = secrets.token_urlsafe(32)
    with connect() as conn:
        if replacement_hash:
            conn.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s AND password_hash = %s",
                (replacement_hash, row["id"], row["password_hash"]),
            )
        # Only successful login rotates the current browser session.
        previous_digest = token_digest(previous_token)
        if previous_digest:
            conn.execute("DELETE FROM sessions WHERE token_hash = %s", (previous_digest,))
        conn.execute("DELETE FROM sessions WHERE idle_expires_at <= clock_timestamp()")
        session = conn.execute(
            """INSERT INTO sessions (token_hash, user_id, expires_at, idle_expires_at)
               VALUES (%s, %s,
                       clock_timestamp() + (%s * interval '1 second'),
                       clock_timestamp() + (%s * interval '1 second'))
               RETURNING idle_expires_at""",
            (token_digest(token), row["id"], SESSION_MAX_SECONDS, SESSION_IDLE_SECONDS),
        ).fetchone()
    return Session(PublicUser.model_validate(row), token, session["idle_expires_at"])


def current_session(token: str | None) -> Session | None:
    digest = token_digest(token)
    if not digest:
        return None
    with connect() as conn:
        row = conn.execute(
            """UPDATE sessions AS s
               SET idle_expires_at = LEAST(s.expires_at,
                   clock_timestamp() + (%s * interval '1 second'))
               FROM users AS u
               WHERE s.token_hash = %s AND u.id = s.user_id
                 AND s.idle_expires_at > clock_timestamp()
                 AND s.expires_at > clock_timestamp()
               RETURNING u.id, u.username, u.display_name, u.must_change_password, s.idle_expires_at""",
            (SESSION_IDLE_SECONDS, digest),
        ).fetchone()
        if not row:
            conn.execute("DELETE FROM sessions WHERE token_hash = %s", (digest,))
            return None
    return Session(PublicUser.model_validate(row), token, row["idle_expires_at"])


def change_password(session: Session, current_password: str, new_password: str) -> PublicUser | None:
    with connect() as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = %s", (session.user.id,)).fetchone()
    try:
        password_hasher.verify(row["password_hash"] if row else _dummy_hash, current_password)
    except (VerificationError, InvalidHashError):
        return None
    if not row:
        return None
    new_hash = password_hasher.hash(new_password)
    with connect() as conn:
        user = conn.execute(
            """UPDATE users SET password_hash = %s, must_change_password = false,
                   password_changed_at = clock_timestamp()
               WHERE id = %s AND password_hash = %s
               RETURNING id, username, display_name, must_change_password""",
            (new_hash, session.user.id, row["password_hash"]),
        ).fetchone()
        if not user:
            return None
        # Sessions opened with the old password end; this browser stays signed in.
        conn.execute("DELETE FROM sessions WHERE user_id = %s AND token_hash <> %s",
                     (session.user.id, token_digest(session.token)))
    return PublicUser.model_validate(user)


def logout(token: str | None) -> None:
    digest = token_digest(token)
    if digest:
        with connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = %s", (digest,))
