"""Account creation shared by the administrative CLI and future auth code."""
import re
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from argon2 import PasswordHasher, Type
from psycopg.errors import UniqueViolation

from .db import connect

# Fixed parameters make the choice reproducible in the project report.
# Each call generates its own random salt; the encoded hash includes parameters.
password_hasher = PasswordHasher(
    time_cost=3, memory_cost=65536, parallelism=4,
    hash_len=32, salt_len=16, type=Type.ID,
)


class UserAlreadyExistsError(ValueError):
    pass


@dataclass(frozen=True)
class User:
    id: UUID
    username: str
    display_name: str
    created_at: datetime


def normalize_username(username: str) -> str:
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", username):
        raise ValueError(
            "Identyfikator musi mieć 3–64 znaki: litery a–z, cyfry, kropkę, "
            "myślnik lub podkreślenie; pierwszy znak musi być literą lub cyfrą."
        )
    return username


def validate_password(password: str) -> None:
    if not 12 <= len(password) <= 128 or password.isspace() or "\x00" in password:
        raise ValueError("Hasło musi mieć 12–128 znaków, nie może składać się z samych spacji ani zawierać NUL.")


def create_user(username: str, display_name: str, password: str, must_change_password: bool = False) -> User:
    username = normalize_username(username)
    display_name = display_name.strip()
    if not 1 <= len(display_name) <= 100 or not all(c.isprintable() for c in display_name):
        raise ValueError("Nazwa wyświetlana musi mieć 1–100 znaków i nie zawierać znaków sterujących.")
    validate_password(password)
    # Do not strip or normalize passwords. Only the encoded hash enters SQL.
    password_hash = password_hasher.hash(password)
    try:
        with connect() as conn:
            row = conn.execute(
                """INSERT INTO users (id, username, display_name, password_hash, must_change_password)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id, username, display_name, created_at""",
                (uuid4(), username, display_name, password_hash, must_change_password),
            ).fetchone()
    except UniqueViolation as exc:
        if exc.diag.constraint_name != "users_username_key":
            raise
        raise UserAlreadyExistsError("Konto o takim identyfikatorze już istnieje.") from None
    return User(**row)
