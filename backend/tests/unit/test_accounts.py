import hashlib
import secrets

from argon2 import extract_parameters, Type
from argon2.exceptions import VerifyMismatchError
import pytest

from app.limits import user_bucket
from app.sessions import token_digest
from app.users import normalize_username, password_hasher, validate_password


@pytest.mark.parametrize("raw, expected", [
    ("jan.kowalski", "jan.kowalski"),
    ("  Jan.Kowalski  ", "jan.kowalski"),
    ("abc", "abc"),
    ("a" * 64, "a" * 64),
    ("dr_nowak-2", "dr_nowak-2"),
])
def test_valid_usernames_are_normalized(raw, expected):
    assert normalize_username(raw) == expected


@pytest.mark.parametrize("username", [
    "", "ab", "a" * 65, ".start", "-start", "_start", "jan kowalski",
    "jan@szpital", "zażółć", "jan/../admin", "jan;drop",
])
def test_invalid_usernames_are_rejected(username):
    with pytest.raises(ValueError):
        normalize_username(username)


@pytest.mark.parametrize("password", ["a" * 12, "a" * 128, "  spacje w środku  ", "zażółć gęślą jaźń"])
def test_valid_passwords(password):
    validate_password(password)


@pytest.mark.parametrize("password", ["a" * 11, "a" * 129, " " * 20, "\t" * 15, "abcdefghijk\x00l"])
def test_invalid_passwords(password):
    with pytest.raises(ValueError):
        validate_password(password)


def test_password_hash_is_argon2id_with_fixed_parameters():
    encoded = password_hasher.hash("correct horse battery")
    parameters = extract_parameters(encoded)
    assert encoded.startswith("$argon2id$")
    assert parameters.type is Type.ID
    assert (parameters.time_cost, parameters.memory_cost, parameters.parallelism) == (3, 65536, 4)
    assert "correct horse battery" not in encoded


def test_password_hash_is_salted_and_verifiable():
    first, second = password_hasher.hash("same password 1"), password_hasher.hash("same password 1")
    assert first != second
    assert password_hasher.verify(first, "same password 1")
    with pytest.raises(VerifyMismatchError):
        password_hasher.verify(first, "same password 2")


def test_login_bucket_ignores_case_and_whitespace():
    assert user_bucket(" Jan.Kowalski ") == user_bucket("jan.kowalski")
    assert user_bucket("jan.kowalski") != user_bucket("anna.nowak")
    assert "jan" not in user_bucket("jan.kowalski")


def test_session_token_digest():
    token = secrets.token_urlsafe(32)
    assert token_digest(token) == hashlib.sha256(token.encode()).hexdigest()
    assert token_digest(token) != token


@pytest.mark.parametrize("token", [None, "", "short", "x" * 44, "a" * 42 + "!", "ą" * 43])
def test_malformed_session_tokens_are_ignored(token):
    assert token_digest(token) is None
