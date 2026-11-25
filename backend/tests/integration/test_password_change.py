import secrets

import pytest
from fastapi.testclient import TestClient

from app import main, users


def login(client, username, password):
    return client.post("/api/v1/auth/login", json={"username": username, "password": password})


def change(client, current, new, **kwargs):
    return client.post("/api/v1/auth/password", json={"currentPassword": current, "newPassword": new}, **kwargs)


@pytest.fixture
def temporary(client):
    password = secrets.token_urlsafe(24)
    users.create_user("new.user", "New User", password, must_change_password=True)
    with TestClient(main.app) as fresh:
        assert login(fresh, "new.user", password).status_code == 200
        yield fresh, password


def test_temporary_password_blocks_api_until_changed(temporary):
    fresh, password = temporary
    me = fresh.get("/api/v1/auth/me")
    assert me.status_code == 200 and me.json()["mustChangePassword"] is True
    for path in ("/models", "/analyses", "/analyses/events"):
        response = fresh.get("/api/v1" + path)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "PASSWORD_CHANGE_REQUIRED"
    new_password = secrets.token_urlsafe(24)
    response = change(fresh, password, new_password)
    assert response.status_code == 200 and response.json()["mustChangePassword"] is False
    assert fresh.get("/api/v1/auth/me").json()["mustChangePassword"] is False
    assert fresh.get("/api/v1/models").status_code == 200
    with TestClient(main.app) as other:
        assert login(other, "new.user", password).status_code == 401
        assert login(other, "new.user", new_password).status_code == 200


def test_invalid_password_changes_are_rejected(temporary):
    fresh, password = temporary
    wrong = change(fresh, "wrong-password-123", secrets.token_urlsafe(24))
    assert wrong.status_code == 403 and wrong.json()["error"]["code"] == "INVALID_CURRENT_PASSWORD"
    for new_password in ("short", " " * 20, password):
        response = change(fresh, password, new_password)
        assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_NEW_PASSWORD"
    foreign = change(fresh, password, secrets.token_urlsafe(24), headers={"Origin": "https://untrusted.example"})
    assert foreign.status_code == 403 and foreign.json()["error"]["code"] == "FORBIDDEN_ORIGIN"
    with TestClient(main.app) as anonymous:
        assert change(anonymous, password, secrets.token_urlsafe(24)).status_code == 401
    assert fresh.get("/api/v1/auth/me").json()["mustChangePassword"] is True


def test_password_change_signs_out_other_sessions(temporary):
    fresh, password = temporary
    with TestClient(main.app) as second:
        assert login(second, "new.user", password).status_code == 200
        assert change(fresh, password, secrets.token_urlsafe(24)).status_code == 200
        assert second.get("/api/v1/auth/me").status_code == 401
    assert fresh.get("/api/v1/auth/me").status_code == 200


def test_current_password_guessing_is_rate_limited(temporary):
    fresh, password = temporary
    for _ in range(5):
        assert change(fresh, "wrong-password-123", secrets.token_urlsafe(24)).status_code == 403
    response = change(fresh, password, secrets.token_urlsafe(24))
    assert response.status_code == 429 and response.json()["error"]["code"] == "RATE_LIMITED"
