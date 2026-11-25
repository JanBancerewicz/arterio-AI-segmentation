from io import BytesIO
import secrets

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app import config, db, main, users


def png(size=(256, 256), **params):
    buffer = BytesIO()
    Image.new("L", size, 90).save(buffer, format="PNG", **params)
    return buffer.getvalue()


def upload(client, content, filename="frame.png"):
    return client.post("/api/v1/analyses", data={"modelId": "smoke-test"},
                       files={"file": (filename, content, "image/png")})


@pytest.fixture
def account(client):
    password = secrets.token_urlsafe(24)
    users.create_user("security.user", "Security User", password)
    return "security.user", password


def test_session_cookie_attributes(account):
    username, password = account
    with TestClient(main.app) as browser:
        response = browser.post("/api/v1/auth/login", json={"username": username, "password": password})
    cookie = response.headers["set-cookie"].lower()
    assert response.status_code == 200
    assert cookie.startswith(config.SESSION_COOKIE_NAME + "=")
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/api" in cookie
    assert "max-age=" in cookie


def test_credentials_are_never_stored_in_plain_text(account):
    username, password = account
    with TestClient(main.app) as browser:
        browser.post("/api/v1/auth/login", json={"username": username, "password": password})
        token = browser.cookies[config.SESSION_COOKIE_NAME]
    with db.connect() as conn:
        user = conn.execute("SELECT password_hash FROM users WHERE username = %s", (username,)).fetchone()
        tokens = [row["token_hash"] for row in conn.execute("SELECT token_hash FROM sessions").fetchall()]
    assert user["password_hash"].startswith("$argon2id$") and password not in user["password_hash"]
    assert token not in tokens and all(len(value) == 64 for value in tokens)


def test_login_does_not_reveal_existing_accounts(account):
    username, _ = account
    with TestClient(main.app) as browser:
        wrong_password = browser.post("/api/v1/auth/login", json={"username": username, "password": "wrong-password-1"})
        unknown_user = browser.post("/api/v1/auth/login", json={"username": "nobody.here", "password": "wrong-password-1"})
    assert wrong_password.status_code == unknown_user.status_code == 401
    first, second = wrong_password.json()["error"], unknown_user.json()["error"]
    assert (first["code"], first["message"]) == (second["code"], second["message"])


def test_login_rotates_session_token(account):
    username, password = account
    with TestClient(main.app) as browser:
        browser.post("/api/v1/auth/login", json={"username": username, "password": password})
        before = browser.cookies[config.SESSION_COOKIE_NAME]
        browser.post("/api/v1/auth/login", json={"username": username, "password": password})
        after = browser.cookies[config.SESSION_COOKIE_NAME]
    assert before != after
    with TestClient(main.app, cookies={config.SESSION_COOKIE_NAME: before}) as attacker:
        assert attacker.get("/api/v1/auth/me").status_code == 401


def test_logout_revokes_session_on_server(client):
    token = client.cookies[config.SESSION_COOKIE_NAME]
    response = client.post("/api/v1/auth/logout")
    assert response.status_code == 204 and "max-age=0" in response.headers["set-cookie"].lower()
    with TestClient(main.app, cookies={config.SESSION_COOKIE_NAME: token}) as replay:
        assert replay.get("/api/v1/auth/me").status_code == 401


@pytest.mark.parametrize("token", ["", "forged", "A" * 43, "' OR '1'='1"])
def test_forged_session_tokens_are_rejected(client, token):
    with TestClient(main.app, cookies={config.SESSION_COOKIE_NAME: token}) as forged:
        response = forged.get("/api/v1/analyses")
    assert response.status_code == 401 and response.json()["error"]["code"] == "UNAUTHENTICATED"


def test_cross_site_fetch_metadata_is_rejected(client):
    response = client.post("/api/v1/auth/logout", headers={"Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 403 and response.json()["error"]["code"] == "FORBIDDEN_ORIGIN"
    assert client.get("/api/v1/auth/me").status_code == 200


@pytest.mark.parametrize("username", ["' OR '1'='1", "admin'--", "x\"; DROP TABLE users; --"])
def test_sql_injection_in_login_is_harmless(client, username):
    with TestClient(main.app) as browser:
        response = browser.post("/api/v1/auth/login", json={"username": username, "password": "' OR '1'='1"})
    assert response.status_code == 401
    with db.connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM users").fetchone()["n"] == 1


@pytest.mark.parametrize("query", ["?status=queued' OR '1'='1", "?status=queued;DROP", "?page=0", "?pageSize=1000"])
def test_malicious_query_parameters_are_rejected(client, query):
    response = client.get("/api/v1/analyses" + query)
    assert response.status_code == 422 and response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_mass_assignment_is_rejected(client):
    response = client.post("/api/v1/auth/login", json={"username": "test.user", "password": "x", "mustChangePassword": False})
    assert response.status_code == 422


@pytest.mark.parametrize("filename, expected", [
    ("../../etc/passwd.png", "passwd.png"),
    ("..\\..\\windows\\system.png", "system.png"),
    ("scan‮gnp.png", "scangnp.png"),
    ("../", "image.png"),
])
def test_uploaded_filename_cannot_escape_storage(client, filename, expected):
    response = upload(client, png(), filename)
    assert response.status_code == 201
    analysis = response.json()
    assert analysis["filename"] == expected
    stored = sorted(path.relative_to(main.STORAGE_DIR).as_posix() for path in main.STORAGE_DIR.rglob("*") if path.is_file())
    assert stored == [f"{analysis['id']}/original.png"]


@pytest.mark.parametrize("content, code", [
    (png((255, 512)), "INVALID_IMAGE_SIZE"),
    (png((4097, 256)), "INVALID_IMAGE_SIZE"),
    (b"GIF89a" + b"\x00" * 100, "UNSUPPORTED_FILE_TYPE"),
    (b"<?php system($_GET['c']); ?>", "UNSUPPORTED_FILE_TYPE"),
])
def test_dangerous_uploads_are_rejected(client, content, code):
    response = upload(client, content)
    assert response.json()["error"]["code"] == code
    assert not list(main.STORAGE_DIR.iterdir())


def test_animated_png_is_rejected(client):
    buffer = BytesIO()
    frames = [Image.new("L", (256, 256), value) for value in (0, 255)]
    frames[0].save(buffer, format="PNG", save_all=True, append_images=frames[1:])
    response = upload(client, buffer.getvalue())
    assert response.status_code == 422 and response.json()["error"]["code"] == "IMAGE_ANIMATED"


def test_decompression_bomb_is_rejected(client, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    response = upload(client, png((256, 256)))
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_IMAGE"


@pytest.mark.parametrize("path", ["/analyses/not-a-uuid", "/analyses/00000000-0000-0000-0000-000000000000/files/..%2F..%2Fetc%2Fpasswd"])
def test_malformed_resource_paths(client, path):
    response = client.get("/api/v1" + path)
    assert response.status_code in (404, 422)
    assert "error" in response.json()


def test_internal_errors_do_not_leak_details(client, monkeypatch):
    def explode(*_):
        raise RuntimeError("secret database password in stack trace")
    monkeypatch.setattr(main, "find_analysis", explode)
    with TestClient(main.app, raise_server_exceptions=False, cookies=client.cookies) as browser:
        response = browser.get("/api/v1/analyses/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in response.text


def test_responses_are_not_cached(client):
    for path in ("/auth/me", "/analyses", "/models"):
        assert client.get("/api/v1" + path).headers["cache-control"] == "no-store"
