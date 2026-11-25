from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from app import config
from app.auth import same_origin
from app.body_limit import BodyLimitMiddleware
from app.errors import ApiError
from app.schemas import ChangePasswordInput, LoginInput, PublicUser


async def echo(request):
    return JSONResponse({"size": len(await request.body())})


@pytest.fixture
def guarded(monkeypatch):
    monkeypatch.setattr(config, "MAX_REQUEST_BYTES", 100)
    monkeypatch.setattr(config, "MAX_LOGIN_BYTES", 10)
    app = Starlette(routes=[Route("/{path:path}", echo, methods=["GET", "POST", "DELETE"])])
    app.add_middleware(BodyLimitMiddleware)
    with TestClient(app) as client:
        yield client


def test_body_within_limit_reaches_application(guarded):
    response = guarded.post("/upload", content=b"x" * 100)
    assert response.status_code == 200 and response.json() == {"size": 100}


def test_declared_oversized_body_is_rejected(guarded):
    response = guarded.post("/upload", content=b"x" * 101)
    assert response.status_code == 413
    error = response.json()["error"]
    assert error["code"] == "REQUEST_TOO_LARGE" and error["details"] == {"maxRequestBytes": 100}


def test_streamed_oversized_body_is_rejected(guarded):
    response = guarded.post("/upload", content=(chunk for chunk in (b"x" * 60, b"x" * 60)))
    assert response.status_code == 413


def test_login_has_stricter_limit(guarded):
    assert guarded.post("/api/v1/auth/login", content=b"x" * 11).status_code == 413
    assert guarded.post("/upload", content=b"x" * 11).status_code == 200


@pytest.mark.parametrize("length", ["-1", "abc"])
def test_invalid_content_length_is_rejected(guarded, length):
    response = guarded.post("/upload", content=b"x", headers={"Content-Length": length})
    assert response.status_code == 400 and response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_read_requests_are_not_limited(guarded):
    assert guarded.get("/anything").status_code == 200


def request_with(headers):
    scope = {"type": "http", "method": "POST", "path": "/", "headers": [
        (name.lower().encode(), value.encode()) for name, value in headers.items()]}
    return Request(scope)


@pytest.mark.parametrize("headers", [
    {},
    {"Origin": config.APP_ORIGIN},
    {"Origin": config.APP_ORIGIN, "Sec-Fetch-Site": "same-origin"},
])
def test_same_origin_requests_are_allowed(headers):
    same_origin(request_with(headers))


@pytest.mark.parametrize("headers", [
    {"Origin": "https://untrusted.example"},
    {"Origin": config.APP_ORIGIN + ".untrusted.example"},
    {"Origin": "null"},
    {"Sec-Fetch-Site": "cross-site"},
])
def test_cross_origin_requests_are_rejected(headers):
    with pytest.raises(ApiError) as error:
        same_origin(request_with(headers))
    assert error.value.status == 403 and error.value.code == "FORBIDDEN_ORIGIN"


def test_login_input_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        LoginInput.model_validate({"username": "jan", "password": "secret", "isAdmin": True})


@pytest.mark.parametrize("payload", [
    {"username": "", "password": "secret"},
    {"username": "jan", "password": ""},
    {"username": "jan", "password": "x" * 129},
    {"username": "jan"},
])
def test_login_input_limits(payload):
    with pytest.raises(ValidationError):
        LoginInput.model_validate(payload)


def test_passwords_are_hidden_from_representation():
    login = LoginInput(username="jan", password="very secret value")
    change = ChangePasswordInput(current_password="old secret value", new_password="new secret value")
    assert "very secret value" not in repr(login)
    assert "secret value" not in repr(change)


def test_api_uses_camel_case():
    user = PublicUser(id="00000000-0000-0000-0000-000000000001", username="jan",
                      display_name="Jan", must_change_password=True)
    assert user.model_dump(by_alias=True).keys() == {"id", "username", "displayName", "mustChangePassword"}
    assert ChangePasswordInput.model_validate({"currentPassword": "a", "newPassword": "b"}).new_password == "b"
