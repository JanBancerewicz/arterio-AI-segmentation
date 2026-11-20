from io import BytesIO
import secrets
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import db, main, users, worker


def upload(client, **extra):
    image = BytesIO()
    Image.new("L", (256, 256), 180).save(image, format="PNG")
    return client.post("/api/v1/analyses", data={"modelId": "smoke-test", **extra},
                       files={"file": ("private.png", image.getvalue(), "image/png")})


@pytest.fixture
def other(client):
    password = secrets.token_urlsafe(24)
    users.create_user("other.user", "Other User", password)
    with TestClient(main.app) as second:
        assert second.post("/api/v1/auth/login", json={"username": "other.user", "password": password}).status_code == 200
        yield second


def test_anonymous_access_denied(client):
    analysis = upload(client).json()
    with TestClient(main.app) as anonymous:
        for path in ("/models", "/analyses", f"/analyses/{analysis['id']}",
                     f"/analyses/{analysis['id']}/files/original"):
            response = anonymous.get("/api/v1" + path)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "UNAUTHENTICATED"
        assert upload(anonymous).status_code == 401
        assert anonymous.post(f"/api/v1/analyses/{analysis['id']}/cancel").status_code == 401
        assert anonymous.get("/api/v1/health").status_code == 200


def test_owner_comes_from_session_and_history_is_private(client, other):
    owner = client.get("/api/v1/auth/me").json()
    stranger = other.get("/api/v1/auth/me").json()
    response = upload(client, userId=stranger["id"])
    assert response.status_code == 201
    analysis = response.json()
    assert "httponly" in response.headers["set-cookie"].lower()
    with db.connect() as conn:
        row = conn.execute("SELECT user_id FROM analyses WHERE id = %s", (analysis["id"],)).fetchone()
    assert str(row["user_id"]) == owner["id"]
    assert client.get("/api/v1/analyses").json()["total"] == 1
    for query in ("", "?status=queued&pageSize=1", "?page=10"):
        history = other.get("/api/v1/analyses" + query).json()
        assert history["items"] == [] and history["total"] == 0
    for path in (f"/analyses/{analysis['id']}", f"/analyses/{analysis['id']}/files/original"):
        assert other.get("/api/v1" + path).status_code == 404
    assert other.post(f"/api/v1/analyses/{analysis['id']}/cancel").status_code == 404
    assert client.get(f"/api/v1/analyses/{analysis['id']}").json()["status"] == "queued"
    assert client.post(f"/api/v1/analyses/{analysis['id']}/cancel").status_code == 200


def test_results_and_downloads_are_owner_only(client, other):
    analysis = upload(client).json()
    worker_id = uuid4()
    row = worker.claim_job(worker_id)
    worker.process_job(row, worker_id)
    assert client.get(f"/api/v1/analyses/{analysis['id']}").json()["status"] == "completed"
    for kind in ("original", "mask", "overlay"):
        for query in ("", "?download=1"):
            path = f"/api/v1/analyses/{analysis['id']}/files/{kind}" + query
            response = client.get(path)
            assert response.status_code == 200
            assert "httponly" in response.headers["set-cookie"].lower()
            assert response.headers["cache-control"] == "no-store"
            assert other.get(path).status_code == 404
    assert other.post(f"/api/v1/analyses/{analysis['id']}/cancel").status_code == 404
    assert client.post(f"/api/v1/analyses/{analysis['id']}/cancel").status_code == 409
    missing = other.get(f"/api/v1/analyses/{uuid4()}")
    foreign = other.get(f"/api/v1/analyses/{analysis['id']}")
    assert missing.status_code == foreign.status_code == 404
    assert missing.json()["error"]["code"] == foreign.json()["error"]["code"]


def test_post_origin_and_expired_session(client):
    analysis = upload(client).json()
    headers = {"Origin": "https://untrusted.example"}
    assert client.post(f"/api/v1/analyses/{analysis['id']}/cancel", headers=headers).status_code == 403
    image = BytesIO()
    Image.new("L", (256, 256), 0).save(image, format="PNG")
    assert client.post("/api/v1/analyses", data={"modelId": "smoke-test"},
                       files={"file": ("test.png", image.getvalue(), "image/png")}, headers=headers).status_code == 403
    with db.connect() as conn:
        conn.execute("UPDATE sessions SET idle_expires_at = clock_timestamp() - interval '1 second'")
    response = client.get(f"/api/v1/analyses/{analysis['id']}/files/original")
    assert response.status_code == 401
    assert "max-age=0" in response.headers["set-cookie"].lower()


def test_legacy_analyses_hidden_and_skipped_by_worker(client):
    legacy_id = uuid4()
    with db.connect() as conn:
        conn.execute("""INSERT INTO analyses (id, filename, model_id, status, width, height)
                        VALUES (%s, 'legacy.png', 'smoke-test', 'queued', 256, 256)""", (legacy_id,))
    assert client.get("/api/v1/analyses").json()["total"] == 0
    assert client.get(f"/api/v1/analyses/{legacy_id}").status_code == 404
    assert client.get(f"/api/v1/analyses/{legacy_id}/files/original").status_code == 404
    assert client.post(f"/api/v1/analyses/{legacy_id}/cancel").status_code == 404
    analysis = upload(client).json()
    assert analysis["queuePosition"] == 1
    row = worker.claim_job(uuid4())
    assert row["id"] == UUID(analysis["id"])
    assert worker.claim_job(uuid4()) is None
    with db.connect() as conn:
        assert conn.execute("SELECT status FROM analyses WHERE id = %s", (legacy_id,)).fetchone()["status"] == "queued"
