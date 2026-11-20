import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from time import monotonic, sleep
from uuid import UUID, uuid4
from unittest.mock import patch

from fastapi.testclient import TestClient
import psycopg
from psycopg import sql

from app import config, db, events, main, sessions, storage, worker
from test_pipeline import create


def test_owner_delete_and_files(client):
    analysis = create(client)
    url = f"/api/v1/analyses/{analysis['id']}"
    assert client.delete(url).status_code == 409
    worker_id = uuid4()
    worker.process_job(worker.claim_job(worker_id), worker_id)
    folder = main.STORAGE_DIR / analysis["id"]
    assert (folder / "mask.png").is_file()
    response = client.delete(url)
    assert response.status_code == 204 and response.content == b""
    assert client.get(url).status_code == 404
    assert client.get(url + "/files/original").status_code == 404
    assert client.delete(url).status_code == 404
    storage.cleanup_pending(main.STORAGE_DIR)
    assert not folder.exists()
    assert client.get("/api/v1/analyses").json()["total"] == 0


def test_deletion_retry_is_durable(client):
    analysis = create(client)
    url = f"/api/v1/analyses/{analysis['id']}"
    client.post(url + "/cancel")
    with patch.object(storage.shutil, "rmtree", side_effect=OSError("synthetic disk failure")):
        assert client.delete(url).status_code == 204
        with db.connect() as conn:
            assert conn.execute("SELECT 1 FROM file_deletions WHERE analysis_id = %s", (analysis["id"],)).fetchone()
        assert (main.STORAGE_DIR / analysis["id"]).is_dir()
    storage.cleanup_pending(main.STORAGE_DIR)
    assert not (main.STORAGE_DIR / analysis["id"]).exists()
    with db.connect() as conn:
        assert not conn.execute("SELECT 1 FROM file_deletions").fetchone()


def test_foreign_delete_and_origin(client):
    import secrets
    from app import users
    password = secrets.token_urlsafe(24)
    users.create_user("delete.other", "Other", password)
    analysis = create(client)
    url = f"/api/v1/analyses/{analysis['id']}"
    with TestClient(main.app) as other:
        assert other.post("/api/v1/auth/login", json={"username": "delete.other", "password": password}).status_code == 200
        assert other.delete(url).status_code == 404
    assert client.delete(url, headers={"Origin": "https://untrusted.example"}).status_code == 403
    assert client.get(url).status_code == 200


def test_active_limit_is_atomic(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_ACTIVE_ANALYSES", 2)
    def upload(_):
        from test_ownership import upload
        return upload(client)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(upload, range(3)))
    assert sorted(result.status_code for result in results) == [201, 201, 429]
    rejected = next(result for result in results if result.status_code == 429)
    assert rejected.json()["error"]["code"] == "TOO_MANY_ACTIVE_ANALYSES"
    assert client.get("/api/v1/analyses").json()["total"] == 2
    assert len(list(main.STORAGE_DIR.iterdir())) == 2
    first = next(result.json() for result in results if result.status_code == 201)
    client.post(f"/api/v1/analyses/{first['id']}/cancel")
    assert upload(0).status_code == 201


def test_global_queue_limit(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_QUEUED_ANALYSES", 1)
    create(client)
    from test_ownership import upload
    response = upload(client)
    assert response.status_code == 503 and response.json()["error"]["code"] == "QUEUE_FULL"
    assert response.headers["Retry-After"] == "5"
    assert client.get("/api/v1/analyses").json()["total"] == 1


def test_login_limit_and_window_reset(client):
    payload = {"username": "TEST.USER", "password": "wrong"}
    for _ in range(5):
        assert client.post("/api/v1/auth/login", json=payload).status_code == 401
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 429 and response.json()["error"]["code"] == "RATE_LIMITED"
    assert 1 <= int(response.headers["Retry-After"]) <= 900
    with db.connect() as conn:
        conn.execute("UPDATE login_limits SET window_started_at = clock_timestamp() - interval '901 seconds'")
    assert client.post("/api/v1/auth/login", json=payload).status_code == 401


def test_ip_limit_across_usernames(client, monkeypatch):
    monkeypatch.setattr(config, "LOGIN_IP_ATTEMPTS", 2)
    assert client.post("/api/v1/auth/login", json={"username": "missing.a", "password": "wrong"}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"username": "missing.b", "password": "wrong"}).status_code == 429


def test_request_limit_before_parser(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_REQUEST_BYTES", 64)
    for body in (b"x" * 65, (part for part in (b"x" * 40, b"x" * 40))):
        response = client.post("/api/v1/analyses", content=body)
        assert response.status_code == 413
        assert response.json()["error"]["requestId"] == response.headers["X-Request-ID"]
    assert client.get("/api/v1/analyses").json()["total"] == 0
    assert not list(main.STORAGE_DIR.iterdir())
    response = client.post("/api/v1/auth/login", content=b"x" * (config.MAX_LOGIN_BYTES + 1))
    assert response.status_code == 413


class ConnectedRequest:
    async def is_disconnected(self):
        return False


async def listener():
    connection = await psycopg.AsyncConnection.connect(db.DATABASE_URL, autocommit=True)
    cursor = await connection.execute("SELECT 'arterio_analyses_' || substr(md5(current_schema()),1,16)")
    channel = (await cursor.fetchone())[0]
    await connection.execute(sql.SQL("LISTEN {}").format(sql.Identifier(channel)))
    return connection


def test_sse_commit_privacy_and_close(client):
    user_id = UUID(client.get("/api/v1/auth/me").json()["id"])
    token = client.cookies[config.SESSION_COOKIE_NAME]
    async def check():
        connection = await listener()
        generator = events.event_stream(ConnectedRequest(), user_id, sessions.token_digest(token), connection)
        try:
            assert "event: resync" in await anext(generator)
            analysis = create(client)
            message = await asyncio.wait_for(anext(generator), timeout=3)
            assert "analysis.changed" in message and analysis["id"] in message
            foreign = {"userId": str(uuid4()), "analysisId": str(uuid4()), "queueChanged": True}
            name, data = events.notification_for_user(json.dumps(foreign), user_id)
            assert name == "queue.changed" and data == {}
            foreign["queueChanged"] = False
            assert events.notification_for_user(json.dumps(foreign), user_id) is None
            with db.connect() as conn:
                conn.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))
            client_id = UUID(analysis["id"])
            with db.connect() as conn:
                conn.execute("UPDATE analyses SET status = 'cancelled' WHERE id = %s", (client_id,))
            expired = await asyncio.wait_for(anext(generator), timeout=3)
            assert "session.expired" in expired
        finally:
            await generator.aclose()
        assert connection.closed
    asyncio.run(check())
    with TestClient(main.app) as anonymous:
        assert anonymous.get("/api/v1/analyses/events").status_code == 401


def test_notifications_are_transactional(client):
    analysis = create(client)
    async def check():
        connection = await listener()
        try:
            with db.connect() as writer:
                writer.execute("UPDATE analyses SET status = 'cancelled' WHERE id = %s", (analysis["id"],))
                before = [n async for n in connection.notifies(timeout=0.1, stop_after=1)]
                assert before == []
            after = [n async for n in connection.notifies(timeout=1, stop_after=1)]
            assert after and json.loads(after[0].payload)["analysisId"] == analysis["id"]
        finally:
            await connection.close()
    asyncio.run(check())


def test_stream_slots_release_on_disconnect(client):
    from app.errors import ApiError
    user_id = UUID(client.get("/api/v1/auth/me").json()["id"])
    async def check():
        connections = []
        try:
            for _ in range(config.SSE_MAX_PER_USER + 1):
                connections.append(await psycopg.AsyncConnection.connect(db.DATABASE_URL, autocommit=True))
            for connection in connections[:-1]:
                await events.reserve_stream(connection, user_id)
            try:
                await events.reserve_stream(connections[-1], user_id)
            except ApiError as exc:
                assert exc.status == 429
            else:
                raise AssertionError("Stream limit not enforced")
            await connections[0].close()
            await events.reserve_stream(connections[-1], user_id)
        finally:
            for connection in connections:
                await connection.close()
    asyncio.run(check())


def test_unknown_endpoint_uses_error_contract(client):
    response = client.get("/api/v1/not-a-route")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
    assert response.json()["error"]["requestId"] == response.headers["X-Request-ID"]
