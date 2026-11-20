from uuid import uuid4
import secrets

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app import db, main, migrate, users, worker


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Real PostgreSQL, isolated schema and storage for each test."""
    original_url = db.DATABASE_URL
    schema = "test_" + uuid4().hex
    with psycopg.connect(original_url) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    monkeypatch.setattr(db, "DATABASE_URL", make_conninfo(original_url, options=f"-csearch_path={schema}"))
    monkeypatch.setattr(main, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(worker, "STORAGE_DIR", tmp_path)
    try:
        migrate.main()
        password = secrets.token_urlsafe(24)
        users.create_user("test.user", "Test User", password)
        with TestClient(main.app) as test_client:
            response = test_client.post("/api/v1/auth/login", json={"username": "test.user", "password": password})
            assert response.status_code == 200
            yield test_client
    finally:
        with psycopg.connect(original_url) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
