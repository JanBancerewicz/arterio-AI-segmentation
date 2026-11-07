import os
from pathlib import Path

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://arterio:arterio@localhost:5432/arterio"
)
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "backend/storage")).resolve()
MAX_FILE_BYTES = 10 * 1024 * 1024
MODEL_ID = "smoke-test"
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", "120"))
if JOB_TIMEOUT_SECONDS < 1:
    raise ValueError("JOB_TIMEOUT_SECONDS must be positive")
WORKER_HEARTBEAT_SECONDS = 2
WORKER_STALE_SECONDS = 10

APP_ORIGIN = os.environ.get("APP_ORIGIN", "http://localhost:8000").rstrip("/")
SESSION_COOKIE_NAME = "arterio_session"
SESSION_COOKIE_PATH = "/api"
_cookie_secure = os.environ.get("SESSION_COOKIE_SECURE", "false").lower()
if _cookie_secure not in ("true", "false"):
    raise ValueError("SESSION_COOKIE_SECURE must be true or false")
SESSION_COOKIE_SECURE = _cookie_secure == "true"
SESSION_IDLE_SECONDS = 8 * 60 * 60
SESSION_MAX_SECONDS = 7 * 24 * 60 * 60
