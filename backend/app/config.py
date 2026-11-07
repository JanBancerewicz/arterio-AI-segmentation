import os
import math
import re
from pathlib import Path

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://arterio:arterio@localhost:5432/arterio"
)
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "backend/storage")).resolve()
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_FILE_BYTES + 64 * 1024
MAX_LOGIN_BYTES = 4096
MAX_ACTIVE_ANALYSES = 5
MAX_QUEUED_ANALYSES = 100
LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_USER_ATTEMPTS = 5
LOGIN_IP_ATTEMPTS = 30
SSE_HEARTBEAT_SECONDS = 15
SSE_SESSION_CHECK_SECONDS = 5
SSE_MAX_PER_USER = 3
SSE_MAX_TOTAL = 50
MODEL_ID = "smoke-test"
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints")).resolve()
UNET_CHECKPOINT_PATH = Path(os.environ.get("UNET_CHECKPOINT_PATH", str(CHECKPOINT_DIR / "unet_best.pth"))).resolve()
UNET_MODEL_VERSION = os.environ.get("UNET_MODEL_VERSION", "unversioned")
UNET_DEVICE = os.environ.get("UNET_DEVICE", "cpu")
if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", UNET_DEVICE):
    raise ValueError("UNET_DEVICE must be cpu, cuda or cuda:N")
UNET_THRESHOLD = float(os.environ.get("UNET_THRESHOLD", "0.5"))
if not math.isfinite(UNET_THRESHOLD) or not 0 < UNET_THRESHOLD < 1:
    raise ValueError("UNET_THRESHOLD must be between 0 and 1")
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
