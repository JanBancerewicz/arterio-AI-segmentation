import os
from pathlib import Path

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://arterio:arterio@localhost:5432/arterio"
)
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "backend/storage")).resolve()
MAX_FILE_BYTES = 10 * 1024 * 1024
MODEL_ID = "smoke-test"
JOB_TIMEOUT_SECONDS = 120
