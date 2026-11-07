"""Local prototype with authentication; analysis ownership is added next."""
import logging
import shutil
import warnings
from io import BytesIO
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from .config import MAX_FILE_BYTES, MODEL_ID, STORAGE_DIR
from .auth import clear_session_cookie, router as auth_router
from .db import connect
from .errors import ApiError
from .repository import serialize
from .schemas import Analysis, AnalysisPage, SegmentationModel

log = logging.getLogger(__name__)
app = FastAPI(
    title="Arterio — infrastructure prototype",
    version="0.2.0",
    description="Local smoke-test API with session endpoints. Analyses are still public; no AI inference. Do not expose publicly.",
)
app.include_router(auth_router)


@app.middleware("http")
async def request_metadata(request: Request, call_next):
    request.state.request_id = str(uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["Cache-Control"] = "no-store"
    return response


def error_response(request, status, code, message, details=None):
    return JSONResponse(status_code=status, content={"error": {
        "code": code, "message": message, "details": details,
        "requestId": getattr(request.state, "request_id", str(uuid4())),
    }}, headers={"Cache-Control": "no-store"})


@app.exception_handler(ApiError)
async def api_error(request: Request, exc: ApiError):
    response = error_response(request, exc.status, exc.code, exc.message, exc.details)
    if exc.code == "UNAUTHENTICATED":
        clear_session_cookie(response)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    fields = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors()]
    return error_response(request, 422, "VALIDATION_ERROR", "Niepoprawne dane żądania.", {"fields": fields})


@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception):
    log.exception("Request %s failed", getattr(request.state, "request_id", "unknown"))
    return error_response(request, 500, "INTERNAL_ERROR", "Błąd serwera.")


@app.get("/api/v1/health")
def health():
    with connect() as conn:
        count = conn.execute(
            "SELECT count(*) AS n FROM workers WHERE heartbeat_at > clock_timestamp() - interval '10 seconds'"
        ).fetchone()["n"]
        queued = conn.execute("SELECT count(*) AS n FROM analyses WHERE status = 'queued'").fetchone()["n"]
    return {"status": "ok" if count else "degraded", "workersOnline": count, "queueLength": queued, "mode": "smoke-test"}


@app.get("/api/v1/models", response_model=list[SegmentationModel])
def models():
    return [SegmentationModel(
        id=MODEL_ID, name="Test infrastruktury — bez modelu AI", version="0.1.0",
        description="Próbny worker progowania obrazu. Wynik nie jest segmentacją naczyń.",
        is_default=True, available=True,
    )]


@app.post("/api/v1/analyses", status_code=201, response_model=Analysis)
def create_analysis(file: Annotated[UploadFile, File()], model_id: Annotated[str, Form(alias="modelId")]):
    if model_id != MODEL_ID:
        raise ApiError(422, "MODEL_NOT_FOUND", "Nieznany model.")
    data = file.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ApiError(413, "FILE_TOO_LARGE", "Maksymalny rozmiar pliku to 10 MiB.")
    if not data:
        raise ApiError(422, "INVALID_IMAGE", "Plik jest pusty.")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ApiError(415, "UNSUPPORTED_FILE_TYPE", "Wymagany jest obraz PNG.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                if min(width, height) < 256 or max(width, height) > 4096:
                    raise ApiError(422, "INVALID_IMAGE_SIZE", "Każdy bok obrazu musi mieć od 256 do 4096 pikseli.")
                if getattr(image, "n_frames", 1) != 1:
                    raise ApiError(422, "IMAGE_ANIMATED", "Wymagana jest pojedyncza klatka PNG.")
                image.verify()
            with Image.open(BytesIO(data)) as image:
                image.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ApiError(422, "INVALID_IMAGE", "Nie można odczytać obrazu PNG.") from None
    analysis_id = uuid4()
    folder = STORAGE_DIR / str(analysis_id)
    filename = (file.filename or "image.png").replace("\\", "/").split("/")[-1]
    filename = "".join(c for c in filename if c.isprintable())[:255] or "image.png"
    try:
        folder.mkdir(parents=True)
        (folder / "original.png").write_bytes(data)
        with connect() as conn:
            row = conn.execute(
                """INSERT INTO analyses (id, filename, model_id, status, width, height)
                   VALUES (%s, %s, %s, 'queued', %s, %s) RETURNING *""",
                (analysis_id, filename, model_id, width, height),
            ).fetchone()
            result = serialize(conn, row)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return JSONResponse(status_code=201, content=result.model_dump(mode="json", by_alias=True),
                        headers={"Location": f"/api/v1/analyses/{analysis_id}"})


@app.get("/api/v1/analyses", response_model=AnalysisPage)
def list_analyses(page: Annotated[int, Query(ge=1)] = 1,
                  page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
                  status: str | None = None):
    statuses = status.split(",") if status else ["queued", "processing", "completed", "failed", "cancelled"]
    if not set(statuses) <= {"queued", "processing", "completed", "failed", "cancelled"}:
        raise ApiError(422, "VALIDATION_ERROR", "Niepoprawny filtr statusu.")
    with connect() as conn:
        total = conn.execute("SELECT count(*) AS n FROM analyses WHERE status = ANY(%s)", (statuses,)).fetchone()["n"]
        rows = conn.execute(
            "SELECT * FROM analyses WHERE status = ANY(%s) ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
            (statuses, page_size, (page - 1) * page_size),
        ).fetchall()
        return AnalysisPage(items=[serialize(conn, row) for row in rows], page=page, page_size=page_size, total=total)


def find_analysis(conn, analysis_id):
    row = conn.execute("SELECT * FROM analyses WHERE id = %s", (analysis_id,)).fetchone()
    if not row:
        raise ApiError(404, "ANALYSIS_NOT_FOUND", "Nie znaleziono analizy.")
    return row


@app.get("/api/v1/analyses/{analysis_id}", response_model=Analysis)
def get_analysis(analysis_id: UUID):
    with connect() as conn:
        return serialize(conn, find_analysis(conn, analysis_id))


@app.post("/api/v1/analyses/{analysis_id}/cancel", response_model=Analysis)
def cancel_analysis(analysis_id: UUID):
    with connect() as conn:
        row = conn.execute(
            """UPDATE analyses SET status = 'cancelled', finished_at = clock_timestamp()
               WHERE id = %s AND status = 'queued' RETURNING *""", (analysis_id,),
        ).fetchone()
        row = row or find_analysis(conn, analysis_id)
        if row["status"] != "cancelled":
            raise ApiError(409, "ANALYSIS_NOT_CANCELLABLE", "Można anulować tylko zadanie oczekujące.", {"status": row["status"]})
        return serialize(conn, row)


@app.get("/api/v1/analyses/{analysis_id}/files/{kind}")
def get_file(analysis_id: UUID, kind: Literal["original", "mask", "overlay"], download: bool = False):
    with connect() as conn:
        row = find_analysis(conn, analysis_id)
    if kind != "original" and row["status"] != "completed":
        raise ApiError(409, "RESULT_NOT_READY", "Wynik nie jest dostępny.")
    path = STORAGE_DIR / str(analysis_id) / f"{kind}.png"
    if not path.is_file():
        raise ApiError(404, "FILE_NOT_FOUND", "Nie znaleziono pliku.")
    return FileResponse(path, media_type="image/png", filename=f"{kind}.png",
                        content_disposition_type="attachment" if download else "inline")
