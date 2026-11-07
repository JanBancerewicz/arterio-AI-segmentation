"""Local prototype with session authentication and per-user analyses."""
import logging
import shutil
import warnings
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import MAX_FILE_BYTES, STORAGE_DIR, WORKER_STALE_SECONDS
from .auth import clear_session_cookie, require_user, router as auth_router, same_origin, set_session_cookie
from .db import connect
from .errors import ApiError
from .repository import serialize
from .schemas import Analysis, AnalysisPage, ErrorResponse, PublicUser, SegmentationModel
from .worker_monitor import monitor_lifetime
from .models import UNET_ID, catalogue, find_model
from .body_limit import BodyLimitMiddleware
from .events import router as events_router
from .limits import check_analysis_limits
from .storage import cleanup_lifetime, cleanup_pending

CurrentUser = Annotated[PublicUser, Depends(require_user)]
AUTH_ERRORS = {401: {"model": ErrorResponse, "description": "UNAUTHENTICATED"}}
ANALYSIS_ERRORS = {**AUTH_ERRORS, 404: {"model": ErrorResponse, "description": "ANALYSIS_NOT_FOUND"}}

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with monitor_lifetime(), cleanup_lifetime(STORAGE_DIR):
        yield


app = FastAPI(
    title="Arterio — infrastructure prototype",
    version="0.4.0",
    description="Local API with sessions, owner-only analyses and optional U-Net inference. Production hardening is incomplete.",
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse, "description": "VALIDATION_ERROR"}},
)
app.include_router(auth_router)
app.include_router(events_router)  # Register /events before /{analysis_id}.
app.add_middleware(BodyLimitMiddleware)


@app.middleware("http")
async def request_metadata(request: Request, call_next):
    request.state.request_id = str(uuid4())
    response = await call_next(request)
    session = getattr(request.state, "auth_session", None)
    if session and response.status_code < 400:
        # Dependencies' Response headers are not merged into FileResponse or
        # JSONResponse. Refresh here so downloads also retain an active session.
        set_session_cookie(response, session)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["Cache-Control"] = "no-store"
    return response


def error_response(request, status, code, message, details=None):
    request_id = getattr(request.state, "request_id", str(uuid4()))
    return JSONResponse(status_code=status, content={"error": {
        "code": code, "message": message, "details": details,
        "requestId": request_id,
    }}, headers={"Cache-Control": "no-store", "X-Request-ID": request_id})


@app.exception_handler(ApiError)
async def api_error(request: Request, exc: ApiError):
    response = error_response(request, exc.status, exc.code, exc.message, exc.details)
    response.headers.update(exc.headers)
    if exc.code == "UNAUTHENTICATED":
        clear_session_cookie(response)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    fields = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors()]
    return error_response(request, 422, "VALIDATION_ERROR", "Niepoprawne dane żądania.", {"fields": fields})


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    code, message = {
        404: ("NOT_FOUND", "Nie znaleziono zasobu."),
        405: ("METHOD_NOT_ALLOWED", "Niedozwolona metoda HTTP."),
    }.get(exc.status_code, ("VALIDATION_ERROR", "Niepoprawne żądanie HTTP."))
    response = error_response(request, exc.status_code, code, message)
    response.headers.update(exc.headers or {})
    return response


@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception):
    log.exception("Request %s failed", getattr(request.state, "request_id", "unknown"))
    return error_response(request, 500, "INTERNAL_ERROR", "Błąd serwera.")


@app.get("/api/v1/health")
def health():
    with connect() as conn:
        count = conn.execute(
            "SELECT count(*) AS n FROM workers WHERE heartbeat_at > clock_timestamp() - (%s * interval '1 second')",
            (WORKER_STALE_SECONDS,),
        ).fetchone()["n"]
        queued = conn.execute("SELECT count(*) AS n FROM analyses WHERE status = 'queued' AND user_id IS NOT NULL").fetchone()["n"]
    return {"status": "ok" if count else "degraded", "workersOnline": count, "queueLength": queued, "mode": "development"}


@app.get("/api/v1/models", response_model=list[SegmentationModel], responses=AUTH_ERRORS)
def models(user: CurrentUser):
    entries = catalogue()
    default = next((model.id for model in entries if model.id == UNET_ID and model.available), entries[0].id)
    return [SegmentationModel(id=model.id, name=model.name, version=model.version,
                              description=model.description, is_default=model.id == default,
                              available=model.available) for model in entries]


@app.post("/api/v1/analyses", status_code=201, response_model=Analysis,
          dependencies=[Depends(same_origin)], responses={
              **AUTH_ERRORS, 403: {"model": ErrorResponse, "description": "FORBIDDEN_ORIGIN"},
              413: {"model": ErrorResponse, "description": "FILE_TOO_LARGE or REQUEST_TOO_LARGE"},
              415: {"model": ErrorResponse, "description": "UNSUPPORTED_FILE_TYPE"},
              429: {"model": ErrorResponse, "description": "TOO_MANY_ACTIVE_ANALYSES"},
              503: {"model": ErrorResponse, "description": "MODEL_UNAVAILABLE or QUEUE_FULL"},
          })
def create_analysis(user: CurrentUser, response: Response, file: Annotated[UploadFile, File()],
                    model_id: Annotated[str, Form(alias="modelId")]):
    model = find_model(model_id)
    if model is None:
        raise ApiError(422, "MODEL_NOT_FOUND", "Nieznany model.")
    if not model.available:
        raise ApiError(503, "MODEL_UNAVAILABLE", "Model jest niedostępny. Brakuje skonfigurowanego checkpointu.")
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
            check_analysis_limits(conn, user.id)
            row = conn.execute(
                """INSERT INTO analyses (id, user_id, filename, model_id, model_name, model_version, status, width, height)
                   VALUES (%s, %s, %s, %s, %s, %s, 'queued', %s, %s) RETURNING *""",
                (analysis_id, user.id, filename, model_id, model.name, model.version, width, height),
            ).fetchone()
            result = serialize(conn, row)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    response.headers["Location"] = f"/api/v1/analyses/{analysis_id}"
    return result


@app.get("/api/v1/analyses", response_model=AnalysisPage, responses=AUTH_ERRORS)
def list_analyses(user: CurrentUser, page: Annotated[int, Query(ge=1)] = 1,
                  page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
                  status: str | None = None):
    statuses = status.split(",") if status else ["queued", "processing", "completed", "failed", "cancelled"]
    if not set(statuses) <= {"queued", "processing", "completed", "failed", "cancelled"}:
        raise ApiError(422, "VALIDATION_ERROR", "Niepoprawny filtr statusu.")
    with connect() as conn:
        total = conn.execute("SELECT count(*) AS n FROM analyses WHERE user_id = %s AND status = ANY(%s)",
                             (user.id, statuses)).fetchone()["n"]
        rows = conn.execute(
            """SELECT * FROM analyses WHERE user_id = %s AND status = ANY(%s)
               ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s""",
            (user.id, statuses, page_size, (page - 1) * page_size),
        ).fetchall()
        return AnalysisPage(items=[serialize(conn, row) for row in rows], page=page, page_size=page_size, total=total)


def find_analysis(conn, analysis_id, user_id):
    row = conn.execute("SELECT * FROM analyses WHERE id = %s AND user_id = %s", (analysis_id, user_id)).fetchone()
    if not row:
        raise ApiError(404, "ANALYSIS_NOT_FOUND", "Nie znaleziono analizy.")
    return row


@app.get("/api/v1/analyses/{analysis_id}", response_model=Analysis, responses=ANALYSIS_ERRORS)
def get_analysis(analysis_id: UUID, user: CurrentUser):
    with connect() as conn:
        return serialize(conn, find_analysis(conn, analysis_id, user.id))


@app.post("/api/v1/analyses/{analysis_id}/cancel", response_model=Analysis,
          dependencies=[Depends(same_origin)], responses={
              **ANALYSIS_ERRORS, 403: {"model": ErrorResponse, "description": "FORBIDDEN_ORIGIN"},
              409: {"model": ErrorResponse, "description": "ANALYSIS_NOT_CANCELLABLE"},
          })
def cancel_analysis(analysis_id: UUID, user: CurrentUser):
    with connect() as conn:
        row = conn.execute(
            """UPDATE analyses SET status = 'cancelled', finished_at = clock_timestamp()
               WHERE id = %s AND user_id = %s AND status = 'queued' RETURNING *""", (analysis_id, user.id),
        ).fetchone()
        row = row or find_analysis(conn, analysis_id, user.id)
        if row["status"] != "cancelled":
            raise ApiError(409, "ANALYSIS_NOT_CANCELLABLE", "Można anulować tylko zadanie oczekujące.", {"status": row["status"]})
        return serialize(conn, row)


@app.delete("/api/v1/analyses/{analysis_id}", status_code=204,
            dependencies=[Depends(same_origin)], responses={
                **ANALYSIS_ERRORS, 403: {"model": ErrorResponse, "description": "FORBIDDEN_ORIGIN"},
                409: {"model": ErrorResponse, "description": "ANALYSIS_ACTIVE"},
            })
def delete_analysis(analysis_id: UUID, user: CurrentUser):
    with connect() as conn:
        row = conn.execute("SELECT status FROM analyses WHERE id = %s AND user_id = %s FOR UPDATE",
                           (analysis_id, user.id)).fetchone()
        if not row:
            raise ApiError(404, "ANALYSIS_NOT_FOUND", "Nie znaleziono analizy.")
        if row["status"] in {"queued", "processing"}:
            raise ApiError(409, "ANALYSIS_ACTIVE", "Nie można usunąć aktywnej analizy. Najpierw anuluj zadanie oczekujące.")
        conn.execute("INSERT INTO file_deletions (analysis_id) VALUES (%s)", (analysis_id,))
        conn.execute("DELETE FROM analyses WHERE id = %s AND user_id = %s", (analysis_id, user.id))
    try:
        cleanup_pending(STORAGE_DIR)
    except Exception:
        log.exception("File cleanup queued for retry after deleting analysis %s", analysis_id)
    return Response(status_code=204)


@app.get("/api/v1/analyses/{analysis_id}/files/{kind}", responses={
    **ANALYSIS_ERRORS, 409: {"model": ErrorResponse, "description": "RESULT_NOT_READY"},
})
def get_file(analysis_id: UUID, kind: Literal["original", "mask", "overlay"], user: CurrentUser, download: bool = False):
    with connect() as conn:
        row = find_analysis(conn, analysis_id, user.id)
    if kind != "original" and row["status"] != "completed":
        raise ApiError(409, "RESULT_NOT_READY", "Wynik nie jest dostępny.")
    path = STORAGE_DIR / str(analysis_id) / f"{kind}.png"
    if not path.is_file():
        raise ApiError(404, "FILE_NOT_FOUND", "Nie znaleziono pliku.")
    return FileResponse(path, media_type="image/png", filename=f"{kind}.png",
                        content_disposition_type="attachment" if download else "inline")
