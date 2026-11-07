"""One authenticated SSE stream per client, backed by PostgreSQL NOTIFY."""
from contextlib import aclosing
import json
import logging
from time import monotonic
from typing import Annotated
from uuid import UUID

from anyio import CancelScope
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
import psycopg
from psycopg import sql

from . import config, db, sessions
from .auth import require_user
from .errors import ApiError
from .schemas import ErrorResponse, PublicUser

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analyses", tags=["Analysis events"])


def frame(event, data):
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def notification_for_user(payload, user_id):
    try:
        data = json.loads(payload)
        owner = UUID(data["userId"])
        analysis_id = str(UUID(data["analysisId"]))
    except (ValueError, TypeError, KeyError):
        return None
    if owner == user_id:
        return "analysis.changed", {"analysisId": analysis_id, "queueChanged": data.get("queueChanged") is True}
    if data.get("queueChanged") is True:
        # Queue positions are global, but foreign analysis/user IDs stay private.
        return "queue.changed", {}
    return None


async def session_alive(connection, digest, user_id):
    cursor = await connection.execute(
        """SELECT 1 FROM sessions WHERE token_hash = %s AND user_id = %s
           AND idle_expires_at > clock_timestamp() AND expires_at > clock_timestamp()""",
        (digest, user_id),
    )
    return await cursor.fetchone() is not None


async def reserve_stream(connection, user_id):
    # Session advisory locks are released automatically when this connection closes.
    cursor = await connection.execute(
        """SELECT slot FROM generate_series(0, %s - 1) AS slots(slot)
           WHERE pg_try_advisory_lock(73190424, hashtext(%s || ':' || slot::text)) LIMIT 1""",
        (config.SSE_MAX_PER_USER, str(user_id)),
    )
    if await cursor.fetchone() is None:
        raise ApiError(429, "TOO_MANY_EVENT_STREAMS", "Osiągnięto limit połączeń aktualizacji statusu.",
                       {"limit": config.SSE_MAX_PER_USER}, headers={"Retry-After": "5"})
    cursor = await connection.execute(
        """SELECT slot FROM generate_series(0, %s - 1) AS slots(slot)
           WHERE pg_try_advisory_lock(73190425, slot) LIMIT 1""", (config.SSE_MAX_TOTAL,),
    )
    if await cursor.fetchone() is None:
        raise ApiError(429, "TOO_MANY_EVENT_STREAMS", "Serwer aktualizacji statusu jest zajęty.",
                       headers={"Retry-After": "5"})


async def event_stream(request, user_id, digest, connection):
    try:
        yield "retry: 2000\n" + frame("resync", {})
        last_heartbeat = last_check = monotonic()
        while not await request.is_disconnected():
            # Close the notification generator before executing SQL on this connection:
            # psycopg holds its connection lock for the lifetime of that generator.
            async with aclosing(connection.notifies(timeout=1, stop_after=100)) as notifications:
                batch = [notification async for notification in notifications]
            now = monotonic()
            if batch or now - last_check >= config.SSE_SESSION_CHECK_SECONDS:
                if not await session_alive(connection, digest, user_id):
                    yield frame("session.expired", {})
                    return
                last_check = now
            emitted = set()
            for notification in batch:
                message = notification_for_user(notification.payload, user_id)
                if message:
                    encoded = frame(*message)
                    if encoded not in emitted:
                        emitted.add(encoded)
                        yield encoded
            if now - last_heartbeat >= config.SSE_HEARTBEAT_SECONDS:
                yield ": heartbeat\n\n"
                last_heartbeat = now
    except psycopg.Error:
        log.exception("SSE database connection lost; client must reconnect and resync")
    finally:
        with CancelScope(shield=True):
            await connection.close()


@router.get("/events", response_class=StreamingResponse, responses={
    200: {"content": {"text/event-stream": {}}, "description": "resync, analysis.changed, queue.changed, session.expired"},
    401: {"model": ErrorResponse, "description": "UNAUTHENTICATED"},
    429: {"model": ErrorResponse, "description": "TOO_MANY_EVENT_STREAMS"},
})
async def events(request: Request, user: Annotated[PublicUser, Depends(require_user)]):
    connection = await psycopg.AsyncConnection.connect(db.DATABASE_URL, autocommit=True,
                                                       connect_timeout=5, application_name="arterio-sse")
    try:
        await connection.execute("SET statement_timeout = '2s'")
        await reserve_stream(connection, user.id)
        cursor = await connection.execute("SELECT 'arterio_analyses_' || substr(md5(current_schema()), 1, 16)")
        channel = (await cursor.fetchone())[0]
        await connection.execute(sql.SQL("LISTEN {}").format(sql.Identifier(channel)))
    except BaseException:
        await connection.close()
        raise
    digest = sessions.token_digest(request.cookies.get(config.SESSION_COOKIE_NAME))
    return StreamingResponse(event_stream(request, user.id, digest, connection), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})
