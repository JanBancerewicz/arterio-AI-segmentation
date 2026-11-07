"""Database-backed admission limits, consistent across API processes."""
import hashlib

from . import config
from .db import connect
from .errors import ApiError


def user_bucket(username):
    return "user:" + hashlib.sha256(username.strip().lower().encode()).hexdigest()


def reserve_login(username, ip):
    buckets = [(user_bucket(username), config.LOGIN_USER_ATTEMPTS),
               ("ip:" + hashlib.sha256(ip.encode()).hexdigest(), config.LOGIN_IP_ATTEMPTS)]
    with connect() as conn:
        for key, limit in sorted(buckets):
            # Reserve before Argon2 verification so simultaneous attempts are bounded too.
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 73190423))", (key,))
            row = conn.execute(
                """INSERT INTO login_limits (bucket_key, attempts) VALUES (%s, 1)
                   ON CONFLICT (bucket_key) DO UPDATE SET
                     attempts = CASE WHEN login_limits.window_started_at <= statement_timestamp() - (%s * interval '1 second')
                         THEN 1 ELSE login_limits.attempts + 1 END,
                     window_started_at = CASE WHEN login_limits.window_started_at <= statement_timestamp() - (%s * interval '1 second')
                         THEN statement_timestamp() ELSE login_limits.window_started_at END
                   RETURNING attempts,
                     GREATEST(1, CEIL(EXTRACT(EPOCH FROM window_started_at
                         + (%s * interval '1 second') - clock_timestamp())))::integer AS retry_after""",
                (key, config.LOGIN_WINDOW_SECONDS, config.LOGIN_WINDOW_SECONDS, config.LOGIN_WINDOW_SECONDS),
            ).fetchone()
            if row["attempts"] > limit:
                # Roll back this reservation; previously committed attempts remain counted.
                raise ApiError(429, "RATE_LIMITED", "Zbyt wiele prób logowania. Spróbuj później.",
                               {"retryAfterSeconds": row["retry_after"]},
                               headers={"Retry-After": str(row["retry_after"])})


def login_succeeded(username):
    with connect() as conn:
        key = user_bucket(username)
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 73190423))", (key,))
        conn.execute("DELETE FROM login_limits WHERE bucket_key = %s", (key,))


def prune_login_limits():
    with connect() as conn:
        conn.execute("DELETE FROM login_limits WHERE window_started_at < clock_timestamp() - interval '1 day'")


def check_analysis_limits(conn, user_id):
    # All admissions use this short transaction lock; worker processing does not.
    conn.execute("SELECT pg_advisory_xact_lock(73190422)")
    active = conn.execute(
        "SELECT count(*) AS n FROM analyses WHERE user_id = %s AND status IN ('queued', 'processing')", (user_id,),
    ).fetchone()["n"]
    if active >= config.MAX_ACTIVE_ANALYSES:
        raise ApiError(429, "TOO_MANY_ACTIVE_ANALYSES", "Osiągnięto limit aktywnych analiz.",
                       {"limit": config.MAX_ACTIVE_ANALYSES, "active": active})
    queued = conn.execute("SELECT count(*) AS n FROM analyses WHERE status = 'queued' AND user_id IS NOT NULL").fetchone()["n"]
    if queued >= config.MAX_QUEUED_ANALYSES:
        raise ApiError(503, "QUEUE_FULL", "Kolejka jest pełna. Spróbuj za chwilę.", headers={"Retry-After": "5"})
