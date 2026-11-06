from .schemas import Analysis


def serialize(conn, row):
    position = None
    if row["status"] == "queued":
        position = max(1, conn.execute(
            """SELECT count(*) AS n FROM analyses WHERE status = 'queued'
               AND (created_at, id) <= (%s, %s)""",
            (row["created_at"], row["id"]),
        ).fetchone()["n"])
    duration = None
    if row["started_at"] and row["finished_at"]:
        duration = max(0, int((row["finished_at"] - row["started_at"]).total_seconds() * 1000))
    base = f"/api/v1/analyses/{row['id']}/files"
    return Analysis.model_validate({
        **row,
        "queue_position": position,
        "processing_ms": duration,
        "files": {
            "original": f"{base}/original",
            "mask": f"{base}/mask" if row["status"] == "completed" else None,
            "overlay": f"{base}/overlay" if row["status"] == "completed" else None,
        },
    })
