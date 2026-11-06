CREATE TABLE analyses (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    processing_stage TEXT,
    error_code TEXT,
    error_message TEXT,
    worker_id UUID
);
CREATE INDEX analyses_queue ON analyses (created_at, id) WHERE status = 'queued';
CREATE TABLE workers (
    id UUID PRIMARY KEY,
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
