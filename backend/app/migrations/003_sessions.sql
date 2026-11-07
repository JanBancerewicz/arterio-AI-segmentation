CREATE TABLE sessions (
    token_hash TEXT PRIMARY KEY CHECK (token_hash ~ '^[0-9a-f]{64}$'),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    expires_at TIMESTAMPTZ NOT NULL,
    idle_expires_at TIMESTAMPTZ NOT NULL,
    CHECK (idle_expires_at <= expires_at)
);
CREATE INDEX sessions_user ON sessions (user_id);
CREATE INDEX sessions_idle_expiry ON sessions (idle_expires_at);
