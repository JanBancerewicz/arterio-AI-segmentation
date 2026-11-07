-- NULL is retained only for legacy analyses created before authentication.
-- They are not exposed by the API and are not picked up by the worker.
ALTER TABLE analyses ADD COLUMN user_id UUID REFERENCES users(id);
CREATE INDEX analyses_user_history ON analyses (user_id, created_at DESC, id DESC);
