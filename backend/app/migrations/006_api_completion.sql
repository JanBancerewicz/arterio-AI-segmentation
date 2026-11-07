CREATE TABLE login_limits (
    bucket_key TEXT PRIMARY KEY,
    window_started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    attempts INTEGER NOT NULL CHECK (attempts > 0)
);
CREATE INDEX login_limits_expiry ON login_limits (window_started_at);

CREATE TABLE file_deletions (
    analysis_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE FUNCTION notify_analysis_change() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    owner_id UUID;
    changed_id UUID;
    queue_changed BOOLEAN;
BEGIN
    IF TG_OP = 'DELETE' THEN
        owner_id := OLD.user_id;
        changed_id := OLD.id;
        queue_changed := OLD.status = 'queued';
    ELSE
        owner_id := NEW.user_id;
        changed_id := NEW.id;
        queue_changed := NEW.status = 'queued';
        IF TG_OP = 'UPDATE' THEN
            IF OLD.status IS NOT DISTINCT FROM NEW.status
               AND OLD.processing_stage IS NOT DISTINCT FROM NEW.processing_stage THEN
                RETURN NULL;
            END IF;
            queue_changed := queue_changed OR OLD.status = 'queued';
        END IF;
    END IF;
    IF owner_id IS NOT NULL THEN
        -- A separate channel for each schema keeps isolated test data separate.
        PERFORM pg_notify('arterio_analyses_' || substr(md5(TG_TABLE_SCHEMA), 1, 16),
            json_build_object('userId', owner_id, 'analysisId', changed_id,
                              'queueChanged', queue_changed)::text);
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER analyses_notifications AFTER INSERT OR DELETE OR UPDATE OF status, processing_stage
    ON analyses FOR EACH ROW EXECUTE FUNCTION notify_analysis_change();
