ALTER TABLE users
    ADD COLUMN must_change_password BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN password_changed_at TIMESTAMPTZ;
