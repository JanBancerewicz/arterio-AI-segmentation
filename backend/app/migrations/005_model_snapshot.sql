-- Defaults preserve the metadata of existing smoke-test analyses.
ALTER TABLE analyses
    ADD COLUMN model_name TEXT NOT NULL DEFAULT 'Test infrastruktury — bez modelu AI',
    ADD COLUMN model_version TEXT NOT NULL DEFAULT '0.1.0';
