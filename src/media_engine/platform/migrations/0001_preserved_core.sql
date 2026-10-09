CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (id),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects (id),
    name TEXT NOT NULL,
    position INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL REFERENCES scenes (id),
    asset_type TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generation_jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (id),
    status TEXT NOT NULL,
    pricing_version TEXT,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_generation_jobs_user ON generation_jobs (user_id, created_at);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    scene_id TEXT,
    asset_id TEXT,
    job_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    storage_provider TEXT NOT NULL,
    bucket TEXT NOT NULL,
    object_key TEXT NOT NULL UNIQUE,
    mime_type TEXT NOT NULL,
    byte_count BIGINT NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    duration_ms INTEGER,
    provider TEXT,
    engine TEXT,
    model TEXT,
    generation_settings JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_artifacts_user ON artifacts (user_id) WHERE deleted_at IS NULL;
CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    meter_type TEXT NOT NULL,
    amount BIGINT NOT NULL,
    pricing_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_runs (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES generation_jobs (id),
    provider TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS project_exports (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects (id),
    object_key TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS credit_accounts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (id),
    pool TEXT NOT NULL,
    UNIQUE (user_id, pool)
);
CREATE TABLE IF NOT EXISTS credit_ledger (
    id TEXT PRIMARY KEY,
    credit_account_id TEXT NOT NULL REFERENCES credit_accounts (id),
    pool TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    amount_units INTEGER NOT NULL,
    job_id TEXT,
    subscription_id TEXT,
    payment_transaction_id TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    pricing_version TEXT,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS pricing_rules (
    id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    service TEXT NOT NULL,
    model TEXT NOT NULL,
    quality TEXT NOT NULL,
    unit TEXT NOT NULL,
    credit_price INTEGER NOT NULL,
    internal_cost_units INTEGER NOT NULL,
    effective_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pricing_lookup ON pricing_rules (service, model, effective_at);
CREATE TABLE IF NOT EXISTS continuity_states (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    clip_index INTEGER NOT NULL,
    state JSONB NOT NULL,
    continuation_prompt TEXT,
    continuation_prompt_edited BOOLEAN NOT NULL DEFAULT FALSE
);
