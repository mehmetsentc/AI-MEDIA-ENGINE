CREATE TABLE IF NOT EXISTS credit_pools (
    pool TEXT PRIMARY KEY
);
INSERT INTO credit_pools (pool) VALUES
    ('trial'),
    ('promo'),
    ('subscription'),
    ('topup')
ON CONFLICT (pool) DO NOTHING;

CREATE TABLE IF NOT EXISTS ledger_entry_types (
    entry_type TEXT PRIMARY KEY
);
INSERT INTO ledger_entry_types (entry_type) VALUES
    ('subscription_grant'),
    ('topup_purchase'),
    ('trial_grant'),
    ('promo_grant'),
    ('reserve'),
    ('consume'),
    ('release'),
    ('refund'),
    ('expiration'),
    ('admin_adjustment')
ON CONFLICT (entry_type) DO NOTHING;

CREATE TABLE IF NOT EXISTS usage_meters (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    service TEXT NOT NULL,
    meter_type TEXT NOT NULL,
    amount BIGINT NOT NULL,
    pricing_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_job ON usage_meters (job_id);

CREATE TABLE IF NOT EXISTS durable_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (id),
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    lease_owner TEXT,
    lease_until TIMESTAMPTZ,
    attempts INTEGER NOT NULL DEFAULT 0,
    pricing_version TEXT,
    max_concurrent INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_durable_jobs_queue ON durable_jobs (status, created_at);

CREATE TABLE IF NOT EXISTS media_artifacts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (id),
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
    generation_settings TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_media_user ON media_artifacts (user_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_media_job ON media_artifacts (job_id);

CREATE TABLE IF NOT EXISTS media_orphans (
    object_key TEXT PRIMARY KEY,
    bucket TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    reconciled_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS kill_switches (
    scope TEXT NOT NULL,
    target TEXT NOT NULL,
    engaged BOOLEAN NOT NULL,
    PRIMARY KEY (scope, target)
);

CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entitlements (
    id TEXT PRIMARY KEY,
    plan_id TEXT REFERENCES plans (id),
    user_id TEXT REFERENCES users (id),
    key TEXT NOT NULL,
    value_json JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entitlements_user ON entitlements (user_id, key);

CREATE TABLE IF NOT EXISTS payment_transactions (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    fee_units INTEGER,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS webhook_events (
    event_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users (id),
    plan_id TEXT NOT NULL REFERENCES plans (id),
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS cost_facts (
    job_id TEXT PRIMARY KEY,
    service TEXT,
    engine TEXT,
    model TEXT,
    provider TEXT,
    gpu_model TEXT,
    gpu_hourly_usd TEXT,
    gpu_seconds DOUBLE PRECISION,
    inference_seconds DOUBLE PRECISION,
    cold_start_seconds DOUBLE PRECISION,
    model_load_seconds DOUBLE PRECISION,
    api_cost_usd TEXT,
    storage_cost_usd TEXT,
    bandwidth_cost_usd TEXT,
    failed_job_cost_usd TEXT,
    actual_cost_usd TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id TEXT PRIMARY KEY,
    service TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    gpu TEXT,
    quality TEXT,
    resolution TEXT,
    duration_ms INTEGER,
    vram_mib INTEGER,
    cold_start_seconds DOUBLE PRECISION,
    warm_start_seconds DOUBLE PRECISION,
    model_load_seconds DOUBLE PRECISION,
    inference_seconds DOUBLE PRECISION,
    jobs_per_hour DOUBLE PRECISION,
    concurrency INTEGER,
    provider_cost_usd TEXT,
    cost_per_job_usd TEXT,
    recorded_at TIMESTAMPTZ NOT NULL,
    version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS platform_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signed_urls (
    token TEXT PRIMARY KEY,
    object_key TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS scene_id TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS characters TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS objects TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS location TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS lighting TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS visual_style TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS camera TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS movement_direction TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS story_state TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS previous_prompt TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS last_frame_artifact_id TEXT;
ALTER TABLE continuity_states ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;
ALTER TABLE provider_runs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;
ALTER TABLE scenes ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_credit_ledger_job ON credit_ledger (job_id, entry_type);

INSERT INTO platform_config (key, value)
VALUES ('pool_priority', 'trial,promo,subscription,topup')
ON CONFLICT (key) DO NOTHING;
INSERT INTO kill_switches (scope, target, engaged)
VALUES ('global', '*', FALSE)
ON CONFLICT (scope, target) DO NOTHING;

DO $$
BEGIN
    ALTER TABLE credit_accounts
        ADD CONSTRAINT credit_accounts_pool_known
        FOREIGN KEY (pool) REFERENCES credit_pools (pool);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE credit_ledger
        ADD CONSTRAINT credit_ledger_pool_known
        FOREIGN KEY (pool) REFERENCES credit_pools (pool);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE credit_ledger
        ADD CONSTRAINT credit_ledger_entry_type_known
        FOREIGN KEY (entry_type) REFERENCES ledger_entry_types (entry_type);
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE OR REPLACE FUNCTION media_engine_ledger_append_only()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'append-only';
END;
$$;

DO $$
BEGIN
    CREATE TRIGGER credit_ledger_block_update
        BEFORE UPDATE ON credit_ledger
        FOR EACH ROW
        EXECUTE FUNCTION media_engine_ledger_append_only();
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE TRIGGER credit_ledger_block_delete
        BEFORE DELETE ON credit_ledger
        FOR EACH ROW
        EXECUTE FUNCTION media_engine_ledger_append_only();
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE TRIGGER credit_ledger_block_truncate
        BEFORE TRUNCATE ON credit_ledger
        FOR EACH STATEMENT
        EXECUTE FUNCTION media_engine_ledger_append_only();
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
