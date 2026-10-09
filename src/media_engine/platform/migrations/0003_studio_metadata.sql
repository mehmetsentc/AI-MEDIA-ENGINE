CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    attempt_count INTEGER NOT NULL,
    result_uri TEXT,
    error_code TEXT,
    input_json JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_client ON jobs (client_id, created_at);

CREATE TABLE IF NOT EXISTS job_attempts (
    attempt_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs (id),
    status TEXT NOT NULL,
    error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS external_resources (
    resource_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    state TEXT NOT NULL,
    create_attempted INTEGER NOT NULL,
    hourly_price_usd TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS host_quarantine (
    provider TEXT NOT NULL,
    offer_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    failure_class TEXT NOT NULL,
    reason TEXT NOT NULL,
    failed_at DOUBLE PRECISION NOT NULL,
    quarantine_until DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (provider, offer_id, machine_id)
);

CREATE TABLE IF NOT EXISTS studio_assets (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    engine TEXT,
    model TEXT,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL,
    artifact_id TEXT,
    job_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    settings_json JSONB NOT NULL,
    history_json JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_studio_assets_scene ON studio_assets (scene_id, created_at);
CREATE INDEX IF NOT EXISTS idx_studio_assets_job ON studio_assets (job_id);

CREATE TABLE IF NOT EXISTS job_artifacts (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    seed INTEGER,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMPTZ NOT NULL,
    byte_count BIGINT NOT NULL,
    sha256 TEXT NOT NULL,
    path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job_artifacts_job ON job_artifacts (job_id);

CREATE TABLE IF NOT EXISTS job_usage (
    job_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    engine_id TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    duration_seconds DOUBLE PRECISION NOT NULL,
    attempt_count INTEGER NOT NULL,
    estimated_cost_usd TEXT,
    status TEXT NOT NULL,
    provider TEXT,
    gpu_model TEXT,
    hourly_price_usd TEXT,
    gpu_seconds DOUBLE PRECISION,
    actual_cost_usd TEXT,
    credit_before TEXT,
    credit_after TEXT,
    credit_delta TEXT,
    calculated_runtime_cost TEXT,
    unexplained_cost_delta TEXT,
    instance_id TEXT,
    machine_id TEXT
);
