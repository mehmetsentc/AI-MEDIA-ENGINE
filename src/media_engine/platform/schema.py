"""Metadata schema. Media bytes are never columns."""

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS credit_accounts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    pool TEXT NOT NULL,
    UNIQUE (user_id, pool)
);
CREATE TABLE IF NOT EXISTS credit_ledger (
    id TEXT PRIMARY KEY,
    credit_account_id TEXT NOT NULL,
    pool TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    amount_units INTEGER NOT NULL,
    job_id TEXT,
    subscription_id TEXT,
    payment_transaction_id TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    pricing_version TEXT,
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS credit_ledger_block_update
BEFORE UPDATE ON credit_ledger
BEGIN
    SELECT RAISE(ABORT, 'append-only');
END;
CREATE TRIGGER IF NOT EXISTS credit_ledger_block_delete
BEFORE DELETE ON credit_ledger
BEGIN
    SELECT RAISE(ABORT, 'append-only');
END;
CREATE TABLE IF NOT EXISTS pricing_rules (
    id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    service TEXT NOT NULL,
    model TEXT NOT NULL,
    quality TEXT NOT NULL,
    unit TEXT NOT NULL,
    credit_price INTEGER NOT NULL,
    internal_cost_units INTEGER NOT NULL,
    effective_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pricing_lookup
    ON pricing_rules (service, model, effective_at);
CREATE TABLE IF NOT EXISTS usage_meters (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    service TEXT NOT NULL,
    meter_type TEXT NOT NULL,
    amount INTEGER NOT NULL,
    pricing_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_job ON usage_meters (job_id);
CREATE TABLE IF NOT EXISTS durable_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    lease_owner TEXT,
    lease_until TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    pricing_version TEXT,
    max_concurrent INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS media_artifacts (
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
    byte_count INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    duration_ms INTEGER,
    provider TEXT,
    engine TEXT,
    model TEXT,
    generation_settings TEXT,
    created_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_media_user ON media_artifacts (user_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_media_job ON media_artifacts (job_id);
CREATE TABLE IF NOT EXISTS media_orphans (
    object_key TEXT PRIMARY KEY,
    bucket TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reconciled_at TEXT
);
CREATE TABLE IF NOT EXISTS kill_switches (
    scope TEXT NOT NULL,
    target TEXT NOT NULL,
    engaged INTEGER NOT NULL,
    PRIMARY KEY (scope, target)
);
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entitlements (
    id TEXT PRIMARY KEY,
    plan_id TEXT,
    user_id TEXT,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS payment_transactions (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    fee_units INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS webhook_events (
    event_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    received_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subscriptions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cost_facts (
    job_id TEXT PRIMARY KEY,
    service TEXT,
    engine TEXT,
    model TEXT,
    provider TEXT,
    gpu_model TEXT,
    gpu_hourly_usd TEXT,
    gpu_seconds REAL,
    inference_seconds REAL,
    cold_start_seconds REAL,
    model_load_seconds REAL,
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
    cold_start_seconds REAL,
    warm_start_seconds REAL,
    model_load_seconds REAL,
    inference_seconds REAL,
    jobs_per_hour REAL,
    concurrency INTEGER,
    provider_cost_usd TEXT,
    cost_per_job_usd TEXT,
    recorded_at TEXT NOT NULL,
    version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS platform_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signed_urls (
    token TEXT PRIMARY KEY,
    object_key TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS continuity_states (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    scene_id TEXT,
    clip_index INTEGER NOT NULL,
    characters TEXT,
    objects TEXT,
    location TEXT,
    lighting TEXT,
    visual_style TEXT,
    camera TEXT,
    movement_direction TEXT,
    story_state TEXT,
    previous_prompt TEXT,
    last_frame_artifact_id TEXT,
    continuation_prompt TEXT,
    continuation_prompt_edited INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS provider_runs (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

POSTGRES_SCHEMA = """
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
"""

MEANINGFUL_STATES = frozenset({
    "queued",
    "reserved",
    "planning",
    "provisioning",
    "worker_starting",
    "generating",
    "saving",
    "completed",
    "failed",
    "cancelled",
})

METER_TYPES = frozenset({
    "input_tokens",
    "output_tokens",
    "images",
    "pixels",
    "steps",
    "audio_seconds",
    "music_seconds",
    "video_seconds",
    "storage_bytes",
    "bandwidth_bytes",
    "gpu_seconds",
    "api_units",
})

PLAN_CATEGORIES = (
    "Free",
    "Trial",
    "Starter",
    "Creator",
    "Publisher",
    "Publisher Pro",
    "Agency",
)

LEDGER_TYPES = frozenset({
    "subscription_grant",
    "topup_purchase",
    "trial_grant",
    "promo_grant",
    "reserve",
    "consume",
    "release",
    "refund",
    "expiration",
    "admin_adjustment",
})

LONG_VIDEO_KEYS = (
    "long_video_enabled",
    "maximum_video_duration",
    "maximum_scene_count",
    "maximum_clip_count",
    "max_resolution",
    "max_concurrent_video_jobs",
    "continuity_enabled",
    "custom_model_access",
)

MEDIA_SERVICES = (
    "text",
    "image",
    "image_edit",
    "tts",
    "voice_clone",
    "music",
    "audio",
    "video",
    "storage",
    "publish",
)


def should_persist_state(state: str) -> bool:
    """Percentage ticks stay out of the database."""
    return state in MEANINGFUL_STATES
