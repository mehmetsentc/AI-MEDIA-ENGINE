"""SQLite access for local metadata.

PostgreSQL can replace this module later. Callers depend on the store
classes, not on SQL dialect details scattered through the orchestrator.
Media bytes are never written here.
"""
from __future__ import annotations

import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    result_uri TEXT,
    error_code TEXT,
    input_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_attempts (
    attempt_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_events (
    job_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    engine_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    attempt_count INTEGER NOT NULL,
    estimated_cost_usd TEXT,
    status TEXT NOT NULL,
    provider TEXT,
    gpu_model TEXT,
    hourly_price_usd TEXT,
    gpu_seconds REAL,
    actual_cost_usd TEXT,
    credit_before TEXT,
    credit_after TEXT,
    credit_delta TEXT,
    calculated_runtime_cost TEXT,
    unexplained_cost_delta TEXT
);
CREATE TABLE IF NOT EXISTS external_resources (
    resource_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    state TEXT NOT NULL,
    create_attempted INTEGER NOT NULL,
    hourly_price_usd TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_error TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT NOT NULL,
    seed INTEGER,
    width INTEGER,
    height INTEGER,
    created_at TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS host_quarantine (
    provider TEXT NOT NULL,
    offer_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    failure_class TEXT NOT NULL,
    reason TEXT NOT NULL,
    failed_at REAL NOT NULL,
    quarantine_until REAL NOT NULL,
    PRIMARY KEY (provider, offer_id, machine_id)
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    if db_path != ":memory:":
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if db_path != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def init_schema(db_path: str) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
