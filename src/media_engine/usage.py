"""Internal usage rows. Not invoices, credits, or payments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from media_engine.db import connect, init_schema


@dataclass
class UsageEvent:
    job_id: str
    client_id: str
    job_type: str
    engine_id: str
    started_at: str
    finished_at: str
    duration_seconds: float
    attempt_count: int
    estimated_cost_usd: Optional[str]
    status: str
    provider: Optional[str] = None
    gpu_model: Optional[str] = None
    hourly_price_usd: Optional[str] = None
    gpu_seconds: Optional[float] = None
    actual_cost_usd: Optional[str] = None


class UsageStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        init_schema(db_path)
        _ensure_usage_columns(db_path)

    def record(self, event: UsageEvent) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO usage_events (
                    job_id, client_id, job_type, engine_id, started_at, finished_at,
                    duration_seconds, attempt_count, estimated_cost_usd, status,
                    provider, gpu_model, hourly_price_usd, gpu_seconds, actual_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.job_id, event.client_id, event.job_type, event.engine_id,
                    event.started_at, event.finished_at, event.duration_seconds,
                    event.attempt_count, event.estimated_cost_usd, event.status,
                    event.provider, event.gpu_model, event.hourly_price_usd,
                    event.gpu_seconds, event.actual_cost_usd,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def for_job(self, job_id: str) -> Optional[UsageEvent]:
        conn = connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM usage_events WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return UsageEvent(
            job_id=row["job_id"],
            client_id=row["client_id"],
            job_type=row["job_type"],
            engine_id=row["engine_id"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_seconds=float(row["duration_seconds"]),
            attempt_count=int(row["attempt_count"]),
            estimated_cost_usd=row["estimated_cost_usd"],
            status=row["status"],
            provider=row["provider"],
            gpu_model=row["gpu_model"],
            hourly_price_usd=row["hourly_price_usd"],
            gpu_seconds=row["gpu_seconds"],
            actual_cost_usd=row["actual_cost_usd"],
        )


def _ensure_usage_columns(db_path: str) -> None:
    columns = (
        ("provider", "TEXT"),
        ("gpu_model", "TEXT"),
        ("hourly_price_usd", "TEXT"),
        ("gpu_seconds", "REAL"),
        ("actual_cost_usd", "TEXT"),
    )
    conn = connect(db_path)
    try:
        have = {row[1] for row in conn.execute("PRAGMA table_info(usage_events)")}
        for name, kind in columns:
            if name not in have:
                conn.execute(f"ALTER TABLE usage_events ADD COLUMN {name} {kind}")
        conn.commit()
    finally:
        conn.close()
