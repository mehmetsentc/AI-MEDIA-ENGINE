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


class UsageStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        init_schema(db_path)

    def record(self, event: UsageEvent) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO usage_events (
                    job_id, client_id, job_type, engine_id, started_at, finished_at,
                    duration_seconds, attempt_count, estimated_cost_usd, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.job_id, event.client_id, event.job_type, event.engine_id,
                    event.started_at, event.finished_at, event.duration_seconds,
                    event.attempt_count, event.estimated_cost_usd, event.status,
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
        )
