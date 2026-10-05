"""Provider-agnostic record of resources that may still be billing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from media_engine.db import connect, init_schema


class ResourceState:
    REQUESTED = "REQUESTED"
    PROVISIONING = "PROVISIONING"
    READY = "READY"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"
    FAILED = "FAILED"
    AMBIGUOUS = "AMBIGUOUS"
    UNCONDITIONAL = frozenset({PROVISIONING, READY, TERMINATING, AMBIGUOUS})


@dataclass
class ResourceRecord:
    resource_id: str
    provider: str
    owner_id: str
    state: str
    create_attempted: bool
    hourly_price_usd: Optional[str]
    created_at: float
    updated_at: float
    last_error: Optional[str] = None


def is_unresolved(record: ResourceRecord) -> bool:
    if record.state in ResourceState.UNCONDITIONAL:
        return True
    if record.state == ResourceState.REQUESTED and record.create_attempted:
        return True
    return False


class ResourceLedger:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        init_schema(db_path)

    def upsert(self, record: ResourceRecord) -> ResourceRecord:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO external_resources (
                    resource_id, provider, owner_id, state, create_attempted,
                    hourly_price_usd, created_at, updated_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(resource_id) DO UPDATE SET
                    provider = excluded.provider,
                    owner_id = excluded.owner_id,
                    state = excluded.state,
                    create_attempted = excluded.create_attempted,
                    hourly_price_usd = excluded.hourly_price_usd,
                    updated_at = excluded.updated_at,
                    last_error = excluded.last_error
                """,
                (
                    record.resource_id, record.provider, record.owner_id, record.state,
                    1 if record.create_attempted else 0, record.hourly_price_usd,
                    record.created_at, record.updated_at, record.last_error,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return record

    def get(self, resource_id: str) -> Optional[ResourceRecord]:
        conn = connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM external_resources WHERE resource_id = ?",
                (resource_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return _row(row)

    def list(self) -> list[ResourceRecord]:
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM external_resources ORDER BY created_at, resource_id"
            ).fetchall()
        finally:
            conn.close()
        return [_row(row) for row in rows]

    def delete(self, resource_id: str) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute("DELETE FROM external_resources WHERE resource_id = ?", (resource_id,))
            conn.commit()
        finally:
            conn.close()

    def unresolved(self) -> list[ResourceRecord]:
        return [record for record in self.list() if is_unresolved(record)]


def _row(row: object) -> ResourceRecord:
    return ResourceRecord(
        resource_id=row["resource_id"],
        provider=row["provider"],
        owner_id=row["owner_id"],
        state=row["state"],
        create_attempted=bool(row["create_attempted"]),
        hourly_price_usd=row["hourly_price_usd"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        last_error=row["last_error"],
    )
