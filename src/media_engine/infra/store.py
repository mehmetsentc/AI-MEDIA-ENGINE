"""Local GPU-manager state. This store never opens Neon."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS gpu_manager_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT NOT NULL,
    instance_id TEXT NOT NULL DEFAULT '',
    machine_id INTEGER NOT NULL DEFAULT 0,
    offer_id INTEGER NOT NULL DEFAULT 0,
    create_attempted INTEGER NOT NULL DEFAULT 0,
    detail_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);
"""


class ManagerStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def save(self, record: dict) -> None:
        self._db.execute(
            """
            INSERT INTO gpu_manager_state (
                id, state, instance_id, machine_id, offer_id, create_attempted, detail_json, updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                state = excluded.state,
                instance_id = excluded.instance_id,
                machine_id = excluded.machine_id,
                offer_id = excluded.offer_id,
                create_attempted = excluded.create_attempted,
                detail_json = excluded.detail_json,
                updated_at = excluded.updated_at
            """,
            (
                record["state"],
                str(record.get("instance_id") or ""),
                int(record.get("machine_id") or 0),
                int(record.get("offer_id") or 0),
                1 if record.get("create_attempted") else 0,
                json.dumps(record.get("detail") or {}),
                str(record.get("updated_at") or ""),
            ),
        )
        self._db.commit()

    def load(self) -> dict | None:
        row = self._db.execute("SELECT * FROM gpu_manager_state WHERE id = 1").fetchone()
        if row is None:
            return None
        detail = json.loads(row["detail_json"] or "{}")
        return {
            "state": row["state"],
            "instance_id": row["instance_id"],
            "machine_id": int(row["machine_id"]),
            "offer_id": int(row["offer_id"]),
            "create_attempted": bool(row["create_attempted"]),
            "detail": detail if isinstance(detail, dict) else {},
            "updated_at": row["updated_at"],
        }
