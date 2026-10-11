"""Read-only view of saved production media. It cannot grant credits or rent a GPU."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from media_engine.platform.repository import PlatformBlocked
from media_engine.storage.r2_media import open_media_store
from media_engine.studio.account import job_group

_POSTGRES = Path("/Users/user/AI-MEDIA-ENGINE/.env.postgres")
_R2 = Path("/Users/user/AI-MEDIA-ENGINE/.env.r2")


class ReadOnlyPlatform:
    def __init__(self, dsn: str, media: object) -> None:
        self._dsn = dsn
        self.media = media
        self.repo = self

    def __repr__(self) -> str:
        return "ReadOnlyPlatform()"

    def balance(self, user_id: str, pool: Optional[str] = None) -> int:
        row = self._one(
            """
            SELECT COALESCE(SUM(credit_ledger.amount_units), 0) AS total
            FROM credit_ledger
            JOIN credit_accounts ON credit_accounts.id = credit_ledger.credit_account_id
            WHERE credit_accounts.user_id = %s
            """,
            (user_id,),
        )
        return 0 if row is None else int(row["total"])

    def image_price(self) -> int:
        return 2

    def list_media(self, user_id: str) -> list[dict]:
        rows = self._all(
            """
            SELECT id, user_id, job_id, sha256, byte_count, width, height, model,
                   created_at::text AS created_at, generation_settings::text AS generation_settings
            FROM media_artifacts
            WHERE user_id = %s AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (user_id,),
        )
        return [
            {
                "id": str(row["id"]),
                "user_id": str(row["user_id"]),
                "job_id": str(row["job_id"]),
                "sha256": str(row["sha256"]),
                "byte_count": int(row["byte_count"]),
                "width": row["width"],
                "height": row["height"],
                "model": row["model"],
                "created_at": str(row["created_at"]),
                "settings": row["generation_settings"],
            }
            for row in rows
        ]

    def media_row(self, artifact_id: str) -> Optional[dict]:
        row = self._one(
            """
            SELECT object_key, bucket, sha256, byte_count, user_id, job_id
            FROM media_artifacts
            WHERE id = %s AND deleted_at IS NULL
            """,
            (artifact_id,),
        )
        if row is None:
            return None
        return {
            "object_key": str(row["object_key"]),
            "bucket": str(row["bucket"]),
            "sha256": str(row["sha256"]),
            "byte_count": int(row["byte_count"]),
            "user_id": str(row["user_id"]),
            "job_id": str(row["job_id"]),
        }

    def read_media(self, artifact_id: str) -> Optional[bytes]:
        row = self.media_row(artifact_id)
        if row is None:
            return None
        return self.media.get(str(row["object_key"]))

    def job_ledger(self, job_id: str) -> list[dict]:
        rows = self._all(
            """
            SELECT entry_type, amount_units
            FROM credit_ledger
            WHERE job_id = %s
            ORDER BY created_at
            """,
            (job_id,),
        )
        return [{"entry_type": str(row["entry_type"]), "amount": int(row["amount_units"])} for row in rows]

    def list_image_jobs(self, user_id: str) -> list[dict]:
        rows = self._all(
            """
            SELECT id, client_id, status, error_code, created_at::text AS created_at,
                   updated_at::text AS updated_at, input_json::text AS input_json
            FROM jobs
            WHERE client_id = %s AND job_type = 'IMAGE_GENERATE'
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (user_id,),
        )
        items = []
        for row in rows:
            payload = _json(row["input_json"])
            status = str(payload.get("public_status") or str(row["status"]).lower())
            items.append({
                "job_id": str(row["id"]),
                "user_id": str(row["client_id"]),
                "status": status,
                "group": job_group(status),
                "prompt": str(payload.get("prompt") or ""),
                "seed": payload.get("seed"),
                "width": payload.get("width"),
                "height": payload.get("height"),
                "artifact_id": payload.get("artifact_id"),
                "error_code": row["error_code"],
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "credit_status": credit_status_from(self.job_ledger(str(row["id"]))),
            })
        return items

    def authorize_paid(self, *_args: object, **_kwargs: object) -> str:
        raise PlatformBlocked("READ_ONLY")

    def grant(self, *_args: object, **_kwargs: object) -> str:
        raise PlatformBlocked("READ_ONLY")

    def _one(self, sql: str, params: tuple) -> Optional[dict]:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def _all(self, sql: str, params: tuple) -> list[dict]:
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(
            self._dsn, row_factory=dict_row, autocommit=False,
            options="-c default_transaction_read_only=on",
        ) as conn:
            conn.execute("SET default_transaction_read_only = on")
            rows = conn.execute(sql, params).fetchall()
            conn.rollback()
        return [dict(row) for row in rows]


def credit_status_from(entries: list[dict]) -> str:
    kinds = [str(item["entry_type"]) for item in entries]
    if "refund" in kinds:
        return "refunded"
    if "consume" in kinds:
        return "settled"
    if "reserve" in kinds:
        return "reserved"
    return "none"


def attach_readonly(controller: object) -> None:
    """Replace the local ledger with production reads. Failure leaves the local ledger."""
    try:
        from media_engine.platform.runtime import assert_production_postgres_ready
        dsn = assert_production_postgres_ready({
            "MEDIA_ENGINE_POSTGRES_FILE": str(_POSTGRES),
        })
        media = open_media_store(_file_env(_R2))
        controller.platform = ReadOnlyPlatform(dsn, media)
        controller.studio_readonly = True
    except Exception as exc:
        print("READONLY_UNAVAILABLE " + type(exc).__name__, flush=True)


def _file_env(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        loaded[name.strip()] = value.strip().strip("'\"")
    return loaded


def _json(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
