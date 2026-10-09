"""PostgreSQL platform repository. Financial writes commit in one transaction."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

import psycopg
from psycopg.types.json import Json

from media_engine.platform.postgres_conn import PgPool
from media_engine.platform.repository import PlatformBlocked


def _iso(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value)


class PostgresRepository:
    def __init__(self, pool: PgPool) -> None:
        self.pool = pool

    def ensure_user(self, user_id: str, now: str) -> None:
        with self.pool.session() as conn:
            self._ensure_user(conn, user_id, now)

    def set_switch(self, scope: str, target: str, engaged: bool) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO kill_switches (scope, target, engaged) VALUES (%s, %s, %s)
                ON CONFLICT (scope, target) DO UPDATE SET engaged = EXCLUDED.engaged
                """,
                (scope, target, engaged),
            )

    def engaged_switches(self) -> list[tuple[str, str]]:
        with self.pool.session() as conn:
            rows = conn.execute(
                "SELECT scope, target FROM kill_switches WHERE engaged"
            ).fetchall()
        return [(str(row["scope"]), str(row["target"])) for row in rows]

    def add_plan(self, plan_id: str, category: str) -> None:
        with self.pool.session() as conn:
            conn.execute(
                "INSERT INTO plans (id, category) VALUES (%s, %s)",
                (plan_id, category),
            )

    def set_entitlement(self, key: str, value_json: str, *, plan_id: str = "", user_id: str = "") -> None:
        with self.pool.session() as conn:
            if user_id:
                self._ensure_user(conn, user_id, _iso(datetime.now(timezone.utc)))
            conn.execute(
                """
                INSERT INTO entitlements (id, plan_id, user_id, key, value_json)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    "ent_" + uuid.uuid4().hex, plan_id or None, user_id or None, key,
                    Json(json.loads(value_json)),
                ),
            )

    def entitlement_json(self, user_id: str, key: str) -> Optional[str]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT value_json FROM entitlements
                WHERE user_id = %s AND key = %s
                ORDER BY id DESC LIMIT 1
                """,
                (user_id, key),
            ).fetchone()
        if row is None:
            return None
        value = row["value_json"]
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(",", ":"))

    def set_config(self, key: str, value: str) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO platform_config (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (key, value),
            )

    def config(self, key: str) -> Optional[str]:
        with self.pool.session() as conn:
            row = conn.execute(
                "SELECT value FROM platform_config WHERE key = %s",
                (key,),
            ).fetchone()
        return None if row is None else str(row["value"])

    def grant(self, *, user_id: str, pool: str, entry_type: str, amount: int, idempotency_key: str,
              now: str, subscription_id: str = "", payment_id: str = "") -> str:
        return self._post(
            user_id=user_id, pool=pool, entry_type=entry_type, amount=amount,
            idempotency_key=idempotency_key, now=now, subscription_id=subscription_id,
            payment_id=payment_id,
        )

    def balance(self, user_id: str, pool: Optional[str] = None) -> int:
        with self.pool.session() as conn:
            if pool is None:
                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(credit_ledger.amount_units), 0) AS total
                    FROM credit_ledger
                    JOIN credit_accounts ON credit_accounts.id = credit_ledger.credit_account_id
                    WHERE credit_accounts.user_id = %s
                    """,
                    (user_id,),
                ).fetchone()
            else:
                row = self._balance(conn, user_id, pool)
                return int(row)
        return int(row["total"])

    def add_price(self, **fields: object) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO pricing_rules (
                    id, version, service, model, quality, unit, credit_price,
                    internal_cost_units, effective_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    "price_" + uuid.uuid4().hex, fields["version"], fields["service"], fields["model"],
                    fields["quality"], fields["unit"], fields["credit_price"],
                    fields["internal_cost_units"], fields["effective_at"],
                ),
            )

    def price_at(self, service: str, model: str, at: str) -> Optional[dict]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT version, credit_price, unit, effective_at
                FROM pricing_rules
                WHERE service = %s AND model = %s AND effective_at <= %s
                ORDER BY effective_at DESC
                LIMIT 1
                """,
                (service, model, at),
            ).fetchone()
        if row is None:
            return None
        return {
            "version": str(row["version"]),
            "credit_price": int(row["credit_price"]),
            "unit": str(row["unit"]),
            "effective_at": _iso(row["effective_at"]),
        }

    def reservation_version(self, job_id: str) -> Optional[str]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT pricing_version FROM credit_ledger
                WHERE job_id = %s AND entry_type = 'reserve'
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        if row is None or row["pricing_version"] is None:
            return None
        return str(row["pricing_version"])

    def reserve(self, *, user_id: str, amount: int, job_id: str, version: str, pools: list[str], now: str) -> None:
        self.reserve_and_enqueue(
            user_id=user_id, amount=amount, job_id=job_id, version=version, pools=pools, now=now,
            idempotency_key=f"job:{job_id}", max_concurrent=1, enqueue=False,
        )

    def reserve_and_enqueue(self, *, user_id: str, amount: int, job_id: str, version: str, pools: list[str],
                            now: str, idempotency_key: str, max_concurrent: int, enqueue: bool = True) -> None:
        with self.pool.session() as conn:
            self._ensure_user(conn, user_id, now)
            conn.execute(
                "SELECT id FROM credit_accounts WHERE user_id = %s FOR UPDATE",
                (user_id,),
            )
            existing = conn.execute(
                """
                SELECT id FROM credit_ledger
                WHERE job_id = %s AND entry_type = 'reserve'
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if existing is None:
                remaining = amount
                takes: list[tuple[str, int]] = []
                for pool in pools:
                    available = self._balance(conn, user_id, pool)
                    if available <= 0:
                        continue
                    take = min(available, remaining)
                    takes.append((pool, take))
                    remaining -= take
                    if remaining <= 0:
                        break
                if remaining > 0:
                    raise PlatformBlocked("INSUFFICIENT_CREDIT")
                for pool, take in takes:
                    self._post_conn(
                        conn, user_id=user_id, pool=pool, entry_type="reserve", amount=-take,
                        idempotency_key=f"reserve:{job_id}:{pool}", now=now, job_id=job_id,
                        pricing_version=version,
                    )
            if enqueue:
                self._enqueue(conn, user_id, job_id, idempotency_key, version, max_concurrent, now)

    def reserved_by_pool(self, job_id: str) -> list[tuple[str, int]]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT pool, COALESCE(SUM(-amount_units), 0) AS total
                FROM credit_ledger
                WHERE job_id = %s AND entry_type = 'reserve'
                GROUP BY pool
                """,
                (job_id,),
            ).fetchall()
        return [(str(row["pool"]), int(row["total"])) for row in rows]

    def reserved_amount(self, job_id: str) -> int:
        return sum(amount for _, amount in self.reserved_by_pool(job_id))

    def post(self, **kwargs: object) -> str:
        return self._post(**kwargs)  # type: ignore[arg-type]

    def post_many(self, job_id: str, entries: list[dict], *, durable_status: str, now: str) -> None:
        with self.pool.session() as conn:
            if not self._settled(conn, job_id):
                for entry in entries:
                    self._post_conn(conn, **entry)
            conn.execute(
                """
                UPDATE durable_jobs
                SET status = %s, lease_owner = NULL, lease_until = NULL, updated_at = %s
                WHERE job_id = %s AND status IN ('queued', 'claimed')
                """,
                (durable_status, now, job_id),
            )
            conn.execute(
                "UPDATE generation_jobs SET status = %s WHERE id = %s",
                (durable_status, job_id),
            )

    def enqueue(self, *, user_id: str, job_id: str, idempotency_key: str, pricing_version: str,
                max_concurrent: int, now: str) -> None:
        with self.pool.session() as conn:
            self._ensure_user(conn, user_id, now)
            self._enqueue(conn, user_id, job_id, idempotency_key, pricing_version, max_concurrent, now)

    def claim(self, worker_id: str, now: str, until: str) -> Optional[str]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT job_id, user_id, max_concurrent FROM durable_jobs
                WHERE status = 'queued'
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            active = conn.execute(
                """
                SELECT COUNT(job_id) AS n FROM durable_jobs
                WHERE user_id = %s AND status = 'claimed'
                """,
                (row["user_id"],),
            ).fetchone()
            if int(active["n"]) >= int(row["max_concurrent"]):
                return None
            updated = conn.execute(
                """
                UPDATE durable_jobs
                SET status = 'claimed', lease_owner = %s, lease_until = %s, updated_at = %s,
                    attempts = attempts + 1
                WHERE job_id = %s AND status = 'queued'
                """,
                (worker_id, until, now, row["job_id"]),
            )
            if updated.rowcount != 1:
                return None
            return str(row["job_id"])

    def claim_job(self, job_id: str, worker_id: str, now: str, until: str) -> bool:
        with self.pool.session() as conn:
            updated = conn.execute(
                """
                UPDATE durable_jobs
                SET status = 'claimed', lease_owner = %s, lease_until = %s, updated_at = %s,
                    attempts = attempts + 1
                WHERE job_id = %s AND status = 'queued'
                """,
                (worker_id, until, now, job_id),
            )
            if updated.rowcount == 1:
                return True
            current = conn.execute(
                "SELECT status, lease_owner FROM durable_jobs WHERE job_id = %s",
                (job_id,),
            ).fetchone()
            return current is not None and current["status"] == "claimed" and current["lease_owner"] == worker_id

    def heartbeat(self, job_id: str, worker_id: str, until: str, now: str) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                UPDATE durable_jobs SET lease_until = %s, updated_at = %s
                WHERE job_id = %s AND lease_owner = %s AND status = 'claimed'
                """,
                (until, now, job_id, worker_id),
            )

    def recover_expired(self, now: str) -> int:
        with self.pool.session() as conn:
            cursor = conn.execute(
                """
                UPDATE durable_jobs
                SET status = 'queued', lease_owner = NULL, lease_until = NULL, updated_at = %s
                WHERE status = 'claimed' AND lease_until < %s
                """,
                (now, now),
            )
            return int(cursor.rowcount)

    def cancel(self, job_id: str, now: str) -> bool:
        with self.pool.session() as conn:
            cursor = conn.execute(
                """
                UPDATE durable_jobs
                SET status = 'cancelled', lease_owner = NULL, lease_until = NULL, updated_at = %s
                WHERE job_id = %s AND status IN ('queued', 'claimed')
                """,
                (now, job_id),
            )
            return cursor.rowcount == 1

    def job_status(self, job_id: str) -> Optional[str]:
        with self.pool.session() as conn:
            row = conn.execute(
                "SELECT status FROM durable_jobs WHERE job_id = %s",
                (job_id,),
            ).fetchone()
        return None if row is None else str(row["status"])

    def claim_execution_slot(self, job_id: str, now: str) -> bool:
        """Allow one GPU start for a reserved job. A retry does not get a second slot."""
        with self.pool.session() as conn:
            row = conn.execute(
                """
                UPDATE durable_jobs
                SET attempts = attempts + 1, updated_at = %s
                WHERE job_id = %s AND attempts = 0 AND status = 'queued'
                RETURNING job_id
                """,
                (now, job_id),
            ).fetchone()
        return row is not None

    def pending_jobs(self, user_id: str) -> list[dict]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT job_id, status FROM durable_jobs
                WHERE user_id = %s AND status IN ('queued', 'claimed')
                ORDER BY created_at
                """,
                (user_id,),
            ).fetchall()
        return [{"job_id": str(row["job_id"]), "status": str(row["status"])} for row in rows]

    def ledger_history(self, user_id: str, limit: int = 50) -> list[dict]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT credit_ledger.entry_type, credit_ledger.amount_units,
                       credit_ledger.job_id, credit_ledger.idempotency_key, credit_ledger.created_at
                FROM credit_ledger
                JOIN credit_accounts ON credit_accounts.id = credit_ledger.credit_account_id
                WHERE credit_accounts.user_id = %s
                ORDER BY credit_ledger.created_at
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        return [
            {
                "entry_type": str(row["entry_type"]),
                "amount_units": int(row["amount_units"]),
                "job_id": None if row["job_id"] is None else str(row["job_id"]),
                "idempotency_key": str(row["idempotency_key"]),
                "created_at": _iso(row["created_at"]),
            }
            for row in rows
        ]

    def artifact_for_job(self, job_id: str) -> Optional[dict]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT id, bucket, object_key FROM media_artifacts
                WHERE job_id = %s AND deleted_at IS NULL
                ORDER BY created_at
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "bucket": str(row["bucket"]),
            "object_key": str(row["object_key"]),
        }

    def insert_artifact(self, row: dict, now: str) -> None:
        settings = row.get("generation_settings") or "{}"
        if isinstance(settings, str):
            settings = json.loads(settings)
        with self.pool.session() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO media_artifacts (
                        id, user_id, project_id, scene_id, asset_id, job_id, artifact_type,
                        storage_provider, bucket, object_key, mime_type, byte_count, sha256,
                        width, height, duration_ms, provider, engine, model, generation_settings, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, 'r2', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        row["id"], row["user_id"], row["project_id"], row.get("scene_id") or None,
                        row.get("asset_id") or None, row["job_id"], row["artifact_type"], row["bucket"],
                        row["object_key"], row["mime_type"], row["byte_count"], row["sha256"],
                        row.get("width"), row.get("height"), row.get("duration_ms"), row.get("provider"),
                        row.get("engine"), row.get("model"), json.dumps(settings, separators=(",", ":")), now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO artifacts (
                        id, user_id, project_id, scene_id, asset_id, job_id, artifact_type,
                        storage_provider, bucket, object_key, mime_type, byte_count, sha256,
                        width, height, provider, engine, model, generation_settings, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, 'r2', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        row["id"], row["user_id"], row["project_id"], row.get("scene_id") or None,
                        row.get("asset_id") or None, row["job_id"], row["artifact_type"], row["bucket"],
                        row["object_key"], row["mime_type"], row["byte_count"], row["sha256"],
                        row.get("width"), row.get("height"), row.get("provider"), row.get("engine"),
                        row.get("model"), Json(settings), now,
                    ),
                )
            except psycopg.Error as exc:
                raise PlatformBlocked("ARTIFACT_METADATA_FAILED") from exc

    def media_row(self, artifact_id: str) -> Optional[dict]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT object_key, bucket, sha256, byte_count
                FROM media_artifacts
                WHERE id = %s AND deleted_at IS NULL
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "object_key": str(row["object_key"]),
            "bucket": str(row["bucket"]),
            "sha256": str(row["sha256"]),
            "byte_count": int(row["byte_count"]),
        }

    def settlement_exists(self, job_id: str) -> bool:
        with self.pool.session() as conn:
            return self._settled(conn, job_id)

    def record_orphan(self, key: str, bucket: str, digest: str, reason: str, now: str) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO media_orphans (object_key, bucket, sha256, reason, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (object_key) DO NOTHING
                """,
                (key, bucket, digest, reason, now),
            )

    def reconcile_orphan(self, key: str, now: str) -> str:
        with self.pool.session() as conn:
            row = conn.execute(
                "SELECT reconciled_at FROM media_orphans WHERE object_key = %s",
                (key,),
            ).fetchone()
            if row is None:
                raise PlatformBlocked("ORPHAN_UNKNOWN")
            if row["reconciled_at"]:
                return "reconciled"
            conn.execute(
                "UPDATE media_orphans SET reconciled_at = %s WHERE object_key = %s",
                (now, key),
            )
        return "reconciled"

    def soft_delete(self, artifact_id: str, now: str) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                UPDATE media_artifacts SET deleted_at = %s
                WHERE id = %s AND deleted_at IS NULL
                """,
                (now, artifact_id),
            )

    def storage_bytes(self, user_id: str, project_id: str = "", kind: str = "") -> int:
        clauses = ["user_id = %s", "deleted_at IS NULL"]
        params: list[object] = [user_id]
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if kind:
            clauses.append("artifact_type = %s")
            params.append(kind)
        with self.pool.session() as conn:
            row = conn.execute(
                f"SELECT COALESCE(SUM(byte_count), 0) AS total FROM media_artifacts WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
        return int(row["total"])

    def list_artifacts(self, user_id: str, limit: int, offset: int) -> list[dict]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT id, object_key, byte_count, sha256, artifact_type
                FROM media_artifacts
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY created_at
                LIMIT %s OFFSET %s
                """,
                (user_id, limit, offset),
            ).fetchall()
        return [
            {
                "id": str(row["id"]),
                "object_key": str(row["object_key"]),
                "byte_count": int(row["byte_count"]),
                "sha256": str(row["sha256"]),
                "artifact_type": str(row["artifact_type"]),
            }
            for row in rows
        ]

    def issue_signed(self, token: str, object_key: str, expires_at: str) -> None:
        with self.pool.session() as conn:
            conn.execute(
                "INSERT INTO signed_urls (token, object_key, expires_at) VALUES (%s, %s, %s)",
                (token, object_key, expires_at),
            )

    def signed_object(self, token: str) -> Optional[dict]:
        with self.pool.session() as conn:
            row = conn.execute(
                "SELECT object_key, expires_at FROM signed_urls WHERE token = %s",
                (token,),
            ).fetchone()
        if row is None:
            return None
        return {"object_key": str(row["object_key"]), "expires_at": _iso(row["expires_at"])}

    def accept_webhook(self, provider: str, event_id: str, now: str) -> bool:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                INSERT INTO webhook_events (event_id, provider, received_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                RETURNING event_id
                """,
                (event_id, provider, now),
            ).fetchone()
        return row is not None

    def record_payment(self, *, provider: str, kind: str, user_id: str, idempotency_key: str, now: str) -> str:
        with self.pool.session() as conn:
            existing = conn.execute(
                "SELECT id FROM payment_transactions WHERE idempotency_key = %s",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return str(existing["id"])
            payment_id = "pay_" + uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO payment_transactions (
                    id, provider, kind, status, idempotency_key, user_id, created_at
                ) VALUES (%s, %s, %s, 'verified', %s, %s, %s)
                """,
                (payment_id, provider, kind, idempotency_key, user_id, now),
            )
            return payment_id

    def record_meter(self, **fields: object) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO usage_meters (id, job_id, service, meter_type, amount, pricing_version, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    "meter_" + uuid.uuid4().hex, fields["job_id"], fields["service"], fields["meter_type"],
                    fields["amount"], fields["pricing_version"], fields["now"],
                ),
            )
            conn.execute(
                """
                INSERT INTO usage_events (id, job_id, meter_type, amount, pricing_version)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    "use_" + uuid.uuid4().hex, fields["job_id"], fields["meter_type"],
                    fields["amount"], fields["pricing_version"],
                ),
            )

    def record_cost(self, job_id: str, fields: dict) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO cost_facts (
                    job_id, service, engine, model, provider, gpu_model, gpu_hourly_usd,
                    gpu_seconds, inference_seconds, cold_start_seconds, model_load_seconds,
                    api_cost_usd, storage_cost_usd, bandwidth_cost_usd, failed_job_cost_usd,
                    actual_cost_usd
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id) DO UPDATE SET actual_cost_usd = EXCLUDED.actual_cost_usd
                """,
                (
                    job_id, fields.get("service"), fields.get("engine"), fields.get("model"),
                    fields.get("provider"), fields.get("gpu_model"), fields.get("gpu_hourly_usd"),
                    fields.get("gpu_seconds"), fields.get("inference_seconds"),
                    fields.get("cold_start_seconds"), fields.get("model_load_seconds"),
                    fields.get("api_cost_usd"), fields.get("storage_cost_usd"),
                    fields.get("bandwidth_cost_usd"), fields.get("failed_job_cost_usd"),
                    str(fields.get("actual_cost_usd", "0")),
                ),
            )

    def ledger_totals(self) -> dict[str, int]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT entry_type, COALESCE(SUM(amount_units), 0) AS total
                FROM credit_ledger GROUP BY entry_type
                """
            ).fetchall()
        return {str(row["entry_type"]): int(row["total"]) for row in rows}

    def actual_cost_sum(self) -> float:
        with self.pool.session() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(CAST(actual_cost_usd AS DOUBLE PRECISION)), 0) AS total FROM cost_facts"
            ).fetchone()
        return float(row["total"])

    def save_continuity(self, row: dict, now: str) -> str:
        state_id = str(row.get("id") or "cont_" + uuid.uuid4().hex)
        state = {
            "characters": row.get("characters"),
            "objects": row.get("objects"),
            "location": row.get("location"),
            "lighting": row.get("lighting"),
            "visual_style": row.get("visual_style"),
            "camera": row.get("camera"),
            "movement_direction": row.get("movement_direction"),
            "story_state": row.get("story_state"),
            "previous_prompt": row.get("previous_prompt"),
            "last_frame_artifact_id": row.get("last_frame_artifact_id"),
        }
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO continuity_states (
                    id, project_id, clip_index, state, continuation_prompt, continuation_prompt_edited,
                    scene_id, characters, objects, location, lighting, visual_style, camera,
                    movement_direction, story_state, previous_prompt, last_frame_artifact_id, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, FALSE, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    state_id, row["project_id"], row["clip_index"], Json(state), row.get("continuation_prompt"),
                    row.get("scene_id"), row.get("characters"), row.get("objects"), row.get("location"),
                    row.get("lighting"), row.get("visual_style"), row.get("camera"),
                    row.get("movement_direction"), row.get("story_state"), row.get("previous_prompt"),
                    row.get("last_frame_artifact_id"), now,
                ),
            )
        return state_id

    def edit_continuation(self, state_id: str, prompt: str) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                UPDATE continuity_states
                SET continuation_prompt = %s, continuation_prompt_edited = TRUE
                WHERE id = %s
                """,
                (prompt, state_id),
            )

    def continuation_prompt(self, state_id: str) -> Optional[str]:
        with self.pool.session() as conn:
            row = conn.execute(
                "SELECT continuation_prompt FROM continuity_states WHERE id = %s",
                (state_id,),
            ).fetchone()
        if row is None or row["continuation_prompt"] is None:
            return None
        return str(row["continuation_prompt"])

    def record_benchmark(self, fields: dict) -> str:
        benchmark_id = "bench_" + uuid.uuid4().hex
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO benchmarks (
                    id, service, model, provider, gpu, quality, resolution, duration_ms,
                    vram_mib, cold_start_seconds, warm_start_seconds, model_load_seconds,
                    inference_seconds, jobs_per_hour, concurrency, provider_cost_usd,
                    cost_per_job_usd, recorded_at, version
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    benchmark_id, fields["service"], fields["model"], fields["provider"],
                    fields.get("gpu"), fields.get("quality"), fields.get("resolution"),
                    fields.get("duration_ms"), fields.get("vram_mib"), fields.get("cold_start_seconds"),
                    fields.get("warm_start_seconds"), fields.get("model_load_seconds"),
                    fields.get("inference_seconds"), fields.get("jobs_per_hour"),
                    fields.get("concurrency"), fields.get("provider_cost_usd"),
                    fields.get("cost_per_job_usd"), fields["recorded_at"], fields["version"],
                ),
            )
        return benchmark_id

    def benchmark_count(self) -> int:
        with self.pool.session() as conn:
            row = conn.execute("SELECT COUNT(id) AS n FROM benchmarks").fetchone()
        return int(row["n"])

    def mutate_ledger_for_test(self) -> None:
        with self.pool.session() as conn:
            conn.execute("UPDATE credit_ledger SET amount_units = 0")

    def _ensure_user(self, conn: object, user_id: str, now: str) -> None:
        conn.execute(
            "INSERT INTO users (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (user_id, now),
        )

    def _balance(self, conn: object, user_id: str, pool: str) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(credit_ledger.amount_units), 0) AS total
            FROM credit_ledger
            JOIN credit_accounts ON credit_accounts.id = credit_ledger.credit_account_id
            WHERE credit_accounts.user_id = %s AND credit_accounts.pool = %s
            """,
            (user_id, pool),
        ).fetchone()
        return int(row["total"])

    def _settled(self, conn: object, job_id: str) -> bool:
        row = conn.execute(
            """
            SELECT id FROM credit_ledger
            WHERE job_id = %s AND entry_type IN ('consume', 'release', 'refund')
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        return row is not None

    def _enqueue(self, conn: object, user_id: str, job_id: str, idempotency_key: str,
                 pricing_version: str, max_concurrent: int, now: str) -> None:
        conn.execute(
            """
            INSERT INTO durable_jobs (
                job_id, user_id, status, idempotency_key, attempts, pricing_version,
                max_concurrent, created_at, updated_at
            ) VALUES (%s, %s, 'queued', %s, 0, %s, %s, %s, %s)
            ON CONFLICT (job_id) DO NOTHING
            """,
            (job_id, user_id, idempotency_key, pricing_version, max_concurrent, now, now),
        )
        conn.execute(
            """
            INSERT INTO generation_jobs (id, user_id, status, pricing_version, created_at)
            VALUES (%s, %s, 'queued', %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (job_id, user_id, pricing_version, now),
        )

    def _post(self, **kwargs: object) -> str:
        with self.pool.session() as conn:
            self._ensure_user(conn, str(kwargs["user_id"]), str(kwargs["now"]))
            return self._post_conn(conn, **kwargs)

    def _post_conn(self, conn: object, **kwargs: object) -> str:
        idempotency_key = str(kwargs["idempotency_key"])
        existing = conn.execute(
            "SELECT id FROM credit_ledger WHERE idempotency_key = %s",
            (idempotency_key,),
        ).fetchone()
        if existing is not None:
            return str(existing["id"])
        user_id = str(kwargs["user_id"])
        pool = str(kwargs["pool"])
        account = conn.execute(
            "SELECT id FROM credit_accounts WHERE user_id = %s AND pool = %s",
            (user_id, pool),
        ).fetchone()
        if account is None:
            account_id = "acct_" + uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO credit_accounts (id, user_id, pool) VALUES (%s, %s, %s)
                ON CONFLICT (user_id, pool) DO NOTHING
                """,
                (account_id, user_id, pool),
            )
            account = conn.execute(
                "SELECT id FROM credit_accounts WHERE user_id = %s AND pool = %s",
                (user_id, pool),
            ).fetchone()
            account_id = str(account["id"])
        else:
            account_id = str(account["id"])
        entry_id = "led_" + uuid.uuid4().hex
        inserted = conn.execute(
            """
            INSERT INTO credit_ledger (
                id, credit_account_id, pool, entry_type, amount_units, job_id,
                subscription_id, payment_transaction_id, idempotency_key, pricing_version, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (
                entry_id, account_id, pool, kwargs["entry_type"], int(kwargs["amount"]),
                kwargs.get("job_id") or None, kwargs.get("subscription_id") or None,
                kwargs.get("payment_id") or None, idempotency_key, kwargs.get("pricing_version") or None,
                kwargs["now"],
            ),
        ).fetchone()
        if inserted is not None:
            return str(inserted["id"])
        again = conn.execute(
            "SELECT id FROM credit_ledger WHERE idempotency_key = %s",
            (idempotency_key,),
        ).fetchone()
        return str(again["id"])
