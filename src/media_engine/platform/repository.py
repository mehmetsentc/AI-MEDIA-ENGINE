"""SQLite repository for platform metadata. Callers do not write SQL."""
from __future__ import annotations

import sqlite3
import uuid
from typing import Optional

from media_engine.db import connect
from media_engine.platform.schema import SQLITE_SCHEMA


class PlatformBlocked(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SqliteRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        conn = connect(db_path)
        try:
            conn.executescript(SQLITE_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO platform_config (key, value) VALUES ('pool_priority', 'trial,promo,subscription,topup')"
            )
            conn.execute(
                "INSERT OR IGNORE INTO kill_switches (scope, target, engaged) VALUES ('global', '*', 0)"
            )
            conn.commit()
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        return connect(self.db_path)

    def ensure_user(self, user_id: str, now: str) -> None:
        conn = self._conn()
        try:
            conn.execute("INSERT OR IGNORE INTO users (id, created_at) VALUES (?, ?)", (user_id, now))
            conn.commit()
        finally:
            conn.close()

    def set_switch(self, scope: str, target: str, engaged: bool) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO kill_switches (scope, target, engaged) VALUES (?, ?, ?)
                ON CONFLICT (scope, target) DO UPDATE SET engaged = excluded.engaged
                """,
                (scope, target, 1 if engaged else 0),
            )
            conn.commit()
        finally:
            conn.close()

    def engaged_switches(self) -> list[tuple[str, str]]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT scope, target FROM kill_switches WHERE engaged = 1"
            ).fetchall()
        finally:
            conn.close()
        return [(str(row["scope"]), str(row["target"])) for row in rows]

    def add_plan(self, plan_id: str, category: str) -> None:
        conn = self._conn()
        try:
            conn.execute("INSERT INTO plans (id, category) VALUES (?, ?)", (plan_id, category))
            conn.commit()
        finally:
            conn.close()

    def set_entitlement(self, key: str, value_json: str, *, plan_id: str = "", user_id: str = "") -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO entitlements (id, plan_id, user_id, key, value_json) VALUES (?, ?, ?, ?, ?)",
                ("ent_" + uuid.uuid4().hex, plan_id or None, user_id or None, key, value_json),
            )
            conn.commit()
        finally:
            conn.close()

    def entitlement_json(self, user_id: str, key: str) -> Optional[str]:
        conn = self._conn()
        try:
            row = conn.execute(
                """
                SELECT value_json FROM entitlements
                WHERE user_id = ? AND key = ?
                ORDER BY id DESC LIMIT 1
                """,
                (user_id, key),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else str(row["value_json"])

    def set_config(self, key: str, value: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO platform_config (key, value) VALUES (?, ?)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
            conn.commit()
        finally:
            conn.close()

    def config(self, key: str) -> Optional[str]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT value FROM platform_config WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
        return None if row is None else str(row["value"])

    def grant(self, *, user_id: str, pool: str, entry_type: str, amount: int, idempotency_key: str,
              now: str, subscription_id: str = "", payment_id: str = "") -> str:
        return self._post(
            user_id=user_id, pool=pool, entry_type=entry_type, amount=amount,
            idempotency_key=idempotency_key, now=now, subscription_id=subscription_id,
            payment_id=payment_id,
        )

    def balance(self, user_id: str, pool: Optional[str] = None) -> int:
        conn = self._conn()
        try:
            if pool is None:
                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(credit_ledger.amount_units), 0) AS total
                    FROM credit_ledger
                    JOIN credit_accounts ON credit_accounts.id = credit_ledger.credit_account_id
                    WHERE credit_accounts.user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT COALESCE(SUM(credit_ledger.amount_units), 0) AS total
                    FROM credit_ledger
                    JOIN credit_accounts ON credit_accounts.id = credit_ledger.credit_account_id
                    WHERE credit_accounts.user_id = ? AND credit_accounts.pool = ?
                    """,
                    (user_id, pool),
                ).fetchone()
        finally:
            conn.close()
        return int(row["total"])

    def add_price(self, **fields: object) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO pricing_rules (
                    id, version, service, model, quality, unit, credit_price,
                    internal_cost_units, effective_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "price_" + uuid.uuid4().hex, fields["version"], fields["service"], fields["model"],
                    fields["quality"], fields["unit"], fields["credit_price"],
                    fields["internal_cost_units"], fields["effective_at"],
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def price_at(self, service: str, model: str, at: str) -> Optional[dict]:
        conn = self._conn()
        try:
            row = conn.execute(
                """
                SELECT version, credit_price, unit, effective_at
                FROM pricing_rules
                WHERE service = ? AND model = ? AND effective_at <= ?
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
                "effective_at": str(row["effective_at"]),
            }
        finally:
            conn.close()

    def reservation_version(self, job_id: str) -> Optional[str]:
        conn = self._conn()
        try:
            row = conn.execute(
                """
                SELECT pricing_version FROM credit_ledger
                WHERE job_id = ? AND entry_type = 'reserve'
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None or row["pricing_version"] is None:
            return None
        return str(row["pricing_version"])

    def reserve(self, *, user_id: str, amount: int, job_id: str, version: str, pools: list[str], now: str) -> None:
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT id FROM credit_ledger WHERE job_id = ? AND entry_type = 'reserve' LIMIT 1",
                (job_id,),
            ).fetchone()
            if existing is not None:
                conn.commit()
                return
            remaining = amount
            takes: list[tuple[str, int]] = []
            for pool in pools:
                available = self._balance_conn(conn, user_id, pool)
                if available <= 0:
                    continue
                take = min(available, remaining)
                takes.append((pool, take))
                remaining -= take
                if remaining <= 0:
                    break
            if remaining > 0:
                conn.rollback()
                raise PlatformBlocked("INSUFFICIENT_CREDIT")
            for pool, take in takes:
                self._post_conn(
                    conn, user_id=user_id, pool=pool, entry_type="reserve", amount=-take,
                    idempotency_key=f"reserve:{job_id}:{pool}", now=now, job_id=job_id,
                    pricing_version=version,
                )
            conn.commit()
        finally:
            conn.close()

    def reserved_by_pool(self, job_id: str) -> list[tuple[str, int]]:
        conn = self._conn()
        try:
            rows = conn.execute(
                """
                SELECT pool, COALESCE(SUM(-amount_units), 0) AS total
                FROM credit_ledger
                WHERE job_id = ? AND entry_type = 'reserve'
                GROUP BY pool
                """,
                (job_id,),
            ).fetchall()
        finally:
            conn.close()
        return [(str(row["pool"]), int(row["total"])) for row in rows]

    def reserved_amount(self, job_id: str) -> int:
        conn = self._conn()
        try:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(-amount_units), 0) AS total
                FROM credit_ledger
                WHERE job_id = ? AND entry_type = 'reserve'
                """,
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        return int(row["total"])

    def post(self, **kwargs: object) -> str:
        return self._post(**kwargs)  # type: ignore[arg-type]

    def enqueue(self, *, user_id: str, job_id: str, idempotency_key: str, pricing_version: str,
                max_concurrent: int, now: str) -> None:
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT job_id FROM durable_jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return
            conn.execute(
                """
                INSERT INTO durable_jobs (
                    job_id, user_id, status, idempotency_key, attempts, pricing_version,
                    max_concurrent, created_at, updated_at
                ) VALUES (?, ?, 'queued', ?, 0, ?, ?, ?, ?)
                """,
                (job_id, user_id, idempotency_key, pricing_version, max_concurrent, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def claim(self, worker_id: str, now: str, until: str) -> Optional[str]:
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT job_id, user_id, max_concurrent FROM durable_jobs
                WHERE status = 'queued'
                ORDER BY created_at
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            active = conn.execute(
                "SELECT COUNT(*) AS n FROM durable_jobs WHERE user_id = ? AND status = 'claimed'",
                (row["user_id"],),
            ).fetchone()
            if int(active["n"]) >= int(row["max_concurrent"]):
                conn.commit()
                return None
            updated = conn.execute(
                """
                UPDATE durable_jobs
                SET status = 'claimed', lease_owner = ?, lease_until = ?, updated_at = ?,
                    attempts = attempts + 1
                WHERE job_id = ? AND status = 'queued'
                """,
                (worker_id, until, now, row["job_id"]),
            )
            conn.commit()
            if updated.rowcount != 1:
                return None
            return str(row["job_id"])
        finally:
            conn.close()

    def heartbeat(self, job_id: str, worker_id: str, until: str, now: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                UPDATE durable_jobs SET lease_until = ?, updated_at = ?
                WHERE job_id = ? AND lease_owner = ? AND status = 'claimed'
                """,
                (until, now, job_id, worker_id),
            )
            conn.commit()
        finally:
            conn.close()

    def recover_expired(self, now: str) -> int:
        conn = self._conn()
        try:
            cursor = conn.execute(
                """
                UPDATE durable_jobs
                SET status = 'queued', lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE status = 'claimed' AND lease_until < ?
                """,
                (now, now),
            )
            conn.commit()
            return int(cursor.rowcount)
        finally:
            conn.close()

    def cancel(self, job_id: str, now: str) -> bool:
        conn = self._conn()
        try:
            cursor = conn.execute(
                """
                UPDATE durable_jobs
                SET status = 'cancelled', lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE job_id = ? AND status IN ('queued', 'claimed')
                """,
                (now, job_id),
            )
            conn.commit()
            return cursor.rowcount == 1
        finally:
            conn.close()

    def job_status(self, job_id: str) -> Optional[str]:
        conn = self._conn()
        try:
            row = conn.execute("SELECT status FROM durable_jobs WHERE job_id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
        return None if row is None else str(row["status"])

    def insert_artifact(self, row: dict, now: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO media_artifacts (
                    id, user_id, project_id, scene_id, asset_id, job_id, artifact_type,
                    storage_provider, bucket, object_key, mime_type, byte_count, sha256,
                    width, height, duration_ms, provider, engine, model, generation_settings, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'r2', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"], row["user_id"], row["project_id"], row.get("scene_id"), row.get("asset_id"),
                    row["job_id"], row["artifact_type"], row["bucket"], row["object_key"], row["mime_type"],
                    row["byte_count"], row["sha256"], row.get("width"), row.get("height"), row.get("duration_ms"),
                    row.get("provider"), row.get("engine"), row.get("model"), row.get("generation_settings"), now,
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise PlatformBlocked("ARTIFACT_METADATA_FAILED") from exc
        finally:
            conn.close()

    def media_row(self, artifact_id: str) -> Optional[dict]:
        conn = self._conn()
        try:
            row = conn.execute(
                """
                SELECT object_key, bucket, sha256, byte_count
                FROM media_artifacts
                WHERE id = ? AND deleted_at IS NULL
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
        finally:
            conn.close()

    def settlement_exists(self, job_id: str) -> bool:
        conn = self._conn()
        try:
            row = conn.execute(
                """
                SELECT id FROM credit_ledger
                WHERE job_id = ? AND entry_type IN ('consume', 'release', 'refund')
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def record_orphan(self, key: str, bucket: str, digest: str, reason: str, now: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO media_orphans (object_key, bucket, sha256, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, bucket, digest, reason, now),
            )
            conn.commit()
        finally:
            conn.close()

    def reconcile_orphan(self, key: str, now: str) -> str:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT reconciled_at FROM media_orphans WHERE object_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                raise PlatformBlocked("ORPHAN_UNKNOWN")
            if row["reconciled_at"]:
                return "reconciled"
            conn.execute(
                "UPDATE media_orphans SET reconciled_at = ? WHERE object_key = ?",
                (now, key),
            )
            conn.commit()
            return "reconciled"
        finally:
            conn.close()

    def soft_delete(self, artifact_id: str, now: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE media_artifacts SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                (now, artifact_id),
            )
            conn.commit()
        finally:
            conn.close()

    def storage_bytes(self, user_id: str, project_id: str = "", kind: str = "") -> int:
        clauses = ["user_id = ?", "deleted_at IS NULL"]
        params: list[str] = [user_id]
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if kind:
            clauses.append("artifact_type = ?")
            params.append(kind)
        conn = self._conn()
        try:
            row = conn.execute(
                f"SELECT COALESCE(SUM(byte_count), 0) AS total FROM media_artifacts WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
        finally:
            conn.close()
        return int(row["total"])

    def list_artifacts(self, user_id: str, limit: int, offset: int) -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                """
                SELECT id, object_key, byte_count, sha256, artifact_type
                FROM media_artifacts
                WHERE user_id = ? AND deleted_at IS NULL
                ORDER BY created_at
                LIMIT ? OFFSET ?
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
        finally:
            conn.close()

    def issue_signed(self, token: str, object_key: str, expires_at: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO signed_urls (token, object_key, expires_at) VALUES (?, ?, ?)",
                (token, object_key, expires_at),
            )
            conn.commit()
        finally:
            conn.close()

    def signed_object(self, token: str) -> Optional[dict]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT object_key, expires_at FROM signed_urls WHERE token = ?",
                (token,),
            ).fetchone()
            if row is None:
                return None
            return {"object_key": str(row["object_key"]), "expires_at": str(row["expires_at"])}
        finally:
            conn.close()

    def accept_webhook(self, provider: str, event_id: str, now: str) -> bool:
        conn = self._conn()
        try:
            try:
                conn.execute(
                    "INSERT INTO webhook_events (event_id, provider, received_at) VALUES (?, ?, ?)",
                    (event_id, provider, now),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
        finally:
            conn.close()

    def record_payment(self, *, provider: str, kind: str, user_id: str, idempotency_key: str, now: str) -> str:
        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT id FROM payment_transactions WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return str(existing["id"])
            payment_id = "pay_" + uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO payment_transactions (
                    id, provider, kind, status, idempotency_key, user_id, created_at
                ) VALUES (?, ?, ?, 'verified', ?, ?, ?)
                """,
                (payment_id, provider, kind, idempotency_key, user_id, now),
            )
            conn.commit()
            return payment_id
        finally:
            conn.close()

    def record_meter(self, **fields: object) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO usage_meters (id, job_id, service, meter_type, amount, pricing_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "meter_" + uuid.uuid4().hex, fields["job_id"], fields["service"], fields["meter_type"],
                    fields["amount"], fields["pricing_version"], fields["now"],
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_cost(self, job_id: str, fields: dict) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO cost_facts (
                    job_id, service, engine, model, provider, gpu_model, gpu_hourly_usd,
                    gpu_seconds, inference_seconds, cold_start_seconds, model_load_seconds,
                    api_cost_usd, storage_cost_usd, bandwidth_cost_usd, failed_job_cost_usd,
                    actual_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (job_id) DO UPDATE SET actual_cost_usd = excluded.actual_cost_usd
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
            conn.commit()
        finally:
            conn.close()

    def ledger_totals(self) -> dict[str, int]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT entry_type, COALESCE(SUM(amount_units), 0) AS total FROM credit_ledger GROUP BY entry_type"
            ).fetchall()
        finally:
            conn.close()
        return {str(row["entry_type"]): int(row["total"]) for row in rows}

    def actual_cost_sum(self) -> float:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(CAST(actual_cost_usd AS REAL)), 0) AS total FROM cost_facts"
            ).fetchone()
        finally:
            conn.close()
        return float(row["total"])

    def save_continuity(self, row: dict, now: str) -> str:
        state_id = str(row.get("id") or "cont_" + uuid.uuid4().hex)
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO continuity_states (
                    id, project_id, scene_id, clip_index, characters, objects, location, lighting,
                    visual_style, camera, movement_direction, story_state, previous_prompt,
                    last_frame_artifact_id, continuation_prompt, continuation_prompt_edited, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    state_id, row["project_id"], row.get("scene_id"), row["clip_index"], row.get("characters"),
                    row.get("objects"), row.get("location"), row.get("lighting"), row.get("visual_style"),
                    row.get("camera"), row.get("movement_direction"), row.get("story_state"),
                    row.get("previous_prompt"), row.get("last_frame_artifact_id"), row.get("continuation_prompt"),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return state_id

    def edit_continuation(self, state_id: str, prompt: str) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                UPDATE continuity_states
                SET continuation_prompt = ?, continuation_prompt_edited = 1
                WHERE id = ?
                """,
                (prompt, state_id),
            )
            conn.commit()
        finally:
            conn.close()

    def continuation_prompt(self, state_id: str) -> Optional[str]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT continuation_prompt, continuation_prompt_edited FROM continuity_states WHERE id = ?",
                (state_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return str(row["continuation_prompt"])

    def record_benchmark(self, fields: dict) -> str:
        benchmark_id = "bench_" + uuid.uuid4().hex
        conn = self._conn()
        try:
            conn.execute(
                """
                INSERT INTO benchmarks (
                    id, service, model, provider, gpu, quality, resolution, duration_ms,
                    vram_mib, cold_start_seconds, warm_start_seconds, model_load_seconds,
                    inference_seconds, jobs_per_hour, concurrency, provider_cost_usd,
                    cost_per_job_usd, recorded_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            conn.commit()
        finally:
            conn.close()
        return benchmark_id

    def benchmark_count(self) -> int:
        conn = self._conn()
        try:
            row = conn.execute("SELECT COUNT(*) AS n FROM benchmarks").fetchone()
        finally:
            conn.close()
        return int(row["n"])

    def mutate_ledger_for_test(self) -> None:
        conn = self._conn()
        try:
            conn.execute("UPDATE credit_ledger SET amount_units = 0")
            conn.commit()
        finally:
            conn.close()

    def _balance_conn(self, conn: sqlite3.Connection, user_id: str, pool: str) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(credit_ledger.amount_units), 0) AS total
            FROM credit_ledger
            JOIN credit_accounts ON credit_accounts.id = credit_ledger.credit_account_id
            WHERE credit_accounts.user_id = ? AND credit_accounts.pool = ?
            """,
            (user_id, pool),
        ).fetchone()
        return int(row["total"])

    def _post(self, **kwargs: object) -> str:
        conn = self._conn()
        try:
            entry_id = self._post_conn(conn, **kwargs)
            conn.commit()
            return entry_id
        finally:
            conn.close()

    def _post_conn(self, conn: sqlite3.Connection, **kwargs: object) -> str:
        idempotency_key = str(kwargs["idempotency_key"])
        existing = conn.execute(
            "SELECT id FROM credit_ledger WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing is not None:
            return str(existing["id"])
        user_id = str(kwargs["user_id"])
        pool = str(kwargs["pool"])
        account = conn.execute(
            "SELECT id FROM credit_accounts WHERE user_id = ? AND pool = ?",
            (user_id, pool),
        ).fetchone()
        if account is None:
            account_id = "acct_" + uuid.uuid4().hex
            conn.execute(
                "INSERT INTO credit_accounts (id, user_id, pool) VALUES (?, ?, ?)",
                (account_id, user_id, pool),
            )
        else:
            account_id = str(account["id"])
        entry_id = "led_" + uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO credit_ledger (
                id, credit_account_id, pool, entry_type, amount_units, job_id,
                subscription_id, payment_transaction_id, idempotency_key, pricing_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id, account_id, pool, kwargs["entry_type"], int(kwargs["amount"]),  # type: ignore[arg-type]
                kwargs.get("job_id") or None, kwargs.get("subscription_id") or None,
                kwargs.get("payment_id") or None, idempotency_key, kwargs.get("pricing_version") or None,
                kwargs["now"],
            ),
        )
        return entry_id
