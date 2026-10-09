"""PostgreSQL stores for jobs, studio records, usage, and provider resources."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from psycopg import errors
from psycopg.types.json import Json

from media_engine.jobs.model import (
    INTERRUPTED_ATTEMPT,
    AttemptStatus,
    Job,
    JobNotFound,
    _input_payload,
    _row_to_job,
)
from media_engine.platform.postgres_conn import PgPool
from media_engine.platform.postgres_repo import _iso
from media_engine.resources.ledger import ResourceRecord
from media_engine.storage.artifacts import Artifact
from media_engine.studio.store import (
    TRACKS,
    StudioError,
    _asset,
    _name,
    _project,
    _prompt,
    _scene,
    _settings,
    _type,
)
from media_engine.usage import UsageEvent


def _coerce(row: dict) -> dict:
    converted: dict = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            converted[key] = _iso(value)
        elif isinstance(value, (dict, list)):
            converted[key] = json.dumps(value, separators=(",", ":"))
        else:
            converted[key] = value
    return converted


class PostgresJobStore:
    def __init__(self, pool: PgPool) -> None:
        self.pool = pool
        self.db_path = ""

    def insert(self, job: Job) -> Job:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, client_id, job_type, status, created_at, updated_at,
                    attempt_count, result_uri, error_code, input_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job.id, job.client_id, job.job_type, job.status, job.created_at,
                    job.updated_at, job.attempt_count, job.result_uri, job.error_code,
                    Json(_input_payload(job)),
                ),
            )
        return job

    def find_by_idempotency(self, idempotency_key: str) -> Optional[Job]:
        with self.pool.session() as conn:
            row = conn.execute(
                "SELECT job_id FROM durable_jobs WHERE idempotency_key = %s",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            return self.get(str(row["job_id"]))
        except JobNotFound:
            return None

    def insert_idempotent(self, job: Job, idempotency_key: str) -> Job:
        try:
            with self.pool.session() as conn:
                conn.execute(
                    "INSERT INTO users (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (job.client_id, job.created_at),
                )
                conn.execute(
                    """
                    INSERT INTO durable_jobs (
                        job_id, user_id, status, idempotency_key, attempts, pricing_version,
                        max_concurrent, created_at, updated_at
                    ) VALUES (%s, %s, 'queued', %s, 0, '', 1, %s, %s)
                    """,
                    (job.id, job.client_id, idempotency_key, job.created_at, job.updated_at),
                )
                conn.execute(
                    """
                    INSERT INTO jobs (
                        id, client_id, job_type, status, created_at, updated_at,
                        attempt_count, result_uri, error_code, input_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job.id, job.client_id, job.job_type, job.status, job.created_at,
                        job.updated_at, job.attempt_count, job.result_uri, job.error_code,
                        Json(_input_payload(job)),
                    ),
                )
        except errors.UniqueViolation:
            found = self.find_by_idempotency(idempotency_key)
            if found is None:
                raise
            return found
        return job

    def save(self, job: Job) -> Job:
        current = self.get(job.id)
        if _progress_only(current, job):
            return job
        with self.pool.session() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                   SET status = %s, updated_at = %s, attempt_count = %s,
                       result_uri = %s, error_code = %s, input_json = %s
                 WHERE id = %s
                """,
                (
                    job.status, job.updated_at, job.attempt_count, job.result_uri,
                    job.error_code, Json(_input_payload(job)), job.id,
                ),
            )
            if cursor.rowcount != 1:
                raise JobNotFound(job.id)
        return job

    def get(self, job_id: str) -> Job:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT id, client_id, job_type, status, created_at, updated_at,
                       attempt_count, result_uri, error_code, input_json
                FROM jobs WHERE id = %s
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return _row_to_job(_coerce(row))

    def list(self) -> list[Job]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT id, client_id, job_type, status, created_at, updated_at,
                       attempt_count, result_uri, error_code, input_json
                FROM jobs
                ORDER BY created_at, id
                """
            ).fetchall()
        return [_row_to_job(_coerce(row)) for row in rows]


class PostgresAttemptStore:
    def __init__(self, pool: PgPool, *, max_attempts: int) -> None:
        self.pool = pool
        self.max_attempts = max_attempts

    def count(self, job_id: str) -> int:
        with self.pool.session() as conn:
            row = conn.execute(
                "SELECT COUNT(attempt_id) AS n FROM job_attempts WHERE job_id = %s",
                (job_id,),
            ).fetchone()
        return int(row["n"])

    def start(self, job_id: str, now_iso: str) -> str:
        if self.count(job_id) >= self.max_attempts:
            from media_engine.jobs.model import JobError
            raise JobError("ATTEMPT_LIMIT")
        attempt_id = "attempt_" + uuid.uuid4().hex
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO job_attempts (
                    attempt_id, job_id, status, error_code, created_at, updated_at
                ) VALUES (%s, %s, %s, NULL, %s, %s)
                """,
                (attempt_id, job_id, AttemptStatus.STARTED, now_iso, now_iso),
            )
        return attempt_id

    def finish(self, attempt_id: str, status: str, now_iso: str, error_code: Optional[str] = None) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                UPDATE job_attempts
                   SET status = %s, error_code = %s, updated_at = %s
                 WHERE attempt_id = %s
                """,
                (status, error_code, now_iso, attempt_id),
            )

    def interrupt_started(self, job_id: str, now_iso: str) -> int:
        with self.pool.session() as conn:
            cursor = conn.execute(
                """
                UPDATE job_attempts
                   SET status = %s, error_code = %s, updated_at = %s
                 WHERE job_id = %s AND status = %s
                """,
                (AttemptStatus.INTERRUPTED, INTERRUPTED_ATTEMPT, now_iso, job_id, AttemptStatus.STARTED),
            )
            return int(cursor.rowcount)


class PostgresUsageStore:
    def __init__(self, pool: PgPool) -> None:
        self.pool = pool

    def record(self, event: UsageEvent) -> None:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO job_usage (
                    job_id, client_id, job_type, engine_id, started_at, finished_at,
                    duration_seconds, attempt_count, estimated_cost_usd, status,
                    provider, gpu_model, hourly_price_usd, gpu_seconds, actual_cost_usd,
                    credit_before, credit_after, credit_delta,
                    calculated_runtime_cost, unexplained_cost_delta,
                    instance_id, machine_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (job_id) DO NOTHING
                """,
                (
                    event.job_id, event.client_id, event.job_type, event.engine_id,
                    event.started_at, event.finished_at, event.duration_seconds,
                    event.attempt_count, event.estimated_cost_usd, event.status,
                    event.provider, event.gpu_model, event.hourly_price_usd,
                    event.gpu_seconds, event.actual_cost_usd,
                    event.credit_before, event.credit_after, event.credit_delta,
                    event.calculated_runtime_cost, event.unexplained_cost_delta,
                    event.instance_id, event.machine_id,
                ),
            )

    def for_job(self, job_id: str) -> Optional[UsageEvent]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT job_id, client_id, job_type, engine_id, started_at, finished_at,
                       duration_seconds, attempt_count, estimated_cost_usd, status,
                       provider, gpu_model, hourly_price_usd, gpu_seconds, actual_cost_usd,
                       credit_before, credit_after, credit_delta, calculated_runtime_cost,
                       unexplained_cost_delta, instance_id, machine_id
                FROM job_usage WHERE job_id = %s
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        row = _coerce(row)
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
            credit_before=row["credit_before"],
            credit_after=row["credit_after"],
            credit_delta=row["credit_delta"],
            calculated_runtime_cost=row["calculated_runtime_cost"],
            unexplained_cost_delta=row["unexplained_cost_delta"],
            instance_id=row["instance_id"],
            machine_id=row["machine_id"],
        )


class PostgresArtifactStore:
    def __init__(self, pool: PgPool) -> None:
        self.pool = pool

    def insert(self, artifact: Artifact) -> Artifact:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO job_artifacts (
                    artifact_id, job_id, engine, model, prompt, seed, width, height,
                    created_at, byte_count, sha256, path
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    artifact.artifact_id, artifact.job_id, artifact.engine, artifact.model,
                    artifact.prompt, artifact.seed, artifact.width, artifact.height,
                    artifact.created_at, artifact.byte_count, artifact.sha256, artifact.path,
                ),
            )
        return artifact

    def get(self, artifact_id: str) -> Optional[Artifact]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT artifact_id, job_id, engine, model, prompt, seed, width, height,
                       created_at, byte_count, sha256, path
                FROM job_artifacts WHERE artifact_id = %s
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        row = _coerce(row)
        return Artifact(
            artifact_id=row["artifact_id"],
            job_id=row["job_id"],
            engine=row["engine"],
            model=row["model"],
            prompt=row["prompt"],
            seed=row["seed"],
            width=row["width"],
            height=row["height"],
            created_at=row["created_at"],
            byte_count=int(row["byte_count"]),
            sha256=row["sha256"],
            path=row["path"],
        )


class PostgresResourceLedger:
    def __init__(self, pool: PgPool) -> None:
        self.pool = pool

    def upsert(self, record: ResourceRecord) -> ResourceRecord:
        with self.pool.session() as conn:
            conn.execute(
                """
                INSERT INTO external_resources (
                    resource_id, provider, owner_id, state, create_attempted,
                    hourly_price_usd, created_at, updated_at, last_error
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (resource_id) DO UPDATE SET
                    provider = EXCLUDED.provider,
                    owner_id = EXCLUDED.owner_id,
                    state = EXCLUDED.state,
                    create_attempted = EXCLUDED.create_attempted,
                    hourly_price_usd = EXCLUDED.hourly_price_usd,
                    updated_at = EXCLUDED.updated_at,
                    last_error = EXCLUDED.last_error
                """,
                (
                    record.resource_id, record.provider, record.owner_id, record.state,
                    1 if record.create_attempted else 0, record.hourly_price_usd,
                    record.created_at, record.updated_at, record.last_error,
                ),
            )
        return record

    def get(self, resource_id: str) -> Optional[ResourceRecord]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT resource_id, provider, owner_id, state, create_attempted,
                       hourly_price_usd, created_at, updated_at, last_error
                FROM external_resources WHERE resource_id = %s
                """,
                (resource_id,),
            ).fetchone()
        return None if row is None else _resource(row)

    def list(self) -> list[ResourceRecord]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT resource_id, provider, owner_id, state, create_attempted,
                       hourly_price_usd, created_at, updated_at, last_error
                FROM external_resources
                ORDER BY created_at, resource_id
                """
            ).fetchall()
        return [_resource(row) for row in rows]

    def delete(self, resource_id: str) -> None:
        with self.pool.session() as conn:
            conn.execute(
                "DELETE FROM external_resources WHERE resource_id = %s",
                (resource_id,),
            )

    def unresolved(self) -> list[ResourceRecord]:
        from media_engine.resources.ledger import is_unresolved
        return [record for record in self.list() if is_unresolved(record)]


class PostgresStudioStore:
    def __init__(self, pool: PgPool, user_id: str) -> None:
        self.pool = pool
        self.user_id = user_id

    def create_project(self, name: str, now: str) -> dict:
        title = _name(name)
        project_id = "proj_" + uuid.uuid4().hex
        with self.pool.session() as conn:
            conn.execute(
                "INSERT INTO users (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (self.user_id, now),
            )
            conn.execute(
                """
                INSERT INTO projects (id, user_id, name, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (project_id, self.user_id, title, now, now),
            )
        return self.get_project(project_id)

    def list_projects(self) -> list[dict]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT id, name, created_at, updated_at
                FROM projects
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """,
                (self.user_id,),
            ).fetchall()
        return [_project(_coerce(row)) for row in rows]

    def get_project(self, project_id: str) -> dict:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT id, name, created_at, updated_at
                FROM projects WHERE id = %s AND user_id = %s
                """,
                (project_id, self.user_id),
            ).fetchone()
        if row is None:
            raise StudioError("PROJECT_NOT_FOUND")
        project = _project(_coerce(row))
        project["scenes"] = self.list_scenes(project_id)
        return project

    def update_project(self, project_id: str, name: str, now: str) -> dict:
        self.get_project(project_id)
        with self.pool.session() as conn:
            conn.execute(
                "UPDATE projects SET name = %s, updated_at = %s WHERE id = %s AND user_id = %s",
                (_name(name), now, project_id, self.user_id),
            )
        return self.get_project(project_id)

    def create_scene(self, project_id: str, name: str, now: str) -> dict:
        self.get_project(project_id)
        title = _name(name or "Scene")
        with self.pool.session() as conn:
            count = conn.execute(
                "SELECT COUNT(id) AS n FROM scenes WHERE project_id = %s",
                (project_id,),
            ).fetchone()
            scene_id = "scn_" + uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO scenes (id, project_id, name, position, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (scene_id, project_id, title, int(count["n"]), now, now),
            )
        for asset_type in TRACKS:
            self.create_asset(scene_id, asset_type, "", now, placeholder=True)
        return self.get_scene(scene_id)

    def find_asset_by_job(self, job_id: str) -> Optional[dict]:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT studio_assets.id, studio_assets.project_id, studio_assets.scene_id
                FROM studio_assets
                JOIN projects ON projects.id = studio_assets.project_id
                WHERE studio_assets.job_id = %s AND projects.user_id = %s
                """,
                (job_id, self.user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "asset_id": str(row["id"]),
            "project_id": str(row["project_id"]),
            "scene_id": str(row["scene_id"]),
        }

    def list_scenes(self, project_id: str) -> list[dict]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT scenes.id
                FROM scenes
                JOIN projects ON projects.id = scenes.project_id
                WHERE scenes.project_id = %s AND projects.user_id = %s
                ORDER BY scenes.position, scenes.created_at
                """,
                (project_id, self.user_id),
            ).fetchall()
        return [self.get_scene(row["id"]) for row in rows]

    def get_scene(self, scene_id: str) -> dict:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT scenes.id, scenes.project_id, scenes.name, scenes.position,
                       scenes.created_at, scenes.updated_at
                FROM scenes
                JOIN projects ON projects.id = scenes.project_id
                WHERE scenes.id = %s AND projects.user_id = %s
                """,
                (scene_id, self.user_id),
            ).fetchone()
        if row is None:
            raise StudioError("SCENE_NOT_FOUND")
        scene = _scene(_coerce(row))
        scene["assets"] = self.list_assets(scene_id)
        return scene

    def update_scene(self, scene_id: str, name: str, now: str) -> dict:
        self.get_scene(scene_id)
        with self.pool.session() as conn:
            conn.execute(
                """
                UPDATE scenes SET name = %s, updated_at = %s
                WHERE id = %s AND project_id IN (
                    SELECT id FROM projects WHERE user_id = %s
                )
                """,
                (_name(name), now, scene_id, self.user_id),
            )
        return self.get_scene(scene_id)

    def delete_scene(self, scene_id: str) -> None:
        self.get_scene(scene_id)
        with self.pool.session() as conn:
            conn.execute("DELETE FROM assets WHERE scene_id = %s", (scene_id,))
            conn.execute("DELETE FROM studio_assets WHERE scene_id = %s", (scene_id,))
            conn.execute("DELETE FROM scenes WHERE id = %s", (scene_id,))

    def create_asset(self, scene_id: str, asset_type: str, prompt: str, now: str,
                     *, settings: Optional[dict] = None, placeholder: bool = False) -> dict:
        with self.pool.session() as conn:
            scene = conn.execute(
                """
                SELECT scenes.project_id
                FROM scenes
                JOIN projects ON projects.id = scenes.project_id
                WHERE scenes.id = %s AND projects.user_id = %s
                """,
                (scene_id, self.user_id),
            ).fetchone()
            if scene is None:
                raise StudioError("SCENE_NOT_FOUND")
            kind = _type(asset_type)
            text = "" if placeholder else _prompt(prompt, required=kind == "image")
            asset_id = "ast_" + uuid.uuid4().hex
            stored = _settings(settings or {})
            engine = "qwen-image" if kind == "image" else None
            model = "Qwen/Qwen-Image" if kind == "image" else None
            conn.execute(
                "INSERT INTO assets (id, scene_id, asset_type) VALUES (%s, %s, %s)",
                (asset_id, scene_id, kind),
            )
            conn.execute(
                """
                INSERT INTO studio_assets (
                    id, project_id, scene_id, asset_type, engine, model, prompt, status,
                    artifact_id, job_id, created_at, updated_at, settings_json, history_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'draft', NULL, NULL, %s, %s, %s, %s)
                """,
                (
                    asset_id, scene["project_id"], scene_id, kind, engine, model, text,
                    now, now, Json(stored), Json([]),
                ),
            )
        return self.get_asset(asset_id)

    def list_assets(self, scene_id: str) -> list[dict]:
        with self.pool.session() as conn:
            rows = conn.execute(
                """
                SELECT studio_assets.id, studio_assets.project_id, studio_assets.scene_id,
                       studio_assets.asset_type, studio_assets.engine, studio_assets.model,
                       studio_assets.prompt, studio_assets.status, studio_assets.artifact_id,
                       studio_assets.job_id, studio_assets.created_at, studio_assets.updated_at,
                       studio_assets.settings_json, studio_assets.history_json
                FROM studio_assets
                JOIN projects ON projects.id = studio_assets.project_id
                WHERE studio_assets.scene_id = %s AND projects.user_id = %s
                ORDER BY studio_assets.created_at
                """,
                (scene_id, self.user_id),
            ).fetchall()
        return [_asset(_coerce(row)) for row in rows]

    def get_asset(self, asset_id: str) -> dict:
        with self.pool.session() as conn:
            row = conn.execute(
                """
                SELECT studio_assets.id, studio_assets.project_id, studio_assets.scene_id,
                       studio_assets.asset_type, studio_assets.engine, studio_assets.model,
                       studio_assets.prompt, studio_assets.status, studio_assets.artifact_id,
                       studio_assets.job_id, studio_assets.created_at, studio_assets.updated_at,
                       studio_assets.settings_json, studio_assets.history_json
                FROM studio_assets
                JOIN projects ON projects.id = studio_assets.project_id
                WHERE studio_assets.id = %s AND projects.user_id = %s
                """,
                (asset_id, self.user_id),
            ).fetchone()
        if row is None:
            raise StudioError("ASSET_NOT_FOUND")
        return _asset(_coerce(row))

    def update_asset(self, asset_id: str, now: str, *, prompt: Optional[str] = None,
                     settings: Optional[dict] = None, job_id: Optional[str] = None,
                     artifact_id: Optional[str] = None, status: Optional[str] = None) -> dict:
        current = self.get_asset(asset_id)
        text = current["prompt"] if prompt is None else _prompt(prompt, required=False)
        stored = current["settings"] if settings is None else _settings({**current["settings"], **settings})
        history = list(current["history"])
        current_job = current["job_id"]
        if job_id:
            current_job = job_id
            if job_id not in history:
                history.append(job_id)
        current_artifact = current["artifact_id"] if artifact_id is None else artifact_id
        current_status = current["status"] if status is None else status
        if job_id and status is None:
            current_status = "queued"
        with self.pool.session() as conn:
            conn.execute(
                """
                UPDATE studio_assets
                   SET prompt = %s, settings_json = %s, job_id = %s, artifact_id = %s,
                       status = %s, history_json = %s, updated_at = %s
                 WHERE id = %s
                """,
                (text, Json(stored), current_job, current_artifact, current_status, Json(history), now, asset_id),
            )
        return self.get_asset(asset_id)


def _progress_only(current: Job, job: Job) -> bool:
    same = (
        current.status == job.status
        and current.public_status == job.public_status
        and current.result_uri == job.result_uri
        and current.error_code == job.error_code
        and current.attempt_count == job.attempt_count
        and current.artifact_id == job.artifact_id
        and current.sha256 == job.sha256
        and current.byte_count == job.byte_count
        and current.prompt == job.prompt
    )
    return same and current.progress != job.progress


def _resource(row: dict) -> ResourceRecord:
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
