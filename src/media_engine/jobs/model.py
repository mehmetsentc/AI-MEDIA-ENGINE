"""Generation job state machine and SQLite metadata store."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from media_engine.db import connect, init_schema

IMAGE_GENERATE = "IMAGE_GENERATE"
TEXT_GENERATE = "TEXT_GENERATE"
SUPPORTED_JOB_TYPES = frozenset({IMAGE_GENERATE, TEXT_GENERATE})


class JobStatus:
    QUEUED = "QUEUED"
    WAITING_FOR_WORKER = "WAITING_FOR_WORKER"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELLED})
    ALL = frozenset({
        QUEUED, WAITING_FOR_WORKER, RUNNING, SUCCEEDED, FAILED, CANCELLED,
    })


ALLOWED_TRANSITIONS = {
    JobStatus.QUEUED: {JobStatus.WAITING_FOR_WORKER, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.WAITING_FOR_WORKER: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.QUEUED},
    JobStatus.RUNNING: {JobStatus.SUCCEEDED, JobStatus.FAILED},
    JobStatus.SUCCEEDED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}

PUBLIC_PROGRESS = {
    "queued": 0.0,
    "planning": 0.1,
    "provisioning": 0.2,
    "booting": 0.35,
    "runtime_preparing": 0.5,
    "model_loading": 0.65,
    "generating": 0.8,
    "saving": 0.9,
    "completed": 1.0,
}

RESTART_DURING_JOB = "RESTART_DURING_JOB"
INTERRUPTED_ATTEMPT = "INTERRUPTED_BY_RESTART"


class JobError(Exception):
    code = "JOB_ERROR"


class IllegalTransition(JobError):
    code = "ILLEGAL_TRANSITION"

    def __init__(self, job_id: str, current: str, target: str) -> None:
        super().__init__(f"{job_id} cannot move from {current} to {target}")
        self.job_id = job_id
        self.current = current
        self.target = target


class JobNotFound(JobError):
    code = "JOB_NOT_FOUND"

    def __init__(self, job_id: str) -> None:
        super().__init__(job_id)
        self.job_id = job_id


class UnsupportedJobType(JobError):
    code = "UNSUPPORTED_JOB_TYPE"


def new_job_id() -> str:
    return "job_" + uuid.uuid4().hex


@dataclass
class Job:
    id: str
    client_id: str
    job_type: str
    status: str
    created_at: str
    updated_at: str
    attempt_count: int
    result_uri: Optional[str]
    error_code: Optional[str]
    prompt: str
    task: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    public_status: Optional[str] = None
    progress: float = 0.0
    artifact_id: Optional[str] = None
    sha256: Optional[str] = None
    byte_count: Optional[int] = None

    def transition_to(self, target: str) -> None:
        allowed = ALLOWED_TRANSITIONS.get(self.status, set())
        if target not in allowed:
            raise IllegalTransition(self.id, self.status, target)
        self.status = target

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "client_id": self.client_id,
            "type": self.job_type,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "attempt_count": self.attempt_count,
            "result_uri": self.result_uri,
            "error_code": self.error_code,
        }

    def to_image_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.id,
            "status": self.public_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "error_code": self.error_code,
        }
        if self.public_status == "completed":
            payload["artifact_id"] = self.artifact_id
            payload["width"] = self.width
            payload["height"] = self.height
            payload["seed"] = self.seed
            payload["sha256"] = self.sha256
        return payload


class JobStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        init_schema(db_path)

    def insert(self, job: Job) -> Job:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, client_id, job_type, status, created_at, updated_at,
                    attempt_count, result_uri, error_code, input_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id, job.client_id, job.job_type, job.status, job.created_at,
                    job.updated_at, job.attempt_count, job.result_uri, job.error_code,
                    json.dumps(_input_payload(job), separators=(",", ":")),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return job

    def save(self, job: Job) -> Job:
        conn = connect(self.db_path)
        try:
            cur = conn.execute(
                """
                UPDATE jobs
                   SET status = ?, updated_at = ?, attempt_count = ?,
                       result_uri = ?, error_code = ?, input_json = ?
                 WHERE id = ?
                """,
                (
                    job.status, job.updated_at, job.attempt_count, job.result_uri,
                    job.error_code, json.dumps(_input_payload(job), separators=(",", ":")),
                    job.id,
                ),
            )
            if cur.rowcount != 1:
                raise JobNotFound(job.id)
            conn.commit()
        finally:
            conn.close()
        return job

    def get(self, job_id: str) -> Job:
        conn = connect(self.db_path)
        try:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise JobNotFound(job_id)
        return _row_to_job(row)

    def list(self) -> list[Job]:
        conn = connect(self.db_path)
        try:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at, id").fetchall()
        finally:
            conn.close()
        return [_row_to_job(row) for row in rows]


def _input_payload(job: Job) -> dict[str, Any]:
    payload: dict[str, Any] = {"prompt": job.prompt}
    if job.task:
        payload["task"] = job.task
    if job.public_status is not None:
        payload["width"] = job.width
        payload["height"] = job.height
        payload["seed"] = job.seed
        payload["public_status"] = job.public_status
        payload["progress"] = job.progress
        if job.artifact_id is not None:
            payload["artifact_id"] = job.artifact_id
        if job.sha256 is not None:
            payload["sha256"] = job.sha256
        if job.byte_count is not None:
            payload["byte_count"] = job.byte_count
    return payload


def _row_to_job(row: Any) -> Job:
    payload = json.loads(row["input_json"])
    return Job(
        id=row["id"],
        client_id=row["client_id"],
        job_type=row["job_type"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        attempt_count=row["attempt_count"],
        result_uri=row["result_uri"],
        error_code=row["error_code"],
        prompt=payload.get("prompt", ""),
        task=payload.get("task"),
        width=payload.get("width"),
        height=payload.get("height"),
        seed=payload.get("seed"),
        public_status=payload.get("public_status"),
        progress=float(payload.get("progress") or 0.0),
        artifact_id=payload.get("artifact_id"),
        sha256=payload.get("sha256"),
        byte_count=payload.get("byte_count"),
    )


class AttemptStatus:
    STARTED = "STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class AttemptStore:
    def __init__(self, db_path: str, *, max_attempts: int) -> None:
        self.db_path = db_path
        self.max_attempts = max_attempts
        init_schema(db_path)

    def count(self, job_id: str) -> int:
        conn = connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM job_attempts WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        return int(row["n"])

    def start(self, job_id: str, now_iso: str) -> str:
        if self.count(job_id) >= self.max_attempts:
            raise JobError("ATTEMPT_LIMIT")
        attempt_id = "attempt_" + uuid.uuid4().hex
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO job_attempts (
                    attempt_id, job_id, status, error_code, created_at, updated_at
                ) VALUES (?, ?, ?, NULL, ?, ?)
                """,
                (attempt_id, job_id, AttemptStatus.STARTED, now_iso, now_iso),
            )
            conn.commit()
        finally:
            conn.close()
        return attempt_id

    def finish(self, attempt_id: str, status: str, now_iso: str, error_code: Optional[str] = None) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE job_attempts
                   SET status = ?, error_code = ?, updated_at = ?
                 WHERE attempt_id = ?
                """,
                (status, error_code, now_iso, attempt_id),
            )
            conn.commit()
        finally:
            conn.close()

    def interrupt_started(self, job_id: str, now_iso: str) -> int:
        conn = connect(self.db_path)
        try:
            cur = conn.execute(
                """
                UPDATE job_attempts
                   SET status = ?, error_code = ?, updated_at = ?
                 WHERE job_id = ? AND status = ?
                """,
                (
                    AttemptStatus.INTERRUPTED, INTERRUPTED_ATTEMPT, now_iso,
                    job_id, AttemptStatus.STARTED,
                ),
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()
