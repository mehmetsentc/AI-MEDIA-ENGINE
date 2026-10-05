"""Smallest controller that runs one IMAGE_GENERATE job on a fake GPU."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from media_engine.auth.clients import ClientDirectory
from media_engine.engines.image.base import ImageEngine
from media_engine.engines.image.fake import FakeImageEngine
from media_engine.jobs.model import (
    IMAGE_GENERATE,
    SUPPORTED_JOB_TYPES,
    AttemptStatus,
    AttemptStore,
    Job,
    JobError,
    JobNotFound,
    JobStatus,
    JobStore,
    UnsupportedJobType,
    new_job_id,
)
from media_engine.jobs.queue import JobQueue
from media_engine.jobs.recovery import RecoveryReport, recover
from media_engine.providers.base import (
    GPUProvider,
    OBSERVE_FOREIGN,
    OBSERVE_GONE,
    OBSERVE_PRESENT,
    CreateResult,
    ProviderCapacityError,
    Quote,
)
from media_engine.providers.fake import FakeGPUProvider
from media_engine.resources.ledger import ResourceLedger, ResourceRecord, ResourceState
from media_engine.safety.limits import SafetyLimits
from media_engine.storage.local import LocalStorage


class ManualClock:
    """Injectable clock. Tests advance this instead of sleeping."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("clock cannot move backwards")
        self.t += seconds

    def iso(self) -> str:
        return datetime.fromtimestamp(self.t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MediaController:
    def __init__(self, db_path: str, storage_root: str, *, clock: Optional[ManualClock] = None,
                 limits: Optional[SafetyLimits] = None, provider: Optional[GPUProvider] = None,
                 image_engine: Optional[ImageEngine] = None,
                 clients: Optional[ClientDirectory] = None,
                 idle_wait: Optional[Callable[[threading.Condition], None]] = None) -> None:
        self.clock = clock or ManualClock()
        self.limits = limits or SafetyLimits()
        self.clients = clients or ClientDirectory()
        self.jobs = JobStore(db_path)
        self.attempts = AttemptStore(db_path, max_attempts=self.limits.max_job_attempts)
        self.ledger = ResourceLedger(db_path)
        self.queue = JobQueue()
        self.provider = provider or FakeGPUProvider(
            self.clock.now, max_workers=self.limits.max_gpu_workers,
        )
        self.image_engine = image_engine or FakeImageEngine()
        self.storage = LocalStorage(storage_root)
        self.trace: list[str] = []
        self.last_shutdown_reason: Optional[str] = None
        self.phase_log: list[str] = []
        self.idle_waits = 0
        self.recovery_complete = False
        self._worker_id: Optional[str] = None
        self._worker_created_at: Optional[float] = None
        self._worker_idle_since: Optional[float] = None
        self._lock = threading.Lock()
        self._wake = threading.Condition()
        self._ready = threading.Event()
        self._settled = threading.Event()
        self._stop_requested = False
        self._thread: Optional[threading.Thread] = None
        self._idle_wait = idle_wait or (lambda cond: cond.wait())

    def startup(self) -> RecoveryReport:
        with self._lock:
            report = recover(self.jobs, self.queue, self.attempts, now_iso=self.clock.iso())
            self._reconcile_ledger()
            self.recovery_complete = True
            self.phase_log.append("recovery_complete")
            return report

    def start(self) -> RecoveryReport:
        """Recover state, then run one background worker. Recovery finishes first."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("runner already started")
        report = self.startup()
        self._stop_requested = False
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_loop, name="media-engine-runner", daemon=False)
        self._thread.start()
        if not self._ready.wait(timeout=2):
            raise RuntimeError("runner did not start")
        return report

    def stop(self) -> None:
        """Stop accepting new runner work, wake the thread, and join it."""
        with self._lock:
            self._stop_requested = True
        with self._wake:
            self._wake.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join()
            self._thread = None

    @property
    def runner_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait_settled(self, timeout: float = 2.0) -> bool:
        return self._settled.wait(timeout)

    def submit_job(self, client_id: str, job_type: str, prompt: str) -> Job:
        with self._lock:
            self.clients.require(client_id)
            if job_type not in SUPPORTED_JOB_TYPES:
                raise UnsupportedJobType(job_type)
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
                raise JobError("PROMPT_REQUIRED")
            now = self.clock.iso()
            job = Job(
                id=new_job_id(),
                client_id=client_id,
                job_type=job_type,
                status=JobStatus.QUEUED,
                created_at=now,
                updated_at=now,
                attempt_count=0,
                result_uri=None,
                error_code=None,
                prompt=prompt.strip(),
            )
            self.jobs.insert(job)
            self.queue.enqueue(job.id)
            self.trace.append(f"{job.id}:QUEUED")
            self._settled.clear()
        self._notify()
        return job

    def get_job(self, job_id: str) -> Job:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> Job:
        with self._lock:
            job = self.jobs.get(job_id)
            job.transition_to(JobStatus.CANCELLED)
            job.updated_at = self.clock.iso()
            job.error_code = "CANCELLED"
            self.jobs.save(job)
            self.queue.remove(job.id)
            self.trace.append(f"{job.id}:CANCELLED")
            return job

    def process_available(self) -> None:
        with self._lock:
            while True:
                job_id = self.queue.dequeue()
                if job_id is None:
                    break
                self._run_one(job_id)
            self._enforce()

    def enforce_lifecycle(self) -> Optional[str]:
        with self._lock:
            return self._enforce()

    def _notify(self) -> None:
        with self._wake:
            self._wake.notify()

    def _run_loop(self) -> None:
        announced = False
        while True:
            with self._wake:
                if self._stop_requested:
                    if not announced:
                        self._ready.set()
                    return
                if not self.queue.pending():
                    self.idle_waits += 1
                    if not announced:
                        self._ready.set()
                        announced = True
                    self._idle_wait(self._wake)
                    continue
                if not announced:
                    self._ready.set()
                    announced = True
            self._process_next()

    def _process_next(self) -> None:
        with self._lock:
            if self._stop_requested:
                return
            job_id = self.queue.dequeue()
            if job_id is None:
                return
            self._run_one(job_id)
            self._enforce()
        self._settled.set()

    def _reconcile_ledger(self) -> None:
        for record in self.ledger.unresolved():
            if record.state not in (ResourceState.READY, ResourceState.TERMINATING):
                continue
            observation = self.provider.observe(record.resource_id, record.owner_id)
            now = self.clock.now()
            if observation == OBSERVE_GONE:
                record.state = ResourceState.TERMINATED
                record.last_error = "RECONCILED_GONE"
                record.updated_at = now
                self.ledger.upsert(record)
                continue
            if observation == OBSERVE_PRESENT:
                self._terminate(record.resource_id, "RECONCILED_TERMINATED")
                continue
            record.last_error = (
                "OWNERSHIP_MISMATCH" if observation == OBSERVE_FOREIGN else "PROVIDER_STATUS_UNKNOWN"
            )
            record.updated_at = now
            self.ledger.upsert(record)

    def _run_one(self, job_id: str) -> None:
        try:
            job = self.jobs.get(job_id)
        except JobNotFound:
            return
        if job.status != JobStatus.QUEUED:
            return
        if self.limits.global_kill_switch:
            self._fail(job, "GLOBAL_KILL_SWITCH")
            return
        if job.attempt_count >= self.limits.max_job_attempts:
            self._fail(job, "ATTEMPT_LIMIT")
            return
        quote = self.provider.quote(job.client_id)
        if quote.hourly_price_usd > self.limits.max_hourly_price_usd:
            self._fail(job, "PRICE_ABOVE_CEILING")
            return
        reusable = self._reusable_worker()
        if reusable is None and self.ledger.unresolved():
            self._fail(job, "UNRESOLVED_RESOURCE")
            return
        self._transition(job, JobStatus.WAITING_FOR_WORKER)
        if reusable is None:
            resource_id = self._create_worker(job, quote)
            if resource_id is None:
                return
        else:
            resource_id = reusable
        self.provider.begin_busy(resource_id)
        self._worker_idle_since = None
        self.trace.append(f"{resource_id}:BUSY")
        self._transition(job, JobStatus.RUNNING)
        now_iso = self.clock.iso()
        attempt_id = self.attempts.start(job.id, now_iso)
        job.attempt_count += 1
        job.updated_at = now_iso
        self.jobs.save(job)
        try:
            data = self.image_engine.render(job.prompt)
            uri = self.storage.put(
                client_id=job.client_id, job_id=job.id, name="image.bin", data=data,
            )
        except Exception:
            self.attempts.finish(attempt_id, AttemptStatus.FAILED, self.clock.iso(), "GENERATION_FAILED")
            self._mark_idle(resource_id)
            self._fail(job, "GENERATION_FAILED")
            return
        self.attempts.finish(attempt_id, AttemptStatus.SUCCEEDED, self.clock.iso())
        job.result_uri = uri
        self._transition(job, JobStatus.SUCCEEDED)
        self._mark_idle(resource_id)

    def _create_worker(self, job: Job, quote: Quote) -> Optional[str]:
        self.phase_log.append("provision")
        provisional = "res_" + uuid.uuid4().hex
        now = self.clock.now()
        self._ledger(provisional, job.client_id, ResourceState.REQUESTED, quote, now, None)
        try:
            result: CreateResult = self.provider.create(job.client_id)
        except ProviderCapacityError:
            self._ledger(provisional, job.client_id, ResourceState.FAILED, quote, now, "MAX_GPU_WORKERS")
            self._fail(job, "MAX_GPU_WORKERS")
            return None
        if result.outcome == "definite_failure":
            self._ledger(
                provisional, job.client_id, ResourceState.FAILED, quote, now, "PROVIDER_CREATE_FAILED",
            )
            self._fail(job, "PROVIDER_CREATE_FAILED")
            return None
        if result.outcome != "created" or not result.resource_id:
            self._ledger(provisional, job.client_id, ResourceState.AMBIGUOUS, quote, now, "AMBIGUOUS_CREATE")
            self._fail(job, "AMBIGUOUS_CREATE")
            return None
        self.ledger.delete(provisional)
        created = self.clock.now()
        self._ledger(result.resource_id, job.client_id, ResourceState.READY, quote, created, None)
        self._worker_id = result.resource_id
        self._worker_created_at = created
        self._worker_idle_since = created
        self.trace.append(f"{result.resource_id}:STARTING")
        self.trace.append(f"{result.resource_id}:READY")
        return result.resource_id

    def _ledger(self, resource_id: str, owner: str, state: str, quote: Quote, now: float,
                error: Optional[str]) -> None:
        existing = self.ledger.get(resource_id)
        created = existing.created_at if existing is not None else now
        self.ledger.upsert(ResourceRecord(
            resource_id=resource_id,
            provider=quote.provider,
            owner_id=owner,
            state=state,
            create_attempted=True,
            hourly_price_usd=str(quote.hourly_price_usd),
            created_at=created,
            updated_at=now,
            last_error=error,
        ))

    def _reusable_worker(self) -> Optional[str]:
        if self._worker_id is None:
            return None
        record = self.ledger.get(self._worker_id)
        if record is None or record.state != ResourceState.READY:
            return None
        if self.provider.status(self._worker_id) != "READY":
            return None
        return self._worker_id

    def _transition(self, job: Job, target: str) -> None:
        job.transition_to(target)
        job.updated_at = self.clock.iso()
        self.jobs.save(job)
        self.trace.append(f"{job.id}:{target}")

    def _fail(self, job: Job, code: str) -> None:
        job.error_code = code
        if job.status != JobStatus.FAILED:
            job.transition_to(JobStatus.FAILED)
        job.updated_at = self.clock.iso()
        self.jobs.save(job)
        self.trace.append(f"{job.id}:FAILED:{code}")

    def _mark_idle(self, resource_id: str) -> None:
        self.provider.end_busy(resource_id)
        self._worker_idle_since = self.clock.now()
        self.trace.append(f"{resource_id}:READY")

    def _enforce(self) -> Optional[str]:
        if self._worker_id is None or self._worker_created_at is None:
            return None
        resource_id = self._worker_id
        status = self.provider.status(resource_id)
        if status == "TERMINATED":
            self._worker_id = None
            return None
        now = self.clock.now()
        if now - self._worker_created_at >= self.limits.max_worker_lifetime_seconds:
            self._terminate(resource_id, "MAX_WORKER_LIFETIME")
            return self.last_shutdown_reason
        if status == "READY" and self._worker_idle_since is not None:
            if now - self._worker_idle_since >= self.limits.gpu_idle_shutdown_seconds:
                self._terminate(resource_id, "GPU_IDLE_SHUTDOWN")
                return self.last_shutdown_reason
        return None

    def _terminate(self, resource_id: str, reason: str) -> None:
        record = self.ledger.get(resource_id)
        now = self.clock.now()
        if record is not None:
            record.state = ResourceState.TERMINATING
            record.updated_at = now
            self.ledger.upsert(record)
        self.provider.terminate(resource_id)
        self.trace.append(f"{resource_id}:STOPPING")
        self.trace.append(f"{resource_id}:TERMINATED")
        if record is not None:
            record.state = ResourceState.TERMINATED
            record.updated_at = now
            record.last_error = reason
            self.ledger.upsert(record)
        self._worker_id = None
        self._worker_created_at = None
        self._worker_idle_since = None
        self.last_shutdown_reason = reason


__all__ = ["IMAGE_GENERATE", "ManualClock", "MediaController"]
