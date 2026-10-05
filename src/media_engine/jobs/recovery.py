"""Restart policy for durable jobs.

QUEUED jobs are requeued. WAITING_FOR_WORKER jobs return to QUEUED and are
requeued. RUNNING jobs become FAILED and are not replayed. A started
attempt left open by the crash is marked interrupted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from media_engine.jobs.model import (
    INTERRUPTED_ATTEMPT,
    RESTART_DURING_JOB,
    AttemptStore,
    JobStatus,
    JobStore,
)
from media_engine.jobs.queue import JobQueue


@dataclass
class RecoveryReport:
    requeued_job_ids: list[str] = field(default_factory=list)
    failed_job_ids: list[str] = field(default_factory=list)
    untouched_job_ids: list[str] = field(default_factory=list)


def recover(jobs: JobStore, queue: JobQueue, attempts: AttemptStore, *, now_iso: str) -> RecoveryReport:
    report = RecoveryReport()
    pending = set(queue.pending())

    def _enqueue(job_id: str) -> None:
        if job_id in pending:
            return
        queue.enqueue(job_id)
        pending.add(job_id)

    for job in jobs.list():
        if job.status in JobStatus.TERMINAL:
            report.untouched_job_ids.append(job.id)
            continue
        if job.status == JobStatus.QUEUED:
            _enqueue(job.id)
            report.requeued_job_ids.append(job.id)
            continue
        if job.status == JobStatus.WAITING_FOR_WORKER:
            job.transition_to(JobStatus.QUEUED)
            job.updated_at = now_iso
            jobs.save(job)
            _enqueue(job.id)
            report.requeued_job_ids.append(job.id)
            continue
        if job.status == JobStatus.RUNNING:
            job.error_code = RESTART_DURING_JOB
            job.transition_to(JobStatus.FAILED)
            job.updated_at = now_iso
            jobs.save(job)
            attempts.interrupt_started(job.id, now_iso)
            report.failed_job_ids.append(job.id)
    return report


__all__ = ["INTERRUPTED_ATTEMPT", "RecoveryReport", "recover"]
