"""The production media run: authorization, one instance, one job, one artifact.

``run_speech_rehearsal`` proves the bookkeeping on a fake instance and refuses
a real provider outright. This is its sibling for a real worker. The chain it
closes is the last structural gap in the GPU media path:

    authorization -> budget -> create -> ship code -> stage weights
                  -> run job -> fetch artifact -> verify -> delete

Three things are deliberate.

**Renting stays switched off.** ``GPU_RENTAL_REFUSED`` was never a bug to route
around, so this module does not quietly drop it. It reads one named switch,
defaulting to refused, and the refusal happens before ``provider.create`` is
called -- so the whole chain below can be written and tested at no cost, and
turning it on is one explicit decision rather than a diff nobody reviews.

**The instance is deleted even when the job fails.** A run that raises after
``create`` and leaves the machine up is the single most expensive bug this
code could have: it bills by the minute with nobody watching. Deletion is in a
``finally``, and the report says whether it was verified.

**The artifact is verified against the worker's own digest.** Bytes that
arrived short or wrong must not become a charged-for artifact.
"""
from __future__ import annotations

import os
from pathlib import Path

from media_engine.models.speech_stage import SpeechPrepError
from media_engine.providers.base import OBSERVE_GONE, WorkerStageError
from media_engine.providers.execution_authorization import accept_speech_record
from media_engine.providers.live6_prepare import claim_execution_id
from media_engine.providers.media_run import (
    WEIGHTS_ROOT,
    RemoteRunError,
    install_command,
    job_command,
    parse_result,
    remote_artifact_path,
    verify_artifact,
)
from media_engine.safety.execution_budget import (
    LIVE7_CEILING_USD,
    LIVE7_RESERVE_SECONDS,
    open_live7_budget,
)

#: The one switch. Absent or anything but the exact word below means refused.
RENTAL_ENV = "MEDIA_ENGINE_GPU_RENTAL_APPROVED"
RENTAL_WORD = "approved"
OUT_DIR = "/workspace/media-out"
IN_DIR = "/workspace/media-in"


def rental_approved(source: dict | None = None) -> bool:
    """Whether a real rental has been approved, by name, in the environment.

    Deliberately not a boolean-ish parse: "1", "true" and "yes" are the kinds
    of value that end up set by accident in a shell profile or a stale export.
    Spending money asks for the whole word.
    """
    env = os.environ if source is None else source
    return str(env.get(RENTAL_ENV, "")).strip().casefold() == RENTAL_WORD


def authorized_models(authorization) -> set[tuple[str, str]]:
    """Every (model, revision) this authorization actually covers.

    A speech session pins a pair; a single-model authorization pins one. Both
    shapes exist, so reading only one of them would either refuse every speech
    job or accept any model on a speech authorization.
    """
    pairs = {
        (str(model), str(revision))
        for model, revision in (getattr(authorization, "speech_models", ()) or ())
    }
    model = str(getattr(authorization, "model", "") or "")
    revision = str(getattr(authorization, "revision", "") or "")
    if model and revision:
        pairs.add((model, revision))
    return pairs


def run_media_job(authorization, *, job: dict, artifact_path, **kwargs) -> dict:
    """One job on one instance. A thin wrapper over the batch below."""
    report = run_media_jobs(
        authorization, jobs=[job], artifact_paths=[artifact_path], **kwargs,
    )
    first = report["jobs"][0]
    merged = dict(report)
    merged.pop("jobs", None)
    merged.update(first)
    merged["status"] = report["status"]
    return merged


def run_media_jobs(
    authorization,
    *,
    jobs: list,
    provider,
    worker,
    platform,
    user_id: str,
    now: float,
    code_sha: str,
    recorded_digest: str | None,
    claim_path: Path,
    machine_id: int,
    offer_id: int,
    instance_id: str,
    clock,
    host: str,
    port: str,
    artifact_paths: list,
    stage_weights,
    volume_id: int | None = None,
    image_ref: dict | None = None,
    env: dict | None = None,
    run_timeout: int = 900,
) -> dict:
    """Run several jobs on ONE rented worker and bring every artifact home.

    One instance, not one per job. Renting a second machine for the second
    capability would double the bill and break the authorization's own
    max_creates=1; the rehearsal has always run both models on one instance
    and the paid path has to match it.
    """
    if not rental_approved(env):
        # Before create. Nothing is rented, nothing is billed, nothing to undo.
        raise SpeechPrepError("GPU_RENTAL_REFUSED")

    if not jobs or len(jobs) != len(artifact_paths):
        raise SpeechPrepError("JOB_INCOMPLETE")
    for job in jobs:
        if not str(job.get("capability") or "") or not str(job.get("model_id") or ""):
            raise SpeechPrepError("JOB_INCOMPLETE")

    accept_speech_record(
        authorization,
        now=now,
        code_sha=code_sha,
        recorded_digest=recorded_digest,
        volume_id=volume_id,
        image_ref=image_ref,
    )
    allowed = authorized_models(authorization)
    for job in jobs:
        # The authorization names the checkpoints, revision and all. Matching
        # on the model name alone would let a different revision of the same
        # repository run on a machine this authorization paid for, which is
        # exactly the substitution the pins exist to prevent.
        if (str(job["model_id"]), str(job.get("revision") or "")) not in allowed:
            raise SpeechPrepError("MODEL_NOT_PINNED")
    # The distinct models decide how many stages this instance needs. No
    # separate cap is checked here: the pin list above already bounds the set,
    # and accept_speech_record checks the stage allowance itself. A third
    # check that can never fire would read as protection without being any.
    wanted = []
    for job in jobs:
        if str(job["model_id"]) not in wanted:
            wanted.append(str(job["model_id"]))
    if (int(machine_id), int(offer_id)) not in authorization.allowed_pairs:
        raise WorkerStageError("MACHINE_SCOPE")
    claim_execution_id(claim_path, authorization.execution_id)

    budget = open_live7_budget(
        execution_id=authorization.execution_id,
        grant_execution_id=authorization.execution_id,
        grant_ceiling=LIVE7_CEILING_USD,
        hourly=authorization.max_hourly_usd,
        wall_now=clock.wall,
        monotonic_now=clock.mono,
    )
    _guard(budget, clock)

    created = provider.create(user_id)
    if getattr(provider, "create_calls", 1) != 1 or not created.resource_id:
        raise SpeechPrepError("INSTANCE_LIMIT")

    report: dict = {
        "rehearsal": False,
        "resource_id": created.resource_id,
        "creates": getattr(provider, "create_calls", 1),
        "models": tuple(wanted),
        "deleted": False,
        "status": "FAILED",
        "jobs": [],
    }
    try:
        _guard(budget, clock)
        package = worker.stage_media(host, port)
        report["package_sha256"] = package["sha256"]

        staged: dict = {}
        for model_id in wanted:
            _guard(budget, clock)
            staged[model_id] = stage_weights(worker, host, port, model_id)
        report["weights"] = staged

        for job, artifact_path in zip(jobs, artifact_paths):
            _guard(budget, clock)
            entry = {
                "id": str(job.get("id") or ""),
                "capability": str(job["capability"]),
                "model_id": str(job["model_id"]),
                "status": "FAILED",
            }
            report["jobs"].append(entry)
            command = job_command(
                job, weights_root=WEIGHTS_ROOT, out_dir=OUT_DIR, in_dir=IN_DIR,
                timeout_seconds=run_timeout,
            )
            code, out, _err = worker.run(host, port, command, run_timeout)
            if code != 0 and not out:
                raise WorkerStageError("WORKER_FAILED")
            receipt = parse_result(out)
            entry["receipt"] = receipt

            _guard(budget, clock)
            remote = remote_artifact_path(OUT_DIR, receipt)
            artifact_path = Path(artifact_path)
            worker.fetch(host, port, remote, artifact_path)
            data = artifact_path.read_bytes()
            verify_artifact(data, receipt)
            entry["bytes"] = len(data)
            entry["sha256"] = receipt["sha256"]
            entry["artifact_path"] = str(artifact_path)
            entry["status"] = "COMPLETED"
        report["status"] = "COMPLETED"
    finally:
        # The machine goes away whether or not the job worked.
        report["deleted"] = _terminate(provider, created.resource_id, user_id)
        report["spending_deadline_wall"] = budget.spending_deadline_wall
        report["deletion_reserve_seconds"] = LIVE7_RESERVE_SECONDS
    return report


def _terminate(provider, resource_id: str, owner: str) -> bool:
    """Best effort, and honest about it: a failure here is reported, not raised.

    Raising would hide whatever actually went wrong in the job, and the caller
    needs the truth about both. A false here means a machine may still be up.
    """
    try:
        provider.terminate(resource_id)
        return provider.observe(resource_id, owner) == OBSERVE_GONE
    except Exception:
        return False


def _guard(budget, clock) -> None:
    if budget.stop(clock.wall, clock.mono):
        raise WorkerStageError("BUDGET_STOP")
