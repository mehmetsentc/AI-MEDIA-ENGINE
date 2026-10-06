"""Phase 2D: one owned Vast GPU, one Qwen-Image PNG, then destroy.

Tests pass a fake generator. The manual command is the only live caller.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Mapping, Optional, TextIO

from media_engine.db import connect, init_schema
from media_engine.engines.image.qwen import (
    FIRST_HEIGHT,
    FIRST_PROMPT,
    FIRST_SEED,
    FIRST_WIDTH,
    INFERENCE_STEPS,
    MODEL_ID,
    QwenImageEngine,
    sha256_hex,
)
from media_engine.engines.image.qwen_remote import __file__ as _REMOTE_FILE
from media_engine.providers.base import OBSERVE_FOREIGN, OBSERVE_GONE, OBSERVE_PRESENT, OBSERVE_UNKNOWN
from media_engine.providers.offers import GpuOffer, model_key
from media_engine.providers.planner import ResourceRequirements, max_estimated_gpu_cost, plan
from media_engine.providers.vast import VastDiscovery
from media_engine.providers.vast_lifecycle import (
    VastLifecycle,
    approval_token,
    instances_list_url,
    live_transport,
    ownership_label,
)
from media_engine.resources.ledger import ResourceLedger, ResourceState
from media_engine.safety.limits import SafetyLimits
from media_engine.usage import UsageEvent, UsageStore

APPROVED_GPU_MODEL = "RTX 5090"
MIN_VRAM_GB = Decimal("24")
MIN_RELIABILITY = Decimal("0.98")
MAX_HOURLY_USD = Decimal("0.40")
MAX_ESTIMATED_COST_USD = Decimal("0.20")
MAX_LIFETIME_SECONDS = 1800
SOFT_RUNTIME_SECONDS = 1200
HARD_CLEANUP_SECONDS = 1500
READY_POLL_SECONDS = 5
MAX_CREATE_ATTEMPTS = 1
OWNER = "phase2d"
INSTANCE_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"
# 54 GB weights, 10 GB image and Python packages, 16 GB cache and unpack headroom.
DISK_GB = 80
# Vast storage_cost is $/GB/month. Live search.diskHour equals storage_cost * GB / 720.
HOURS_PER_STORAGE_MONTH = Decimal(720)
RUNTYPE = "ssh"
SSH_DIR = Path("runtime/phase2d-ssh")
# Vast's SSH banner says publickey auth can lag a few seconds after attach.
PROPAGATION_SECONDS = 45
_PROPAGATION_HINT = "try again after a few seconds"
# A host CDI/NVIDIA failure needs the machine operator to repair it.
# Six hours skips that machine for the rest of a work session and then expires.
QUARANTINE_SECONDS = 6 * 60 * 60
INFRASTRUCTURE_QUARANTINE = frozenset({
    "CONTAINER_START_FAILED",
    "IMAGE_PULL_FAILED",
    "HOST_OFFLINE",
})


class Phase2DStop(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


def policy_fields() -> dict[str, str]:
    return {
        "provider": "vast",
        "gpu_model": APPROVED_GPU_MODEL,
        "min_vram_gb": "24",
        "gpu_count": "1",
        "min_reliability": "0.98",
        "max_hourly_price_usd": "0.40",
        "max_gpu_lifetime_seconds": str(MAX_LIFETIME_SECONDS),
        "max_create_attempts": str(MAX_CREATE_ATTEMPTS),
        "max_gpu_workers": "1",
        "max_estimated_gpu_cost_usd": "0.20",
        "disk_gb": str(DISK_GB),
    }


def fingerprint_for(fields: Mapping[str, str]) -> str:
    raw = json.dumps(dict(fields), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def policy_fingerprint() -> str:
    return fingerprint_for(policy_fields())


def format_policy() -> str:
    fields = policy_fields()
    lines = ["PHASE 2D POLICY"]
    for key in (
        "provider", "gpu_model", "min_vram_gb", "gpu_count", "min_reliability",
        "max_hourly_price_usd", "max_gpu_lifetime_seconds", "max_create_attempts",
        "max_gpu_workers", "max_estimated_gpu_cost_usd", "disk_gb",
    ):
        lines.append(f"{key}: {fields[key]}")
    lines.append("POLICY FINGERPRINT: " + policy_fingerprint())
    lines.append("STATUS: HUMAN_POLICY_APPROVAL_REQUIRED")
    return "\n".join(lines) + "\n"


def policy_requirements() -> ResourceRequirements:
    return ResourceRequirements(
        min_vram_gb=MIN_VRAM_GB,
        gpu_count=1,
        max_hourly_price_usd=MAX_HOURLY_USD,
        min_reliability=MIN_RELIABILITY,
        compatible_gpu_models=(APPROVED_GPU_MODEL,),
        max_job_seconds=MAX_LIFETIME_SECONDS,
    )


def preconditions(env: Mapping[str, str]) -> str:
    key = env.get("VAST_API_KEY") or ""
    if not key.strip():
        raise Phase2DStop("VAST_API_KEY_MISSING")
    if env.get("LIVE_EXTERNAL_PROVIDERS") != "true":
        raise Phase2DStop("LIVE_EXTERNAL_PROVIDERS_DISABLED")
    if env.get("VAST_PROVISIONING") != "1":
        raise Phase2DStop("PROVISION_DISABLED")
    approval = env.get("VAST_HUMAN_POLICY_APPROVAL") or ""
    if approval.strip() == "":
        raise Phase2DStop("HUMAN_POLICY_APPROVAL_REQUIRED")
    if approval != "approve-policy:" + policy_fingerprint():
        raise Phase2DStop("APPROVAL_MISMATCH")
    if SafetyLimits().max_gpu_workers != 1 or SafetyLimits().max_job_attempts != 1:
        raise Phase2DStop("MAX_GPU_WORKERS")
    return key


def storage_hourly_usd(storage_per_gb_month: Decimal, disk_gb: int) -> Decimal:
    return storage_per_gb_month * Decimal(disk_gb) / HOURS_PER_STORAGE_MONTH


def quote_all_in(offer: GpuOffer, disk_gb: int = DISK_GB) -> Optional[tuple[Decimal, Decimal, Decimal]]:
    """Return GPU hourly, storage hourly, and total hourly for this disk allocation.

    storage_cost of zero or missing cannot prove a total, so the quote is refused.
    """
    gpu = offer.gpu_hourly_price_usd
    monthly = offer.storage_price_per_gb_month
    if gpu is None or monthly is None or monthly <= 0 or gpu < 0:
        return None
    if offer.disk_gb is None or offer.disk_gb < disk_gb:
        return None
    storage = storage_hourly_usd(monthly, disk_gb)
    total = gpu + storage
    if offer.hourly_price_usd > total:
        total = offer.hourly_price_usd
    return gpu, storage, total


def projected_max_cost(total_hourly: Decimal, seconds: int = MAX_LIFETIME_SECONDS) -> Decimal:
    return max_estimated_gpu_cost(total_hourly, seconds)


def budget_status(total_hourly: Decimal, *, seconds: int = MAX_LIFETIME_SECONDS) -> Optional[str]:
    if total_hourly > MAX_HOURLY_USD:
        return "PRICE_ABOVE_CEILING"
    if projected_max_cost(total_hourly, seconds) > MAX_ESTIMATED_COST_USD:
        return "COST_ABOVE_CEILING"
    return None


def accept_offer(offer: GpuOffer) -> GpuOffer:
    if offer.provider != "vast":
        raise Phase2DStop("OFFER_MISMATCH")
    if model_key(offer.gpu_model) != model_key(APPROVED_GPU_MODEL):
        raise Phase2DStop("OFFER_MISMATCH")
    if offer.gpu_count != 1 or offer.vram_gb < MIN_VRAM_GB:
        raise Phase2DStop("OFFER_MISMATCH")
    if offer.reliability is None or offer.reliability < MIN_RELIABILITY:
        raise Phase2DStop("OFFER_MISMATCH")
    if offer.availability not in (None, "rentable"):
        raise Phase2DStop("OFFER_GONE")
    quote = quote_all_in(offer, DISK_GB)
    if quote is None:
        raise Phase2DStop("PRICE_UNPROVEN")
    _gpu, _storage, total = quote
    status = budget_status(total)
    if status is not None:
        raise Phase2DStop(status)
    return replace(offer, hourly_price_usd=total)


def record_host_quarantine(
    db_path: str,
    *,
    offer: GpuOffer,
    failure_class: str,
    reason: str,
    now: float,
) -> None:
    """Remember one infrastructure failure. Application failures are ignored."""
    if failure_class not in INFRASTRUCTURE_QUARANTINE:
        return
    if offer.provider != "vast" or not offer.offer_id:
        return
    init_schema(db_path)
    machine_id = offer.machine_id or ""
    safe_reason = sanitize_text(reason or failure_class)[:300]
    conn = connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO host_quarantine (
                provider, offer_id, machine_id, failure_class, reason, failed_at, quarantine_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, offer_id, machine_id) DO UPDATE SET
                failure_class = excluded.failure_class,
                reason = excluded.reason,
                failed_at = excluded.failed_at,
                quarantine_until = excluded.quarantine_until
            """,
            (
                offer.provider,
                offer.offer_id,
                machine_id,
                failure_class,
                safe_reason,
                float(now),
                float(now) + QUARANTINE_SECONDS,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def active_quarantine(db_path: str, now: float) -> tuple[frozenset[str], frozenset[str]]:
    """Return machine ids and offer ids whose cooldown has not expired."""
    init_schema(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT offer_id, machine_id FROM host_quarantine
            WHERE provider = ? AND quarantine_until > ?
            """,
            ("vast", float(now)),
        ).fetchall()
    finally:
        conn.close()
    machines = {row["machine_id"] for row in rows if row["machine_id"]}
    offers = {row["offer_id"] for row in rows if row["offer_id"]}
    return frozenset(machines), frozenset(offers)


def backfill_quarantine_machine_id(db_path: str, *, offer_id: str, machine_id: str, provider: str = "vast") -> None:
    """Fill a missing machine id. Expiry, reason, and failure class stay unchanged."""
    if provider != "vast" or not offer_id.isdigit() or not machine_id.isdigit():
        raise Phase2DStop("IDENTITY_CONFLICT")
    init_schema(db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT machine_id, failure_class, reason, failed_at, quarantine_until
            FROM host_quarantine WHERE provider = ? AND offer_id = ?
            """,
            (provider, offer_id),
        ).fetchall()
        if len(rows) != 1:
            raise Phase2DStop("IDENTITY_CONFLICT")
        current = rows[0]["machine_id"] or ""
        if current == machine_id:
            return
        if current:
            raise Phase2DStop("IDENTITY_CONFLICT")
        original = (rows[0]["failure_class"], rows[0]["reason"], rows[0]["failed_at"], rows[0]["quarantine_until"])
        updated = conn.execute(
            """
            UPDATE host_quarantine SET machine_id = ?
            WHERE provider = ? AND offer_id = ? AND (machine_id = '' OR machine_id IS NULL)
            """,
            (machine_id, provider, offer_id),
        )
        after = conn.execute(
            """
            SELECT machine_id, failure_class, reason, failed_at, quarantine_until
            FROM host_quarantine WHERE provider = ? AND offer_id = ?
            """,
            (provider, offer_id),
        ).fetchone()
        unchanged = (
            after["failure_class"], after["reason"], after["failed_at"], after["quarantine_until"],
        )
        if updated.rowcount != 1 or after["machine_id"] != machine_id or unchanged != original:
            conn.rollback()
            raise Phase2DStop("IDENTITY_CONFLICT")
        conn.commit()
    except Phase2DStop:
        conn.rollback()
        raise
    finally:
        conn.close()


def persist_selected_identity(directory: Path, offer: GpuOffer, now: float) -> None:
    """Save the chosen offer identity before any create. host_id is diagnostic only."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "offer_id": offer.offer_id,
        "machine_id": offer.machine_id,
        "host_id": offer.host_id,
        "selected_at": _iso(now),
    }
    (directory / "selection-before-create.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def select_one(
    offers: list[GpuOffer],
    *,
    excluded_machine_ids: frozenset[str] = frozenset(),
    excluded_offer_ids: frozenset[str] = frozenset(),
) -> GpuOffer:
    eligible: list[GpuOffer] = []
    failures: list[str] = []
    quarantined = 0
    for offer in offers:
        try:
            accepted = accept_offer(offer)
        except Phase2DStop as exc:
            failures.append(exc.code)
            continue
        machine_id = accepted.machine_id or ""
        if accepted.offer_id in excluded_offer_ids or (machine_id and machine_id in excluded_machine_ids):
            quarantined += 1
            continue
        eligible.append(accepted)
    if not eligible:
        if quarantined:
            raise Phase2DStop("NO_HEALTHY_OFFER_WITHIN_BUDGET")
        if failures and all(code == "PRICE_UNPROVEN" for code in failures):
            raise Phase2DStop("PRICE_UNPROVEN")
        if failures and all(code == "PRICE_ABOVE_CEILING" for code in failures):
            raise Phase2DStop("PRICE_ABOVE_CEILING")
        if failures and all(code == "COST_ABOVE_CEILING" for code in failures):
            raise Phase2DStop("COST_ABOVE_CEILING")
        raise Phase2DStop("OFFER_GONE")
    selected = plan(eligible, policy_requirements()).selected
    if selected is None:
        raise Phase2DStop("OFFER_GONE")
    return selected


def run_phase2d(
    env: Mapping[str, str],
    *,
    db_path: str,
    artifact_dir: str,
    search_transport,
    lifecycle_transport,
    generate: Callable[[str, GpuOffer, float], tuple[bytes, dict]],
    now: Callable[[], float],
    stdout: Optional[TextIO] = None,
) -> int:
    """One discovery, one create, one image, then destroy. No second create."""
    out = sys.stdout if stdout is None else stdout
    try:
        key = preconditions(env)
        if ResourceLedger(db_path).unresolved():
            raise Phase2DStop("UNRESOLVED_RESOURCE")
    except Phase2DStop as exc:
        _say(out, exc.code)
        return 2
    started = now()
    deadline = started + MAX_LIFETIME_SECONDS
    requirements = policy_requirements()
    try:
        offers = VastDiscovery(transport=search_transport).search(
            replace(
                requirements.to_search(gpu_model=APPROVED_GPU_MODEL),
                allocated_storage_gb=DISK_GB,
                sort_by_hourly=True,
            ),
            api_key=key,
            read_only=True,
            limit=64,
        )
        excluded_machines, excluded_offers = active_quarantine(db_path, started)
        selected = select_one(
            offers,
            excluded_machine_ids=excluded_machines,
            excluded_offer_ids=excluded_offers,
        )
    except Phase2DStop as exc:
        _say(out, exc.code)
        return 3
    _say(out, "OFFER " + selected.offer_id)
    _say(out, "GPU " + selected.gpu_model)
    _say(out, "VRAM " + str(selected.vram_gb))
    _say(out, "HOURLY " + str(selected.hourly_price_usd))
    _say(out, "RELIABILITY " + str(selected.reliability))
    _say(out, "PERFORMANCE " + str(selected.performance))
    _say(out, "COST600 " + str(max_estimated_gpu_cost(selected.hourly_price_usd, 600)))
    _say(out, "COST1800 " + str(max_estimated_gpu_cost(selected.hourly_price_usd, MAX_LIFETIME_SECONDS)))
    persist_selected_identity(Path(artifact_dir), selected, started)
    lifecycle = VastLifecycle(
        db_path,
        limits=SafetyLimits(
            live_external_providers=True,
            max_hourly_price_usd=MAX_HOURLY_USD,
            max_gpu_workers=1,
            max_job_attempts=1,
            max_worker_lifetime_seconds=float(MAX_LIFETIME_SECONDS),
        ),
        transport=lifecycle_transport,
        now=now,
    )
    try:
        created = lifecycle.create(
            offers=[selected],
            requirements=requirements,
            offer_id=selected.offer_id,
            owner=OWNER,
            api_key=key,
            provision=True,
            human_approval=approval_token(selected.offer_id),
            image=INSTANCE_IMAGE,
            disk_gb=DISK_GB,
            runtype=RUNTYPE,
        )
    except Exception as exc:
        _say(out, str(getattr(exc, "code", "CREATE_BLOCKED")))
        return 4
    if created.outcome == "definite_failure":
        _say(out, "CREATE_FAILED")
        return 4
    if created.outcome != "created" or not created.resource_id:
        _say(out, "AMBIGUOUS_CREATE")
        return 4
    instance_id = created.resource_id
    if instance_id == selected.offer_id:
        _say(out, "CREATE_FAILED")
        return 4
    try:
        kind = lifecycle.observe(instance_id, OWNER, api_key=key, provision=True)
    except Exception:
        _destroy_if_present(lifecycle, instance_id, key)
        _say(out, OBSERVE_UNKNOWN)
        return 5
    if kind != OBSERVE_PRESENT:
        _say(out, "FOREIGN" if kind == OBSERVE_FOREIGN else "OWNERSHIP_UNKNOWN")
        return 5
    png = b""
    report: dict = {}
    generated = False
    try:
        if now() >= deadline or _over_budget(selected, now() - started):
            raise Phase2DStop("COST_ABOVE_CEILING")
        cleanup_deadline = min(deadline, started + HARD_CLEANUP_SECONDS)
        runtime_deadline = min(cleanup_deadline, started + SOFT_RUNTIME_SECONDS)
        if now() >= runtime_deadline:
            raise Phase2DStop("SSH_NOT_READY")
        png, report = generate(instance_id, selected, runtime_deadline)
        image = QwenImageEngine(lambda **_kwargs: png).render(
            FIRST_PROMPT, width=FIRST_WIDTH, height=FIRST_HEIGHT, seed=FIRST_SEED,
        )
        path = _write_artifact(artifact_dir, instance_id, image, selected, report, started, now())
        generated = True
        _say(out, "ARTIFACT " + str(path))
        _say(out, "SHA256 " + sha256_hex(image))
    except Exception as exc:
        code = getattr(exc, "code", "GENERATION_FAILED")
        _say(out, code)
        if code in INFRASTRUCTURE_QUARANTINE:
            record_host_quarantine(
                db_path,
                offer=selected,
                failure_class=code,
                reason=getattr(exc, "detail", "") or code,
                now=now(),
            )
    destroyed = _destroy_owned(lifecycle, instance_id, key)
    if not destroyed:
        _say(out, "MANUAL_CLEANUP_REQUIRED")
        _say(out, "INSTANCE " + instance_id)
        return 8
    try:
        after = lifecycle.observe(instance_id, OWNER, api_key=key, provision=True)
    except Exception:
        after = OBSERVE_UNKNOWN
    elapsed = now() - started
    final = "TERMINATED" if after == OBSERVE_GONE else "STILL_PRESENT"
    _mark_terminated(lifecycle, instance_id, final)
    _record_usage(db_path, selected, instance_id, started, now(), final, elapsed, report)
    _say(out, final)
    _say(out, "INSTANCE " + instance_id)
    if not generated:
        return 7
    return 0 if final == "TERMINATED" else 6


def ssh_generate(instance_id: str, offer: GpuOffer, deadline: float) -> tuple[bytes, dict]:
    """Copy the runner to the owned instance, generate once, and return the PNG."""
    key = os.environ.get("VAST_API_KEY") or ""
    diagnostic_dir = Path("runtime/artifacts/phase2d")

    def fetch() -> Optional[dict]:
        headers = {
            "Authorization": "Bearer " + key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        raw = live_transport("GET", instances_list_url(), b"", headers)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise Phase2DStop("SSH_NOT_READY") from exc
        for row in payload.get("instances") or []:
            if str(row.get("id")) != instance_id:
                continue
            _reject_over_price(row)
            return row
        return None

    preview = fetch()
    if not isinstance(preview, dict):
        raise Phase2DStop("SSH_NOT_READY")
    identity, public = _new_ephemeral_ssh_key(SSH_DIR / instance_id)
    fingerprint = public_key_fingerprint(public)

    def attach() -> None:
        parsed = attach_ssh_key(_ssh_attach_transport, instance_id, public, key)
        _write_attach_metadata(SSH_DIR / instance_id, instance_id, fingerprint, time.time(), parsed)

    return run_ssh_runtime(
        instance_id=instance_id,
        fetch=fetch,
        attach=attach,
        ssh=lambda host, port, command, data, timeout: _ssh_live(
            identity, host, port, command, data, timeout,
        ),
        scp=lambda host, port, remote, local, timeout: _scp_live(
            identity, host, port, remote, local, timeout,
        ),
        deadline=deadline,
        now=time.time,
        sleep=time.sleep,
        diagnostic_dir=diagnostic_dir,
        key_fingerprint=fingerprint,
        logs=lambda: _instance_logs_live(instance_id, key),
        script=Path(_REMOTE_FILE).read_bytes(),
        job=json.dumps({
            "prompt": FIRST_PROMPT,
            "seed": FIRST_SEED,
            "width": FIRST_WIDTH,
            "height": FIRST_HEIGHT,
            "steps": INFERENCE_STEPS,
        }).encode("utf-8"),
        secrets=(key,),
    )


def ssh_endpoint(row: Mapping) -> Optional[tuple[str, str]]:
    """Return the SSH host and port using the official vastai ssh-url contract."""
    if str(row.get("actual_status") or "") != "running":
        return None
    if str(row.get("intended_status") or "") != "running":
        return None
    ports = row.get("ports")
    if isinstance(ports, dict):
        mapped = ports.get("22/tcp")
        if isinstance(mapped, list) and mapped and isinstance(mapped[0], dict):
            host = str(row.get("public_ipaddr") or "")
            port = str(mapped[0].get("HostPort") or "")
            if host and port.isdigit():
                return host, port
    host = str(row.get("ssh_host") or "")
    raw_port = row.get("ssh_port")
    port = "" if raw_port is None else str(raw_port)
    if host and port.isdigit():
        if "jupyter" in str(row.get("image_runtype") or ""):
            port = str(int(port) + 1)
        return host, port
    return None


MAX_READINESS_SNAPSHOTS = 20
# Failure phrases from the official vastai status classifier. Bare words such as
# "pull" or "docker" are omitted so a normal loading message does not abort.
_IMAGE_PULL_FAILED = re.compile(
    r"no such image|repository does not exist|pull access denied|"
    r"requested access to the resource is denied|manifest unknown|"
    r"manifest invalid|failed to pull|pull failed|error pulling",
    re.IGNORECASE,
)
_CONTAINER_START_FAILED = re.compile(
    r"oci runtime|failed to start|exec format error|nvidia-container|start container",
    re.IGNORECASE,
)


def terminal_ssh_status(row: Mapping) -> bool:
    actual = str(row.get("actual_status") or "")
    intended = str(row.get("intended_status") or "")
    return actual in {"stopped", "exited", "offline", "unknown"} or intended in {"stopped", "exited"}


def classify_startup_failure(row: Mapping) -> Optional[str]:
    """Return a startup code, or None while Vast is still transitioning."""
    text = row.get("status_msg")
    message = text.strip() if isinstance(text, str) else ""
    if message and _IMAGE_PULL_FAILED.search(message):
        return "IMAGE_PULL_FAILED"
    if message and _CONTAINER_START_FAILED.search(message):
        return "CONTAINER_START_FAILED"
    if not terminal_ssh_status(row):
        return None
    actual = str(row.get("actual_status") or "")
    intended = str(row.get("intended_status") or "")
    if actual == "offline":
        return "HOST_OFFLINE"
    if actual == "exited" or intended == "exited":
        return "INSTANCE_EXITED"
    if actual == "stopped" or intended == "stopped":
        return "INSTANCE_STOPPED"
    return "UNKNOWN_STARTUP_FAILURE"


def readiness_snapshot(row: Mapping, now: float, secrets: tuple[str, ...] = ()) -> dict:
    """One bounded, secret-free view of a Vast instance row."""
    text = row.get("status_msg")
    message = text.strip() if isinstance(text, str) else ""
    start = row.get("start_date")
    start_date = float(start) if isinstance(start, (int, float)) and not isinstance(start, bool) else None
    elapsed = None
    if start_date is not None and 1_000_000_000 <= start_date <= 20_000_000_000:
        elapsed = now - start_date
    return {
        "at": _iso(now),
        "actual_status": sanitize_text(str(row.get("actual_status") or ""), secrets),
        "intended_status": sanitize_text(str(row.get("intended_status") or ""), secrets),
        "status_msg": sanitize_text(message, secrets),
        "ssh_endpoint_available": ssh_endpoint(row) is not None,
        "public_ip_available": bool(str(row.get("public_ipaddr") or "")),
        "start_date": start_date,
        "time_since_create": elapsed,
    }


def classify_ssh_stderr(stderr: bytes) -> str:
    text = stderr.decode("utf-8", "replace").lower()
    if "permission denied" in text or "authentication failed" in text:
        return "SSH_AUTH_FAILED"
    if (
        "connection refused" in text
        or "connection timed out" in text
        or "operation timed out" in text
        or "no route to host" in text
    ):
        return "SSH_NOT_READY"
    return "SSH_CONNECT_FAILED"


def sanitize_text(text: str, secrets: tuple[str, ...] = ()) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[redacted]",
        text,
        flags=re.S,
    )
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", text)
    text = re.sub(r"\b[0-9a-fA-F]{32,}\b", "[redacted]", text)
    if len(text) > 500:
        text = text[-500:]
    return text


def write_runtime_diagnostic(
    directory: Path,
    instance_id: str,
    *,
    stage: str,
    exception: str,
    ssh_exit_code: Optional[int],
    remote_exit_code: Optional[int],
    stdout: bytes,
    stderr: bytes,
    now: float,
    secrets: tuple[str, ...] = (),
    evidence: Optional[Mapping] = None,
    key_fingerprint: str = "",
    instance_log: bytes = b"",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "exception": exception,
        "ssh_exit_code": ssh_exit_code,
        "remote_exit_code": remote_exit_code,
        "stdout_tail": sanitize_text(stdout.decode("utf-8", "replace"), secrets),
        "stderr_tail": sanitize_text(stderr.decode("utf-8", "replace"), secrets),
        "at": _iso(now),
    }
    if evidence:
        for key in (
            "final_actual_status",
            "final_intended_status",
            "final_status_msg",
            "time_since_create",
            "transition_count",
            "transitions",
        ):
            if key in evidence:
                payload[key] = evidence[key]
    if key_fingerprint:
        payload["public_key_fingerprint"] = key_fingerprint
        payload["attachment_stage"] = stage
    if instance_log:
        payload["instance_log_tail"] = sanitize_text(instance_log.decode("utf-8", "replace"), secrets)
    path = directory / f"phase2d-{instance_id}.diagnostic.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def startup_evidence(snapshots: list) -> dict:
    kept = snapshots[-MAX_READINESS_SNAPSHOTS:]
    last = kept[-1] if kept else {}
    return {
        "final_actual_status": last.get("actual_status"),
        "final_intended_status": last.get("intended_status"),
        "final_status_msg": last.get("status_msg"),
        "time_since_create": last.get("time_since_create"),
        "transition_count": len(snapshots),
        "transitions": kept,
    }


def write_readiness_log(directory: Path, instance_id: str, snapshots: list) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"phase2d-{instance_id}.readiness.json"
    path.write_text(json.dumps(startup_evidence(snapshots), indent=2) + "\n", encoding="utf-8")
    return path


def wait_for_ssh_ready(
    fetch,
    deadline: float,
    now,
    sleep,
    on_snapshot: Optional[Callable[[dict], None]] = None,
    secrets: tuple[str, ...] = (),
) -> tuple[dict, str, str]:
    """Poll until Vast reports running and an SSH endpoint. Sleeps between checks."""
    while now() < deadline:
        row = fetch()
        if not isinstance(row, dict):
            row = {}
        snapshot = readiness_snapshot(row, now(), secrets)
        if on_snapshot is not None:
            on_snapshot(snapshot)
        if row:
            failure = classify_startup_failure(row)
            if failure:
                message = row.get("status_msg")
                detail = message.strip() if isinstance(message, str) else ""
                raise Phase2DStop(failure, sanitize_text(detail, secrets))
            endpoint = ssh_endpoint(row)
            if endpoint is not None:
                return row, endpoint[0], endpoint[1]
        remaining = deadline - now()
        if remaining <= 0:
            break
        sleep(min(READY_POLL_SECONDS, remaining))
    raise Phase2DStop("SSH_NOT_READY_TIMEOUT")


def run_ssh_runtime(
    *,
    instance_id: str,
    fetch,
    attach,
    ssh,
    scp,
    deadline: float,
    now,
    sleep,
    diagnostic_dir: Path,
    script: bytes,
    job: bytes,
    secrets: tuple[str, ...] = (),
    key_fingerprint: str = "",
    logs=None,
) -> tuple[bytes, dict]:
    """Wait until SSH accepts a command, then run the remote generator once."""
    snapshots: list[dict] = []

    def on_snapshot(snapshot: dict) -> None:
        snapshots.append(snapshot)
        write_readiness_log(diagnostic_dir, instance_id, snapshots)

    def record(stage: str, ssh_code: Optional[int], remote_code: Optional[int], stdout: bytes, stderr: bytes, exception: str, instance_log: bytes = b"") -> None:
        write_runtime_diagnostic(
            diagnostic_dir,
            instance_id,
            stage=stage,
            exception=exception,
            ssh_exit_code=ssh_code,
            remote_exit_code=remote_code,
            stdout=stdout,
            stderr=stderr,
            now=now(),
            secrets=secrets,
            evidence=startup_evidence(snapshots),
            key_fingerprint=key_fingerprint,
            instance_log=instance_log,
        )

    try:
        attach()
    except Phase2DStop as exc:
        record(exc.code, None, None, b"", b"", "Phase2DStop")
        raise
    except Exception as exc:
        record("SSH_INSTANCE_KEY_ATTACH_FAILED", None, None, b"", b"", type(exc).__name__)
        raise Phase2DStop("SSH_INSTANCE_KEY_ATTACH_FAILED") from exc
    try:
        _row, host, port = wait_for_ssh_ready(
            fetch, deadline, now, sleep, on_snapshot=on_snapshot, secrets=secrets,
        )
    except Phase2DStop as exc:
        record(exc.code, None, None, b"", b"", "Phase2DStop")
        raise
    first_denial = None
    while now() < deadline:
        code, out, err = ssh(host, port, "true", None, 20)
        if code == 0:
            break
        kind = classify_ssh_stderr(err)
        if kind == "SSH_AUTH_FAILED":
            if first_denial is None:
                first_denial = now()
            decision = auth_failure_code(err, now() - first_denial)
            if decision == "":
                remaining = min(deadline - now(), PROPAGATION_SECONDS - (now() - first_denial))
                if remaining <= 0:
                    decision = "SSH_KEY_PROPAGATION_TIMEOUT"
                else:
                    sleep(min(READY_POLL_SECONDS, remaining))
                    continue
            record(decision, code, None, out, err, "Phase2DStop", instance_log=_safe_logs(logs, secrets))
            raise Phase2DStop(decision)
        if kind != "SSH_NOT_READY":
            record("SSH_CONNECT_FAILED", code, None, out, err, "Phase2DStop")
            raise Phase2DStop("SSH_CONNECT_FAILED")
        remaining = deadline - now()
        if remaining <= 0:
            break
        sleep(min(READY_POLL_SECONDS, remaining))
    else:
        record("SSH_NOT_READY", None, None, b"", b"", "Phase2DStop")
        raise Phase2DStop("SSH_NOT_READY")
    if now() >= deadline:
        record("SSH_NOT_READY", None, None, b"", b"", "Phase2DStop")
        raise Phase2DStop("SSH_NOT_READY")
    for command, data, timeout in (
        ("mkdir -p /workspace && cat > /workspace/phase2d_gen.py", script, 60),
        ("cat > /workspace/phase2d_job.json", job, 30),
    ):
        code, out, err = ssh(host, port, command, data, timeout)
        if code != 0:
            record("REMOTE_COMMAND_FAILED", code, code, out, err, "Phase2DStop")
            raise Phase2DStop("REMOTE_COMMAND_FAILED")
    remaining = max(1, int(deadline - now()))
    code, out, err = ssh(
        host,
        port,
        "python3 -m pip install -q 'diffusers>=0.35' transformers accelerate safetensors pillow "
        "&& python3 /workspace/phase2d_gen.py",
        None,
        remaining,
    )
    local_dir = diagnostic_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    report_path = local_dir / f".{instance_id}.remote.json"
    png_path = local_dir / f".{instance_id}.download.png"
    scp(host, port, "/workspace/phase2d_report.json", report_path, 60)
    report: dict = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    if code != 0:
        detail = str(report.get("error") or "")
        if "DOWNLOAD" in detail:
            stage = "MODEL_DOWNLOAD_FAILED"
        elif "IMPORT" in detail or "LOAD" in detail:
            stage = "MODEL_LOAD_FAILED"
        else:
            stage = "REMOTE_COMMAND_FAILED"
        record(stage, None, code, out, err, "Phase2DStop")
        raise Phase2DStop(stage)
    if not report_path.exists():
        record("REMOTE_REPORT_MISSING", None, code, out, err, "Phase2DStop")
        raise Phase2DStop("REMOTE_REPORT_MISSING")
    scp(host, port, "/workspace/phase2d_output.png", png_path, 60)
    if not png_path.exists() or png_path.stat().st_size <= 0:
        record("GENERATION_FAILED", None, code, out, err, "Phase2DStop")
        raise Phase2DStop("GENERATION_FAILED")
    return png_path.read_bytes(), report


def auth_failure_code(stderr: bytes, elapsed: float) -> str:
    """Return a retry sentinel or the terminal auth code.

    Vast's SSH banner says to retry after a few seconds. A publickey
    rejection without that hint is a deterministic authentication failure.
    """
    text = stderr.decode("utf-8", "replace").lower()
    if _PROPAGATION_HINT in text and elapsed < PROPAGATION_SECONDS:
        return ""
    if _PROPAGATION_HINT in text:
        return "SSH_KEY_PROPAGATION_TIMEOUT"
    return "SSH_AUTH_FAILED"


def _safe_logs(logs, secrets: tuple[str, ...]) -> bytes:
    if logs is None:
        return b""
    try:
        text = logs()
    except Exception:
        return b""
    if not isinstance(text, str):
        return b""
    return sanitize_text(text, secrets).encode("utf-8")


def public_key_fingerprint(public_key: str) -> str:
    blob = base64.b64decode(public_key.strip().split()[1])
    digest = base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii").rstrip("=")
    return "SHA256:" + digest


def ssh_argv(identity: str, host: str, port: str, command: str) -> list[str]:
    return [
        "ssh", "-i", identity, "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=20", "-p", port, "root@" + host, command,
    ]


def _ssh_live(identity: str, host: str, port: str, command: str, data: Optional[bytes], timeout: int) -> tuple[int, bytes, bytes]:
    try:
        proc = subprocess.run(
            ssh_argv(identity, host, port, command),
            input=data,
            capture_output=True,
            timeout=max(1, timeout),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, bytes) else b""
        stderr = exc.stderr if isinstance(exc.stderr, bytes) else b"connection timed out"
        return 255, stdout, stderr
    return proc.returncode, proc.stdout, proc.stderr


def _scp_live(identity: str, host: str, port: str, remote: str, local: Path, timeout: int) -> None:
    subprocess.run(
        [
            "scp", "-i", identity, "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-P", port, "root@" + host + ":" + remote, str(local),
        ],
        capture_output=True,
        timeout=max(1, timeout),
        check=False,
    )


_SSH_ATTACH = re.compile(r"https://console\.vast\.ai/api/v0/instances/([0-9]+)/ssh/")


def ssh_attach_allowed(method: str, url: str) -> bool:
    return method == "POST" and _SSH_ATTACH.fullmatch(url) is not None


def parse_instance_attach_response(raw: bytes) -> dict:
    """Confirm the documented instance attach response. It has no key id."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
        raise Phase2DStop("SSH_INSTANCE_KEY_ATTACH_FAILED") from exc
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise Phase2DStop("SSH_INSTANCE_KEY_ATTACH_FAILED")
    msg = payload.get("msg")
    message = msg.strip() if isinstance(msg, str) else ""
    if message.lower().startswith("ssh-"):
        message = ""
    key_id = payload.get("id")
    recorded = str(key_id) if isinstance(key_id, int) and not isinstance(key_id, bool) else None
    return {"success": True, "msg": message, "key_id": recorded, "http_status": 200}


def attach_ssh_key(transport, instance_id: str, public_key: str, api_key: str) -> dict:
    """Attach one public key to one owned instance. The private key never leaves this machine."""
    if not instance_id.isdigit():
        raise Phase2DStop("OWNERSHIP_UNKNOWN")
    text = public_key.strip()
    if not text.startswith("ssh-ed25519 ") or "PRIVATE" in text or "\n" in text:
        raise Phase2DStop("SSH_KEY_REFUSED")
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/ssh/"
    if not ssh_attach_allowed("POST", url):
        raise Phase2DStop("COMMAND_BLOCKED")
    body = json.dumps({"ssh_key": text}, separators=(",", ":")).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    try:
        raw = transport("POST", url, body, headers)
    except Phase2DStop:
        raise
    except Exception as exc:
        raise Phase2DStop("SSH_INSTANCE_KEY_ATTACH_FAILED") from exc
    return parse_instance_attach_response(raw if isinstance(raw, bytes) else b"")


def _write_attach_metadata(directory: Path, instance_id: str, fingerprint: str, now: float, parsed: Mapping) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance_id": instance_id,
        "fingerprint": fingerprint,
        "attached_at": _iso(now),
        "http_status": parsed.get("http_status"),
        "success": True,
        "key_id": parsed.get("key_id"),
    }
    msg = parsed.get("msg") or ""
    if isinstance(msg, str) and msg and "ssh-" not in msg.lower():
        payload["msg"] = sanitize_text(msg)
    (directory / "attach.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ssh_attach_transport(method: str, url: str, body: bytes, headers: dict) -> bytes:
    if not ssh_attach_allowed(method, url):
        raise Phase2DStop("COMMAND_BLOCKED")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise Phase2DStop("SSH_INSTANCE_KEY_ATTACH_FAILED") from exc


def _new_ephemeral_ssh_key(directory: Path) -> tuple[str, str]:
    """One new ed25519 pair for this instance. The private key stays in this directory."""
    directory.mkdir(parents=True, exist_ok=True)
    private = directory / "id_ed25519"
    public = directory / "id_ed25519.pub"
    if not private.exists() or not public.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(private), "-N", "", "-q", "-C", "phase2d"],
            check=True,
            capture_output=True,
        )
    os.chmod(private, 0o600)
    text = public.read_text(encoding="utf-8").strip()
    if not text.startswith("ssh-ed25519 ") or "PRIVATE" in text or "\n" in text:
        raise Phase2DStop("SSH_KEY_REFUSED")
    return str(private), text


_INSTANCE_LOGS = re.compile(r"https://console\.vast\.ai/api/v0/instances/request_logs/([0-9]+)/")


def _instance_logs_live(instance_id: str, api_key: str) -> str:
    """Ask Vast for a short log tail. Failure leaves the SSH error intact."""
    if not instance_id.isdigit():
        return ""
    url = f"https://console.vast.ai/api/v0/instances/request_logs/{instance_id}/"
    if _INSTANCE_LOGS.fullmatch(url) is None:
        return ""
    body = json.dumps({"tail": "80"}).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return ""
    if isinstance(payload, dict) and isinstance(payload.get("msg"), str) and "result_url" not in payload:
        return payload.get("msg") or ""
    result_url = payload.get("result_url") if isinstance(payload, dict) else None
    if not isinstance(result_url, str) or not result_url.startswith("https://"):
        return ""
    host = urllib.parse.urlparse(result_url).hostname or ""
    if host != "vast.ai" and not host.endswith(".vast.ai"):
        return ""
    for _ in range(3):
        time.sleep(0.3)
        try:
            with urllib.request.urlopen(result_url, timeout=10) as response:
                if response.status == 200:
                    return response.read().decode("utf-8", "replace")
        except Exception:
            continue
    return ""


def _reject_over_price(row: dict) -> None:
    raw = row.get("dph_total")
    if raw is None:
        return
    try:
        hourly = Decimal(str(raw))
    except Exception:
        return
    if hourly > MAX_HOURLY_USD:
        raise Phase2DStop("PRICE_ABOVE_CEILING")
    if max_estimated_gpu_cost(hourly, MAX_LIFETIME_SECONDS) > MAX_ESTIMATED_COST_USD:
        raise Phase2DStop("COST_ABOVE_CEILING")


def _over_budget(offer: GpuOffer, elapsed: float) -> bool:
    if elapsed <= 0:
        return False
    return max_estimated_gpu_cost(offer.hourly_price_usd, int(elapsed) + 1) > MAX_ESTIMATED_COST_USD


def _destroy_owned(lifecycle: VastLifecycle, instance_id: str, key: str) -> bool:
    for _ in range(2):
        try:
            lifecycle.terminate(
                instance_id,
                OWNER,
                api_key=key,
                provision=True,
                human_approval="approve-terminate:" + instance_id,
            )
            return True
        except Exception:
            continue
    return False


def _destroy_if_present(lifecycle: VastLifecycle, instance_id: str, key: str) -> None:
    try:
        kind = lifecycle.observe(instance_id, OWNER, api_key=key, provision=True)
    except Exception:
        return
    if kind == OBSERVE_PRESENT:
        _destroy_owned(lifecycle, instance_id, key)


def _mark_terminated(lifecycle: VastLifecycle, instance_id: str, status: str) -> None:
    record = lifecycle.ledger.get(instance_id)
    if record is None or status not in {"TERMINATED", "STILL_PRESENT"}:
        return
    if status == "TERMINATED":
        record.state = ResourceState.TERMINATED
        record.last_error = status
        record.updated_at = lifecycle._now()
        lifecycle.ledger.upsert(record)


def _write_artifact(
    artifact_dir: str,
    instance_id: str,
    png: bytes,
    offer: GpuOffer,
    report: dict,
    started: float,
    finished: float,
) -> Path:
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"phase2d-{instance_id}.png"
    if path.exists():
        stamp = datetime.fromtimestamp(finished, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = root / f"phase2d-{instance_id}-{stamp}.png"
    path.write_bytes(png)
    meta = {
        "model_id": MODEL_ID,
        "prompt": FIRST_PROMPT,
        "seed": FIRST_SEED,
        "width": FIRST_WIDTH,
        "height": FIRST_HEIGHT,
        "steps": INFERENCE_STEPS,
        "provider": "vast",
        "gpu_model": offer.gpu_model,
        "offer_id": offer.offer_id,
        "instance_id": instance_id,
        "hourly_price_usd": str(offer.hourly_price_usd),
        "owner_label": ownership_label(OWNER),
        "sha256": sha256_hex(png),
        "bytes": len(png),
        "created_at": _iso(started),
        "finished_at": _iso(finished),
        "remote": report,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def _record_usage(
    db_path: str,
    offer: GpuOffer,
    instance_id: str,
    started: float,
    finished: float,
    status: str,
    elapsed: float,
    report: dict,
) -> None:
    estimated = offer.hourly_price_usd * Decimal(str(max(elapsed, 0))) / Decimal(3600)
    UsageStore(db_path).record(UsageEvent(
        job_id="phase2d-" + instance_id,
        client_id=OWNER,
        job_type="PHASE2D_IMAGE",
        engine_id="qwen-image",
        started_at=_iso(started),
        finished_at=_iso(finished),
        duration_seconds=float(elapsed),
        attempt_count=1,
        estimated_cost_usd=str(estimated),
        status=status,
        provider="vast",
        gpu_model=offer.gpu_model,
        hourly_price_usd=str(offer.hourly_price_usd),
        gpu_seconds=float(elapsed),
        actual_cost_usd=None,
    ))
    _ = report


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _say(out: TextIO, line: str) -> None:
    out.write(line + "\n")


def main(environ: Optional[Mapping[str, str]] = None, stdout: Optional[TextIO] = None) -> int:
    env = os.environ if environ is None else environ
    root = Path("runtime")
    root.mkdir(parents=True, exist_ok=True)
    return run_phase2d(
        env,
        db_path=str(root / "phase2d.sqlite3"),
        artifact_dir=str(root / "artifacts" / "phase2d"),
        search_transport=None,
        lifecycle_transport=live_transport,
        generate=ssh_generate,
        now=lambda: datetime.now(timezone.utc).timestamp(),
        stdout=stdout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
