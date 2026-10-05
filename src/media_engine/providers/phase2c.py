"""One approved Vast policy: discover once, create once, then destroy.

The manual command is the only caller that may arm the live transport.
Tests pass a fake transport. This module does not run on import.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Mapping, Optional, TextIO

from media_engine.providers.base import OBSERVE_FOREIGN, OBSERVE_GONE, OBSERVE_PRESENT, OBSERVE_UNKNOWN
from media_engine.providers.offers import GpuOffer, model_key
from media_engine.providers.planner import ResourceRequirements, max_estimated_gpu_cost, plan
from media_engine.providers.vast import VastDiscovery
from media_engine.providers.vast_lifecycle import (
    VastLifecycle,
    approval_token,
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
MAX_ESTIMATED_COST_USD = Decimal("0.07")
MAX_LIFETIME_SECONDS = 600
MAX_CREATE_ATTEMPTS = 1
OWNER = "phase2c"
INSTANCE_IMAGE = "ubuntu:22.04"
DISK_GB = 8


class Phase2CStop(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def policy_fields() -> dict[str, str]:
    return {
        "provider": "vast",
        "gpu_model": APPROVED_GPU_MODEL,
        "min_vram_gb": "24",
        "gpu_count": "1",
        "min_reliability": "0.98",
        "max_hourly_price_usd": "0.40",
        "max_job_seconds": "600",
        "max_estimated_gpu_cost_usd": "0.07",
        "max_create_attempts": "1",
        "max_gpu_workers": "1",
    }


def fingerprint_for(fields: Mapping[str, str]) -> str:
    raw = json.dumps(dict(fields), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def policy_fingerprint() -> str:
    return fingerprint_for(policy_fields())


def format_policy() -> str:
    fields = policy_fields()
    lines = ["PHASE 2C POLICY"]
    for key in (
        "provider", "gpu_model", "min_vram_gb", "gpu_count", "min_reliability",
        "max_hourly_price_usd", "max_job_seconds", "max_estimated_gpu_cost_usd",
        "max_create_attempts", "max_gpu_workers",
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
        raise Phase2CStop("VAST_API_KEY_MISSING")
    if env.get("LIVE_EXTERNAL_PROVIDERS") != "true":
        raise Phase2CStop("LIVE_EXTERNAL_PROVIDERS_DISABLED")
    if env.get("VAST_PROVISIONING") != "1":
        raise Phase2CStop("PROVISION_DISABLED")
    approval = env.get("VAST_HUMAN_POLICY_APPROVAL") or ""
    if approval.strip() == "":
        raise Phase2CStop("HUMAN_POLICY_APPROVAL_REQUIRED")
    if approval != "approve-policy:" + policy_fingerprint():
        raise Phase2CStop("APPROVAL_MISMATCH")
    if SafetyLimits().max_gpu_workers != 1 or SafetyLimits().max_job_attempts != 1:
        raise Phase2CStop("MAX_GPU_WORKERS")
    if MAX_CREATE_ATTEMPTS != 1:
        raise Phase2CStop("CREATE_LIMIT")
    return key


def accept_offer(offer: GpuOffer) -> None:
    if offer.provider != "vast":
        raise Phase2CStop("OFFER_MISMATCH")
    if model_key(offer.gpu_model) != model_key(APPROVED_GPU_MODEL):
        raise Phase2CStop("OFFER_MISMATCH")
    if offer.gpu_count != 1:
        raise Phase2CStop("OFFER_MISMATCH")
    if offer.vram_gb < MIN_VRAM_GB:
        raise Phase2CStop("OFFER_MISMATCH")
    if offer.reliability is None or offer.reliability < MIN_RELIABILITY:
        raise Phase2CStop("OFFER_MISMATCH")
    if offer.availability not in (None, "rentable"):
        raise Phase2CStop("OFFER_GONE")
    cost = max_estimated_gpu_cost(offer.hourly_price_usd, MAX_LIFETIME_SECONDS)
    if cost > MAX_ESTIMATED_COST_USD:
        raise Phase2CStop("COST_ABOVE_CEILING")
    if offer.hourly_price_usd > MAX_HOURLY_USD:
        raise Phase2CStop("PRICE_ABOVE_CEILING")


def select_one(offers: list[GpuOffer]) -> GpuOffer:
    selected = plan(offers, policy_requirements()).selected
    if selected is None:
        raise Phase2CStop("OFFER_GONE")
    accept_offer(selected)
    return selected


def run_phase2c(
    env: Mapping[str, str],
    *,
    db_path: str,
    search_transport,
    lifecycle_transport,
    now: Callable[[], float],
    stdout: Optional[TextIO] = None,
) -> int:
    """Return a process status. Mutation happens only after every gate passes."""
    out = sys.stdout if stdout is None else stdout
    try:
        key = preconditions(env)
        if ResourceLedger(db_path).unresolved():
            raise Phase2CStop("UNRESOLVED_RESOURCE")
    except Phase2CStop as exc:
        _say(out, exc.code)
        return 2
    started = now()
    requirements = policy_requirements()
    try:
        offers = VastDiscovery(transport=search_transport).search(
            requirements.to_search(gpu_model=APPROVED_GPU_MODEL),
            api_key=key,
            read_only=True,
            limit=64,
        )
        selected = select_one(offers)
    except Phase2CStop as exc:
        _say(out, exc.code)
        return 3
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
        created = _create_once(
            lifecycle,
            selected=selected,
            requirements=requirements,
            api_key=key,
        )
    except Exception as exc:
        code = getattr(exc, "code", "CREATE_BLOCKED")
        _say(out, str(code))
        return 4
    if created.outcome == "ambiguous" or not created.resource_id:
        owned = lifecycle.list_owned(OWNER, api_key=key, provision=True)
        _persist(
            db_path, selected, created.resource_id or selected.offer_id, started, now(),
            "AMBIGUOUS", None,
        )
        _say(out, "AMBIGUOUS")
        _say(out, "OWNED_LIST " + ",".join(owned))
        return 4
    instance_id = created.resource_id
    try:
        kind = lifecycle.observe(instance_id, OWNER, api_key=key, provision=True)
    except Exception:
        _persist(db_path, selected, instance_id, started, now(), "OWNERSHIP_UNKNOWN", None)
        _say(out, OBSERVE_UNKNOWN)
        return 5
    if kind != OBSERVE_PRESENT:
        status = "FOREIGN" if kind == OBSERVE_FOREIGN else "OWNERSHIP_UNKNOWN"
        _persist(db_path, selected, instance_id, started, now(), status, None)
        _say(out, status)
        return 5
    elapsed = now() - started
    destroyed = False
    try:
        _destroy_owned(lifecycle, instance_id, key)
        destroyed = True
    except Exception:
        try:
            _destroy_owned(lifecycle, instance_id, key)
            destroyed = True
        except Exception:
            destroyed = False
    if not destroyed:
        _persist(db_path, selected, instance_id, started, now(), "DESTROY_FAILED", elapsed)
        _say(out, "DESTROY_FAILED")
        return 8
    try:
        after = lifecycle.observe(instance_id, OWNER, api_key=key, provision=True)
    except Exception:
        after = OBSERVE_UNKNOWN
    final = "LIFETIME_EXCEEDED" if elapsed > MAX_LIFETIME_SECONDS else (
        "TERMINATED" if after == OBSERVE_GONE else "STILL_PRESENT"
    )
    _mark_terminated(lifecycle, instance_id, final)
    _persist(db_path, selected, instance_id, started, now(), final, elapsed)
    _say(out, final)
    _say(out, f"INSTANCE {instance_id}")
    return 0 if final == "TERMINATED" else 6


def _create_once(lifecycle: VastLifecycle, *, selected: GpuOffer,
                 requirements: ResourceRequirements, api_key: str):
    if getattr(lifecycle, "_phase2c_creates", 0) >= MAX_CREATE_ATTEMPTS:
        raise Phase2CStop("CREATE_LIMIT")
    lifecycle._phase2c_creates = getattr(lifecycle, "_phase2c_creates", 0) + 1
    return lifecycle.create(
        offers=[selected],
        requirements=requirements,
        offer_id=selected.offer_id,
        owner=OWNER,
        api_key=api_key,
        provision=True,
        human_approval=approval_token(selected.offer_id),
        image=INSTANCE_IMAGE,
        disk_gb=DISK_GB,
    )


def _destroy_owned(lifecycle: VastLifecycle, instance_id: str, key: str) -> None:
    lifecycle.terminate(
        instance_id,
        OWNER,
        api_key=key,
        provision=True,
        human_approval="approve-terminate:" + instance_id,
    )


def _mark_terminated(lifecycle: VastLifecycle, instance_id: str, status: str) -> None:
    record = lifecycle.ledger.get(instance_id)
    if record is None or status not in {"TERMINATED", "LIFETIME_EXCEEDED"}:
        return
    record.state = ResourceState.TERMINATED
    record.last_error = status
    record.updated_at = lifecycle._now()
    lifecycle.ledger.upsert(record)


def _persist(
    db_path: str,
    offer: GpuOffer,
    instance_id: str,
    started: float,
    finished: float,
    status: str,
    elapsed: Optional[float],
    estimated: Optional[Decimal] = None,
) -> None:
    if estimated is None and elapsed is not None:
        if elapsed <= 0:
            estimated = Decimal("0")
        else:
            estimated = offer.hourly_price_usd * Decimal(str(elapsed)) / Decimal(3600)
    seconds = None if elapsed is None else float(elapsed)
    payload = {
        "provider": "vast",
        "instance_id": instance_id,
        "offer_id": offer.offer_id,
        "owner_label": ownership_label(OWNER),
        "created_at": _iso(started),
        "hourly_price_usd": str(offer.hourly_price_usd),
        "gpu_model": offer.gpu_model,
        "status": status,
        "gpu_seconds": seconds,
        "estimated_gpu_cost_usd": None if estimated is None else str(estimated),
        "cost_basis": "estimated_from_runtime",
    }
    path = Path(db_path).with_suffix(".phase2c.json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if elapsed is None and status == "AMBIGUOUS":
        estimated_text = None
    else:
        estimated_text = None if estimated is None else str(estimated)
    UsageStore(db_path).record(UsageEvent(
        job_id="phase2c-" + instance_id,
        client_id=OWNER,
        job_type="PHASE2C_LIFECYCLE",
        engine_id="vast-lifecycle",
        started_at=_iso(started),
        finished_at=_iso(finished),
        duration_seconds=0 if seconds is None else seconds,
        attempt_count=1,
        estimated_cost_usd=estimated_text,
        status=status,
        provider="vast",
        gpu_model=offer.gpu_model,
        hourly_price_usd=str(offer.hourly_price_usd),
        gpu_seconds=seconds,
        actual_cost_usd=None,
    ))


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _say(out: TextIO, line: str) -> None:
    out.write(line + "\n")


def main(environ: Optional[Mapping[str, str]] = None, stdout: Optional[TextIO] = None) -> int:
    env = os.environ if environ is None else environ
    root = Path("runtime")
    root.mkdir(parents=True, exist_ok=True)
    return run_phase2c(
        env,
        db_path=str(root / "phase2c.sqlite3"),
        search_transport=None,
        lifecycle_transport=live_transport,
        now=lambda: datetime.now(timezone.utc).timestamp(),
        stdout=stdout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
