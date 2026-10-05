"""One approved Vast instance: create, verify ownership, destroy.

The manual command is the only caller that may arm the live transport.
Tests pass a fake transport. This module does not run on import.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Mapping, Optional, TextIO

from media_engine.providers.base import OBSERVE_FOREIGN, OBSERVE_GONE, OBSERVE_PRESENT, OBSERVE_UNKNOWN
from media_engine.providers.offers import GpuOffer, GpuRequirements, model_key
from media_engine.providers.planner import ResourceRequirements, max_estimated_gpu_cost
from media_engine.providers.vast import VastDiscovery
from media_engine.providers.vast_lifecycle import (
    VastLifecycle,
    approval_token,
    live_transport,
    ownership_label,
)
from media_engine.resources.ledger import ResourceState
from media_engine.safety.limits import SafetyLimits
from media_engine.usage import UsageEvent, UsageStore

APPROVED_OFFER_ID = "43994879"
APPROVED_GPU_MODEL = "RTX 5090"
MAX_HOURLY_USD = Decimal("0.40")
MAX_ESTIMATED_COST_USD = Decimal("0.07")
MAX_LIFETIME_SECONDS = 600
OWNER = "phase2c"
INSTANCE_IMAGE = "ubuntu:22.04"
DISK_GB = 8


class Phase2CStop(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def preconditions(env: Mapping[str, str]) -> str:
    key = env.get("VAST_API_KEY") or ""
    if not key.strip():
        raise Phase2CStop("VAST_API_KEY_MISSING")
    if env.get("LIVE_EXTERNAL_PROVIDERS") != "true":
        raise Phase2CStop("LIVE_EXTERNAL_PROVIDERS_DISABLED")
    if env.get("VAST_PROVISIONING") != "1":
        raise Phase2CStop("PROVISION_DISABLED")
    if env.get("VAST_APPROVED_OFFER_ID") != APPROVED_OFFER_ID:
        raise Phase2CStop("OFFER_NOT_APPROVED")
    if env.get("VAST_HUMAN_APPROVAL") != approval_token(APPROVED_OFFER_ID):
        raise Phase2CStop("HUMAN_APPROVAL_REQUIRED")
    if SafetyLimits().max_gpu_workers != 1 or SafetyLimits().max_job_attempts != 1:
        raise Phase2CStop("MAX_GPU_WORKERS")
    return key


def accept_offer(offer: GpuOffer) -> None:
    if offer.offer_id != APPROVED_OFFER_ID:
        raise Phase2CStop("NO_FALLBACK_OFFER")
    if model_key(offer.gpu_model) != model_key(APPROVED_GPU_MODEL):
        raise Phase2CStop("OFFER_MISMATCH")
    if offer.availability not in (None, "rentable"):
        raise Phase2CStop("OFFER_GONE")
    if offer.hourly_price_usd > MAX_HOURLY_USD:
        raise Phase2CStop("PRICE_ABOVE_CEILING")
    cost = max_estimated_gpu_cost(offer.hourly_price_usd, MAX_LIFETIME_SECONDS)
    if cost > MAX_ESTIMATED_COST_USD:
        raise Phase2CStop("COST_ABOVE_CEILING")


def choose_approved(offers: list[GpuOffer]) -> GpuOffer:
    matches = [offer for offer in offers if offer.offer_id == APPROVED_OFFER_ID]
    if not matches:
        raise Phase2CStop("NO_FALLBACK_OFFER" if offers else "OFFER_GONE")
    if len(matches) != 1:
        raise Phase2CStop("NO_FALLBACK_OFFER")
    accept_offer(matches[0])
    return matches[0]


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
    except Phase2CStop as exc:
        _say(out, exc.code)
        return 2
    started = now()
    try:
        offers = VastDiscovery(transport=search_transport).search(
            GpuRequirements(offer_id=APPROVED_OFFER_ID, gpu_count=1),
            api_key=key,
            read_only=True,
            limit=5,
        )
        selected = choose_approved(offers)
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
    requirements = ResourceRequirements(
        min_vram_gb=Decimal("24"),
        gpu_count=1,
        max_hourly_price_usd=MAX_HOURLY_USD,
        min_reliability=Decimal("0"),
        max_job_seconds=MAX_LIFETIME_SECONDS,
    )
    try:
        created = lifecycle.create(
            offers=[selected],
            requirements=requirements,
            offer_id=APPROVED_OFFER_ID,
            owner=OWNER,
            api_key=key,
            provision=True,
            human_approval=approval_token(APPROVED_OFFER_ID),
            image=INSTANCE_IMAGE,
            disk_gb=DISK_GB,
        )
    except Exception as exc:
        code = getattr(exc, "code", "CREATE_BLOCKED")
        _say(out, str(code))
        return 4
    if created.outcome == "ambiguous" or not created.resource_id:
        owned = lifecycle.list_owned(OWNER, api_key=key, provision=True)
        _persist(
            db_path, selected, created.resource_id or APPROVED_OFFER_ID, started, now(),
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
        "offer_id": APPROVED_OFFER_ID,
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
