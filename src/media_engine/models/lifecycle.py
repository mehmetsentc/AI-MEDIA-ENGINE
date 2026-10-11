"""One resident model. This module does not create a worker or a volume."""
from __future__ import annotations

from media_engine.models.registry import ModelSpec
from media_engine.providers.vast_watchdog import HEARTBEAT_STALE_SECONDS

SCRATCH_BYTES = 8 * 1024 * 1024 * 1024


def plan_load(spec: ModelSpec, *, resident: str | None, free_disk_bytes: int,
              active_workers: int, execution_authorized: bool,
              busy: bool = False) -> dict:
    if active_workers > 1:
        return {"action": "REFUSE", "error": "SECOND_GPU_REFUSED"}
    if spec.readiness not in {"GPU_VERIFIED", "PRODUCTION_READY"} or not spec.executable:
        return {"action": "REFUSE", "error": "CAPABILITY_UNAVAILABLE"}
    if not execution_authorized:
        return {"action": "REFUSE", "error": "AUTHORIZATION_REQUIRED"}
    needed = int(spec.storage_bytes or 0) + SCRATCH_BYTES
    if spec.storage_bytes and free_disk_bytes < needed:
        return {"action": "REFUSE", "error": "DISK_CAPACITY", "needed_bytes": needed}
    if resident and resident != spec.model_id:
        if busy or active_workers:
            return {"action": "SWITCH_WHEN_IDLE", "error": "MODEL_BUSY", "unload": resident}
        return {"action": "UNLOAD_THEN_LOAD", "unload": resident, "load": spec.model_id}
    if resident == spec.model_id:
        return {"action": "REUSE", "model_id": spec.model_id}
    return {"action": "LOAD", "model_id": spec.model_id, "workers": 1}


def recover_worker(resident: str | None) -> dict:
    """Clear the resident model. Deletion stays on the existing watchdog."""
    return {
        "action": "UNLOAD_AND_DELETE",
        "resident_cleared": True,
        "previous": resident,
        "stale_after_seconds": HEARTBEAT_STALE_SECONDS,
        "spending_deadline": "unchanged",
    }
