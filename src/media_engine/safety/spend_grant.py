"""One execution may spend only inside an explicit grant. A later job needs a new grant."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class SpendGrant:
    execution_id: str
    ceiling_usd: Decimal
    max_instances: int
    issued_at: float
    expires_at: float
    machine_id: int | None = None
    profile: str = "live6"
    gpu_family: str = ""
    min_vram_mib: int = 0
    max_hourly_usd: Decimal | None = None
    disk_gb: int | None = None
    cold_stage: bool = False
    max_transfer_seconds: int = 0
    max_execution_seconds: int = 0
    cuda_min: str = ""
    forbidden_volume_id: int | None = None


def evaluate_grant(
    grant: SpendGrant | None,
    *,
    now: float,
    execution_id: str,
    ceiling_usd: Decimal,
    instances: int,
    machine_id: int | None = None,
) -> dict:
    """The planner, the controller, and the watchdog call this with the same ceiling."""
    if grant is None:
        return _stop("MISSING_AUTHORIZATION")
    if grant.execution_id != execution_id:
        return _stop("STALE_AUTHORIZATION")
    if now >= grant.expires_at:
        return _stop("EXPIRED_AUTHORIZATION")
    if Decimal(ceiling_usd) != grant.ceiling_usd:
        return _stop("BUDGET_MISMATCH")
    if instances > grant.max_instances or grant.max_instances < 1:
        return _stop("INSTANCE_LIMIT")
    if grant.machine_id is not None and machine_id != grant.machine_id:
        return _stop("MACHINE_SCOPE")
    if grant.expires_at <= grant.issued_at:
        return _stop("EXPIRED_AUTHORIZATION")
    return {
        "action": "continue",
        "reason": "AUTHORIZED",
        "ceiling_usd": str(grant.ceiling_usd),
        "max_instances": grant.max_instances,
        "expires_at": grant.expires_at,
    }


def _stop(reason: str) -> dict:
    return {"action": "stop", "reason": reason, "ceiling_usd": None, "max_instances": 0, "expires_at": None}
