"""Central safety defaults. External providers fail closed."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


class ExternalProvidersDisabled(Exception):
    code = "LIVE_EXTERNAL_PROVIDERS_DISABLED"


class KillSwitchEngaged(Exception):
    code = "GLOBAL_KILL_SWITCH"


@dataclass(frozen=True)
class SafetyLimits:
    max_gpu_workers: int = 1
    max_job_attempts: int = 1
    gpu_idle_shutdown_seconds: float = 300
    max_worker_lifetime_seconds: float = 1800
    live_external_providers: bool = False
    max_hourly_price_usd: Decimal = Decimal("0.60")
    # Placeholders. Phase 1 does not meter monthly spend or per-client concurrency.
    global_monthly_budget_usd: Optional[Decimal] = None
    client_monthly_budget_usd: Optional[Decimal] = None
    max_concurrent_jobs_per_client: Optional[int] = None
    global_kill_switch: bool = False


def external_provider_permitted(limits: SafetyLimits) -> bool:
    """Credentials cannot enable a live provider. Both flags must allow it,
    and the live flag defaults off."""
    if limits.global_kill_switch:
        return False
    return limits.live_external_providers is True


def require_live_external_provider(limits: SafetyLimits) -> None:
    if not external_provider_permitted(limits):
        if limits.global_kill_switch:
            raise KillSwitchEngaged("GLOBAL_KILL_SWITCH")
        raise ExternalProvidersDisabled("LIVE_EXTERNAL_PROVIDERS is false")
