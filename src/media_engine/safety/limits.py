"""Central safety defaults. External providers fail closed."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
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


def limits_from_env(limits: Optional[SafetyLimits] = None) -> SafetyLimits:
    """Apply operator ceilings. Invalid values fail closed."""
    current = limits or SafetyLimits()
    updates: dict[str, object] = {}
    raw_price = os.environ.get("MEDIA_ENGINE_MAX_GPU_HOURLY_USD", "").strip()
    if raw_price:
        try:
            price = Decimal(raw_price)
        except Exception as exc:
            raise ExternalProvidersDisabled("MEDIA_ENGINE_MAX_GPU_HOURLY_USD") from exc
        if price <= 0:
            raise ExternalProvidersDisabled("MEDIA_ENGINE_MAX_GPU_HOURLY_USD")
        updates["max_hourly_price_usd"] = price
    raw_life = os.environ.get("MEDIA_ENGINE_MAX_WORKER_LIFETIME_SECONDS", "").strip()
    if raw_life:
        try:
            lifetime = float(raw_life)
        except ValueError as exc:
            raise ExternalProvidersDisabled("MEDIA_ENGINE_MAX_WORKER_LIFETIME_SECONDS") from exc
        if lifetime <= 0:
            raise ExternalProvidersDisabled("MEDIA_ENGINE_MAX_WORKER_LIFETIME_SECONDS")
        updates["max_worker_lifetime_seconds"] = lifetime
    if not updates:
        return current
    return replace(current, **updates)


def require_live_external_provider(limits: SafetyLimits) -> None:
    if not external_provider_permitted(limits):
        if limits.global_kill_switch:
            raise KillSwitchEngaged("GLOBAL_KILL_SWITCH")
        raise ExternalProvidersDisabled("LIVE_EXTERNAL_PROVIDERS is false")
