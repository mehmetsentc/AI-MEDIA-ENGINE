"""Read-only offer ranking. This command never rents a GPU.

Required, or the command exits before any request:

    VAST_API_KEY=your-key-here
    VAST_READ_ONLY_DISCOVERY=1
    VAST_MIN_VRAM_GB
    VAST_MAX_HOURLY_PRICE_USD
    VAST_MIN_RELIABILITY
    VAST_GPU_COUNT
    VAST_MAX_JOB_SECONDS

Optional:

    VAST_GPU_MODEL
    VAST_LIMIT
    VAST_MIN_PERFORMANCE
    VAST_EXCLUDED_GPU_MODELS
    VAST_COMPATIBLE_GPU_MODELS
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal
from typing import Mapping, Optional, TextIO

from media_engine.providers.planner import ResourceRequirements, format_preflight, plan
from media_engine.providers.vast import VastDiscovery

_FLAG = "VAST_READ_ONLY_DISCOVERY"


def main(
    environ: Optional[Mapping[str, str]] = None,
    transport=None,
    stdout: Optional[TextIO] = None,
) -> int:
    env = os.environ if environ is None else environ
    out = sys.stdout if stdout is None else stdout
    key = env.get("VAST_API_KEY") or ""
    if env.get(_FLAG) != "1" or not key.strip():
        print("vast preflight is fail-closed", file=sys.stderr)
        return 2
    try:
        requirements = _requirements(env)
        limit = _limit(env)
    except Exception:
        print("vast preflight is fail-closed", file=sys.stderr)
        return 2
    try:
        offers = VastDiscovery(transport=transport).search(
            requirements.to_search(gpu_model=_model(env)),
            api_key=key,
            read_only=True,
            limit=limit,
        )
        result = plan(offers, requirements)
    except Exception:
        print("vast preflight failed", file=sys.stderr)
        return 1
    out.write(format_preflight(result))
    return 0 if result.status == "HUMAN_APPROVAL_REQUIRED" else 3


def _requirements(env: Mapping[str, str]) -> ResourceRequirements:
    return ResourceRequirements(
        min_vram_gb=Decimal(env["VAST_MIN_VRAM_GB"]),
        gpu_count=int(env["VAST_GPU_COUNT"]),
        max_hourly_price_usd=Decimal(env["VAST_MAX_HOURLY_PRICE_USD"]),
        min_reliability=Decimal(env["VAST_MIN_RELIABILITY"]),
        min_performance=_optional_decimal(env.get("VAST_MIN_PERFORMANCE")),
        compatible_gpu_models=_names(env.get("VAST_COMPATIBLE_GPU_MODELS")),
        excluded_gpu_models=_names(env.get("VAST_EXCLUDED_GPU_MODELS")),
        max_job_seconds=int(env["VAST_MAX_JOB_SECONDS"]),
    )


def _model(env: Mapping[str, str]) -> Optional[str]:
    value = (env.get("VAST_GPU_MODEL") or "").strip()
    return value or None


def _limit(env: Mapping[str, str]) -> int:
    raw = (env.get("VAST_LIMIT") or "").strip()
    if not raw:
        return 64
    return int(raw)


def _optional_decimal(value: Optional[str]) -> Optional[Decimal]:
    if value is None or not value.strip():
        return None
    return Decimal(value)


def _names(value: Optional[str]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
