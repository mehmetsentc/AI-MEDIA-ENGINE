"""Manual read-only Vast offer probe. Does not rent a GPU.

Both of these must be set or the command exits without a request:

    VAST_API_KEY=your-key-here
    VAST_READ_ONLY_DISCOVERY=1

Optional generic filters, unset by default:

    VAST_MIN_VRAM_GB
    VAST_MAX_HOURLY_PRICE_USD
    VAST_MIN_RELIABILITY
    VAST_GPU_COUNT
"""
from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from typing import Mapping, Optional, TextIO

from media_engine.providers.offers import GpuRequirements
from media_engine.providers.vast import SEARCH_URL, VastDiscovery, _redact

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
        print("vast discovery is fail-closed", file=sys.stderr)
        return 2
    try:
        offers = VastDiscovery(transport=transport).search(
            _requirements(env),
            api_key=key,
            read_only=True,
            limit=5,
        )
    except Exception as exc:
        print(_redact(str(exc), key), file=sys.stderr)
        return 1
    json.dump([offer.as_dict() for offer in offers], out, indent=2)
    out.write("\n")
    return 0


def _requirements(env: Mapping[str, str]) -> GpuRequirements:
    return GpuRequirements(
        min_vram_gb=_optional_decimal(env.get("VAST_MIN_VRAM_GB")),
        max_hourly_price_usd=_optional_decimal(env.get("VAST_MAX_HOURLY_PRICE_USD")),
        min_reliability=_optional_decimal(env.get("VAST_MIN_RELIABILITY")),
        gpu_count=_optional_int(env.get("VAST_GPU_COUNT")),
    )


def _optional_decimal(value: Optional[str]) -> Optional[Decimal]:
    if value is None or not value.strip():
        return None
    return Decimal(value)


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or not value.strip():
        return None
    return int(value)


if __name__ == "__main__":
    if SEARCH_URL != "https://console.vast.ai/api/v0/bundles/":
        raise SystemExit(2)
    raise SystemExit(main())
