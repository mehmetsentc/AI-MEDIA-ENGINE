"""Read-only Vast offer search. No instance create, start, stop, or destroy."""
from __future__ import annotations

import json
import urllib.request
from decimal import Decimal
from typing import Callable, Optional

from media_engine.providers.offers import GpuOffer, GpuRequirements, model_key

SEARCH_METHOD = "POST"
SEARCH_URL = "https://console.vast.ai/api/v0/bundles/"
_MAX_BODY = 1_000_000

Transport = Callable[[str, str, bytes, dict[str, str]], bytes]


class VastDiscoveryError(Exception):
    code = "VAST_DISCOVERY_ERROR"


class DiscoveryDisabled(VastDiscoveryError):
    code = "VAST_DISCOVERY_DISABLED"


class MissingApiKey(VastDiscoveryError):
    code = "VAST_API_KEY_MISSING"


class VastMutationBlocked(VastDiscoveryError):
    code = "VAST_MUTATION_BLOCKED"


def _decimal(value: object) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str, Decimal)):
        try:
            return Decimal(str(value))
        except Exception:
            return None
    return None


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise VastMutationBlocked("VAST_MUTATION_BLOCKED")


def _urllib_transport(method: str, url: str, body: bytes, headers: dict[str, str]) -> bytes:
    if method != SEARCH_METHOD or url != SEARCH_URL:
        raise VastMutationBlocked("VAST_MUTATION_BLOCKED")
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=20) as response:
        raw = response.read(_MAX_BODY + 1)
    if len(raw) > _MAX_BODY:
        raise VastDiscoveryError("vast search response is too large")
    return raw


class VastDiscovery:
    """Search marketplace offers. This object has no provision methods."""

    def __init__(self, transport: Optional[Transport] = None) -> None:
        self._transport = transport or _urllib_transport

    def search(
        self,
        requirements: GpuRequirements,
        *,
        api_key: str = "",
        read_only: bool = False,
        limit: int = 5,
    ) -> list[GpuOffer]:
        if read_only is not True:
            raise DiscoveryDisabled("VAST_READ_ONLY_DISCOVERY is required")
        if not isinstance(api_key, str) or not api_key.strip():
            raise MissingApiKey("VAST_API_KEY is required")
        if limit < 1 or limit > 100:
            raise VastDiscoveryError("limit must be from 1 to 100")
        query = _query(requirements, limit)
        payload = json.dumps(query, separators=(",", ":")).encode("utf-8")
        if api_key.encode("utf-8") in payload:
            raise VastDiscoveryError("refusing to send the API key in the query")
        headers = {
            "Authorization": "Bearer " + api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        try:
            if SEARCH_METHOD != "POST" or SEARCH_URL != "https://console.vast.ai/api/v0/bundles/":
                raise VastMutationBlocked("VAST_MUTATION_BLOCKED")
            raw = self._transport(SEARCH_METHOD, SEARCH_URL, payload, headers)
        except VastMutationBlocked:
            raise
        except Exception:
            raise VastDiscoveryError("vast search failed") from None
        return _normalize(raw, requirements)


def _query(requirements: GpuRequirements, limit: int) -> dict:
    query: dict = {
        "limit": limit,
        "type": "on-demand",
        "rentable": {"eq": True},
    }
    if requirements.gpu_model:
        query["gpu_name"] = {"eq": requirements.gpu_model}
    if requirements.gpu_count is not None:
        query["num_gpus"] = {"eq": requirements.gpu_count}
    if requirements.min_vram_gb is not None:
        query["gpu_ram"] = {"gte": int(requirements.min_vram_gb * 1024)}
    if requirements.max_hourly_price_usd is not None:
        query["dph_total"] = {"lte": float(requirements.max_hourly_price_usd)}
    if requirements.min_reliability is not None:
        query["reliability"] = {"gte": float(requirements.min_reliability)}
    if requirements.offer_id:
        if not requirements.offer_id.isdigit():
            raise VastDiscoveryError("offer id must be numeric")
        query["id"] = {"eq": int(requirements.offer_id)}
    if requirements.allocated_storage_gb is not None:
        query["allocated_storage"] = requirements.allocated_storage_gb
    if requirements.sort_by_hourly:
        query["order"] = [["dph_total", "asc"]]
    return query


def _normalize(raw: bytes, requirements: GpuRequirements) -> list[GpuOffer]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VastDiscoveryError("vast search failed") from exc
    rows = payload.get("offers") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise VastDiscoveryError("vast search failed")
    offers: list[GpuOffer] = []
    for row in rows:
        offer = _one(row)
        if offer is not None and _matches(offer, requirements):
            offers.append(offer)
    offers.sort(key=_offer_sort_key)
    return offers


def _offer_sort_key(offer: GpuOffer) -> tuple:
    if offer.offer_id.isdigit():
        return (0, int(offer.offer_id))
    return (1, offer.offer_id)


def _one(row: object) -> Optional[GpuOffer]:
    if not isinstance(row, dict):
        return None
    offer_id = row.get("id")
    gpu_model = row.get("gpu_name")
    gpu_count = row.get("num_gpus")
    vram_mb = _decimal(row.get("gpu_ram"))
    price = _decimal(row.get("dph_total"))
    if offer_id is None or not isinstance(gpu_model, str) or isinstance(gpu_count, bool):
        return None
    if not isinstance(gpu_count, int) or vram_mb is None or price is None:
        return None
    reliability = _decimal(row.get("reliability2"))
    if reliability is None:
        reliability = _decimal(row.get("reliability"))
    cuda_value = row.get("cuda_max_good")
    if cuda_value is None:
        cuda_value = row.get("compute_cap")
    rentable = row.get("rentable")
    machine_id = _identity(row.get("machine_id"))
    host_id = _identity(row.get("host_id"))
    return GpuOffer(
        provider="vast",
        offer_id=str(offer_id),
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        vram_gb=vram_mb / Decimal(1024),
        hourly_price_usd=price,
        reliability=reliability,
        performance=_decimal(row.get("dlperf")),
        cuda=None if cuda_value is None else str(cuda_value),
        disk_gb=_decimal(row.get("disk_space")),
        location=row.get("geolocation") if isinstance(row.get("geolocation"), str) else None,
        availability="rentable" if rentable is True else "unavailable",
        gpu_hourly_price_usd=_decimal(row.get("dph_base")),
        storage_price_per_gb_month=_decimal(row.get("storage_cost")),
        machine_id=machine_id,
        host_id=host_id,
    )


def _identity(value: object) -> Optional[str]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return str(value)


def _matches(offer: GpuOffer, requirements: GpuRequirements) -> bool:
    if requirements.gpu_count is not None and offer.gpu_count != requirements.gpu_count:
        return False
    if requirements.min_vram_gb is not None and offer.vram_gb < requirements.min_vram_gb:
        return False
    if requirements.max_hourly_price_usd is not None and offer.hourly_price_usd > requirements.max_hourly_price_usd:
        return False
    if requirements.min_reliability is not None:
        if offer.reliability is None or offer.reliability < requirements.min_reliability:
            return False
    if requirements.gpu_model:
        if model_key(offer.gpu_model) != model_key(requirements.gpu_model):
            return False
    if requirements.offer_id and offer.offer_id != requirements.offer_id:
        return False
    return True
