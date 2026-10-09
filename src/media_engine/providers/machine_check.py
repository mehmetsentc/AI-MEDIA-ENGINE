"""Read-only suitability check for one warm-cache machine.

This module never creates, starts, or destroys an instance. Search uses the
existing bundles endpoint. A missing credential returns a closed result.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from decimal import Decimal
from typing import Callable, Optional

from media_engine.providers.vast_image import WARM_DISK_GB, _hourly
from media_engine.providers.vast_lifecycle import instances_list_url

MACHINE_ID = 47281
VOLUME_ID = 54653022
MOUNT_PATH = "/models"
MAX_HOURLY_USD = Decimal("0.375")
MIN_VRAM_MIB = 44 * 1024
BUNDLES = "https://console.vast.ai/api/v0/bundles/"
VOLUMES = "https://console.vast.ai/api/v0/volumes/?owner=me"

Transport = Callable[[str, str, bytes, dict[str, str]], tuple[int, bytes]]


def classify_machine(
    *,
    offers: list,
    volumes: list,
    instances: list,
    credential: str = "ok",
    max_hourly: Decimal = MAX_HOURLY_USD,
) -> dict:
    """Classify one inventory snapshot. `suitable` is true only for a free attach."""
    result = {
        "machine_id": MACHINE_ID,
        "volume_id": VOLUME_ID,
        "credential": credential,
        "offer_state": "MACHINE_UNAVAILABLE",
        "volume_state": "VOLUME_MISSING",
        "suitable": False,
        "code": "MACHINE_UNAVAILABLE",
        "hourly_usd": None,
        "gpu_name": "",
        "gpu_ram_mib": None,
        "rentable": None,
        "rented": None,
        "offer_id": None,
        "active_instance_ids": [],
        "creates_instance": False,
    }
    if credential == "missing":
        result["code"] = "CREDENTIAL_MISSING"
        return result
    if credential == "rejected":
        result["code"] = "CREDENTIAL_REJECTED"
        return result

    volume = _volume(volumes)
    if volume is None:
        result["volume_state"] = "VOLUME_MISSING"
    elif int(volume.get("machine_id") or 0) != MACHINE_ID:
        result["volume_state"] = "VOLUME_INCOMPATIBLE"
    elif volume.get("instances"):
        result["volume_state"] = "VOLUME_INCOMPATIBLE"
    else:
        result["volume_state"] = "VOLUME_READY"

    offer = _offer(offers)
    if offer is None:
        result["offer_state"] = "MACHINE_UNAVAILABLE"
    else:
        result["offer_id"] = offer.get("id")
        result["gpu_name"] = str(offer.get("gpu_name") or "")
        result["rentable"] = offer.get("rentable")
        result["rented"] = offer.get("rented")
        try:
            result["gpu_ram_mib"] = float(offer.get("gpu_ram") or 0)
            hourly = _hourly(offer, WARM_DISK_GB)
        except Exception:
            result["offer_state"] = "GPU_MISMATCH"
            result["code"] = "GPU_MISMATCH"
            return result
        result["hourly_usd"] = str(hourly)
        compact = result["gpu_name"].casefold().replace(" ", "").replace("_", "")
        ram = result["gpu_ram_mib"] or 0
        if "a6000" not in compact or "ada" in compact or ram < MIN_VRAM_MIB:
            result["offer_state"] = "GPU_MISMATCH"
        elif hourly > max_hourly:
            result["offer_state"] = "PRICE_CHANGED"
        elif offer.get("rented") is True:
            result["offer_state"] = "OFFER_RENTED"
        elif offer.get("rentable") is not True:
            result["offer_state"] = "OFFER_EXISTS_NOT_RENTABLE"
        else:
            result["offer_state"] = "OFFER_RENTABLE"

    active = []
    for row in instances or []:
        if not isinstance(row, dict):
            continue
        try:
            machine = int(row.get("machine_id") or 0)
        except (TypeError, ValueError):
            continue
        if machine == MACHINE_ID and row.get("id") is not None:
            active.append(str(row.get("id")))
    result["active_instance_ids"] = active
    result["code"] = result["offer_state"]
    if result["offer_state"] == "OFFER_RENTABLE" and result["volume_state"] != "VOLUME_READY":
        result["code"] = result["volume_state"]
    result["suitable"] = (
        result["code"] == "OFFER_RENTABLE"
        and result["volume_state"] == "VOLUME_READY"
        and not active
    )
    if active and result["code"] == "OFFER_RENTABLE":
        result["suitable"] = False
        result["code"] = "OFFER_RENTED"
    return result


def _offer(offers: list) -> Optional[dict]:
    for row in offers or []:
        if isinstance(row, dict) and int(row.get("machine_id") or 0) == MACHINE_ID:
            return row
    return None


def _volume(volumes: list) -> Optional[dict]:
    for row in volumes or []:
        if isinstance(row, dict) and int(row.get("id") or 0) == VOLUME_ID:
            return row
    return None


class ReadOnlyInventory:
    """One inventory read. PUT and DELETE are refused before any transport call."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self.mutations: list[str] = []

    def check(self, api_key: str) -> dict:
        if not isinstance(api_key, str) or not api_key.strip():
            return classify_machine(offers=[], volumes=[], instances=[], credential="missing")
        headers = {
            "Authorization": "Bearer " + api_key.strip(),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        volume_status, volumes = self._get(VOLUMES, headers)
        instance_status, instance_payload = self._get(instances_list_url(), headers)
        query = {
            "limit": 16,
            "type": "on-demand",
            "num_gpus": {"eq": 1},
            "machine_id": {"eq": MACHINE_ID},
            "gpu_ram": {"gte": MIN_VRAM_MIB},
        }
        offer_status, offer_payload = self._send("POST", BUNDLES, query, headers)
        if volume_status in {401, 403} or instance_status in {401, 403} or offer_status in {401, 403}:
            return classify_machine(offers=[], volumes=[], instances=[], credential="rejected")
        return classify_machine(
            offers=_rows(offer_payload, "offers"),
            volumes=volumes,
            instances=_rows(instance_payload, "instances"),
        )

    def _get(self, url: str, headers: dict[str, str]) -> tuple[int, list]:
        status, payload = self._send("GET", url, None, headers)
        key = "volumes" if "volumes" in url else "instances"
        return status, _rows(payload, key) if isinstance(payload, dict) else []

    def _send(self, method: str, url: str, body: Optional[dict], headers: dict[str, str]) -> tuple[int, dict]:
        if method in {"PUT", "DELETE", "PATCH"}:
            self.mutations.append(method)
            raise RuntimeError("MUTATION_BLOCKED")
        raw = b"" if body is None else json.dumps(body).encode("utf-8")
        status, data = self._transport(method, url, raw, headers)
        if status in {401, 403}:
            return status, {}
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, AttributeError):
            return status, {}
        return status, payload if isinstance(payload, dict) else {}


def _rows(payload: dict, key: str) -> list:
    rows = payload.get(key) if isinstance(payload, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _live_transport(method: str, url: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    if method in {"PUT", "DELETE", "PATCH"}:
        raise RuntimeError("MUTATION_BLOCKED")
    request = urllib.request.Request(url, data=body or None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.status), response.read(1_000_000)
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(20_000)
    except Exception:
        return 0, b""


def main(argv: Optional[list[str]] = None) -> int:
    """Print one JSON classification. Requires the read-only discovery flag."""
    del argv
    if os.environ.get("VAST_READ_ONLY_DISCOVERY") != "1":
        print(json.dumps({"code": "READ_ONLY_FLAG_REQUIRED", "creates_instance": False}))
        return 2
    key = os.environ.get("VAST_API_KEY", "")
    report = ReadOnlyInventory(_live_transport).check(key)
    print(json.dumps(report))
    return 0 if report["credential"] != "rejected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
