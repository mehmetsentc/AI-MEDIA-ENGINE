"""Read-only Vast inventory. Create, delete, and volume changes are refused."""
from __future__ import annotations

import json

from media_engine.providers.live7_contract import LIVE7_DISK_GB, LIVE7_MIN_VRAM_MIB
from media_engine.providers.vast_image import BUNDLES
from media_engine.providers.vast_lifecycle import collect_instance_pages

SEARCH_QUERY = {
    "limit": 200,
    "type": "on-demand",
    "rentable": {"eq": True},
    "num_gpus": {"eq": 1},
    "gpu_ram": {"gte": LIVE7_MIN_VRAM_MIB},
    "disk_space": {"gte": LIVE7_DISK_GB},
}


class ReadOnlyRefusal(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class VastReadOnly:
    """Same bundle search and instance list the image provider already uses."""

    def __init__(self, transport) -> None:
        self.transport = transport
        self.calls: list[str] = []

    def search(self) -> list:
        return self._offers(SEARCH_QUERY)

    def machine_offers(self, machine_id: int) -> list:
        query = {
            "limit": 16,
            "type": "on-demand",
            "num_gpus": {"eq": 1},
            "machine_id": {"eq": int(machine_id)},
        }
        return self._offers(query)

    def instances(self) -> list[dict]:
        def request(url: str) -> tuple[int, bytes]:
            return self._call("GET", url, None)

        return collect_instance_pages(request)

    def _offers(self, query: dict) -> list:
        status, raw = self._call("POST", BUNDLES, query)
        if int(status) != 200:
            raise ReadOnlyRefusal("PROVIDER_READ_FAILED")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReadOnlyRefusal("PROVIDER_READ_FAILED") from exc
        offers = payload.get("offers") if isinstance(payload, dict) else None
        if not isinstance(offers, list):
            raise ReadOnlyRefusal("PROVIDER_READ_FAILED")
        return [row for row in offers if isinstance(row, dict)]

    def _call(self, method: str, url: str, body: dict | None) -> tuple[int, bytes]:
        if method not in {"GET", "POST"}:
            raise ReadOnlyRefusal("PAID_ACTIONS_DISABLED")
        if method == "POST" and url != BUNDLES:
            raise ReadOnlyRefusal("PAID_ACTIONS_DISABLED")
        if method == "GET" and "instances" not in url:
            raise ReadOnlyRefusal("PAID_ACTIONS_DISABLED")
        self.calls.append(method + " " + url.split("?", 1)[0])
        return self.transport(method, url, body)


def offline_transport(method: str, url: str, body: dict | None) -> tuple[int, bytes]:
    """Parser path with no network. A live key is not read."""
    del body
    if method == "POST" and url == BUNDLES:
        return 200, b'{"offers":[]}'
    if method == "GET" and "instances" in url:
        return 200, b'{"instances":[]}'
    raise ReadOnlyRefusal("PAID_ACTIONS_DISABLED")
