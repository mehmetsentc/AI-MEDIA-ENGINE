"""Gated Vast create/observe/destroy. No entrypoint calls this in the current phase."""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable, Optional
from urllib.parse import quote_plus

from media_engine.providers.base import (
    OBSERVE_FOREIGN,
    OBSERVE_GONE,
    OBSERVE_PRESENT,
    OBSERVE_UNKNOWN,
    CreateResult,
)
from media_engine.providers.offers import GpuOffer
from media_engine.providers.planner import ResourceRequirements, plan
from media_engine.providers.vast import VastMutationBlocked, _NoRedirect
from media_engine.resources.ledger import ResourceLedger, ResourceRecord, ResourceState
from media_engine.safety.limits import SafetyLimits, require_live_external_provider

_OWNER = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
_DIGITS = re.compile(r"^[0-9]+$")
_ASKS = re.compile(r"^https://console.vast.ai/api/v0/asks/([0-9]+)/$")
_INSTANCE = re.compile(r"^https://console.vast.ai/api/v0/instances/([0-9]+)/$")
_INSTANCES_V1 = "https://console.vast.ai/api/v1/instances/"
_MAX_BODY = 1_000_000
_MAX_INSTANCE_PAGES = 8

Transport = Callable[[str, str, bytes, dict[str, str]], bytes]


class ProvisionBlocked(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def ownership_label(owner: str) -> str:
    if _OWNER.fullmatch(owner) is None:
        raise ProvisionBlocked("OWNER_REQUIRED")
    return "media-engine:" + owner


def approval_token(offer_id: str) -> str:
    if _DIGITS.fullmatch(offer_id) is None:
        raise ProvisionBlocked("OFFER_ID_INVALID")
    return "approve:" + offer_id


def _not_armed(method: str, url: str, body: bytes, headers: dict[str, str]) -> bytes:
    raise VastMutationBlocked("VAST_MUTATION_BLOCKED")


def instances_list_url(after_token: Optional[str] = None) -> str:
    """Read-only list URL used by the current Vast CLI."""
    pairs = [
        ("select_filters", json.dumps({})),
        ("order_by", json.dumps([{"col": "id", "dir": "asc"}])),
        ("limit", json.dumps(25)),
    ]
    if after_token is not None:
        if after_token == "" or any(char.isspace() for char in after_token):
            raise ProvisionBlocked("VAST_RESPONSE_INVALID")
        pairs.append(("after_token", after_token))
    query = "&".join(name + "=" + quote_plus(value) for name, value in pairs)
    return _INSTANCES_V1 + "?" + query


def _instances_list_allowed(url: str) -> bool:
    first = instances_list_url()
    if url == first:
        return True
    prefix = first + "&after_token="
    if not url.startswith(prefix):
        return False
    token = url[len(prefix):]
    return token != "" and "&" not in token and "/" not in token


def live_transport(method: str, url: str, body: bytes, headers: dict[str, str]) -> bytes:
    """Live Vast call. Only the phase 2C manual command may pass this in."""
    if method == "PUT" and _ASKS.fullmatch(url):
        allowed = True
    elif method == "GET" and _instances_list_allowed(url):
        allowed = True
    elif method == "DELETE" and _INSTANCE.fullmatch(url):
        allowed = True
    else:
        allowed = False
    if not allowed:
        raise VastMutationBlocked("VAST_MUTATION_BLOCKED")
    request = urllib.request.Request(url, data=body or None, headers=headers, method=method)
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=20) as response:
        raw = response.read(_MAX_BODY + 1)
    if len(raw) > _MAX_BODY:
        raise ProvisionBlocked("VAST_RESPONSE_TOO_LARGE")
    return raw


class VastLifecycle:
    def __init__(self, db_path: str, *, limits: Optional[SafetyLimits] = None,
                 transport: Optional[Transport] = None, now: Optional[Callable[[], float]] = None) -> None:
        self.limits = limits or SafetyLimits()
        self.ledger = ResourceLedger(db_path)
        self._transport = transport or _not_armed
        self._now = now or (lambda: 0.0)

    def create(
        self,
        *,
        offers: list[GpuOffer],
        requirements: ResourceRequirements,
        offer_id: str,
        owner: str,
        api_key: str = "",
        provision: bool = False,
        human_approval: str = "",
        image: str = "",
        disk_gb: int = 0,
    ) -> CreateResult:
        self._require_common(provision=provision, api_key=api_key)
        if human_approval != approval_token(offer_id):
            raise ProvisionBlocked("HUMAN_APPROVAL_REQUIRED")
        if not image.strip() or not isinstance(disk_gb, int) or isinstance(disk_gb, bool) or disk_gb < 1:
            raise ProvisionBlocked("WORKLOAD_IMAGE_REQUIRED")
        selected = plan(offers, requirements).selected
        if selected is None or selected.offer_id != offer_id:
            raise ProvisionBlocked("OFFER_NOT_SELECTED")
        if selected.hourly_price_usd > self.limits.max_hourly_price_usd:
            raise ProvisionBlocked("PRICE_ABOVE_CEILING")
        self._require_capacity(owner)
        label = ownership_label(owner)
        payload = json.dumps(
            {"label": label, "image": image, "disk": disk_gb},
            separators=(",", ":"),
        ).encode("utf-8")
        url = f"https://console.vast.ai/api/v0/asks/{offer_id}/"
        try:
            raw = self._transport("PUT", url, payload, self._headers(api_key))
        except ProvisionBlocked:
            raise
        except Exception:
            self._mark(offer_id, owner, ResourceState.AMBIGUOUS, selected, "AMBIGUOUS_CREATE")
            return CreateResult(outcome="ambiguous")
        instance_id = _new_contract(raw)
        if instance_id is None:
            self._mark(offer_id, owner, ResourceState.AMBIGUOUS, selected, "AMBIGUOUS_CREATE")
            return CreateResult(outcome="ambiguous")
        self._mark(instance_id, owner, ResourceState.READY, selected, None)
        return CreateResult(outcome="created", resource_id=instance_id)

    def observe(self, resource_id: str, owner: str, *, api_key: str = "", provision: bool = False) -> str:
        self._require_common(provision=provision, api_key=api_key)
        return self._classify(resource_id, owner, api_key)

    def list_owned(self, owner: str, *, api_key: str = "", provision: bool = False) -> list[str]:
        self._require_common(provision=provision, api_key=api_key)
        label = ownership_label(owner)
        return [
            str(row["id"])
            for row in self._instances(api_key)
            if str(row.get("label") or "") == label
        ]

    def terminate(
        self,
        resource_id: str,
        owner: str,
        *,
        api_key: str = "",
        provision: bool = False,
        human_approval: str = "",
    ) -> None:
        self._require_common(provision=provision, api_key=api_key)
        if _DIGITS.fullmatch(resource_id) is None:
            raise ProvisionBlocked("OFFER_ID_INVALID")
        if human_approval != "approve-terminate:" + resource_id:
            raise ProvisionBlocked("HUMAN_APPROVAL_REQUIRED")
        kind = self._classify(resource_id, owner, api_key)
        if kind != OBSERVE_PRESENT:
            raise ProvisionBlocked("OWNERSHIP_MISMATCH" if kind == OBSERVE_FOREIGN else "OWNERSHIP_UNKNOWN")
        self._transport(
            "DELETE",
            f"https://console.vast.ai/api/v0/instances/{resource_id}/",
            b"",
            self._headers(api_key),
        )

    def _require_common(self, *, provision: bool, api_key: str) -> None:
        if provision is not True:
            raise ProvisionBlocked("PROVISION_DISABLED")
        require_live_external_provider(self.limits)
        if self.limits.max_gpu_workers != 1 or self.limits.max_job_attempts != 1:
            raise ProvisionBlocked("MAX_GPU_WORKERS")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ProvisionBlocked("VAST_API_KEY_MISSING")

    def _require_capacity(self, owner: str) -> None:
        rows = self.ledger.list()
        active = [
            row for row in rows
            if row.owner_id == owner and row.state in {ResourceState.READY, ResourceState.PROVISIONING}
        ]
        if len(active) >= self.limits.max_gpu_workers:
            raise ProvisionBlocked("MAX_GPU_WORKERS")
        if self.ledger.unresolved():
            raise ProvisionBlocked("UNRESOLVED_RESOURCE")

    def _classify(self, resource_id: str, owner: str, api_key: str) -> str:
        if _DIGITS.fullmatch(resource_id) is None:
            return OBSERVE_UNKNOWN
        wanted = ownership_label(owner)
        found = None
        for row in self._instances(api_key):
            if str(row.get("id")) == resource_id:
                found = row
                break
        if found is None:
            return OBSERVE_GONE
        label = str(found.get("label") or "")
        if label == wanted:
            return OBSERVE_PRESENT
        if label.startswith("media-engine:"):
            return OBSERVE_FOREIGN
        return OBSERVE_UNKNOWN

    def _instances(self, api_key: str) -> list[dict]:
        rows: list[dict] = []
        token: Optional[str] = None
        seen: set[str] = set()
        for _ in range(_MAX_INSTANCE_PAGES):
            raw = self._transport("GET", instances_list_url(token), b"", self._headers(api_key))
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                return []
            page = payload.get("instances") if isinstance(payload, dict) else None
            if not isinstance(page, list):
                return []
            rows.extend(row for row in page if isinstance(row, dict))
            nxt = payload.get("next_token") if isinstance(payload, dict) else None
            if nxt in (None, ""):
                return rows
            if not isinstance(nxt, str) or nxt in seen:
                return []
            seen.add(nxt)
            token = nxt
        return []

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": "Bearer " + api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _mark(self, resource_id: str, owner: str, state: str, offer: GpuOffer, error: Optional[str]) -> None:
        now = self._now()
        self.ledger.upsert(ResourceRecord(
            resource_id=resource_id,
            provider="vast",
            owner_id=owner,
            state=state,
            create_attempted=True,
            hourly_price_usd=str(offer.hourly_price_usd),
            created_at=now,
            updated_at=now,
            last_error=error,
        ))


def _new_contract(raw: bytes) -> Optional[str]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("new_contract")
    if isinstance(value, bool) or value is None:
        return None
    text = str(value)
    if _DIGITS.fullmatch(text) is None:
        return None
    return text
