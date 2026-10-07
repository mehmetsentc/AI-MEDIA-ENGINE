"""Paid-execution policy. The ledger in the repository stays authoritative."""
from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from media_engine.platform.keys import MEDIA_BUCKET, media_bucket, object_key, storage_limit
from media_engine.platform.repository import PlatformBlocked, SqliteRepository
from media_engine.platform.schema import LEDGER_TYPES, METER_TYPES, PLAN_CATEGORIES

__all__ = ["ArtifactMeta", "MemoryMedia", "PaymentProvider", "PaddlePaymentProvider", "Platform", "PlatformBlocked"]


@dataclass(frozen=True)
class ArtifactMeta:
    id: str
    object_key: str
    sha256: str
    byte_count: int
    state: str


class MemoryMedia:
    """In-process stand-in for the private media bucket."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_next = False
        self.bucket = MEDIA_BUCKET

    def put(self, key: str, data: bytes) -> None:
        if self.fail_next:
            self.fail_next = False
            raise PlatformBlocked("ARTIFACT_UPLOAD_FAILED")
        self.objects[key] = data

    def get(self, key: str) -> bytes:
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]

    def head(self, key: str) -> int:
        if key not in self.objects:
            raise PlatformBlocked("ARTIFACT_VERIFY_FAILED")
        return len(self.objects[key])


class Platform:
    def __init__(self, db_path: str, *, media: Optional[MemoryMedia] = None,
                 now: Optional[Callable[[], datetime]] = None) -> None:
        self.repo = SqliteRepository(db_path)
        self.media = media or MemoryMedia()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.bucket = str(getattr(self.media, "bucket", "") or media_bucket())

    def now_iso(self) -> str:
        return self._now().strftime("%Y-%m-%dT%H:%M:%SZ")

    def require_user(self, user_id: str, *, production: bool) -> str:
        if production and not str(user_id or "").strip():
            raise PlatformBlocked("ANONYMOUS_PAID_EXECUTION")
        if not str(user_id or "").strip():
            raise PlatformBlocked("ANONYMOUS_PAID_EXECUTION")
        self.repo.ensure_user(user_id, self.now_iso())
        return user_id

    def set_switch(self, scope: str, target: str, engaged: bool) -> None:
        self.repo.set_switch(scope, target, engaged)

    def execution_blocked(self, *, service: str = "", provider: str = "", model: str = "") -> Optional[str]:
        for scope, target in self.repo.engaged_switches():
            if scope == "global":
                return "GLOBAL_KILL_SWITCH"
            if scope == "service" and target == service:
                return "SERVICE_KILL_SWITCH"
            if scope == "provider" and target == provider:
                return "PROVIDER_KILL_SWITCH"
            if scope == "model" and target == model:
                return "MODEL_KILL_SWITCH"
        return None

    def add_plan(self, plan_id: str, category: str) -> None:
        if category not in PLAN_CATEGORIES:
            raise ValueError("plan category")
        self.repo.add_plan(plan_id, category)

    def set_entitlement(self, *, key: str, value: object, plan_id: str = "", user_id: str = "") -> None:
        self.repo.set_entitlement(key, json.dumps(value), plan_id=plan_id, user_id=user_id)

    def set_config(self, key: str, value: str) -> None:
        self.repo.set_config(key, value)

    def entitlement(self, user_id: str, key: str, default: object = None) -> object:
        raw = self.repo.entitlement_json(user_id, key)
        if raw is None:
            return default
        return json.loads(raw)

    def set_pool_priority(self, pools: list[str]) -> None:
        self.repo.set_config("pool_priority", ",".join(pools))

    def pool_priority(self) -> list[str]:
        raw = self.repo.config("pool_priority") or ""
        return [item for item in raw.split(",") if item]

    def grant(self, user_id: str, pool: str, amount: int, entry_type: str, *, idempotency_key: str,
              subscription_id: str = "", payment_id: str = "") -> str:
        if entry_type not in LEDGER_TYPES or amount == 0:
            raise ValueError("ledger entry")
        self.require_user(user_id, production=True)
        return self.repo.grant(
            user_id=user_id, pool=pool, entry_type=entry_type, amount=amount,
            idempotency_key=idempotency_key, now=self.now_iso(),
            subscription_id=subscription_id, payment_id=payment_id,
        )

    def balance(self, user_id: str, pool: Optional[str] = None) -> int:
        return self.repo.balance(user_id, pool)

    def add_price(self, **fields: object) -> None:
        self.repo.add_price(**fields)

    def authorize_paid(self, user_id: str, job_id: str, *, service: str = "image",
                       model: str = "qwen-image", provider: str = "vast") -> str:
        """Reserve credits and enqueue. A failure happens before any provider create."""
        self.require_user(user_id, production=True)
        blocked = self.execution_blocked(service=service, provider=provider, model=model)
        if blocked:
            raise PlatformBlocked(blocked)
        existing = self.repo.reservation_version(job_id)
        if existing:
            limit = int(self.entitlement(user_id, "max_concurrent_jobs", 1) or 1)
            self.repo.enqueue(
                user_id=user_id, job_id=job_id, idempotency_key=f"job:{job_id}",
                pricing_version=existing, max_concurrent=limit, now=self.now_iso(),
            )
            return existing
        rule = self.repo.price_at(service, model, self.now_iso())
        if rule is None:
            raise PlatformBlocked("PRICING_UNAVAILABLE")
        version = str(rule["version"])
        estimate = int(rule["credit_price"])
        self._enforce_spend_cap(user_id, provider, estimate)
        if self.balance(user_id) < estimate:
            raise PlatformBlocked("INSUFFICIENT_CREDIT")
        limit = int(self.entitlement(user_id, "max_concurrent_jobs", 1) or 1)
        self.repo.reserve(
            user_id=user_id, amount=estimate, job_id=job_id, version=version,
            pools=self.pool_priority(), now=self.now_iso(),
        )
        self.repo.enqueue(
            user_id=user_id, job_id=job_id, idempotency_key=f"job:{job_id}",
            pricing_version=version, max_concurrent=limit, now=self.now_iso(),
        )
        return version

    def settle(self, user_id: str, job_id: str, actual: int) -> None:
        """Return the hold, then record the actual consumption once."""
        if self.repo.settlement_exists(job_id):
            return
        remaining = max(actual, 0)
        for pool, reserved in self.repo.reserved_by_pool(job_id):
            charge = min(remaining, reserved)
            remaining -= charge
            self.repo.post(
                user_id=user_id, pool=pool, entry_type="release", amount=reserved,
                idempotency_key=f"release:{job_id}:{pool}", now=self.now_iso(), job_id=job_id,
            )
            if charge:
                self.repo.post(
                    user_id=user_id, pool=pool, entry_type="consume", amount=-charge,
                    idempotency_key=f"consume:{job_id}:{pool}", now=self.now_iso(), job_id=job_id,
                )

    def settle_failure(self, user_id: str, job_id: str, legitimate_cost: int = 0) -> None:
        """Return the hold, then keep only the legitimate failure cost."""
        if self.repo.settlement_exists(job_id):
            return
        remaining = max(legitimate_cost, 0)
        for pool, reserved in self.repo.reserved_by_pool(job_id):
            charge = min(remaining, reserved)
            remaining -= charge
            self.repo.post(
                user_id=user_id, pool=pool, entry_type="refund", amount=reserved,
                idempotency_key=f"refund:{job_id}:{pool}", now=self.now_iso(), job_id=job_id,
            )
            if charge:
                self.repo.post(
                    user_id=user_id, pool=pool, entry_type="consume", amount=-charge,
                    idempotency_key=f"fail-consume:{job_id}:{pool}", now=self.now_iso(), job_id=job_id,
                )

    def record_meter(self, job_id: str, service: str, meter_type: str, amount: int, pricing_version: str) -> None:
        if meter_type not in METER_TYPES:
            raise ValueError("meter")
        self.repo.record_meter(
            job_id=job_id, service=service, meter_type=meter_type, amount=amount,
            pricing_version=pricing_version, now=self.now_iso(),
        )

    def record_cost(self, job_id: str, **fields: object) -> None:
        self.repo.record_cost(job_id, fields)

    def cancel(self, job_id: str) -> bool:
        return self.repo.cancel(job_id, self.now_iso())

    def claim(self, worker_id: str, *, lease_seconds: int = 60) -> Optional[str]:
        until = (self._now() + timedelta(seconds=lease_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.repo.claim(worker_id, self.now_iso(), until)

    def heartbeat(self, job_id: str, worker_id: str, *, lease_seconds: int = 60) -> None:
        until = (self._now() + timedelta(seconds=lease_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.repo.heartbeat(job_id, worker_id, until, self.now_iso())

    def recover_expired(self) -> int:
        return self.repo.recover_expired(self.now_iso())

    def persist_media(self, *, user_id: str, project_id: str, scene_id: str, job_id: str,
                      artifact_id: str, kind: str, ext: str, data: bytes, mime_type: str,
                      settings: dict, local_path: Optional[Path] = None, asset_id: str = "",
                      width: Optional[int] = None, height: Optional[int] = None) -> ArtifactMeta:
        key = object_key(
            user_id=user_id, project_id=project_id, scene_id=scene_id,
            artifact_id=artifact_id, ext=ext, kind=kind,
        )
        limit_name = {
            "images": "MEDIA_MAX_IMAGE_BYTES",
            "video": "MEDIA_MAX_VIDEO_BYTES",
        }.get(kind, "MEDIA_MAX_AUDIO_BYTES")
        default = 20_000_000 if kind == "images" else 500_000_000
        if len(data) > storage_limit(limit_name, default):
            raise PlatformBlocked("STORAGE_LIMIT")
        if self.repo.storage_bytes(user_id) + len(data) > storage_limit("MEDIA_MAX_USER_STORAGE_BYTES", 5_000_000_000):
            raise PlatformBlocked("STORAGE_LIMIT")
        digest = hashlib.sha256(data).hexdigest()
        try:
            self.media.put(key, data)
        except PlatformBlocked:
            raise
        head = getattr(self.media, "head", None)
        if head is not None and int(head(key)) != len(data):
            raise PlatformBlocked("ARTIFACT_VERIFY_FAILED")
        stored = self.media.get(key)
        if hashlib.sha256(stored).hexdigest() != digest:
            raise PlatformBlocked("ARTIFACT_VERIFY_FAILED")
        try:
            self.repo.insert_artifact({
                "id": artifact_id, "user_id": user_id, "project_id": project_id, "scene_id": scene_id,
                "asset_id": asset_id, "job_id": job_id, "artifact_type": kind, "bucket": self.bucket,
                "object_key": key, "mime_type": mime_type, "byte_count": len(data), "sha256": digest,
                "width": width, "height": height, "provider": "vast", "engine": "qwen-image",
                "model": "Qwen/Qwen-Image", "generation_settings": json.dumps(settings),
            }, self.now_iso())
        except PlatformBlocked:
            self.repo.record_orphan(key, self.bucket, digest, "metadata_write_failed", self.now_iso())
            raise
        if local_path is not None:
            local_path.unlink(missing_ok=True)
        return ArtifactMeta(artifact_id, key, digest, len(data), "completed")

    def read_media(self, artifact_id: str) -> Optional[bytes]:
        row = self.repo.media_row(artifact_id)
        if row is None:
            return None
        return self.media.get(str(row["object_key"]))

    def reconcile(self, key: str) -> str:
        return self.repo.reconcile_orphan(key, self.now_iso())

    def soft_delete(self, artifact_id: str) -> None:
        self.repo.soft_delete(artifact_id, self.now_iso())

    def storage_bytes(self, user_id: str, *, project_id: str = "", kind: str = "") -> int:
        return self.repo.storage_bytes(user_id, project_id, kind)

    def issue_signed_url(self, object_key_value: str, *, ttl_seconds: int = 300) -> str:
        token = secrets.token_urlsafe(24)
        expires = (self._now() + timedelta(seconds=ttl_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.repo.issue_signed(token, object_key_value, expires)
        return token

    def resolve_signed_url(self, token: str) -> Optional[str]:
        row = self.repo.signed_object(token)
        if row is None or str(row["expires_at"]) <= self.now_iso():
            return None
        return str(row["object_key"])

    def accept_webhook(self, provider: str, event_id: str) -> bool:
        return self.repo.accept_webhook(provider, event_id, self.now_iso())

    def record_payment(self, *, provider: str, kind: str, user_id: str, idempotency_key: str) -> str:
        return self.repo.record_payment(
            provider=provider, kind=kind, user_id=user_id, idempotency_key=idempotency_key, now=self.now_iso(),
        )

    def save_continuity(self, **row: object) -> str:
        return self.repo.save_continuity(row, self.now_iso())

    def edit_continuation(self, state_id: str, prompt: str) -> None:
        self.repo.edit_continuation(state_id, prompt)

    def record_benchmark(self, **fields: object) -> str:
        payload = dict(fields)
        payload.setdefault("recorded_at", self.now_iso())
        return self.repo.record_benchmark(payload)

    def finops(self) -> dict[str, int | float]:
        totals: dict[str, int | float] = dict(self.repo.ledger_totals())
        totals["actual_cost_usd"] = self.repo.actual_cost_sum()
        return totals

    def _enforce_spend_cap(self, user_id: str, provider: str, estimate: int) -> None:
        user_cap = self.repo.config(f"user_spend_cap:{user_id}")
        if user_cap is not None and estimate > int(user_cap):
            raise PlatformBlocked("USER_SPEND_CAP")
        provider_cap = self.repo.config(f"provider_spend_cap:{provider}")
        if provider_cap and estimate > int(provider_cap):
            raise PlatformBlocked("PROVIDER_SPEND_CAP")
        daily_cap = self.repo.config("daily_spend_cap")
        if daily_cap and estimate > int(daily_cap):
            raise PlatformBlocked("DAILY_SPEND_CAP")


class PaymentProvider:
    """A browser return is not payment. The verified webhook is."""

    name = "abstract"

    def create_checkout(self, user_id: str, kind: str) -> str:
        raise PlatformBlocked("PAYMENT_PROVIDER_NOT_CONFIGURED")

    def verify_webhook(self, payload: bytes, signature: str) -> str:
        raise PlatformBlocked("PAYMENT_PROVIDER_NOT_CONFIGURED")


class PaddlePaymentProvider(PaymentProvider):
    name = "paddle"

    def create_checkout(self, user_id: str, kind: str) -> str:
        raise PlatformBlocked("PADDLE_NOT_CONFIGURED")

    def verify_webhook(self, payload: bytes, signature: str) -> str:
        raise PlatformBlocked("PADDLE_NOT_CONFIGURED")
