"""In-memory GPU provider. No sockets and no sleeps."""
from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Callable, Optional

from media_engine.providers.base import (
    GPUProvider,
    OBSERVE_FOREIGN,
    OBSERVE_GONE,
    OBSERVE_PRESENT,
    OBSERVE_UNKNOWN,
    CreateResult,
    ProviderCapacityError,
    ProviderError,
    Quote,
    WorkerStageError,
)

OFF = "OFF"
STARTING = "STARTING"
READY = "READY"
BUSY = "BUSY"
STOPPING = "STOPPING"
TERMINATED = "TERMINATED"
ACTIVE = frozenset({STARTING, READY, BUSY, STOPPING})


class _Resource:
    def __init__(self, resource_id: str, owner: str, price: Decimal, created_at: float) -> None:
        self.resource_id = resource_id
        self.owner = owner
        self.hourly_price_usd = price
        self.created_at = created_at
        self.idle_since: Optional[float] = None
        self.status = OFF
        self.history = [OFF]


class FakeGPUProvider(GPUProvider):
    def __init__(self, now: Callable[[], float], *, hourly_price: Decimal = Decimal("0.50"),
                 outcome: str = "created", max_workers: int = 1) -> None:
        self.hourly_price = hourly_price
        self.outcome = outcome
        self._now = now
        self.max_workers = max_workers
        self.create_calls = 0
        self.terminate_calls = 0
        self.reachable = True
        self.unavailable = False
        self.fail_stage: Optional[str] = None
        self.name = "fake"
        self.gpu_model = "fake"
        self.machine_id = "local"
        self._resources: dict[str, _Resource] = {}

    def quote(self, owner: str) -> Quote:
        if not owner:
            raise ProviderError("owner is required")
        if self.unavailable:
            raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
        return Quote(hourly_price_usd=self.hourly_price, provider="fake")

    def boot(self, resource_id: str) -> None:
        self._require(resource_id)
        self._fail_stage("boot", "WORKER_BOOT_FAILED")
        self._fail_stage("timeout", "WORKER_TIMEOUT")
        self._fail_stage("connect", "CONNECT_TIMEOUT")

    def confirm_cache(self, resource_id: str) -> None:
        self._require(resource_id)
        self._fail_stage("cache", "MODEL_CACHE_INVALID")
        self._fail_stage("cache_stall", "CACHE_STALL")

    def prepare_runtime(self, resource_id: str) -> None:
        self._require(resource_id)
        self._fail_stage("runtime", "RUNTIME_PREPARE_FAILED")

    def ensure_model(self, resource_id: str) -> None:
        self._require(resource_id)
        self._fail_stage("load", "MODEL_LOAD_FAILED")

    def _fail_stage(self, stage: str, code: str) -> None:
        if self.fail_stage == stage:
            raise WorkerStageError(code)

    def create(self, owner: str) -> CreateResult:
        self.create_calls += 1
        if self.outcome == "definite_failure":
            return CreateResult(outcome="definite_failure")
        if self.outcome == "ambiguous":
            return CreateResult(outcome="ambiguous")
        if self._active_count() >= self.max_workers:
            raise ProviderCapacityError("MAX_GPU_WORKERS")
        resource_id = "gpu_" + uuid.uuid4().hex
        resource = _Resource(resource_id, owner, self.hourly_price, self._now())
        resource.history.append(STARTING)
        resource.status = STARTING
        resource.history.append(READY)
        resource.status = READY
        resource.idle_since = resource.created_at
        self._resources[resource_id] = resource
        return CreateResult(outcome="created", resource_id=resource_id)

    def status(self, resource_id: str) -> str:
        return self._require(resource_id).status

    def history(self, resource_id: str) -> list[str]:
        return list(self._require(resource_id).history)

    def created_at(self, resource_id: str) -> float:
        return self._require(resource_id).created_at

    def idle_since(self, resource_id: str) -> Optional[float]:
        return self._require(resource_id).idle_since

    def begin_busy(self, resource_id: str) -> None:
        resource = self._require(resource_id)
        if resource.status != READY:
            raise ProviderError("worker is not ready")
        resource.status = BUSY
        resource.history.append(BUSY)
        resource.idle_since = None

    def end_busy(self, resource_id: str) -> None:
        resource = self._require(resource_id)
        if resource.status != BUSY:
            raise ProviderError("worker is not busy")
        resource.status = READY
        resource.history.append(READY)
        resource.idle_since = self._now()

    def observe(self, resource_id: str, owner: str) -> str:
        if not self.reachable:
            return OBSERVE_UNKNOWN
        resource = self._resources.get(resource_id)
        if resource is None or resource.status == TERMINATED:
            return OBSERVE_GONE
        if resource.owner != owner:
            return OBSERVE_FOREIGN
        return OBSERVE_PRESENT

    def plant(self, resource_id: str, owner: str, *, status: str = READY) -> None:
        """Test hook for a resource that already exists before this process."""
        if resource_id in self._resources:
            raise ProviderError("resource already exists")
        resource = _Resource(resource_id, owner, self.hourly_price, self._now())
        resource.status = status
        resource.history = [OFF, status] if status != OFF else [OFF]
        if status == READY:
            resource.idle_since = resource.created_at
        self._resources[resource_id] = resource

    def terminate(self, resource_id: str) -> None:
        self.terminate_calls += 1
        resource = self._require(resource_id)
        if resource.status == TERMINATED:
            return
        resource.status = STOPPING
        resource.history.append(STOPPING)
        resource.status = TERMINATED
        resource.history.append(TERMINATED)
        resource.idle_since = None

    def list_owned(self) -> list[dict]:
        return [
            {
                "resource_id": item.resource_id,
                "owner": item.owner,
                "status": item.status,
                "hourly_price_usd": str(item.hourly_price_usd),
                "created_at": item.created_at,
            }
            for item in self._resources.values()
        ]

    def _require(self, resource_id: str) -> _Resource:
        try:
            return self._resources[resource_id]
        except KeyError as exc:
            raise ProviderError("unknown resource") from exc

    def _active_count(self) -> int:
        return sum(1 for item in self._resources.values() if item.status in ACTIVE)
