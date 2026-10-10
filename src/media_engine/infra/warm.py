"""One loaded model, sequential jobs. A valid lease survives a successful inference."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WarmLease:
    revision: str
    expires_at: float
    paid: bool = False


@dataclass
class WarmWorker:
    revision: str
    lease: WarmLease
    loader: object
    idle_after_seconds: float = 300.0
    now: object = field(default=lambda: 0.0)
    state: str = "STOPPED"
    loaded: bool = False
    loads: int = 0
    queue: list = field(default_factory=list)
    results: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    memory_mib: int = 0
    idle_since: float | None = None
    create_calls: int = 0

    def start(self) -> None:
        if self.lease.paid:
            raise RuntimeError("PAID_WARM_SESSION_DISABLED")
        if self.lease.revision != self.revision:
            raise RuntimeError("MODEL_MISMATCH")
        self._load()
        self.state = "READY"

    def _load(self) -> None:
        self.memory_mib = int(self.loader.load(self.revision))
        self.loaded = True
        self.loads += 1
        self.state = "LOADING"

    def submit(self, job: dict) -> None:
        if self.state not in {"READY", "IDLE"}:
            raise RuntimeError("WORKER_NOT_READY")
        self.queue.append(dict(job))

    def run_available(self) -> list:
        finished = []
        while self.queue:
            if self.state not in {"READY", "IDLE", "BUSY"}:
                break
            job = self.queue.pop(0)
            self.state = "BUSY"
            self.idle_since = None
            try:
                finished.append(self.loader.generate(job))
                self.results.append(job["id"])
            except Exception as exc:
                self.errors.append({"id": job.get("id"), "error": type(exc).__name__})
                self.state = "FAILED"
                return finished
            if self._lease_valid():
                self.state = "READY"
            else:
                self.state = "IDLE"
                self.idle_since = float(self.now())
        if self.state == "READY":
            self.state = "IDLE"
            self.idle_since = float(self.now())
        return finished

    def tick(self, now: float) -> str:
        if self.state == "IDLE" and self.idle_since is not None:
            if not self._lease_valid(now) and now - self.idle_since >= self.idle_after_seconds:
                self.stop()
        return self.state

    def crash(self) -> None:
        self.state = "FAILED"
        self.loaded = False
        self.memory_mib = 0

    def restart(self) -> None:
        if self.lease.paid:
            raise RuntimeError("PAID_WARM_SESSION_DISABLED")
        self._load()
        self.state = "READY"
        self.errors.append({"id": "", "error": "RESTARTED"})

    def stop(self) -> None:
        self.state = "STOPPED"
        self.loaded = False
        self.memory_mib = 0

    def status(self) -> dict:
        return {
            "state": self.state,
            "revision": self.revision,
            "model_loaded": self.loaded,
            "memory_mib": self.memory_mib,
            "queue": len(self.queue),
            "lease_expires_at": self.lease.expires_at,
            "paid": self.lease.paid,
            "loads": self.loads,
        }

    def _lease_valid(self, now: float | None = None) -> bool:
        current = float(self.now()) if now is None else float(now)
        return current < self.lease.expires_at
