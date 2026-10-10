"""One session over the existing queue, cache, and GPU manager. It does not rent."""
from __future__ import annotations

from decimal import Decimal

from media_engine.infra.ownership import SharedOwnership
from media_engine.infra.policies import catalog
from media_engine.platform.service import PlatformBlocked


class PipelineError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class IntegratedSession:
    def __init__(self, *, manager, worker, cache, platform, source, policy_name: str = "1h",
                 started_at: float = 0.0, now=None) -> None:
        self.manager = manager
        self.worker = worker
        self.cache = cache
        self.platform = platform
        self.source = source
        self.policy = catalog()[policy_name]
        self.started_at = float(started_at)
        self.now = now or manager.now
        self.ownership = SharedOwnership(platform.repo)
        self.completed: list[str] = []
        self.cache_result = ""

    def refresh(self) -> list:
        return self.manager.pull(self.source, ceiling=Decimal("5.00"))

    def submit(self, user_id: str, job_id: str) -> str:
        if not self._open():
            self.manager.alerts.append({"code": "BUDGET_EXHAUSTED"})
            raise PipelineError("BUDGET_EXHAUSTED")
        if len(self.completed) >= self.policy.max_jobs:
            raise PipelineError("MAX_JOBS")
        decision = self.manager.evaluate_capacity(self.policy.name, None)
        if decision["allowed"]:
            raise PipelineError("CREATE_GATE")
        if not self.platform.pool_priority():
            self.platform.set_pool_priority(["topup"])
        self.platform.set_entitlement(
            key="max_concurrent_jobs", value=self.policy.max_jobs, user_id=user_id,
        )
        self.platform.authorize_paid(user_id, job_id)
        return job_id

    def run(self, user_id: str, revision: str, files: list[dict], source) -> dict:
        if self.worker.state == "STOPPED":
            self.worker.start()
        if self.cache.lookup(revision) == "HIT":
            self.cache_result = "HIT"
        else:
            self.cache_result = self.cache.stage(revision, source, files)
        finished: list[str] = []
        while len(self.completed) < self.policy.max_jobs:
            if not self._open():
                self.manager.alerts.append({"code": "BUDGET_EXHAUSTED"})
                break
            job_id = self.platform.claim("warm-worker")
            if not job_id:
                break
            self.worker.submit({"id": job_id, "revision": revision})
            self.worker.run_available()
            amount = self.platform.repo.reserved_amount(job_id)
            self.platform.settle(user_id, job_id, amount)
            self.completed.append(job_id)
            finished.append(job_id)
        return {"cache": self.cache_result, "jobs": finished, "creates": self.manager.provider_creates}

    def fail(self, user_id: str, job_id: str) -> str:
        try:
            return self.platform.recover_paid(user_id, job_id)
        except PlatformBlocked as exc:
            raise PipelineError(exc.code) from exc

    def _open(self) -> bool:
        return float(self.now()) < self.started_at + self.policy.session_seconds()
