"""Queue a generation without creating a GPU unless a closed gate opens."""
from __future__ import annotations

from media_engine.infra.policies import PAID_ACTIONS_DISABLED


class DispatchError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class GenerationDispatch:
    def __init__(self, *, manager, worker, credits, artifacts) -> None:
        self.manager = manager
        self.worker = worker
        self.credits = credits
        self.artifacts = artifacts
        self.jobs: dict[str, dict] = {}
        self.idempotency: dict[str, str] = {}

    def submit(self, request: dict) -> dict:
        if not request.get("authenticated"):
            raise DispatchError("UNAUTHORIZED")
        key = str(request.get("idempotency_key") or "")
        if key and key in self.idempotency:
            return dict(self.jobs[self.idempotency[key]])
        user_id = str(request["user_id"])
        price = int(request["credit_price"])
        if self.credits.available(user_id) < price:
            raise DispatchError("INSUFFICIENT_CREDIT")
        job_id = str(request["job_id"])
        job = {
            "id": job_id,
            "user_id": user_id,
            "status": "queued",
            "credit_price": price,
            "revision": request["revision"],
            "settled": False,
            "refunded": False,
            "artifact_id": "",
        }
        self.credits.reserve(user_id, job_id, price)
        self.jobs[job_id] = job
        if key:
            self.idempotency[key] = job_id
        if self._warm_ready(request["revision"]):
            self.worker.submit({"id": job_id, "revision": request["revision"]})
            job["status"] = "dispatched"
            return dict(job)
        decision = self.manager.evaluate_capacity(
            str(request.get("policy") or "1h"), request.get("authorization"),
        )
        job["status"] = "queued"
        job["capacity"] = decision["reason"]
        if decision["reason"] != PAID_ACTIONS_DISABLED and decision["allowed"]:
            raise DispatchError("CREATE_GATE")
        return dict(job)

    def complete(self, job_id: str, artifact: dict) -> dict:
        job = self._job(job_id)
        if job["settled"] or job["refunded"]:
            return dict(job)
        if job["status"] == "failed":
            raise DispatchError("JOB_FAILED")
        stored = self.artifacts.put(job_id, artifact)
        self.credits.settle(job["user_id"], job_id, job["credit_price"])
        job["settled"] = True
        job["status"] = "completed"
        job["artifact_id"] = stored["id"]
        return dict(job)

    def fail(self, job_id: str, code: str) -> dict:
        job = self._job(job_id)
        if job["settled"]:
            raise DispatchError("ALREADY_SETTLED")
        if not job["refunded"]:
            self.credits.refund(job["user_id"], job_id, job["credit_price"])
            job["refunded"] = True
        job["status"] = "failed"
        job["error"] = code
        return dict(job)

    def _warm_ready(self, revision: str) -> bool:
        status = self.worker.status()
        return status["state"] in {"READY", "IDLE"} and status["model_loaded"] and status["revision"] == revision

    def _job(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            raise DispatchError("JOB_NOT_FOUND")
        return job
