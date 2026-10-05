"""Phase 1 zero-cost lifecycle tests. No network and no real GPU."""
from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from urllib.parse import unquote, urlparse

from media_engine.api.app import dispatch
from media_engine.auth.clients import UnknownClient
from media_engine.engines.image.fake import FakeImageEngine
from media_engine.jobs.model import (
    IMAGE_GENERATE,
    RESTART_DURING_JOB,
    Job,
    JobStatus,
    new_job_id,
)
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.providers.base import ProviderCapacityError
from media_engine.providers.fake import FakeGPUProvider
from media_engine.safety.limits import (
    ExternalProvidersDisabled,
    KillSwitchEngaged,
    SafetyLimits,
    external_provider_permitted,
    require_live_external_provider,
)


def _db_bytes(path: Path) -> bytes:
    blob = path.read_bytes()
    for suffix in ("-wal", "-shm"):
        extra = Path(str(path) + suffix)
        if extra.exists():
            blob += extra.read_bytes()
    return blob


class Phase1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.db_path = str(root / "meta.sqlite3")
        self.storage_root = str(root / "artifacts")
        self.clock = ManualClock()

    def _controller(self, **kwargs) -> MediaController:
        kwargs.setdefault("clock", self.clock)
        return MediaController(self.db_path, self.storage_root, **kwargs)

    def test_valid_transitions(self) -> None:
        job = Job(
            id="job_demo", client_id="film-studio", job_type=IMAGE_GENERATE,
            status=JobStatus.QUEUED, created_at="t", updated_at="t", attempt_count=0,
            result_uri=None, error_code=None, prompt="x",
        )
        job.transition_to(JobStatus.WAITING_FOR_WORKER)
        job.transition_to(JobStatus.RUNNING)
        job.transition_to(JobStatus.SUCCEEDED)
        self.assertEqual(job.status, JobStatus.SUCCEEDED)

    def test_illegal_transition_rejected(self) -> None:
        job = Job(
            id="job_demo", client_id="film-studio", job_type=IMAGE_GENERATE,
            status=JobStatus.QUEUED, created_at="t", updated_at="t", attempt_count=0,
            result_uri=None, error_code=None, prompt="x",
        )
        with self.assertRaises(Exception) as caught:
            job.transition_to(JobStatus.SUCCEEDED)
        self.assertEqual(caught.exception.code, "ILLEGAL_TRANSITION")
        self.assertEqual(job.status, JobStatus.QUEUED)

    def test_queued_job_survives_restart(self) -> None:
        first = self._controller()
        job = first.submit_job("film-studio", IMAGE_GENERATE, "after restart")
        second = self._controller()
        report = second.startup()
        self.assertIn(job.id, report.requeued_job_ids)
        second.process_available()
        restored = second.get_job(job.id)
        self.assertEqual(restored.status, JobStatus.SUCCEEDED)
        self.assertTrue(restored.result_uri)

    def test_running_job_after_restart_becomes_failed(self) -> None:
        first = self._controller()
        job = first.submit_job("film-studio", IMAGE_GENERATE, "in flight")
        stored = first.jobs.get(job.id)
        stored.transition_to(JobStatus.WAITING_FOR_WORKER)
        stored.transition_to(JobStatus.RUNNING)
        stored.updated_at = self.clock.iso()
        first.jobs.save(stored)
        first.attempts.start(job.id, self.clock.iso())
        second = self._controller()
        report = second.startup()
        self.assertEqual(report.failed_job_ids, [job.id])
        failed = second.get_job(job.id)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.error_code, RESTART_DURING_JOB)
        self.assertNotIn(job.id, second.queue.pending())
        second.process_available()
        self.assertEqual(second.provider.create_calls, 0)
        conn_count = second.attempts.count(job.id)
        self.assertEqual(conn_count, 1)

    def test_only_one_fake_gpu(self) -> None:
        provider = FakeGPUProvider(self.clock.now, max_workers=1)
        created = provider.create("film-studio")
        with self.assertRaises(ProviderCapacityError):
            provider.create("film-studio")
        active = [row for row in provider.list_owned() if row["status"] != "TERMINATED"]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["resource_id"], created.resource_id)
        self.assertEqual(active[0]["owner"], "film-studio")
        self.assertEqual(active[0]["hourly_price_usd"], "0.50")

    def test_provider_create_does_not_retry(self) -> None:
        provider = FakeGPUProvider(self.clock.now, outcome="definite_failure")
        result = provider.create("film-studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertIsNone(result.resource_id)
        self.assertEqual(provider.create_calls, 1)
        self.assertEqual(provider.list_owned(), [])
        controller = self._controller(provider=provider)
        job = controller.submit_job("film-studio", IMAGE_GENERATE, "no retry")
        controller.process_available()
        controller.process_available()
        self.assertEqual(provider.create_calls, 2)
        failed = controller.get_job(job.id)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.error_code, "PROVIDER_CREATE_FAILED")
        self.assertEqual(failed.attempt_count, 0)

    def test_unresolved_resource_blocks_duplicate_create(self) -> None:
        provider = FakeGPUProvider(self.clock.now, outcome="ambiguous")
        controller = self._controller(provider=provider)
        first = controller.submit_job("film-studio", IMAGE_GENERATE, "maybe created")
        controller.process_available()
        self.assertEqual(provider.create_calls, 1)
        self.assertEqual(controller.get_job(first.id).error_code, "AMBIGUOUS_CREATE")
        self.assertTrue(controller.ledger.unresolved())
        second = controller.submit_job("film-studio", IMAGE_GENERATE, "must not create")
        controller.process_available()
        self.assertEqual(provider.create_calls, 1)
        self.assertEqual(controller.get_job(second.id).error_code, "UNRESOLVED_RESOURCE")

    def test_price_ceiling_rejection(self) -> None:
        provider = FakeGPUProvider(self.clock.now, hourly_price=Decimal("5.00"))
        controller = self._controller(provider=provider)
        job = controller.submit_job("film-studio", IMAGE_GENERATE, "too expensive")
        controller.process_available()
        self.assertEqual(provider.create_calls, 0)
        self.assertEqual(controller.get_job(job.id).error_code, "PRICE_ABOVE_CEILING")

    def test_job_belongs_to_client(self) -> None:
        controller = self._controller()
        job = controller.submit_job("film-studio", IMAGE_GENERATE, "owned")
        self.assertEqual(job.client_id, "film-studio")
        with self.assertRaises(UnknownClient):
            controller.submit_job("nahaber", IMAGE_GENERATE, "other product")
        controller.clients.allow("future-app")
        later = controller.submit_job("future-app", IMAGE_GENERATE, "later client")
        self.assertEqual(later.client_id, "future-app")
        self.assertNotEqual(job.id, later.id)

    def test_artifact_stored_outside_sqlite(self) -> None:
        controller = self._controller()
        job = controller.submit_job("film-studio", IMAGE_GENERATE, "store me")
        controller.process_available()
        saved = controller.get_job(job.id)
        self.assertIsNotNone(saved.result_uri)
        path = Path(unquote(urlparse(saved.result_uri).path))
        body = path.read_bytes()
        self.assertTrue(body.startswith(b"FAKEIMG1\n"))
        self.assertNotIn(self.db_path, str(path))
        self.assertNotIn(b"FAKEIMG1", _db_bytes(Path(self.db_path)))

    def test_successful_image_generate_lifecycle(self) -> None:
        controller = self._controller()
        status, created = dispatch(
            controller, "POST", "/v1/jobs",
            json.dumps({
                "client_id": "film-studio",
                "type": "IMAGE_GENERATE",
                "input": {"prompt": "test image"},
            }).encode("utf-8"),
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["status"], "QUEUED")
        controller.process_available()
        self.clock.advance(300)
        reason = controller.enforce_lifecycle()
        status, done = dispatch(controller, "GET", f"/v1/jobs/{created['id']}", None)
        self.assertEqual(status, 200)
        self.assertEqual(done["status"], "SUCCEEDED")
        self.assertEqual(done["client_id"], "film-studio")
        self.assertEqual(done["attempt_count"], 1)
        self.assertTrue(done["result_uri"])
        self.assertEqual(reason, "GPU_IDLE_SHUTDOWN")
        starting = next(line for line in controller.trace if line.endswith(":STARTING"))
        resource_id = starting.split(":", 1)[0]
        self.assertEqual(
            controller.provider.history(resource_id),
            ["OFF", "STARTING", "READY", "BUSY", "READY", "STOPPING", "TERMINATED"],
        )
        self.assertIn(f"{created['id']}:WAITING_FOR_WORKER", controller.trace)
        self.assertIn(f"{created['id']}:RUNNING", controller.trace)
        self.assertIn(f"{created['id']}:SUCCEEDED", controller.trace)
        self.assertEqual(controller.provider.status(resource_id), "TERMINATED")

    def test_failed_generation_becomes_failed(self) -> None:
        engine = FakeImageEngine(fail=True)
        controller = self._controller(image_engine=engine)
        job = controller.submit_job("film-studio", IMAGE_GENERATE, "fail")
        controller.process_available()
        failed = controller.get_job(job.id)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.error_code, "GENERATION_FAILED")
        self.assertIsNone(failed.result_uri)
        self.assertEqual(failed.attempt_count, 1)
        self.assertEqual(engine.calls, 1)
        controller.process_available()
        self.assertEqual(engine.calls, 1)
        self.assertEqual(controller.provider.create_calls, 1)

    def test_attempt_limit_enforced(self) -> None:
        controller = self._controller()
        now = self.clock.iso()
        job = Job(
            id=new_job_id(), client_id="film-studio", job_type=IMAGE_GENERATE,
            status=JobStatus.QUEUED, created_at=now, updated_at=now, attempt_count=1,
            result_uri=None, error_code=None, prompt="already tried",
        )
        controller.jobs.insert(job)
        controller.queue.enqueue(job.id)
        controller.process_available()
        failed = controller.get_job(job.id)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.error_code, "ATTEMPT_LIMIT")
        self.assertEqual(controller.provider.create_calls, 0)
        controller.attempts.start(job.id, now)
        with self.assertRaises(Exception) as caught:
            controller.attempts.start(job.id, now)
        self.assertIn("ATTEMPT_LIMIT", str(caught.exception))

    def test_live_external_providers_default_false(self) -> None:
        limits = SafetyLimits()
        self.assertFalse(limits.live_external_providers)
        self.assertEqual(limits.max_gpu_workers, 1)
        self.assertEqual(limits.max_job_attempts, 1)
        self.assertEqual(limits.gpu_idle_shutdown_seconds, 300)
        self.assertEqual(limits.max_worker_lifetime_seconds, 1800)
        self.assertIsNone(limits.global_monthly_budget_usd)
        self.assertIsNone(limits.client_monthly_budget_usd)
        self.assertIsNone(limits.max_concurrent_jobs_per_client)
        self.assertFalse(limits.global_kill_switch)
        self.assertFalse(external_provider_permitted(limits))
        with self.assertRaises(ExternalProvidersDisabled):
            require_live_external_provider(limits)

    def test_global_kill_switch_fails_closed(self) -> None:
        limits = SafetyLimits(live_external_providers=True, global_kill_switch=True)
        self.assertFalse(external_provider_permitted(limits))
        with self.assertRaises(KillSwitchEngaged):
            require_live_external_provider(limits)
        controller = self._controller(limits=limits)
        job = controller.submit_job("film-studio", IMAGE_GENERATE, "stopped")
        controller.process_available()
        self.assertEqual(controller.get_job(job.id).error_code, "GLOBAL_KILL_SWITCH")
        self.assertEqual(controller.provider.create_calls, 0)

    def test_cancel_queued_job(self) -> None:
        controller = self._controller()
        job = controller.submit_job("film-studio", IMAGE_GENERATE, "cancel me")
        status, payload = dispatch(controller, "POST", f"/v1/jobs/{job.id}/cancel", b"")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "CANCELLED")
        controller.process_available()
        self.assertEqual(controller.provider.create_calls, 0)
        self.assertEqual(controller.get_job(job.id).status, JobStatus.CANCELLED)

    def test_cancel_terminal_job_rejected(self) -> None:
        controller = self._controller()
        job = controller.submit_job("film-studio", IMAGE_GENERATE, "done")
        controller.process_available()
        status, payload = dispatch(controller, "POST", f"/v1/jobs/{job.id}/cancel", b"")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"], "ILLEGAL_TRANSITION")
        self.assertEqual(controller.get_job(job.id).status, JobStatus.SUCCEEDED)

    def test_idle_shutdown_uses_fake_clock(self) -> None:
        controller = self._controller()
        controller.submit_job("film-studio", IMAGE_GENERATE, "idle")
        controller.process_available()
        resource_id = controller.provider.list_owned()[0]["resource_id"]
        self.clock.advance(299)
        self.assertIsNone(controller.enforce_lifecycle())
        self.assertEqual(controller.provider.status(resource_id), "READY")
        self.clock.advance(1)
        self.assertEqual(controller.enforce_lifecycle(), "GPU_IDLE_SHUTDOWN")
        self.assertEqual(controller.provider.status(resource_id), "TERMINATED")
        self.assertFalse(controller.ledger.unresolved())

    def test_hard_lifetime_shutdown_uses_fake_clock(self) -> None:
        limits = SafetyLimits(gpu_idle_shutdown_seconds=10_000, max_worker_lifetime_seconds=1800)
        controller = self._controller(limits=limits)
        controller.submit_job("film-studio", IMAGE_GENERATE, "lifetime")
        controller.process_available()
        resource_id = controller.provider.list_owned()[0]["resource_id"]
        self.clock.advance(1799)
        self.assertIsNone(controller.enforce_lifecycle())
        self.assertEqual(controller.provider.status(resource_id), "READY")
        self.clock.advance(1)
        self.assertEqual(controller.enforce_lifecycle(), "MAX_WORKER_LIFETIME")
        self.assertEqual(controller.provider.status(resource_id), "TERMINATED")


if __name__ == "__main__":
    unittest.main()
