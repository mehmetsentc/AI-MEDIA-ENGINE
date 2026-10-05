"""Phase 1.1 runner and reconciliation tests. No network and no real GPU."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from media_engine.api.app import dispatch
from media_engine.jobs.model import IMAGE_GENERATE, RESTART_DURING_JOB, JobStatus
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.providers.fake import READY, FakeGPUProvider
from media_engine.resources.ledger import ResourceRecord, ResourceState
from media_engine.safety.limits import (
    ExternalProvidersDisabled,
    SafetyLimits,
    external_provider_permitted,
    require_live_external_provider,
)


def _post(controller: MediaController, prompt: str = "test image") -> tuple[int, dict]:
    return dispatch(
        controller, "POST", "/v1/jobs",
        json.dumps({
            "client_id": "film-studio",
            "type": "IMAGE_GENERATE",
            "input": {"prompt": prompt},
        }).encode("utf-8"),
    )


class Phase11Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.db_path = str(root / "meta.sqlite3")
        self.storage_root = str(root / "artifacts")
        self.clock = ManualClock()
        self.controllers: list[MediaController] = []

    def tearDown(self) -> None:
        for controller in self.controllers:
            controller.stop()

    def _controller(self, **kwargs) -> MediaController:
        kwargs.setdefault("clock", self.clock)
        controller = MediaController(self.db_path, self.storage_root, **kwargs)
        self.controllers.append(controller)
        return controller

    def _remember(self, controller: MediaController, resource_id: str, owner: str = "film-studio") -> None:
        now = self.clock.now()
        controller.ledger.upsert(ResourceRecord(
            resource_id=resource_id,
            provider="fake",
            owner_id=owner,
            state=ResourceState.READY,
            create_attempted=True,
            hourly_price_usd="0.50",
            created_at=now,
            updated_at=now,
        ))

    def test_post_is_processed_by_background_runner(self) -> None:
        controller = self._controller()
        controller.start()
        status, queued = _post(controller)
        self.assertEqual(status, 201)
        self.assertEqual(queued["status"], "QUEUED")
        self.assertTrue(controller.wait_settled())
        done = controller.get_job(queued["id"])
        self.assertEqual(done.status, JobStatus.SUCCEEDED)
        self.assertTrue(done.result_uri)
        self.assertIn(f"{done.id}:WAITING_FOR_WORKER", controller.trace)
        self.assertIn(f"{done.id}:RUNNING", controller.trace)

    def test_runner_waits_without_busy_polling(self) -> None:
        calls = {"n": 0}
        entered = threading.Event()

        def idle_wait(cond: threading.Condition) -> None:
            calls["n"] += 1
            entered.set()
            cond.wait()

        controller = self._controller(idle_wait=idle_wait)
        controller.start()
        self.assertTrue(entered.wait(timeout=2))
        self.assertEqual(calls["n"], 1)
        self.assertEqual(controller.idle_waits, 1)
        self.assertEqual(calls["n"], 1)

    def test_new_job_wakes_runner(self) -> None:
        controller = self._controller()
        controller.start()
        self.assertEqual(controller.idle_waits, 1)
        _, queued = _post(controller, "wake")
        self.assertTrue(controller.wait_settled())
        self.assertEqual(controller.get_job(queued["id"]).status, JobStatus.SUCCEEDED)
        self.assertGreaterEqual(controller.idle_waits, 1)
        self.assertEqual(controller.provider.create_calls, 1)

    def test_runner_start_is_deterministic(self) -> None:
        controller = self._controller()
        report = controller.start()
        self.assertEqual(report.requeued_job_ids, [])
        self.assertTrue(controller.runner_alive)
        self.assertFalse(controller._thread.daemon)
        self.assertTrue(controller.recovery_complete)
        self.assertEqual(controller.phase_log[0], "recovery_complete")
        with self.assertRaises(RuntimeError):
            controller.start()

    def test_runner_stop_joins(self) -> None:
        controller = self._controller()
        controller.start()
        thread = controller._thread
        controller.stop()
        self.assertFalse(thread.is_alive())
        self.assertFalse(controller.runner_alive)
        controller.stop()
        self.assertFalse(thread.is_alive())

    def test_startup_requeues_and_runs_queued_job(self) -> None:
        first = self._controller()
        job = first.submit_job("film-studio", IMAGE_GENERATE, "persisted")
        second = self._controller()
        report = second.start()
        self.assertIn(job.id, report.requeued_job_ids)
        self.assertTrue(second.wait_settled())
        self.assertLess(
            second.phase_log.index("recovery_complete"),
            second.phase_log.index("provision"),
        )
        self.assertEqual(second.get_job(job.id).status, JobStatus.SUCCEEDED)

    def test_startup_running_job_becomes_failed(self) -> None:
        first = self._controller()
        job = first.submit_job("film-studio", IMAGE_GENERATE, "in flight")
        stored = first.jobs.get(job.id)
        stored.transition_to(JobStatus.WAITING_FOR_WORKER)
        stored.transition_to(JobStatus.RUNNING)
        stored.updated_at = self.clock.iso()
        first.jobs.save(stored)
        first.attempts.start(job.id, self.clock.iso())
        provider = FakeGPUProvider(self.clock.now)
        second = self._controller(provider=provider)
        report = second.start()
        self.assertEqual(report.failed_job_ids, [job.id])
        failed = second.get_job(job.id)
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.error_code, RESTART_DURING_JOB)
        self.assertEqual(provider.create_calls, 0)

    def test_confirmed_gone_resource_is_resolved(self) -> None:
        provider = FakeGPUProvider(self.clock.now)
        controller = self._controller(provider=provider)
        self._remember(controller, "gpu_missing")
        controller.start()
        record = controller.ledger.get("gpu_missing")
        self.assertEqual(record.state, ResourceState.TERMINATED)
        self.assertEqual(record.last_error, "RECONCILED_GONE")
        self.assertEqual(provider.terminate_calls, 0)
        self.assertEqual(controller.ledger.unresolved(), [])
        _, queued = _post(controller, "after gone")
        self.assertTrue(controller.wait_settled())
        self.assertEqual(controller.get_job(queued["id"]).status, JobStatus.SUCCEEDED)
        self.assertEqual(provider.create_calls, 1)

    def test_confirmed_present_owned_resource_is_terminated(self) -> None:
        provider = FakeGPUProvider(self.clock.now)
        provider.plant("gpu_live", "film-studio", status=READY)
        controller = self._controller(provider=provider)
        self._remember(controller, "gpu_live")
        controller.start()
        self.assertEqual(provider.terminate_calls, 1)
        self.assertEqual(provider.status("gpu_live"), "TERMINATED")
        record = controller.ledger.get("gpu_live")
        self.assertEqual(record.state, ResourceState.TERMINATED)
        self.assertEqual(record.last_error, "RECONCILED_TERMINATED")
        _, queued = _post(controller, "after cleanup")
        self.assertTrue(controller.wait_settled())
        self.assertEqual(controller.get_job(queued["id"]).status, JobStatus.SUCCEEDED)
        self.assertEqual(provider.create_calls, 1)

    def test_unknown_provider_state_blocks_provisioning(self) -> None:
        provider = FakeGPUProvider(self.clock.now)
        provider.reachable = False
        provider.plant("gpu_live", "film-studio", status=READY)
        controller = self._controller(provider=provider)
        self._remember(controller, "gpu_live")
        controller.start()
        self.assertEqual(provider.terminate_calls, 0)
        self.assertEqual(provider.status("gpu_live"), READY)
        record = controller.ledger.get("gpu_live")
        self.assertEqual(record.state, ResourceState.READY)
        self.assertEqual(record.last_error, "PROVIDER_STATUS_UNKNOWN")
        _, queued = _post(controller, "must block")
        self.assertTrue(controller.wait_settled())
        failed = controller.get_job(queued["id"])
        self.assertEqual(failed.status, JobStatus.FAILED)
        self.assertEqual(failed.error_code, "UNRESOLVED_RESOURCE")
        self.assertEqual(provider.create_calls, 0)

    def test_foreign_resource_is_not_terminated(self) -> None:
        provider = FakeGPUProvider(self.clock.now)
        provider.plant("gpu_foreign", "other-engine", status=READY)
        provider.plant("gpu_stranger", "other-engine", status=READY)
        controller = self._controller(provider=provider)
        self._remember(controller, "gpu_foreign", owner="film-studio")
        controller.start()
        self.assertEqual(provider.terminate_calls, 0)
        self.assertEqual(provider.status("gpu_foreign"), READY)
        self.assertEqual(provider.status("gpu_stranger"), READY)
        record = controller.ledger.get("gpu_foreign")
        self.assertEqual(record.last_error, "OWNERSHIP_MISMATCH")
        self.assertEqual(record.state, ResourceState.READY)
        _, queued = _post(controller, "do not replace foreign")
        self.assertTrue(controller.wait_settled())
        self.assertEqual(controller.get_job(queued["id"]).error_code, "UNRESOLVED_RESOURCE")
        self.assertEqual(provider.create_calls, 0)
        self.assertEqual(provider.terminate_calls, 0)

    def test_unresolved_resource_still_blocks_duplicate_create(self) -> None:
        provider = FakeGPUProvider(self.clock.now, outcome="ambiguous")
        controller = self._controller(provider=provider)
        controller.start()
        _, first = _post(controller, "ambiguous")
        self.assertTrue(controller.wait_settled())
        self.assertEqual(controller.get_job(first["id"]).error_code, "AMBIGUOUS_CREATE")
        self.assertEqual(provider.create_calls, 1)
        _, second = _post(controller, "again")
        self.assertTrue(controller.wait_settled())
        self.assertEqual(controller.get_job(second["id"]).error_code, "UNRESOLVED_RESOURCE")
        self.assertEqual(provider.create_calls, 1)

    def test_successful_create_drops_provisional_row(self) -> None:
        controller = self._controller()
        controller.start()
        _, queued = _post(controller, "clean ledger")
        self.assertTrue(controller.wait_settled())
        rows = controller.ledger.list()
        self.assertTrue(rows)
        self.assertFalse(any(row.resource_id.startswith("res_") for row in rows))
        self.assertFalse(any(row.last_error == "superseded" for row in rows))
        self.assertFalse(any(row.resource_id.startswith("res_") for row in controller.ledger.unresolved()))
        live = [row for row in rows if row.resource_id.startswith("gpu_")]
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0].state, ResourceState.READY)

    def test_external_providers_remain_fail_closed(self) -> None:
        limits = SafetyLimits()
        self.assertFalse(limits.live_external_providers)
        self.assertFalse(limits.global_kill_switch)
        self.assertIsNone(limits.global_monthly_budget_usd)
        self.assertIsNone(limits.client_monthly_budget_usd)
        self.assertIsNone(limits.max_concurrent_jobs_per_client)
        self.assertFalse(external_provider_permitted(limits))
        with self.assertRaises(ExternalProvidersDisabled):
            require_live_external_provider(limits)
        engaged = SafetyLimits(live_external_providers=True, global_kill_switch=True)
        self.assertFalse(external_provider_permitted(engaged))


if __name__ == "__main__":
    unittest.main()
