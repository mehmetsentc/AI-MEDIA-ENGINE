"""Offline studio pipeline. SIMULATED_E2E_PASS is not a real Qwen-Image generation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.platform.service import MemoryMedia, Platform
from media_engine.providers.base import WorkerStageError
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.vast_image import decide_staging_budget
from media_engine.studio.store import StudioStore
from tests.test_phase2e import PNG, PngEngine

SIMULATED_E2E_PASS = "SIMULATED_E2E_PASS"
REAL_IMAGE_E2E_PASS = "REAL_IMAGE_E2E_PASS"


class StageProvider(FakeGPUProvider):
    def __init__(self, now, stage: str | None = None) -> None:
        super().__init__(now)
        self.fail_stage = stage

    def confirm_cache(self, resource_id: str) -> None:
        if self.fail_stage == "transfer":
            raise WorkerStageError("CACHE_TIMEOUT")
        super().confirm_cache(resource_id)


class SimulatedPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = ManualClock()

    def test_successful_simulated_image_is_visible_in_studio(self) -> None:
        controller, platform = self._pipeline()
        store = StudioStore(controller.jobs.db_path)
        project = store.create_project("Night desk", self.clock.iso())
        scene = store.create_scene(project["id"], "Opening", self.clock.iso())
        asset = next(item for item in scene["assets"] if item["type"] == "image")
        job = controller.submit_image("fixture prompt", seed=20260804)
        store.update_asset(asset["id"], self.clock.iso(), job_id=job.id)
        controller.process_available()
        done = controller.get_job(job.id)
        linked = store.get_asset(asset["id"])
        self.assertEqual(done.public_status, "completed")
        self.assertEqual(controller.artifact_bytes(done.artifact_id), PNG)
        self.assertEqual(linked["artifact_id"], done.artifact_id)
        self.assertEqual(platform.balance("nahaber"), 1)
        self.assertEqual(controller.provider.create_calls, 1)
        self.assertEqual(SIMULATED_E2E_PASS, "SIMULATED_E2E_PASS")
        self.assertNotEqual(done.error_code, REAL_IMAGE_E2E_PASS)
        self.assertNotIn(b"Qwen-Image weights", PNG)

    def test_worker_cache_load_and_transfer_failures_refund_once(self) -> None:
        cases = {
            "boot": "WORKER_BOOT_FAILED",
            "cache": "MODEL_CACHE_INVALID",
            "load": "MODEL_LOAD_FAILED",
            "transfer": "CACHE_TIMEOUT",
        }
        for stage, code in cases.items():
            with self.subTest(stage=stage):
                controller, platform = self._pipeline(stage=stage)
                job = controller.submit_image("fixture prompt", seed=20260804)
                controller.process_available()
                failed = controller.get_job(job.id)
                self.assertEqual(failed.error_code, code)
                self.assertEqual(failed.public_status, "failed")
                self.assertEqual(platform.balance("nahaber"), 3)
                platform.settle_failure("nahaber", job.id, legitimate_cost=0)
                self.assertEqual(platform.balance("nahaber"), 3)

    def test_upload_and_persistence_failures_do_not_complete(self) -> None:
        controller, platform = self._pipeline()
        platform.media.fail_next = True
        uploaded = controller.submit_image("upload", seed=1)
        controller.process_available()
        self.assertEqual(controller.get_job(uploaded.id).error_code, "ARTIFACT_UPLOAD_FAILED")
        self.assertEqual(platform.balance("nahaber"), 3)

        def persist_fails(**_kwargs):
            raise RuntimeError("sqlite stand-in for a persistence outage")

        platform.persist_media = persist_fails
        stored = controller.submit_image("persist", seed=2)
        controller.process_available()
        self.assertEqual(controller.get_job(stored.id).error_code, "ARTIFACT_TRANSFER_FAILED")
        self.assertNotEqual(controller.get_job(stored.id).public_status, "completed")
        self.assertEqual(platform.balance("nahaber"), 3)

    def test_duplicate_delivery_and_settlement_are_idempotent(self) -> None:
        controller, platform = self._pipeline()
        job = controller.submit_image("once", seed=3)
        platform.authorize_paid("nahaber", job.id)
        self.assertEqual(platform.balance("nahaber"), 1)
        platform.authorize_paid("nahaber", job.id)
        self.assertEqual(platform.balance("nahaber"), 1)
        conn = platform.repo._conn()
        try:
            rows = conn.execute("SELECT COUNT(*) AS n FROM durable_jobs").fetchone()
            self.assertEqual(int(rows["n"]), 1)
        finally:
            conn.close()
        controller.process_available()
        self.assertEqual(platform.balance("nahaber"), 1)
        platform.settle("nahaber", job.id, 2)
        self.assertEqual(platform.balance("nahaber"), 1)
        platform.settle_failure("nahaber", job.id, legitimate_cost=0)
        self.assertEqual(platform.balance("nahaber"), 1)

    def test_budget_deadline_stops_before_a_slow_transfer_is_called_healthy(self) -> None:
        decision = decide_staging_budget(
            [(0.0, 0), (30.0, 1000)],
            now=400.0,
            deadline=400.0,
            reserve_seconds=30.0,
            total_bytes=100_000,
        )
        self.assertEqual(decision["reason"], "BUDGET_DEADLINE")
        self.assertEqual(decision["action"], "stop")

    def _pipeline(self, stage: str | None = None) -> tuple[MediaController, Platform]:
        root = Path(self.tmp.name) / (stage or "ok")
        root.mkdir(parents=True, exist_ok=True)
        platform = Platform(str(root / "platform.sqlite3"), media=MemoryMedia())
        platform.add_price(
            version="sim-rule", service="image", model="qwen-image", quality="sim",
            unit="image", credit_price=2, internal_cost_units=1, effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 3, "topup_purchase", idempotency_key="sim-" + (stage or "ok"))
        controller = MediaController(
            str(root / "jobs.sqlite3"),
            str(root / "artifacts"),
            clock=self.clock,
            provider=StageProvider(self.clock.now, stage),
            image_engine=PngEngine(),
        )
        controller.platform = platform
        return controller, platform
