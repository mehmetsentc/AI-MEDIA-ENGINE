"""Core platform foundation. No GPU and no live payment provider."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.platform.keys import object_key
from media_engine.platform.pool import (
    ConnectionPool,
    PoolExhausted,
    open_production_pool,
    production_database_status,
)
from media_engine.platform.schema import (
    LONG_VIDEO_KEYS,
    METER_TYPES,
    POSTGRES_SCHEMA,
    SQLITE_SCHEMA,
    should_persist_state,
)
from media_engine.platform.service import (
    MemoryMedia,
    PaddlePaymentProvider,
    Platform,
    PlatformBlocked,
)
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.ssh_surface import (
    MIGRATION_PLAN,
    PORTABLE_BOOTSTRAP_USES_SSH,
    REMAINING_SSH_AFTER_READY,
)
from media_engine.studio.boot import build_controller

PINNED = (
    "ghcr.io/mehmetsentc/ai-media-engine-worker"
    "@sha256:be2c96a09947b15ff1fd8fb1d2aaa4692dcc9c06083ef1486674ac4fbc4a907c"
)
ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 10, 8, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


class LyingMedia(MemoryMedia):
    def get(self, key: str) -> bytes:
        return b"not-the-upload"


class PlatformFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.platform = Platform(str(Path(self.tmp.name) / "platform.sqlite3"), now=self.clock)

    def test_postgres_contract_and_sqlite_dev_path(self) -> None:
        schema = POSTGRES_SCHEMA.upper()
        self.assertIn("JSONB", schema)
        self.assertIn("REFERENCES", schema)
        self.assertNotIn("BYTEA", schema)
        self.assertNotIn("BLOB", SQLITE_SCHEMA.upper())
        for name in ("USERS", "PROJECTS", "SCENES", "ARTIFACTS", "CREDIT_LEDGER", "PRICING_RULES"):
            self.assertIn(name, schema)
        for path in (ROOT / "src/media_engine/platform").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("SELECT *", text.upper())
            self.assertNotIn("9.99", text)
        self.assertTrue(should_persist_state("saving"))
        self.assertFalse(should_persist_state("booting"))
        with self.assertRaises(RuntimeError) as dsn:
            open_production_pool("sqlite:///dev.db")
        self.assertEqual(str(dsn.exception), "POSTGRES_DSN_REQUIRED")
        self.assertEqual(production_database_status(None), "POSTGRES_CONFIGURATION_REQUIRED")
        with self.assertRaises(RuntimeError):
            production_database_status("sqlite:///dev.db")
        pool = open_production_pool("postgres://example.invalid/media")
        self.assertEqual(pool.created, 0)
        reused = ConnectionPool(lambda: object(), max_size=1)
        first = reused.checkout()
        reused.checkin(first)
        self.assertIs(reused.checkout(), first)
        with self.assertRaises(PoolExhausted):
            reused.checkout()

    def test_object_keys_are_isolated_and_delivery_expires(self) -> None:
        key = object_key(
            user_id="alice", project_id="project1", scene_id="scene1",
            artifact_id="art1", ext="png", kind="images",
        )
        self.assertEqual(
            key, "users/alice/projects/project1/scenes/scene1/images/art1.png",
        )
        self.assertNotIn("bob", key)
        export = object_key(
            user_id="alice", project_id="project1", artifact_id="exp1", ext="mp4", kind="exports",
        )
        self.assertTrue(export.endswith("/exports/exp1.mp4"))
        with self.assertRaises(ValueError):
            object_key(
                user_id="../bob", project_id="project1", scene_id="scene1",
                artifact_id="art1", ext="png",
            )
        token = self.platform.issue_signed_url(key, ttl_seconds=60)
        self.assertNotIn("R2", token)
        self.assertEqual(self.platform.resolve_signed_url(token), key)
        self.clock.advance(61)
        self.assertIsNone(self.platform.resolve_signed_url(token))

    def test_artifact_verify_orphan_and_storage_accounting(self) -> None:
        lying = Platform(str(Path(self.tmp.name) / "lie.sqlite3"), media=LyingMedia(), now=self.clock)
        with self.assertRaises(PlatformBlocked) as verify:
            lying.persist_media(
                user_id="alice", project_id="project1", scene_id="scene1", job_id="job1",
                artifact_id="art1", kind="images", ext="png", data=b"png-bytes",
                mime_type="image/png", settings={"seed": 1},
            )
        self.assertEqual(verify.exception.code, "ARTIFACT_VERIFY_FAILED")
        self.assertEqual(lying.storage_bytes("alice"), 0)
        path = Path(self.tmp.name) / "worker.png"
        path.write_bytes(b"png-bytes")
        self.platform.media.fail_next = True
        with self.assertRaises(PlatformBlocked):
            self.platform.persist_media(
                user_id="alice", project_id="project1", scene_id="scene1", job_id="job1",
                artifact_id="art1", kind="images", ext="png", data=b"png-bytes",
                mime_type="image/png", settings={"seed": 1}, local_path=path,
            )
        self.assertTrue(path.is_file())
        saved = self.platform.persist_media(
            user_id="alice", project_id="project1", scene_id="scene1", job_id="job1",
            artifact_id="art1", kind="images", ext="png", data=b"png-bytes",
            mime_type="image/png", settings={"seed": 1}, local_path=path, width=1024, height=1024,
        )
        self.assertEqual(saved.state, "completed")
        self.assertFalse(path.exists())
        self.assertEqual(self.platform.storage_bytes("alice", kind="images"), len(b"png-bytes"))
        with self.assertRaises(PlatformBlocked) as orphan:
            self.platform.persist_media(
                user_id="alice", project_id="project1", scene_id="scene2", job_id="job1",
                artifact_id="art1", kind="images", ext="png", data=b"other",
                mime_type="image/png", settings={"seed": 2},
            )
        self.assertEqual(orphan.exception.code, "ARTIFACT_METADATA_FAILED")
        second_key = object_key(
            user_id="alice", project_id="project1", scene_id="scene2",
            artifact_id="art1", ext="png", kind="images",
        )
        self.assertEqual(self.platform.reconcile(second_key), "reconciled")
        self.platform.persist_media(
            user_id="alice", project_id="project1", scene_id="scene1", job_id="job2",
            artifact_id="art2", kind="images", ext="png", data=b"second",
            mime_type="image/png", settings={},
        )
        page = self.platform.repo.list_artifacts("alice", limit=1, offset=1)
        self.assertEqual(len(page), 1)
        self.platform.soft_delete("art2")
        self.assertEqual(self.platform.storage_bytes("alice"), len(b"png-bytes"))

    def test_ledger_is_append_only_and_settlement_is_idempotent(self) -> None:
        self._price(10)
        self.platform.grant("user1", "trial", 4, "trial_grant", idempotency_key="trial")
        self.platform.grant("user1", "subscription", 3, "subscription_grant", idempotency_key="sub")
        self.platform.grant("user1", "topup", 8, "topup_purchase", idempotency_key="top")
        self.platform.grant("user1", "topup", 8, "topup_purchase", idempotency_key="top")
        self.assertEqual(self.platform.balance("user1", "trial"), 4)
        self.assertEqual(self.platform.balance("user1", "subscription"), 3)
        self.assertEqual(self.platform.balance("user1", "topup"), 8)
        with self.assertRaises(sqlite3.Error):
            self.platform.repo.mutate_ledger_for_test()
        version = self.platform.authorize_paid("user1", "job1")
        again = self.platform.authorize_paid("user1", "job1")
        self.assertEqual(again, version)
        self.assertEqual(self.platform.balance("user1"), 5)
        self.platform.settle("user1", "job1", 4)
        self.platform.settle("user1", "job1", 4)
        self.assertEqual(self.platform.balance("user1"), 11)
        self.platform.authorize_paid("user1", "job2")
        self.platform.settle_failure("user1", "job2", legitimate_cost=2)
        self.platform.settle_failure("user1", "job2", legitimate_cost=2)
        self.assertEqual(self.platform.balance("user1"), 9)
        with self.assertRaises(PlatformBlocked) as short:
            self.platform.authorize_paid("user1", "job3")
        self.assertEqual(short.exception.code, "INSUFFICIENT_CREDIT")
        self.assertEqual(self.platform.balance("user1"), 9)

    def test_gates_block_provider_create(self) -> None:
        controller = self._controller()
        controller.platform = Platform(str(Path(self.tmp.name) / "gate.sqlite3"), now=self.clock)
        job = controller.submit_image("a terrace")
        controller.process_available()
        self.assertEqual(controller.provider.create_calls, 0)
        self.assertEqual(controller.get_job(job.id).error_code, "PRICING_UNAVAILABLE")
        self._price(10, controller.platform)
        controller.platform.grant("nahaber", "topup", 1, "topup_purchase", idempotency_key="small")
        second = controller.submit_image("another terrace")
        controller.process_available()
        self.assertEqual(controller.provider.create_calls, 0)
        self.assertEqual(controller.get_job(second.id).error_code, "INSUFFICIENT_CREDIT")
        controller.platform.set_switch("global", "*", True)
        controller.platform.grant("nahaber", "topup", 50, "topup_purchase", idempotency_key="more")
        third = controller.submit_image("blocked terrace")
        controller.process_available()
        self.assertEqual(controller.get_job(third.id).error_code, "GLOBAL_KILL_SWITCH")
        self.assertEqual(controller.provider.create_calls, 0)

    def test_pricing_meters_queue_and_entitlements(self) -> None:
        self._price(10, version="v1", effective_at="2026-01-01T00:00:00Z")
        self._price(12, version="v2", effective_at="2026-09-01T00:00:00Z")
        self.platform.grant("user1", "topup", 30, "topup_purchase", idempotency_key="g")
        self.platform.set_entitlement(key="max_concurrent_jobs", value=1, user_id="user1")
        for key in LONG_VIDEO_KEYS:
            self.platform.set_entitlement(key=key, value=True, user_id="user1")
        self.assertTrue(self.platform.entitlement("user1", "continuity_enabled"))
        self.platform.add_plan("plan1", "Creator")
        with self.assertRaises(ValueError):
            self.platform.add_plan("plan2", "Custom")
        bound = self.platform.authorize_paid("user1", "job-a")
        self.assertEqual(bound, "v2")
        self.platform.authorize_paid("user1", "job-b")
        first = self.platform.claim("worker-a", lease_seconds=30)
        self.assertIsNotNone(first)
        self.assertIsNone(self.platform.claim("worker-b"))
        self.platform.heartbeat(first, "worker-a", lease_seconds=30)
        self.clock.advance(31)
        self.assertEqual(self.platform.recover_expired(), 1)
        self.assertTrue(self.platform.cancel("job-b"))
        self.assertEqual(self.platform.repo.job_status("job-b"), "cancelled")
        claimed = self.platform.claim("worker-c")
        self.assertEqual(claimed, "job-a")
        for meter in METER_TYPES:
            self.platform.record_meter("job-a", "image", meter, 1, bound)
        self.platform.record_cost(
            "job-a", service="image", engine="qwen-image", model="Qwen/Qwen-Image",
            provider="vast", gpu_seconds=1, actual_cost_usd="0.10", failed_job_cost_usd="0",
        )
        self.platform.record_benchmark(
            service="image", model="Qwen/Qwen-Image", provider="vast", version="bench-0",
            provider_cost_usd=None,
        )
        self.assertEqual(self.platform.repo.benchmark_count(), 1)
        report = self.platform.finops()
        self.assertEqual(report["topup_purchase"], 30)
        self.assertEqual(report["reserve"], -24)
        self.assertEqual(report["actual_cost_usd"], 0.10)
        self.platform.set_config("provider_spend_cap:vast", "1")
        with self.assertRaises(PlatformBlocked) as cap:
            self.platform.authorize_paid("user1", "job-c")
        self.assertEqual(cap.exception.code, "PROVIDER_SPEND_CAP")

    def test_payments_continuity_and_real_mode_gate(self) -> None:
        paddle = PaddlePaymentProvider()
        with self.assertRaises(PlatformBlocked) as checkout:
            paddle.create_checkout("user1", "topup")
        self.assertEqual(checkout.exception.code, "PADDLE_NOT_CONFIGURED")
        self.assertTrue(self.platform.accept_webhook("paddle", "evt1"))
        self.assertFalse(self.platform.accept_webhook("paddle", "evt1"))
        payment = self.platform.record_payment(
            provider="paddle", kind="topup", user_id="user1", idempotency_key="pay1",
        )
        self.assertEqual(self.platform.record_payment(
            provider="paddle", kind="topup", user_id="user1", idempotency_key="pay1",
        ), payment)
        state_id = self.platform.save_continuity(
            project_id="project1", scene_id="scene1", clip_index=1,
            previous_prompt="walks in", continuation_prompt="continues",
            last_frame_artifact_id="art1",
        )
        self.platform.edit_continuation(state_id, "turns toward the window")
        self.assertEqual(self.platform.repo.continuation_prompt(state_id), "turns toward the window")
        with self.assertRaises(PlatformBlocked):
            self.platform.authorize_paid("", "job")
        development = build_controller(Path(self.tmp.name) / "dev", "development")
        self.assertIsNone(development.platform)
        real = build_controller(
            Path(self.tmp.name) / "real", "real", {"VAST_API_KEY": "unit-test-vast-secret"},
        )
        self.addCleanup(real.stop)
        self.assertIsNotNone(real.platform)
        self.assertFalse(PORTABLE_BOOTSTRAP_USES_SSH)
        self.assertIn("generate", REMAINING_SSH_AFTER_READY)
        self.assertGreaterEqual(len(MIGRATION_PLAN), 3)
        pinned = json.loads((ROOT / "deploy/worker/image_ref.json").read_text(encoding="utf-8"))
        template = json.loads((ROOT / "deploy/worker/vast_template.json").read_text(encoding="utf-8"))
        self.assertEqual(pinned["pinned_reference"], PINNED)
        self.assertEqual(template["image"], PINNED)

    def _price(self, credits: int, platform: Platform | None = None, *,
               version: str = "v1", effective_at: str = "2026-01-01T00:00:00Z") -> None:
        target = self.platform if platform is None else platform
        target.add_price(
            version=version, service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=credits, internal_cost_units=1, effective_at=effective_at,
        )

    def _controller(self) -> MediaController:
        root = Path(self.tmp.name) / "jobs"
        root.mkdir()
        return MediaController(
            str(root / "meta.sqlite3"),
            str(root / "artifacts"),
            clock=ManualClock(),
            provider=FakeGPUProvider(ManualClock().now),
        )
