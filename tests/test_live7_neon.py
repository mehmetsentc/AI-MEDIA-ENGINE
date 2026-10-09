"""LIVE-7 credit ledger against an isolated local database. Production Neon is never opened."""
from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from media_engine.engines.image.base import ImageEngineError
from media_engine.engines.image.fake import FakeImageEngine
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.platform.migrate import apply_isolated_migrations, apply_migrations
from media_engine.platform.postgres_app import ProductionStores
from media_engine.platform.postgres_conn import PgPool
from media_engine.platform.postgres_repo import PostgresRepository
from media_engine.platform.postgres_stores import (
    PostgresArtifactStore,
    PostgresAttemptStore,
    PostgresJobStore,
    PostgresResourceLedger,
    PostgresStudioStore,
    PostgresUsageStore,
)
from media_engine.platform.repository import PlatformBlocked
from media_engine.platform.service import MemoryMedia, Platform
from media_engine.safety.execution_budget import LIVE6_CEILING_USD, billable_seconds
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.live7_contract import LIVE7_CEILING_USD
from media_engine.studio.boot import build_controller

DATABASE = "daidoi_live7_ledger"
DSN = "dbname=" + DATABASE
USER = "live7-ledger"


def _admin():
    return psycopg.connect("dbname=postgres", autocommit=True)


def _reset_database() -> None:
    admin = _admin()
    admin.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
        (DATABASE,),
    )
    admin.execute(f"DROP DATABASE IF EXISTS {DATABASE}")
    admin.execute(f"CREATE DATABASE {DATABASE}")
    admin.close()
    connection = psycopg.connect(DSN, autocommit=False, row_factory=dict_row)
    apply_isolated_migrations(connection)
    connection.close()


class _Refuse:
    def execute(self, _sql: str):
        return self

    def fetchone(self):
        return {"name": "neondb", "addr": "10.1.1.1"}


class IsolatedLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _reset_database()
        cls.pool = PgPool(DSN)
        cls.repo = PostgresRepository(cls.pool)
        cls.platform = Platform(repository=cls.repo, media=MemoryMedia(), now=lambda: datetime.now(timezone.utc))
        cls.platform.set_pool_priority(["topup"])
        cls.platform.add_price(
            version="live7-local", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=1, effective_at="2026-01-01T00:00:00Z",
        )
        cls.platform.grant(USER, "topup", 6, "topup_purchase", idempotency_key="live7-local-grant")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.pool.close()

    def _account(self, name: str, credits: int) -> str:
        user = USER + "-" + name
        self.platform.grant(user, "topup", credits, "topup_purchase", idempotency_key="grant-" + name)
        return user

    def test_isolated_apply_refuses_any_other_database(self) -> None:
        with self.assertRaises(RuntimeError) as blocked:
            apply_migrations()
        self.assertEqual(str(blocked.exception), "MIGRATION_APPLY_BLOCKED")
        with self.assertRaises(RuntimeError) as foreign:
            apply_isolated_migrations(_Refuse())
        self.assertEqual(str(foreign.exception), "ISOLATED_DATABASE_REQUIRED")

    def test_reservation_is_atomic_and_idempotent(self) -> None:
        user = self._account("hold", 6)
        self.assertEqual(self.platform.balance(user), 6)
        version = self.platform.authorize_paid(user, "job-hold")
        self.platform.authorize_paid(user, "job-hold")
        self.assertEqual(version, "live7-local")
        self.assertEqual(self.platform.balance(user), 4)
        self.assertEqual(self.platform.repo.job_status("job-hold"), "queued")
        self.assertEqual(self.platform.pending_jobs(user), [{"job_id": "job-hold", "status": "queued"}])
        kinds = [row["entry_type"] for row in self.platform.transaction_history(user)]
        self.assertEqual(kinds, ["topup_purchase", "reserve"])

    def test_concurrent_reservations_cannot_overspend(self) -> None:
        user = self._account("race", 2)
        barrier = threading.Barrier(2)
        errors: list[str] = []

        def reserve(job_id: str) -> None:
            barrier.wait(timeout=5)
            try:
                self.platform.authorize_paid(user, job_id)
            except PlatformBlocked as exc:
                errors.append(exc.code)

        threads = [
            threading.Thread(target=reserve, args=("job-race-a",)),
            threading.Thread(target=reserve, args=("job-race-b",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, ["INSUFFICIENT_CREDIT"])
        self.assertEqual(self.platform.balance(user), 0)

    def test_settlement_and_refund_are_idempotent(self) -> None:
        user = self._account("settle", 6)
        self.platform.authorize_paid(user, "job-ok")
        before = self.platform.balance(user)
        self.platform.settle(user, "job-ok", 2)
        self.platform.settle(user, "job-ok", 2)
        self.platform.settle_failure(user, "job-ok", legitimate_cost=0)
        self.assertEqual(self.platform.balance(user), before)
        self.assertEqual(self.platform.repo.job_status("job-ok"), "completed")
        self.assertNotIn(
            "refund",
            [row["entry_type"] for row in self.platform.transaction_history(user) if row["job_id"] == "job-ok"],
        )
        self.platform.authorize_paid(user, "job-bad")
        held = self.platform.balance(user)
        self.platform.settle_failure(user, "job-bad", legitimate_cost=0)
        self.platform.settle_failure(user, "job-bad", legitimate_cost=0)
        self.assertEqual(self.platform.balance(user), held + 2)
        self.assertEqual(self.platform.repo.job_status("job-bad"), "cancelled")

    def test_upload_without_settlement_recovers_without_another_charge(self) -> None:
        user = self._account("crash", 4)
        self.platform.authorize_paid(user, "job-crash")
        self.platform.repo.insert_artifact({
            "id": "art-crash", "user_id": user, "project_id": "direct", "scene_id": "image",
            "asset_id": "", "job_id": "job-crash", "artifact_type": "images", "bucket": "daidoi-media",
            "object_key": "private/job-crash.png", "mime_type": "image/png", "byte_count": 8,
            "sha256": "ab" * 32, "width": 1024, "height": 1024, "provider": "vast",
            "engine": "qwen-image", "model": "Qwen/Qwen-Image",
            "generation_settings": '{"seed": 20260804}',
        }, self.platform.now_iso())
        before = self.platform.balance(user)
        self.assertEqual(self.platform.recover_paid(user, "job-crash"), "settled_from_artifact")
        self.assertEqual(self.platform.recover_paid(user, "job-crash"), "settled")
        self.assertEqual(self.platform.balance(user), before)
        self.assertEqual(self.platform.repo.artifact_for_job("job-crash")["object_key"], "private/job-crash.png")

    def test_execution_slot_is_single_use(self) -> None:
        user = self._account("slot", 4)
        self.platform.authorize_paid(user, "job-slot")
        self.assertTrue(self.platform.repo.claim_execution_slot("job-slot", self.platform.now_iso()))
        self.assertFalse(self.platform.repo.claim_execution_slot("job-slot", self.platform.now_iso()))
        self.assertEqual(self.platform.recover_paid(user, "job-slot"), "refunded")
        self.assertEqual(self.platform.balance(user), 4)


class PaidExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_database()
        self.pool = PgPool(DSN)
        self.addCleanup(self.pool.close)
        repo = PostgresRepository(self.pool)
        self.platform = Platform(repository=repo, media=MemoryMedia(), now=lambda: datetime.now(timezone.utc))
        self.platform.set_pool_priority(["topup"])
        self.platform.add_price(
            version="live7-run", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=1, effective_at="2026-01-01T00:00:00Z",
        )
        self.platform.grant("nahaber", "topup", 4, "topup_purchase", idempotency_key="live7-run-grant")
        self.root = Path(tempfile.mkdtemp())
        self.provider = FakeGPUProvider(ManualClock().now)
        stores = ProductionStores(
            jobs=PostgresJobStore(self.pool),
            attempts=PostgresAttemptStore(self.pool, max_attempts=1),
            ledger=PostgresResourceLedger(self.pool),
            usage=PostgresUsageStore(self.pool),
            artifacts=PostgresArtifactStore(self.pool),
            studio=PostgresStudioStore(self.pool, user_id="nahaber"),
            platform=self.platform,
            pool=self.pool,
        )
        self.controller = MediaController(
            str(self.root / "unused.sqlite3"),
            str(self.root / "artifacts"),
            clock=ManualClock(),
            provider=self.provider,
            stores=stores,
        )

    def test_database_down_before_reservation_does_not_create(self) -> None:
        closed = PgPool("postgresql://127.0.0.1:1/daidoi_live7_ledger")
        self.controller.platform = Platform(
            repository=PostgresRepository(closed), media=MemoryMedia(),
            now=lambda: datetime.now(timezone.utc),
        )
        self.controller.submit_image("A cinematic Mediterranean newsroom")
        self.controller.process_available()
        self.assertEqual(self.provider.create_calls, 0)
        self.assertFalse((self.root / "unused.sqlite3").exists())

    def test_provider_failures_refund_once_and_do_not_create_again(self) -> None:
        cases = ("definite_failure", "cache", "load")
        for case in cases:
            provider = FakeGPUProvider(self.controller.clock.now)
            if case == "definite_failure":
                provider.outcome = "definite_failure"
            else:
                provider.fail_stage = case
            self.controller.provider = provider
            job = self.controller.submit_image("A cinematic Mediterranean newsroom", seed=20260804)
            self.controller.process_available()
            self.controller.process_available()
            self.assertEqual(provider.create_calls, 1)
            self.assertEqual(self.platform.balance("nahaber"), 4)
            self.assertEqual(self.controller.get_job(job.id).status, "FAILED")

    def test_duplicate_http_request_reserves_once(self) -> None:
        first = self.controller.submit_image(
            "A cinematic Mediterranean newsroom", seed=20260804, idempotency_key="http-1",
        )
        second = self.controller.submit_image(
            "A cinematic Mediterranean newsroom", seed=20260804, idempotency_key="http-1",
        )
        self.assertEqual(first.id, second.id)
        self.platform.authorize_paid("nahaber", first.id)
        self.platform.authorize_paid("nahaber", first.id)
        self.assertEqual(self.platform.balance("nahaber"), 2)
        self.assertEqual(self.provider.create_calls, 0)

    def test_inference_and_upload_failures_refund_without_a_second_create(self) -> None:
        class _Boom:
            def render(self, *_args, **_kwargs):
                raise ImageEngineError("GENERATION_FAILED")

        self.controller.image_engine = _Boom()
        job = self.controller.submit_image("A cinematic Mediterranean newsroom", seed=20260804)
        self.controller.process_available()
        self.controller.process_available()
        self.assertEqual(self.provider.create_calls, 1)
        self.assertEqual(self.controller.get_job(job.id).status, "FAILED")
        self.assertEqual(self.platform.balance("nahaber"), 4)
        self.setUp()

        class _UploadFail(MemoryMedia):
            def put(self, key: str, data: bytes) -> None:
                raise PlatformBlocked("ARTIFACT_UPLOAD_FAILED")

        self.controller.image_engine = FakeImageEngine()
        self.controller.platform.media = _UploadFail()
        self.provider.create_calls = 0
        uploaded = self.controller.submit_image("A cinematic Mediterranean newsroom", seed=20260805)
        self.controller.process_available()
        self.controller.process_available()
        self.assertEqual(self.provider.create_calls, 1)
        self.assertEqual(self.controller.get_job(uploaded.id).status, "FAILED")
        self.assertEqual(self.platform.balance("nahaber"), 4)

    def test_failed_settlement_can_be_retried_once(self) -> None:
        self.platform.authorize_paid("nahaber", "job-settle-retry")
        original = self.platform.repo.post_many

        def unavailable(*_args, **_kwargs):
            raise RuntimeError("settlement unavailable")

        self.platform.repo.post_many = unavailable
        with self.assertRaises(RuntimeError):
            self.platform.settle("nahaber", "job-settle-retry", 2)
        self.platform.repo.post_many = original
        self.platform.settle("nahaber", "job-settle-retry", 2)
        self.platform.settle("nahaber", "job-settle-retry", 2)
        self.assertEqual(self.platform.balance("nahaber"), 2)


class BootSelectionTests(unittest.TestCase):
    def test_paid_profiles_fail_closed(self) -> None:
        root = Path(tempfile.mkdtemp())
        environ = {
            "STUDIO_MODE": "production",
            "VAST_API_KEY": "unit-test-vast-secret",
            "DATABASE_URL": "postgresql://127.0.0.1:1/daidoi_live7_ledger",
        }
        with self.assertRaises(RuntimeError) as production:
            build_controller(root / "production", "real", environ)
        self.assertEqual(str(production.exception), "POSTGRES_CONNECTION_FAILED")
        live7 = dict(environ)
        live7["STUDIO_MODE"] = "real"
        live7["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live7"
        with self.assertRaises(RuntimeError) as profile:
            build_controller(root / "live7", "real", live7)
        self.assertEqual(str(profile.exception), "POSTGRES_CONNECTION_FAILED")
        self.assertFalse(any((root / "production").glob("*.sqlite3")))
        self.assertFalse(any((root / "live7").glob("*.sqlite3")))

    def test_live6_budget_constants_are_unchanged(self) -> None:
        hourly = Decimal("0.3748148148148148511111111111")
        self.assertEqual(billable_seconds(hourly), 3841)
        self.assertEqual(str(LIVE6_CEILING_USD), "0.50")
        self.assertEqual(str(LIVE7_CEILING_USD), "1.50")


if __name__ == "__main__":
    unittest.main()
