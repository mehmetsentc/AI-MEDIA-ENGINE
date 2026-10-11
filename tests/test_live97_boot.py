"""A loading GPU is still booting. A dead session does not search for another."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.qwen import MODEL_ID
from media_engine.jobs.model import JobStatus
from media_engine.orchestrator.controller import MediaController, _DESTROY_ON
from media_engine.platform.service import Platform
from media_engine.providers.base import CreateResult, Quote, WorkerStageError
from media_engine.providers.execution_authorization import issue_authorization
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.phase_clock import (
    INSTANCE_STARTUP_SECONDS,
    MISSING_INSTANCE_SECONDS,
    PHASE_LIMITS,
    UNKNOWN_STATUS_SECONDS,
)
from media_engine.providers.vast_image import VastImageProvider
from media_engine.providers.vast_watchdog import HEARTBEAT_STALE_SECONDS
from media_engine.safety.execution_budget import LIVE7_RESERVE_SECONDS

PIN = "ghcr.io/mehmetsentc/ai-media-engine-worker@sha256:" + ("b" * 64)
SHA = "b86b741eec8d1b07afd08f699e6839f96b92c8d9"


class Clock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _running() -> dict:
    return {
        "id": 7,
        "actual_status": "running",
        "machine_id": 26359,
        "public_ipaddr": "203.0.113.10",
        "ports": {"8080/tcp": [{"HostPort": "18080"}]},
    }


class BootStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = os.environ.get("MEDIA_ENGINE_WORKER_IMAGE")
        os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = PIN

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("MEDIA_ENGINE_WORKER_IMAGE", None)
        else:
            os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = self.previous

    def _provider(self, transport, clock: Clock) -> VastImageProvider:
        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=lambda *args: (_ for _ in ()).throw(AssertionError("ssh")),
        )
        provider.machine_id = "26359"
        provider._health_get = lambda url: (200, b'{"state":"worker_ready","runtime":true}')
        return provider

    def test_loading_reaches_a_ready_worker_before_inference(self) -> None:
        clock = Clock()
        health_at: list[float] = []

        def transport(method, url, body, headers):
            if clock.t < 1_080:
                row = {"id": 7, "actual_status": "loading", "machine_id": 26359}
            else:
                row = _running()
            return 200, json.dumps({"instances": [row]}).encode()

        provider = self._provider(transport, clock)

        def health(url):
            health_at.append(clock.t)
            return 200, b'{"state":"worker_ready","runtime":true}'

        provider._health_get = health
        provider.boot("7")
        self.assertTrue(provider.runtime_ready)
        self.assertEqual(provider._session_closed_reason, "")
        self.assertGreaterEqual(clock.t - 1_000, 80)
        self.assertTrue(health_at)
        self.assertGreaterEqual(health_at[0], 1_080)

    def test_slow_loading_still_inside_the_startup_budget_succeeds(self) -> None:
        clock = Clock()
        limit = 1_000 + INSTANCE_STARTUP_SECONDS - 30

        def transport(method, url, body, headers):
            row = {"id": 7, "actual_status": "loading", "machine_id": 26359} if clock.t < limit else _running()
            return 200, json.dumps({"instances": [row]}).encode()

        provider = self._provider(transport, clock)
        provider.boot("7")
        elapsed = clock.t - 1_000
        self.assertGreater(elapsed, PHASE_LIMITS["worker_ready"][0])
        self.assertLess(elapsed, INSTANCE_STARTUP_SECONDS)
        self.assertTrue(provider.runtime_ready)
        self.assertEqual(provider._session_closed_reason, "")

    def test_an_exited_instance_fails_closed_without_waiting_out_startup(self) -> None:
        clock = Clock()

        def transport(method, url, body, headers):
            status = "loading" if clock.t < 1_015 else "exited"
            row = {"id": 7, "actual_status": status, "machine_id": 26359}
            return 200, json.dumps({"instances": [row]}).encode()

        provider = self._provider(transport, clock)
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "INSTANCE_EXITED")
        self.assertEqual(provider._session_closed_reason, "INSTANCE_EXITED")
        self.assertLess(clock.t - 1_000, 60)

    def test_loading_past_the_startup_budget_closes_the_session(self) -> None:
        clock = Clock()

        def transport(method, url, body, headers):
            row = {"id": 7, "actual_status": "loading", "machine_id": 26359}
            return 200, json.dumps({"instances": [row]}).encode()

        provider = self._provider(transport, clock)
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "STARTUP_DEADLINE")
        elapsed = clock.t - 1_000
        self.assertGreater(elapsed, INSTANCE_STARTUP_SECONDS)
        self.assertLess(elapsed, INSTANCE_STARTUP_SECONDS + 10)
        self.assertIn("STARTUP_DEADLINE", _DESTROY_ON)

    def test_an_unknown_status_is_verified_then_rejected(self) -> None:
        clock = Clock()

        def transport(method, url, body, headers):
            row = {"id": 7, "actual_status": "rebooting", "machine_id": 26359}
            return 200, json.dumps({"instances": [row]}).encode()

        provider = self._provider(transport, clock)
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "UNKNOWN_STARTUP_FAILURE")
        elapsed = clock.t - 1_000
        self.assertGreaterEqual(elapsed, UNKNOWN_STATUS_SECONDS)
        self.assertLess(elapsed, INSTANCE_STARTUP_SECONDS)

    def test_a_missing_row_is_not_treated_as_immediate_death(self) -> None:
        clock = Clock()

        def transport(method, url, body, headers):
            if clock.t < 1_000 + 20:
                return 200, json.dumps({"instances": []}).encode()
            row = {"id": 7, "actual_status": "loading", "machine_id": 26359}
            if clock.t >= 1_040:
                row = _running()
            return 200, json.dumps({"instances": [row]}).encode()

        provider = self._provider(transport, clock)
        provider.boot("7")
        self.assertTrue(provider.runtime_ready)
        self.assertGreaterEqual(clock.t - 1_000, 40)
        self.assertLess(clock.t - 1_000, MISSING_INSTANCE_SECONDS + 40)

    def test_a_closed_session_does_not_search_or_create(self) -> None:
        posts: list[str] = []

        def transport(method, url, body, headers):
            if method == "POST":
                posts.append(url)
            if method in {"PUT", "DELETE", "PATCH"}:
                raise AssertionError(method)
            return 200, b'{"offers":[],"instances":[]}'

        provider = VastImageProvider(api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()))
        provider._session_closed_reason = "STARTUP_DEADLINE"
        with self.assertRaises(WorkerStageError) as caught:
            provider.quote("nahaber")
        self.assertEqual(caught.exception.code, "SESSION_CLOSED")
        created = provider.create("nahaber")
        self.assertEqual(created.outcome, "definite_failure")
        self.assertFalse(provider.create_request_sent)
        self.assertEqual(provider.live6_stop, "SESSION_CLOSED")
        self.assertEqual(provider._session_closed_reason, "STARTUP_DEADLINE")
        self.assertEqual(posts, [])

    def test_watchdog_reserve_and_worker_clock_stay_put(self) -> None:
        self.assertEqual(LIVE7_RESERVE_SECONDS, 840)
        self.assertEqual(PHASE_LIMITS["worker_ready"], (420.0, 180.0))
        self.assertEqual(HEARTBEAT_STALE_SECONDS, 300.0)
        self.assertGreater(INSTANCE_STARTUP_SECONDS, PHASE_LIMITS["worker_ready"][0])

    def test_image_counts_one_two_and_three_still_match_attempts(self) -> None:
        for count in (1, 2, 3):
            record = issue_authorization(
                authorization_id=f"live9-auth-boot{count}",
                execution_id=f"live9-run-boot{count}",
                code_sha=SHA,
                allowed_machine_ids=(150706,),
                allowed_offer_ids=(52273768,),
                allowed_pairs=((150706, 52273768),),
                selection_policy="just_in_time",
                selection_constraints={"gpu_families": ["6000ada"]},
                max_hourly_usd=Decimal("0.95"),
                max_total_usd=Decimal("5.00") if count > 1 else Decimal("1.50"),
                max_instances=1,
                max_creates=1,
                max_inference_attempts=count,
                max_image_count=count,
                max_disk_gb=140,
                model=MODEL_ID,
                revision=REVISION,
                issued_at=1_000,
                expires_at=20_000,
            )
            self.assertEqual(record.max_image_count, count)
            self.assertEqual(record.max_inference_attempts, count)
        with self.assertRaises(WorkerStageError):
            issue_authorization(
                authorization_id="live9-auth-boot4",
                execution_id="live9-run-boot4",
                code_sha=SHA,
                allowed_machine_ids=(150706,),
                allowed_offer_ids=(52273768,),
                allowed_pairs=((150706, 52273768),),
                selection_policy="just_in_time",
                selection_constraints={"gpu_families": ["6000ada"]},
                max_hourly_usd=Decimal("0.95"),
                max_total_usd=Decimal("5.00"),
                max_instances=1,
                max_creates=1,
                max_inference_attempts=4,
                max_image_count=4,
                max_disk_gb=140,
                model=MODEL_ID,
                revision=REVISION,
                issued_at=1_000,
                expires_at=20_000,
            )


class SessionFailureTests(unittest.TestCase):
    def test_the_queued_job_is_refunded_once_and_does_not_rent_again(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.creates = 0
                self.searches = 0
                self.terminated: list[str] = []
                self._session_closed_reason = ""

            def quote(self, owner: str) -> Quote:
                if self._session_closed_reason:
                    raise WorkerStageError("SESSION_CLOSED")
                self.searches += 1
                return Quote(hourly_price_usd=Decimal("0.50"), provider="vast")

            def create(self, owner: str) -> CreateResult:
                self.creates += 1
                return CreateResult(outcome="created", resource_id="77")

            def boot(self, resource_id: str) -> None:
                self._session_closed_reason = "STARTUP_DEADLINE"
                raise WorkerStageError("STARTUP_DEADLINE")

            def terminate(self, resource_id: str) -> None:
                self.terminated.append(resource_id)

            def begin_busy(self, resource_id: str) -> None:
                return None

            def status(self, resource_id: str) -> str:
                return "TERMINATED"

        root = Path(tempfile.mkdtemp())
        db = root / "platform.sqlite3"
        platform = Platform(str(db), now=lambda: datetime(2026, 10, 11, tzinfo=timezone.utc))
        platform.set_pool_priority(["topup"])
        platform.repo.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=1, effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 4, "topup_purchase", idempotency_key="live97-seed")
        provider = Provider()
        controller = MediaController(
            str(root / "studio.sqlite3"),
            str(root / "artifacts"),
            provider=provider,
        )
        controller.platform = platform
        first = controller.submit_image("an alpine lake at dawn, no text", seed=20261016)
        second = controller.submit_image("a greenhouse path at morning, no text", seed=20261017)
        platform.authorize_paid("nahaber", first.id)
        platform.authorize_paid("nahaber", second.id)
        controller.process_available()
        first = controller.jobs.get(first.id)
        second = controller.jobs.get(second.id)
        self.assertEqual(provider.creates, 1)
        self.assertEqual(provider.searches, 1)
        self.assertEqual(provider.terminated, ["77"])
        self.assertEqual(first.status, JobStatus.FAILED)
        self.assertEqual(first.error_code, "STARTUP_DEADLINE")
        self.assertEqual(second.status, JobStatus.FAILED)
        self.assertEqual(second.error_code, "SESSION_CLOSED")
        self.assertIn(f"{second.id}:SESSION_CLOSED:STARTUP_DEADLINE", controller.trace)
        platform.settle_failure("nahaber", first.id)
        platform.settle_failure("nahaber", second.id)
        self.assertEqual(platform.balance("nahaber"), 4)
        with sqlite3.connect(db) as conn:
            counts = dict(conn.execute(
                "SELECT entry_type, COUNT(*) FROM credit_ledger GROUP BY entry_type"
            ).fetchall())
        self.assertEqual(counts.get("reserve"), 2)
        self.assertEqual(counts.get("refund"), 2)
        self.assertIsNone(counts.get("consume"))


if __name__ == "__main__":
    unittest.main()
