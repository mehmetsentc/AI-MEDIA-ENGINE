"""Shared LIVE-6 spending deadline. Clocks and Vast responses are local."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from media_engine.api.app import dispatch
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.providers.base import WorkerStageError
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.vast_image import MACHINE_ID, VOLUME_ID, VastImageProvider
from media_engine.providers.vast_watchdog import run_until_stopped
from media_engine.safety.execution_budget import (
    LIVE6_CEILING_USD,
    LIVE6_PLANNING_USD,
    LIVE6_RESERVE_USD,
    billable_seconds,
    conservative_billing_start,
    open_budget,
    retain_deadline,
)
from media_engine.safety.spend_grant import SpendGrant
from media_engine.studio.__main__ import supervisor_action
from tests.test_phase2e import PngEngine

VERIFIED_HOURLY = Decimal("0.3748148148148148511111111111")


class _TimeoutEngine:
    def render(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = 0) -> bytes:
        raise WorkerStageError("WORKER_TIMEOUT")


class _Host:
    def __init__(self, budget) -> None:
        self.execution_budget = budget

    def spending_stop(self, wall_now: float, monotonic_now: float) -> bool:
        return self.execution_budget.stop(wall_now, monotonic_now)


class _Controller:
    def __init__(self, provider) -> None:
        self.provider = provider


class SharedDeadlineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls: list[str] = []
        self._saved = {
            name: os.environ.get(name)
            for name in (
                "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD",
                "MEDIA_ENGINE_FORBID_R2_STAGE",
                "STUDIO_GPU_BUDGET_USD",
            )
        }
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "0.50"
        os.environ["MEDIA_ENGINE_FORBID_R2_STAGE"] = "1"
        os.environ.pop("STUDIO_GPU_BUDGET_USD", None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _grant(self, execution_id: str = "exec-1", *, issued: float = 1_000.0, expires: float = 9_000.0) -> SpendGrant:
        return SpendGrant(
            execution_id=execution_id,
            ceiling_usd=LIVE6_CEILING_USD,
            max_instances=1,
            issued_at=issued,
            expires_at=expires,
            machine_id=MACHINE_ID,
        )

    def _provider(self) -> VastImageProvider:
        def transport(method, url, body, headers):
            self.calls.append(method)
            if method == "PUT":
                return 200, b'{"success": true, "new_contract": 9}'
            if "volumes" in url:
                return 200, json.dumps({"volumes": [{"id": VOLUME_ID, "machine_id": MACHINE_ID}]}).encode()
            if "instances" in url:
                return 200, b'{"instances": []}'
            return 200, b'{"offers": []}'

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=lambda: 1_000.0,
            sleep=lambda _seconds: None,
            runtime=Path(self.tmp.name),
        )
        provider._monotonic = lambda: 10.0
        provider.offer_id = "30565040"
        provider.cache_source = "warm_cache"
        provider.machine_id = str(MACHINE_ID)
        provider.gpu_model = "RTX A6000"
        provider.hourly_amount = VERIFIED_HOURLY
        provider.disk_gb = 40
        provider._watchdog_popen = lambda *_args, **_kwargs: type("Proc", (), {"pid": 41})()
        return provider

    def test_verified_price_floors_to_3841_seconds_inside_the_reserve(self) -> None:
        seconds = billable_seconds(VERIFIED_HOURLY)
        self.assertEqual(seconds, 3841)
        budget = open_budget(
            execution_id="exec-1",
            grant_execution_id="exec-1",
            grant_ceiling=LIVE6_CEILING_USD,
            hourly=VERIFIED_HOURLY,
            wall_now=1_000.0,
            monotonic_now=10.0,
        )
        self.assertEqual(budget.allowed_seconds, 3841)
        self.assertEqual(budget.spending_deadline_wall, 1_000.0 + 3841)
        self.assertEqual(budget.spending_deadline_monotonic, 10.0 + 3841)
        self.assertEqual(budget.billing_start_event, "create_request")
        self.assertLessEqual(budget.planning_cost_usd(), LIVE6_PLANNING_USD)
        self.assertLess(budget.planning_cost_usd(), LIVE6_CEILING_USD)
        self.assertEqual(budget.planning_usd + budget.reserve_usd, LIVE6_CEILING_USD)
        self.assertFalse(budget.stop(1_000.0 + 3840, 10.0 + 3840))
        self.assertTrue(budget.stop(1_000.0 + 3841, 10.0 + 3840))
        host = _Host(budget)
        before = supervisor_action(
            _Controller(host), wall_now=1_000.0 + 3840, monotonic_now=10.0 + 3840, finished=False,
        )
        at_deadline = supervisor_action(
            _Controller(host), wall_now=1_000.0 + 3841, monotonic_now=10.0 + 3841, finished=False,
        )
        self.assertIsNone(before)
        self.assertEqual(at_deadline, "SPENDING_DEADLINE")
        self.assertNotEqual(at_deadline, "BUDGET_STOP")

    def test_create_freezes_the_deadline_and_a_restart_cannot_extend_it(self) -> None:
        provider = self._provider()
        provider.bind_execution(self._grant(), execution_id="exec-1")
        created = provider.create("daidoi")
        self.assertEqual(created.outcome, "created")
        self.assertEqual(provider.launch_trace, ["watchdog", "put"])
        spec_path = Path(self.tmp.name) / "watchdog.json"
        stored = json.loads(spec_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["allowed_seconds"], 3841)
        self.assertEqual(stored["spending_deadline_wall"], stored["deadline_epoch"])
        self.assertEqual(stored["execution_id"], "exec-1")
        self.assertNotEqual(stored["spending_deadline_wall"], stored["authorization_expires_at"])
        self.assertEqual(stored["billing_start_event"], "create_request")
        provider.hourly_amount = Decimal("0.20")
        provider._now = lambda: 50_000.0
        provider._monotonic = lambda: 80_000.0
        self.assertTrue(provider._arm_guard())
        again = json.loads(spec_path.read_text(encoding="utf-8"))
        self.assertEqual(again["spending_deadline_wall"], stored["spending_deadline_wall"])
        self.assertEqual(again["allowed_seconds"], 3841)
        self.assertEqual(retain_deadline(stored, stored["spending_deadline_wall"] + 500, "exec-1"), stored["spending_deadline_wall"])

    def test_slow_provisioning_does_not_move_the_deadline(self) -> None:
        deadline = 1_000.0 + 3841
        seen = {"now": 1_000.0}
        deleted: list[str] = []

        def fetch():
            if seen["now"] >= 3_000.0:
                return [{"id": "9", "machine_id": MACHINE_ID, "label": "media-engine:one"}]
            return []

        def now():
            return seen["now"]

        def sleep(_seconds):
            if seen["now"] == 1_000.0:
                seen["now"] = 3_000.0
            else:
                seen["now"] = deadline

        outcome = run_until_stopped(
            self._spec(deadline),
            fetch=fetch,
            delete=lambda instance_id: deleted.append(instance_id) or 200,
            now=now,
            sleep=sleep,
        )
        self.assertEqual(deleted, ["9"])
        self.assertEqual(outcome["deadline"], deadline)
        self.assertNotEqual(outcome["deadline"], 3_000.0 + 3841)

    def test_authorization_expiry_is_not_the_spending_deadline(self) -> None:
        provider = self._provider()
        clock = {"t": 1_000.0}
        provider._now = lambda: clock["t"]
        with self.assertRaises(WorkerStageError) as expired:
            provider.bind_execution(self._grant(expires=1_000.0), execution_id="exec-1")
        self.assertEqual(expired.exception.code, "EXPIRED_AUTHORIZATION")
        provider.bind_execution(self._grant(expires=2_000.0), execution_id="exec-1")
        clock["t"] = 2_000.0
        self.assertEqual(provider.create("daidoi").outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "EXPIRED_AUTHORIZATION")
        self.assertNotIn("PUT", self.calls)

        deleted: list[str] = []
        ticks = iter((1_600.0, 1_700.0, 5_000.0))
        outcome = run_until_stopped(
            self._spec(5_000.0, authorization_expires_at=1_500.0, deadline_epoch=1_500.0),
            fetch=lambda: [{"id": "9", "machine_id": MACHINE_ID, "label": "media-engine:one"}],
            delete=lambda instance_id: deleted.append(instance_id) or 200,
            now=lambda: next(ticks),
            sleep=lambda _seconds: None,
        )
        self.assertEqual(deleted, ["9"])
        self.assertEqual(outcome["deadline"], 5_000.0)

    def test_watchdog_restart_reports_a_failed_delete_and_a_duplicate_cleanup(self) -> None:
        spec = self._spec(5_000.0)
        rows = [{"id": "9", "machine_id": MACHINE_ID, "label": "media-engine:one"}]
        failed: list[str] = []
        first = run_until_stopped(
            {**spec, "fail_delete": True},
            fetch=lambda: rows,
            delete=lambda instance_id: failed.append(instance_id) or 500,
            now=lambda: 5_000.0,
            sleep=lambda _seconds: None,
        )
        self.assertEqual(first["termination_failures"], [{"id": "9", "status": 500}])
        self.assertEqual(first["deadline"], 5_000.0)
        deleted: list[str] = []
        second = run_until_stopped(
            spec,
            fetch=lambda: rows,
            delete=lambda instance_id: deleted.append(instance_id) or 200,
            now=lambda: 5_000.0,
            sleep=lambda _seconds: None,
        )
        third = run_until_stopped(
            spec,
            fetch=lambda: rows,
            delete=lambda instance_id: deleted.append(instance_id) or 200,
            now=lambda: 5_000.0,
            sleep=lambda _seconds: None,
        )
        self.assertEqual(deleted, ["9", "9"])
        self.assertEqual(second["deadline"], third["deadline"])
        self.assertEqual(spec["spending_deadline_wall"], 5_000.0)
        with self.assertRaises(WorkerStageError) as mismatch:
            retain_deadline(spec, 9_000.0, "other-exec")
        self.assertEqual(mismatch.exception.code, "STALE_AUTHORIZATION")

    def test_price_and_budget_mismatches_stop_before_create(self) -> None:
        self.assertIsNone(billable_seconds(Decimal("0.376")))
        costly = self._provider()
        costly.hourly_amount = Decimal("0.376")
        costly.bind_execution(self._grant("price"), execution_id="price")
        self.assertEqual(costly.create("daidoi").outcome, "definite_failure")
        self.assertEqual(costly.live6_stop, "OFFER_REJECTED")
        self.assertNotIn("PUT", self.calls)
        with self.assertRaises(WorkerStageError) as ceiling:
            open_budget(
                execution_id="exec-1",
                grant_execution_id="exec-1",
                grant_ceiling=Decimal("0.49"),
                hourly=VERIFIED_HOURLY,
                wall_now=1_000.0,
                monotonic_now=10.0,
            )
        self.assertEqual(ceiling.exception.code, "BUDGET_MISMATCH")
        with self.assertRaises(WorkerStageError) as identity:
            open_budget(
                execution_id="exec-1",
                grant_execution_id="exec-2",
                grant_ceiling=LIVE6_CEILING_USD,
                hourly=VERIFIED_HOURLY,
                wall_now=1_000.0,
                monotonic_now=10.0,
            )
        self.assertEqual(identity.exception.code, "STALE_AUTHORIZATION")
        os.environ["STUDIO_GPU_BUDGET_USD"] = "0.40"
        mismatched = self._provider()
        mismatched.bind_execution(self._grant("studio"), execution_id="studio")
        self.assertEqual(mismatched.create("daidoi").outcome, "definite_failure")
        self.assertEqual(mismatched.live6_stop, "BUDGET_MISMATCH")
        self.assertNotIn("PUT", self.calls)

    def test_uncertain_billing_start_uses_the_earliest_observation(self) -> None:
        self.assertEqual(conservative_billing_start([8_000.0, None, 4_200.0]), 4_200.0)
        budget = open_budget(
            execution_id="exec-1",
            grant_execution_id="exec-1",
            grant_ceiling=LIVE6_CEILING_USD,
            hourly=VERIFIED_HOURLY,
            wall_now=5_000.0,
            monotonic_now=100.0,
            observed_starts=[5_000.0, 8_000.0, 4_200.0],
        )
        self.assertEqual(budget.billing_started_wall, 4_200.0)
        self.assertEqual(budget.billing_started_monotonic, 100.0 - (5_000.0 - 4_200.0))
        self.assertEqual(budget.spending_deadline_wall, 4_200.0 + 3841)
        later = open_budget(
            execution_id="exec-1",
            grant_execution_id="exec-1",
            grant_ceiling=LIVE6_CEILING_USD,
            hourly=VERIFIED_HOURLY,
            wall_now=5_000.0,
            monotonic_now=100.0,
            observed_starts=[6_500.0],
        )
        self.assertEqual(later.billing_started_wall, 5_000.0)
        self.assertEqual(later.spending_deadline_wall - later.billing_started_wall, 3841)

    def test_stage_failures_release_the_worker_before_the_ceiling(self) -> None:
        cases = (
            ("connect", "CONNECT_TIMEOUT", None),
            ("ready", "WORKER_READY_TIMEOUT", None),
            ("cache", "MODEL_CACHE_INVALID", None),
            ("load", "MODEL_LOAD_FAILED", None),
            ("infer", "WORKER_TIMEOUT", _TimeoutEngine()),
        )
        for stage, code, engine in cases:
            provider = FakeGPUProvider(lambda: 1_000.0)
            provider.execution_grant = self._grant(stage)
            provider.spending_stop = lambda _wall, _mono: False
            if stage != "infer":
                provider.fail_stage = stage
            root = Path(self.tmp.name) / stage
            root.mkdir()
            controller = MediaController(
                str(root / "meta.sqlite3"),
                str(root / "artifacts"),
                clock=ManualClock(),
                provider=provider,
                image_engine=engine or PngEngine(),
            )
            dispatch(controller, "POST", "/v1/images/generations", json.dumps({"prompt": stage}).encode())
            controller.process_available()
            controller.process_available()
            self.assertEqual(controller.jobs.list()[0].error_code, code, stage)
            self.assertEqual(provider.create_calls, 1, stage)
            self.assertEqual(provider.terminate_calls, 1, stage)

    def _spec(self, deadline: float, *, authorization_expires_at: float = 500.0, deadline_epoch: float | None = None) -> dict:
        root = Path(self.tmp.name)
        return {
            "execution_id": "exec-1",
            "machine_id": MACHINE_ID,
            "safe_seconds": 3841,
            "spending_deadline_wall": deadline,
            "authorization_expires_at": authorization_expires_at,
            "deadline_epoch": deadline if deadline_epoch is None else deadline_epoch,
            "heartbeat": str(root / "beat"),
            "stop": str(root / "stop"),
            "poll_seconds": 1,
            "ceiling_usd": "0.50",
            "reserve_usd": "0.10",
            "planning_usd": "0.40",
        }
