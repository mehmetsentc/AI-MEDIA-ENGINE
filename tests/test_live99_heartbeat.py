"""The supervising controller refreshes the watchdog heartbeat. The limits stay put."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from media_engine.providers.base import WorkerStageError
from media_engine.providers.phase_clock import INSTANCE_STARTUP_SECONDS, PHASE_LIMITS
from media_engine.providers.session_heartbeat import HEARTBEAT_INTERVAL_SECONDS
from media_engine.providers.vast_image import VastImageProvider
from media_engine.providers.vast_watchdog import (
    HEARTBEAT_STALE_SECONDS,
    instances_to_stop,
    run_until_stopped,
)
from media_engine.safety.execution_budget import (
    LIVE7_MAX_HOURLY_USD,
    LIVE7_RESERVE_SECONDS,
    LIVE8_JIT_CEILING_USD,
)

PIN = "ghcr.io/mehmetsentc/ai-media-engine-worker@sha256:" + ("c" * 64)


class Clock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _row(status: str) -> dict:
    row = {
        "id": 7,
        "actual_status": status,
        "machine_id": 26359,
        "label": "media-engine:live9-run-a",
    }
    if status == "running":
        row["public_ipaddr"] = "203.0.113.10"
        row["ports"] = {"8080/tcp": [{"HostPort": "18080"}]}
    return row


class HeartbeatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = os.environ.get("MEDIA_ENGINE_WORKER_IMAGE")
        os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = PIN
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()

    def tearDown(self) -> None:
        self.tmp.cleanup()
        if self.previous is None:
            os.environ.pop("MEDIA_ENGINE_WORKER_IMAGE", None)
        else:
            os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = self.previous

    def _provider(self, transport) -> VastImageProvider:
        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=self.clock.now,
            sleep=self.clock.sleep,
            ssh_run=lambda *args: (_ for _ in ()).throw(AssertionError("ssh")),
            runtime=Path(self.tmp.name),
        )
        provider.machine_id = "26359"
        provider.execution_id = "live9-run-a"
        provider._resource_id = "7"
        provider.execution_budget = SimpleNamespace(spending_deadline_wall=1_000.0 + 18_107)
        provider._health_get = lambda url: (200, b'{"state":"worker_ready","runtime":true}')
        return provider

    def _beat(self, provider: VastImageProvider) -> dict:
        return json.loads((Path(self.tmp.name) / "watchdog.heartbeat").read_text(encoding="utf-8"))

    def test_safety_limits_are_unchanged(self) -> None:
        self.assertEqual(HEARTBEAT_STALE_SECONDS, 300.0)
        self.assertEqual(HEARTBEAT_INTERVAL_SECONDS, 20.0)
        self.assertLess(HEARTBEAT_INTERVAL_SECONDS * 2, HEARTBEAT_STALE_SECONDS)
        self.assertEqual(INSTANCE_STARTUP_SECONDS, 900.0)
        self.assertEqual(PHASE_LIMITS["worker_ready"], (420.0, 180.0))
        self.assertEqual(LIVE7_RESERVE_SECONDS, 840)
        self.assertEqual(LIVE7_MAX_HOURLY_USD, Decimal("0.95"))
        self.assertEqual(LIVE8_JIT_CEILING_USD, Decimal("5.00"))

    def test_loading_past_300_seconds_stays_under_the_startup_deadline(self) -> None:
        def transport(method, url, body, headers):
            status = "loading" if self.clock.t < 1_350 else "running"
            return 200, json.dumps({"instances": [_row(status)]}).encode()

        provider = self._provider(transport)
        provider.boot("7")
        self.assertGreater(self.clock.t - 1_000, HEARTBEAT_STALE_SECONDS)
        self.assertLess(self.clock.t - 1_000, INSTANCE_STARTUP_SECONDS)
        beat = self._beat(provider)
        self.assertEqual(beat["execution_id"], "live9-run-a")
        self.assertEqual(beat["instance_id"], "7")
        self.assertLessEqual(self.clock.t - beat["at"], HEARTBEAT_INTERVAL_SECONDS)
        self.assertNotIn("fixture-key", Path(self.tmp.name, "watchdog.heartbeat").read_text(encoding="utf-8"))
        self.assertEqual(provider.execution_budget.spending_deadline_wall, 1_000.0 + 18_107)
        self.assertEqual(
            instances_to_stop(
                [_row("running")],
                machine_id=26359,
                now=self.clock.t,
                deadline=provider.execution_budget.spending_deadline_wall,
                heartbeat_age=self.clock.t - beat["at"],
            ),
            [],
        )

    def test_a_dead_supervisor_stops_and_the_watchdog_deletes(self) -> None:
        provider = self._provider(lambda *args: (200, b'{"instances":[]}'))
        before = threading.active_count()
        provider.note_supervision("7")
        self.assertEqual(threading.active_count(), before)
        stamped = self._beat(provider)["at"]
        self.clock.t += HEARTBEAT_STALE_SECONDS + 1
        provider.note_supervision("7")
        self.assertEqual(self._beat(provider)["at"], self.clock.t)

        def stopped() -> None:
            provider.note_supervision("7")

        worker = threading.Thread(target=stopped)
        worker.start()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        frozen = self._beat(provider)["at"]
        self.clock.t += 50
        self.assertEqual(self._beat(provider)["at"], frozen)
        self.assertEqual(provider.heartbeat_writes, 2)

        deleted: list[str] = []
        beat = Path(self.tmp.name) / "watchdog.heartbeat"
        os.utime(beat, (1_000.0, 1_000.0))
        ticks = iter((1_000.0, 1_000.0 + HEARTBEAT_STALE_SECONDS + 5))
        outcome = run_until_stopped(
            self._spec(10_000.0),
            fetch=lambda: [_row("loading")],
            delete=lambda instance_id: deleted.append(instance_id) or 200,
            now=lambda: next(ticks),
            sleep=lambda _seconds: None,
        )
        self.assertEqual(deleted, ["7"])
        self.assertEqual(outcome["deadline"], 10_000.0)
        self.assertEqual(stamped, 1_000.0)

    def test_a_hung_controller_does_not_keep_the_heartbeat_fresh(self) -> None:
        provider = self._provider(lambda *args: (200, b'{"instances":[]}'))
        provider.note_supervision("7")
        stamped = self._beat(provider)["at"]
        self.clock.t += HEARTBEAT_STALE_SECONDS + 50
        self.assertEqual(self._beat(provider)["at"], stamped)
        self.assertEqual(
            instances_to_stop(
                [_row("loading")],
                machine_id=26359,
                now=self.clock.t,
                deadline=10_000.0,
                heartbeat_age=self.clock.t - stamped,
            ),
            ["7"],
        )

    def test_a_failed_write_leaves_watchdog_protection_in_place(self) -> None:
        blocked = Path(self.tmp.name) / "not-a-directory"
        blocked.write_text("x", encoding="utf-8")
        provider = self._provider(lambda *args: (200, b'{"instances":[]}'))
        provider.runtime = blocked
        provider.note_supervision("7")
        self.assertEqual(provider.heartbeat_writes, 0)
        self.assertTrue(provider.heartbeat_error)
        self.assertNotIn("fixture-key", provider.heartbeat_error)
        self.assertIn("heartbeat_write_failed", provider.launch_trace)
        self.assertEqual(
            instances_to_stop(
                [_row("loading")],
                machine_id=26359,
                now=1_000.0 + HEARTBEAT_STALE_SECONDS + 1,
                deadline=10_000.0,
                heartbeat_age=HEARTBEAT_STALE_SECONDS + 1,
            ),
            ["7"],
        )

    def test_one_execution_cannot_refresh_another(self) -> None:
        provider = self._provider(lambda *args: (200, b'{"instances":[]}'))
        provider.note_supervision("7")
        other = self._provider(lambda *args: (200, b'{"instances":[]}'))
        other.execution_id = "live9-run-b"
        other.note_supervision("7")
        self.assertEqual(other.heartbeat_error, "HEARTBEAT_OWNERSHIP")
        self.assertEqual(self._beat(provider)["execution_id"], "live9-run-a")
        same = self._provider(lambda *args: (200, b'{"instances":[]}'))
        same._resource_id = "8"
        same.note_supervision("8")
        self.assertEqual(same.heartbeat_error, "HEARTBEAT_OWNERSHIP")
        self.assertEqual(self._beat(provider)["instance_id"], "7")
        self.clock.t += HEARTBEAT_INTERVAL_SECONDS
        provider.note_supervision("7")
        self.clock.t += HEARTBEAT_INTERVAL_SECONDS
        provider.note_supervision("7")
        self.assertEqual(provider.heartbeat_writes, 3)
        self.assertEqual(self._beat(provider)["execution_id"], "live9-run-a")

    def test_a_fresh_heartbeat_does_not_override_the_spending_deadline(self) -> None:
        beat = Path(self.tmp.name) / "watchdog.heartbeat"
        beat.write_text("{}\n", encoding="utf-8")
        os.utime(beat, (9_000.0, 9_000.0))
        deleted: list[str] = []
        outcome = run_until_stopped(
            self._spec(5_000.0),
            fetch=lambda: [_row("running")],
            delete=lambda instance_id: deleted.append(instance_id) or 200,
            now=lambda: 5_000.0,
            sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("sleep")),
        )
        self.assertEqual(deleted, ["7"])
        self.assertEqual(outcome["deadline"], 5_000.0)
        self.assertEqual(
            instances_to_stop(
                [_row("running")],
                machine_id=26359,
                now=5_000.0,
                deadline=5_000.0,
                heartbeat_age=0,
            ),
            ["7"],
        )

    def test_startup_still_fails_at_900_seconds(self) -> None:
        def transport(method, url, body, headers):
            return 200, json.dumps({"instances": [_row("loading")]}).encode()

        provider = self._provider(transport)
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "STARTUP_DEADLINE")
        elapsed = self.clock.t - 1_000
        self.assertGreater(elapsed, INSTANCE_STARTUP_SECONDS)
        self.assertLess(elapsed, INSTANCE_STARTUP_SECONDS + 10)
        self.assertLessEqual(self.clock.t - self._beat(provider)["at"], HEARTBEAT_INTERVAL_SECONDS)
        writes = provider.heartbeat_writes
        provider._mark_gone("7")
        self.clock.t += HEARTBEAT_INTERVAL_SECONDS
        provider.note_supervision("7")
        self.assertEqual(provider.heartbeat_writes, writes)

    def test_worker_readiness_still_fails_at_420_seconds(self) -> None:
        def transport(method, url, body, headers):
            return 200, json.dumps({"instances": [_row("running")]}).encode()

        provider = self._provider(transport)
        provider._health_get = lambda url: (200, b'{"state":"container_starting","runtime":false}')
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "WORKER_READY_TIMEOUT")
        elapsed = self.clock.t - 1_000
        self.assertGreater(elapsed, PHASE_LIMITS["worker_ready"][0])
        self.assertLess(elapsed, PHASE_LIMITS["worker_ready"][0] + 10)
        self.assertLessEqual(self.clock.t - self._beat(provider)["at"], HEARTBEAT_INTERVAL_SECONDS)

    def test_a_provider_list_failure_does_not_keep_beating(self) -> None:
        def transport(method, url, body, headers):
            raise WorkerStageError("LIST_SCHEMA")

        provider = self._provider(transport)
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "LIST_SCHEMA")
        self.assertEqual(provider.heartbeat_writes, 1)
        self.assertEqual(self.clock.t, 1_000.0)
        frozen = provider.heartbeat_writes
        self.clock.t += 5_000
        self.assertEqual(provider.heartbeat_writes, frozen)

    def _spec(self, deadline: float) -> dict:
        root = Path(self.tmp.name)
        (root / "watchdog.instance").write_text("7", encoding="utf-8")
        return {
            "machine_id": 26359,
            "safe_seconds": 18_107,
            "heartbeat": str(root / "watchdog.heartbeat"),
            "stop": str(root / "watchdog.stop"),
            "instance_file": str(root / "watchdog.instance"),
            "spending_deadline_wall": deadline,
            "poll_seconds": 15,
        }


if __name__ == "__main__":
    unittest.main()
