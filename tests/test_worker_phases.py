"""Phase clocks, cold-stage cost, and the runtime-ready worker. No GPU."""
from __future__ import annotations

import json
import os
import unittest
from decimal import Decimal
from pathlib import Path

from media_engine.orchestrator.controller import _DESTROY_ON
from media_engine.providers.base import WorkerStageError
from media_engine.providers.phase2d import INSTANCE_IMAGE
from media_engine.providers.phase_clock import PHASE_LIMITS, PhaseWatch, note_transfer_text
from media_engine.providers.vast_image import (
    MACHINE_ID,
    VOLUME_ID,
    _generate_command,
    choose_offer,
    cold_stage_allowed,
    plan_cold_stage,
    worker_image,
    VastImageProvider,
)


class Clock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def _offer(offer_id: int, machine: int, *, inet_down: float, dph_base: str) -> dict:
    return {
        "id": offer_id,
        "gpu_name": "A40",
        "gpu_ram": 49140,
        "machine_id": machine,
        "rentable": True,
        "dph_base": dph_base,
        "dph_total": dph_base,
        "storage_cost": "0.20",
        "inet_down": inet_down,
        "disk_space": 400,
        "reliability2": 0.99,
    }


class PhaseClockTests(unittest.TestCase):
    def test_each_phase_has_its_own_limit(self) -> None:
        self.assertEqual(
            set(PHASE_LIMITS),
            {
                "provisioning",
                "connecting",
                "cache_staging",
                "runtime_preparing",
                "model_loading",
                "generating",
                "saving",
            },
        )
        connecting = PhaseWatch.start("connecting", 0)
        staging = PhaseWatch.start("cache_staging", 0)
        self.assertLess(connecting.max_seconds, 900)
        self.assertGreater(staging.max_seconds, connecting.max_seconds)

    def test_moving_bytes_keep_a_long_transfer_alive(self) -> None:
        watch = PhaseWatch.start("cache_staging", 0)
        now = 0.0
        sent = 0
        while now < watch.stall_seconds + 30:
            now += 30
            sent += 1_000_000_000
            self.assertTrue(note_transfer_text(watch, json.dumps({"bytes_done": sent}), now))
            self.assertIsNone(watch.expired(now))
        self.assertGreater(watch.bytes_transferred, 0)
        self.assertLess(now, watch.max_seconds)

    def test_a_stalled_transfer_expires(self) -> None:
        watch = PhaseWatch.start("cache_staging", 0)
        note_transfer_text(watch, '{"bytes_done": 100}', 0)
        self.assertIsNone(watch.expired(watch.stall_seconds))
        self.assertEqual(watch.expired(watch.stall_seconds + 1), "PHASE_STALL")
        self.assertEqual(note_transfer_text(watch, '{"bytes_done": 100}', 10), False)

    def test_connect_timeout_is_not_the_cache_budget(self) -> None:
        clock = Clock()
        calls = {"ssh": 0}

        class Connector:
            def public_key(self) -> str:
                return "ssh-ed25519 AAAAC3Rlc3Q="

            def run(self, host: str, port: str, command: str, timeout: int):
                calls["ssh"] += 1
                return 1, "", "connection refused"

        def transport(method, url, body, headers):
            if method == "GET":
                row = {
                    "id": 7,
                    "actual_status": "running",
                    "ssh_host": "203.0.113.10",
                    "ssh_port": 2222,
                    "machine_id": 26359,
                }
                return 200, json.dumps({"instances": [row]}).encode()
            if method == "POST" and str(url).endswith("/ssh/"):
                return 200, b'{"success": true}'
            return 200, b"{}"

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=clock.now,
            sleep=clock.sleep,
        )
        provider.machine_id = "26359"
        provider._connect_worker = Connector()
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "CONNECT_TIMEOUT")
        self.assertGreater(calls["ssh"], 0)
        self.assertLess(clock.t - 1_000, PHASE_LIMITS["connecting"][0] + 20)
        self.assertLess(clock.t - 1_000, 900)

    def test_stalled_cache_poll_stops_the_copy(self) -> None:
        clock = Clock()
        killed = []

        def ssh(host, port, command, timeout):
            if "cache-sync.pid" in command and "kill" in command:
                killed.append(command)
                return 0, "", ""
            if "echo STARTED" in command:
                return 0, "STARTED\n", ""
            if "r2-sync-progress" in command:
                return 0, '{"bytes_done": 4096, "bytes_total": 57704594653, "MB_per_sec": 0.1}\n', ""
            if "cache-sync.exit" in command:
                return 0, "RUNNING\n", ""
            return 0, "", ""

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=lambda *args: (200, b"{}"),
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=ssh,
        )
        provider._host = "203.0.113.10"
        provider._port = "22"
        with self.assertRaises(WorkerStageError) as caught:
            provider._watch_cache_sync()
        self.assertEqual(caught.exception.code, "CACHE_STALL")
        self.assertEqual(len(killed), 1)
        self.assertLess(clock.t - 1_000, PHASE_LIMITS["cache_staging"][0])

    def test_progressing_cache_poll_runs_past_the_stall_window(self) -> None:
        clock = Clock()
        sent = {"n": 0}

        def ssh(host, port, command, timeout):
            if "echo STARTED" in command:
                return 0, "STARTED\n", ""
            if "r2-sync-progress" in command:
                sent["n"] += 2_000_000_000
                body = json.dumps({"bytes_done": sent["n"], "MB_per_sec": 20})
                return 0, body + "\n", ""
            if "cache-sync.exit" in command:
                if clock.t - 1_000 > PHASE_LIMITS["cache_staging"][1]:
                    return 0, "0\n", ""
                return 0, "RUNNING\n", ""
            return 0, "", ""

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=lambda *args: (200, b"{}"),
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=ssh,
        )
        provider._host = "203.0.113.10"
        provider._port = "22"
        provider._watch_cache_sync()
        self.assertGreater(clock.t - 1_000, PHASE_LIMITS["cache_staging"][1])
        self.assertGreater(sent["n"], 0)

    def test_runtime_ready_image_skips_pip(self) -> None:
        seen: list[str] = []

        def ssh(host, port, command, timeout):
            seen.append(command)
            if "runtime-ready" in command:
                return 0, "RUNTIME_READY\n", ""
            return 0, "", ""

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=lambda *args: (200, b"{}"),
            now=Clock().now,
            sleep=lambda _s: None,
            ssh_run=ssh,
        )
        provider._host = "worker"
        provider._port = "22"
        provider.prepare_runtime("1")
        self.assertTrue(provider.runtime_ready)
        self.assertFalse(any("pip install" in command for command in seen))
        command = _generate_command("a terrace", 1024, 1024, 3, python="python3")
        self.assertIn("python3 -u /workspace/qwen_remote.py generate", command)
        self.assertNotIn("phase2d-venv", command)

    def test_cold_stage_is_off_and_over_budget_copies_are_rejected(self) -> None:
        os.environ.pop("MEDIA_ENGINE_ALLOW_COLD_STAGE", None)
        self.assertFalse(cold_stage_allowed())
        slow = plan_cold_stage(Decimal("0.48"), 20)
        self.assertIsNotNone(slow)
        assert slow is not None
        self.assertIn("hourly", slow)
        self.assertGreater(float(slow["transfer_seconds"]), 0)
        self.assertGreater(slow["projected_cost"], slow["cost_before_generation"])
        self.assertGreater(slow["projected_cost"], Decimal("0.50"))
        choice = choose_offer(
            [_offer(3, 26359, inet_down=20, dph_base="0.32")],
            volume_machine=MACHINE_ID,
            volume_ready=True,
            max_hourly=Decimal("0.60"),
            operation_budget=Decimal("0.50"),
            allow_cold_stage=True,
        )
        self.assertIsNone(choice)
        blocked = choose_offer(
            [_offer(21, 151326, inet_down=8000, dph_base="0.20")],
            volume_machine=MACHINE_ID,
            volume_ready=True,
            max_hourly=Decimal("0.60"),
            operation_budget=Decimal("1.00"),
        )
        self.assertIsNone(blocked)

    def test_warm_volume_still_wins_and_template_has_no_weights(self) -> None:
        choice = choose_offer(
            [
                _offer(21, 151326, inet_down=8000, dph_base="0.20"),
                _offer(8, MACHINE_ID, inet_down=100, dph_base="0.40"),
            ],
            volume_machine=MACHINE_ID,
            volume_ready=True,
            max_hourly=Decimal("0.60"),
            operation_budget=Decimal("1.00"),
            allow_cold_stage=True,
        )
        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice.source, "warm_cache")
        self.assertEqual(choice.machine_id, MACHINE_ID)
        root = Path(__file__).resolve().parents[1]
        docker = (root / "deploy/worker/Dockerfile").read_text(encoding="utf-8")
        template = json.loads((root / "deploy/worker/vast_template.json").read_text(encoding="utf-8"))
        self.assertIn("diffusers==0.41.0", docker)
        self.assertIn("/opt/ai-media-engine/runtime-ready", docker)
        self.assertNotIn("hf-cache", docker.split("ENV", 1)[0])
        self.assertFalse(template["model_weights_baked"])
        self.assertEqual(template["warm_volume"]["id"], VOLUME_ID)
        self.assertEqual(template["image_env"], "MEDIA_ENGINE_WORKER_IMAGE")
        previous = os.environ.pop("MEDIA_ENGINE_WORKER_IMAGE", None)

        def restore() -> None:
            if previous is None:
                os.environ.pop("MEDIA_ENGINE_WORKER_IMAGE", None)
            else:
                os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = previous

        self.addCleanup(restore)
        self.assertEqual(worker_image(), INSTANCE_IMAGE)
        os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = "example.invalid/ai-media-engine-image-worker:runtime"
        self.assertEqual(worker_image(), "example.invalid/ai-media-engine-image-worker:runtime")

    def test_phase_failures_are_cleaned_up(self) -> None:
        for code in ("CONNECT_TIMEOUT", "CACHE_STALL", "CACHE_TIMEOUT", "RUNTIME_TIMEOUT"):
            self.assertIn(code, _DESTROY_ON)
