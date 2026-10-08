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
    LOCAL_HEALTH_COMMAND,
    PORT_MAP_GRACE_SECONDS,
    choose_offer,
    classify_connect_failure,
    classify_readiness,
    cold_stage_allowed,
    parse_health_output,
    pinned_worker_reference,
    plan_cold_stage,
    worker_http_endpoint,
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


def _restore_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _which(name: str) -> str:
    import shutil
    return shutil.which(name) or ""


def _stop_pid(pid: int) -> None:
    import time
    if not _alive(pid):
        return
    os.kill(pid, 15)
    for _ in range(20):
        if not _alive(pid):
            return
        time.sleep(0.05)
    if _alive(pid):
        os.kill(pid, 9)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_health(port: int, seconds: float) -> dict:
    import time
    import urllib.request
    deadline = time.time() + seconds
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%s/health" % port, timeout=1) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            last = exc
        time.sleep(0.05)
    raise AssertionError(last)


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
                "worker_ready",
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

        previous = os.environ.get("MEDIA_ENGINE_WORKER_IMAGE")
        os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = INSTANCE_IMAGE

        def restore_image() -> None:
            if previous is None:
                os.environ.pop("MEDIA_ENGINE_WORKER_IMAGE", None)
            else:
                os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = previous

        self.addCleanup(restore_image)
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
        self.assertFalse(template["ssh_required_for_bootstrap"])
        self.assertIn("worker_ready.py", template["auto_start"])
        self.assertNotIn(":latest", template["image"])
        previous = os.environ.pop("MEDIA_ENGINE_WORKER_IMAGE", None)

        def restore() -> None:
            if previous is None:
                os.environ.pop("MEDIA_ENGINE_WORKER_IMAGE", None)
            else:
                os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = previous

        self.addCleanup(restore)
        pinned = pinned_worker_reference()
        if pinned:
            self.assertEqual(worker_image(), pinned)
            self.assertIn("@sha256:", pinned)
        else:
            self.assertEqual(worker_image(), INSTANCE_IMAGE)
        os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = "example.invalid/ai-media-engine-image-worker:runtime"
        self.assertEqual(worker_image(), "example.invalid/ai-media-engine-image-worker:runtime")

    def test_phase_failures_are_cleaned_up(self) -> None:
        for code in (
            "CONNECT_TIMEOUT", "CACHE_STALL", "CACHE_TIMEOUT", "RUNTIME_TIMEOUT",
            "WORKER_READY_TIMEOUT", "PORT_UNPUBLISHED", "STARTUP_FAILED",
        ):
            self.assertIn(code, _DESTROY_ON)


PIN = "ghcr.io/mehmetsentc/ai-media-engine-worker@sha256:" + ("a" * 64)
_SECRET_MARKERS = ("R2_ACCESS_KEY", "R2_SECRET", "VAST_API_KEY", "AI_MEDIA_ENGINE_API_KEY", ".env.r2", "COPY .env")


class PortableWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = os.environ.get("MEDIA_ENGINE_WORKER_IMAGE")
        os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = PIN

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("MEDIA_ENGINE_WORKER_IMAGE", None)
        else:
            os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = self.previous

    def test_readiness_uses_the_published_host_port(self) -> None:
        mapped = {
            "actual_status": "running",
            "public_ipaddr": "203.0.113.10",
            "ports": {"8080/tcp": [{"HostPort": "41523"}]},
        }
        self.assertEqual(worker_http_endpoint(mapped), ("203.0.113.10", "41523"))
        ssh_only = {
            "actual_status": "running",
            "public_ipaddr": "203.0.113.10",
            "ports": {"22/tcp": [{"HostPort": "34567"}]},
        }
        self.assertIsNone(worker_http_endpoint(ssh_only))
        self.assertIsNone(worker_http_endpoint({"actual_status": "running", "public_ipaddr": "203.0.113.10"}))
        self.assertEqual(classify_readiness("running", False, None), "instance_running")

    def test_health_answers_before_any_model_download(self) -> None:
        import importlib.util
        import threading
        import urllib.request
        root = Path(__file__).resolve().parents[1]
        path = root / "deploy/worker/worker_ready.py"
        text = path.read_text(encoding="utf-8")
        for marker in ("snapshot_download", "hf_hub_download", "from_pretrained", "R2_", "VAST_API_KEY", "import torch", "import diffusers"):
            self.assertNotIn(marker, text)
        spec = importlib.util.spec_from_file_location("worker_ready_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        previous_port = os.environ.pop("WORKER_PORT", None)
        self.addCleanup(lambda: os.environ.__setitem__("WORKER_PORT", previous_port) if previous_port is not None else None)
        self.assertEqual(module.listen_address(), ("0.0.0.0", 8080))
        server = module.serve(("127.0.0.1", 0))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop() -> None:
            server.shutdown()
            server.server_close()

        self.addCleanup(stop)
        host, port = server.server_address
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            content_type = response.headers.get("Content-Type")
        self.assertEqual(content_type, "application/json")
        self.assertEqual(payload["state"], "container_starting")
        self.assertIs(payload["runtime"], False)
        self.assertIs(payload["model_loaded"], False)

    def test_worker_ready_does_not_claim_the_model_is_loaded(self) -> None:
        import importlib.util
        import tempfile
        root = Path(__file__).resolve().parents[1]
        path = root / "deploy/worker/worker_ready.py"
        spec = importlib.util.spec_from_file_location("worker_ready_marker", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        marker = base / "runtime-ready"
        marker.write_text("ok\n", encoding="utf-8")
        files = []
        for name in ("qwen_remote.py", "model_cache.py", "r2_cache.py"):
            item = base / name
            item.write_text("# generation file\n", encoding="utf-8")
            files.append(str(item))
        previous = {
            "WORKER_RUNTIME_MARKER": os.environ.get("WORKER_RUNTIME_MARKER"),
            "WORKER_GENERATION_FILES": os.environ.get("WORKER_GENERATION_FILES"),
        }
        os.environ["WORKER_RUNTIME_MARKER"] = str(marker)
        os.environ["WORKER_GENERATION_FILES"] = ",".join(files)
        self.addCleanup(lambda: _restore_env(previous))
        status = module.assess_runtime()
        self.assertEqual(status["state"], "worker_ready")
        self.assertIs(status["runtime"], True)
        self.assertIs(status["model_loaded"], False)
        os.environ["WORKER_RUNTIME_MARKER"] = str(base / "absent")
        failed = module.assess_runtime()
        self.assertEqual(failed["state"], "startup_failed")
        self.assertIs(failed["model_loaded"], False)

    def test_startup_failure_is_reported_without_secrets(self) -> None:
        import importlib.util
        root = Path(__file__).resolve().parents[1]
        path = root / "deploy/worker/worker_ready.py"
        spec = importlib.util.spec_from_file_location("worker_ready_failure", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        status = module.failure_status(RuntimeError("token=supersecret bearer: abc"))
        encoded = json.dumps(status)
        self.assertEqual(status["state"], "startup_failed")
        self.assertEqual(status["error"], "RuntimeError")
        self.assertNotIn("supersecret", encoded)
        self.assertIn("[redacted]", encoded)

    def test_detached_process_answers_before_imports_finish(self) -> None:
        import socket
        import subprocess
        import sys
        import tempfile
        import time
        root = Path(__file__).resolve().parents[1]
        script = root / "deploy/worker/worker_ready.py"
        holder = socket.socket()
        holder.bind(("127.0.0.1", 0))
        port = holder.getsockname()[1]
        holder.close()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Path(tmp.name) / "worker-ready.log"
        pidfile = Path(tmp.name) / "worker-ready.pid"
        env = os.environ.copy()
        env.update({
            "WORKER_PORT": str(port),
            "WORKER_LOG": str(log),
            "WORKER_PID_FILE": str(pidfile),
            "WORKER_SCRIPT": str(script),
            "WORKER_PYTHON": sys.executable,
            "VAST_API_KEY": "do-not-log-this-secret",
        })
        launcher = root / "deploy/worker/start_worker.sh"
        if _which("setsid"):
            completed = subprocess.run(["bash", str(launcher)], env=env, timeout=15, check=False)
            self.assertEqual(completed.returncode, 0)
            pid = int(pidfile.read_text(encoding="utf-8").strip())
        else:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                env=env,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pid = proc.pid
        self.addCleanup(lambda: _stop_pid(pid))
        payload = _wait_health(port, 10)
        self.assertIn(payload["state"], {"container_starting", "startup_failed", "worker_ready"})
        self.assertIs(payload["model_loaded"], False)
        self.assertTrue(_alive(pid))
        terminal = payload
        deadline = time.time() + 20
        while time.time() < deadline and terminal["state"] == "container_starting":
            time.sleep(0.1)
            terminal = _wait_health(port, 2)
        self.assertIn(terminal["state"], {"startup_failed", "worker_ready"})
        text = log.read_text(encoding="utf-8")
        self.assertIn("pid=", text)
        self.assertIn("event=start", text)
        self.assertNotIn("do-not-log-this-secret", text)
        if terminal["state"] == "startup_failed":
            self.assertTrue(terminal.get("error"))
            self.assertNotIn("do-not-log-this-secret", json.dumps(terminal))

    def test_running_instance_is_not_worker_ready(self) -> None:
        self.assertEqual(classify_readiness("running", False, None), "instance_running")
        self.assertEqual(
            classify_readiness("running", True, {"state": "container_starting", "runtime": False}),
            "container_starting",
        )
        self.assertEqual(
            classify_readiness("running", True, {"state": "worker_ready", "runtime": True}),
            "worker_ready",
        )
        self.assertNotEqual(classify_readiness("running", False, None), "worker_ready")
        self.assertEqual(
            classify_readiness("running", True, {"state": "startup_failed", "runtime": False, "error": "OSError"}),
            "startup_failed",
        )

    def test_create_starts_the_worker_without_ssh_bootstrap(self) -> None:
        sent: list[dict] = []

        def transport(method, url, body, headers):
            sent.append(json.loads(body.decode("utf-8")))
            return 200, b'{"success": true, "new_contract": 5}'

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=Clock().now,
            sleep=lambda _s: None,
            ssh_run=lambda *args: (_ for _ in ()).throw(AssertionError("ssh")),
        )
        provider.offer_id = "9"
        provider.disk_gb = 40
        provider.cache_source = "warm_cache"
        provider.machine_id = str(MACHINE_ID)
        provider.create("nahaber")
        body = sent[0]
        self.assertEqual(body["image"], PIN)
        self.assertEqual(body["runtype"], "ssh")
        self.assertEqual(body["onstart"], "bash /workspace/start_worker.sh")
        self.assertNotIn("pip", body["onstart"])
        self.assertNotIn("args_str", body)
        launcher = Path(__file__).resolve().parents[1] / "deploy/worker/start_worker.sh"
        script = launcher.read_text(encoding="utf-8")
        self.assertIn("setsid", script)
        self.assertIn('"$python" "$script"', script)
        self.assertIn("/workspace/worker_ready.py", script)
        self.assertIn("/workspace/worker-ready.log", script)
        self.assertNotIn("pip", script)
        self.assertEqual(body["env"]["WORKER_PORT"], "8080")
        self.assertEqual(body["env"]["-p 8080:8080"], "1")
        self.assertEqual(body["env"]["HF_HUB_OFFLINE"], "1")
        encoded = json.dumps(body)
        for marker in _SECRET_MARKERS:
            self.assertNotIn(marker, encoded)

    def test_readiness_waits_for_the_health_signal(self) -> None:
        clock = Clock()
        ssh_calls = {"n": 0}

        def ssh(*args):
            ssh_calls["n"] += 1
            return 0, "", ""

        def transport(method, url, body, headers):
            if clock.t < 1_015:
                row = {"id": 7, "actual_status": "running", "machine_id": 26359, "public_ipaddr": "203.0.113.10"}
            else:
                row = {
                    "id": 7,
                    "actual_status": "running",
                    "machine_id": 26359,
                    "public_ipaddr": "203.0.113.10",
                    "ports": {"8080/tcp": [{"HostPort": "18080"}]},
                }
            return 200, json.dumps({"instances": [row]}).encode()

        seen_urls: list[str] = []

        def health(url):
            seen_urls.append(url)
            if clock.t < 1_030:
                return 200, b'{"state":"container_starting","runtime":false}'
            return 200, b'{"state":"worker_ready","runtime":true}'

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=ssh,
        )
        provider.machine_id = "26359"
        provider._health_get = health
        provider.boot("7")
        self.assertEqual(provider.worker_state, "worker_ready")
        self.assertTrue(provider.runtime_ready)
        self.assertEqual(ssh_calls["n"], 0)
        self.assertIn("http://203.0.113.10:18080/health", seen_urls)
        self.assertFalse(any(url.endswith(":8080/health") for url in seen_urls))
        provider.prepare_runtime("7")
        self.assertEqual(ssh_calls["n"], 0)

    def test_readiness_timeout_does_not_use_ssh(self) -> None:
        clock = Clock()

        def transport(method, url, body, headers):
            row = {"id": 7, "actual_status": "running", "machine_id": 26359, "public_ipaddr": "203.0.113.10"}
            return 200, json.dumps({"instances": [row]}).encode()

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=lambda *args: (_ for _ in ()).throw(AssertionError("ssh")),
        )
        provider.machine_id = "26359"
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "PORT_UNPUBLISHED")
        self.assertEqual(provider.worker_state, "instance_running")
        elapsed = clock.t - 1_000
        self.assertGreaterEqual(elapsed, PORT_MAP_GRACE_SECONDS)
        self.assertLess(elapsed, PORT_MAP_GRACE_SECONDS + 5)
        self.assertLess(elapsed, PHASE_LIMITS["worker_ready"][1])

    def test_ssh_local_health_does_not_use_the_public_port(self) -> None:
        clock = Clock()
        commands: list[tuple] = []
        public: list[str] = []

        def transport(method, url, body, headers):
            row = {
                "id": 7,
                "actual_status": "running",
                "machine_id": 26359,
                "public_ipaddr": "203.0.113.10",
                "ssh_host": "ssh4.example",
                "ssh_port": 11982,
            }
            return 200, json.dumps({"instances": [row]}).encode()

        def ssh(host, port, command, timeout):
            commands.append((host, port, command, timeout))
            if clock.t < 1_020:
                return 1, "", "Connection refused"
            return 0, '{"state":"worker_ready","runtime":true,"model_loaded":false}\n', ""

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=ssh,
        )
        provider.machine_id = "26359"
        provider._health_get = lambda url: public.append(url) or (200, b"{}")
        provider.boot("7")
        self.assertEqual(provider.health_transport, "ssh")
        self.assertEqual(provider.worker_state, "worker_ready")
        self.assertEqual(public, [])
        self.assertIn("http://127.0.0.1:8080/health", commands[-1][2])
        self.assertNotIn("203.0.113.10", commands[-1][2])
        self.assertEqual(commands[-1][0], "ssh4.example")
        self.assertEqual(LOCAL_HEALTH_COMMAND, commands[-1][2])

    def test_public_endpoint_uses_the_mapped_host_port(self) -> None:
        endpoint = worker_http_endpoint({
            "actual_status": "running",
            "public_ipaddr": "203.0.113.10",
            "ports": {"8080/tcp": [{"HostPort": "41044"}]},
        })
        self.assertEqual(endpoint, ("203.0.113.10", "41044"))
        self.assertIsNone(worker_http_endpoint({
            "actual_status": "running",
            "public_ipaddr": "203.0.113.10",
        }))

    def test_connect_failures_distinguish_timeout_from_refusal(self) -> None:
        import socket
        import urllib.error
        self.assertEqual(classify_connect_failure(TimeoutError("timed out")), "timeout")
        self.assertEqual(classify_connect_failure(urllib.error.URLError("timed out")), "timeout")
        self.assertEqual(classify_connect_failure(urllib.error.URLError(TimeoutError("timed out"))), "timeout")
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        closed.close()
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1)
            self.fail("closed port accepted a connection")
        except OSError as exc:
            self.assertEqual(classify_connect_failure(exc), "refused")
        banner = parse_health_output('Welcome\n{"state":"worker_ready","runtime":true}\n')
        self.assertEqual(banner["state"], "worker_ready")
        self.assertIsNone(parse_health_output("Connection refused"))

    def test_failed_boot_does_not_create_a_second_job(self) -> None:
        import tempfile

        from media_engine.orchestrator.controller import ManualClock, MediaController
        from media_engine.providers.base import CreateResult, Quote

        class Once:
            name = "vast"

            def __init__(self) -> None:
                self.creates = 0
                self.terminated: list[str] = []

            def quote(self, owner: str) -> Quote:
                return Quote(hourly_price_usd=Decimal("0.10"), provider="vast")

            def create(self, owner: str) -> CreateResult:
                self.creates += 1
                return CreateResult(outcome="created", resource_id="42")

            def boot(self, resource_id: str) -> None:
                raise WorkerStageError("WORKER_READY_TIMEOUT")

            def terminate(self, resource_id: str) -> None:
                self.terminated.append(resource_id)

            def status(self, resource_id: str) -> str:
                return "READY"

            def begin_busy(self, resource_id: str) -> None:
                return None

            def end_busy(self, resource_id: str) -> None:
                return None

            def observe(self, resource_id: str, owner: str) -> str:
                return "gone"

            def list_owned(self) -> list[dict]:
                return []

        provider = Once()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        controller = MediaController(
            str(root / "studio.sqlite3"),
            str(root / "artifacts"),
            clock=ManualClock(),
            provider=provider,
        )
        controller.submit_image("a newsroom", seed=20260804)
        controller.process_available()
        controller.process_available()
        self.assertEqual(provider.creates, 1)
        self.assertEqual(provider.terminated, ["42"])

    def test_loading_instance_keeps_the_port_grace(self) -> None:
        clock = Clock()

        def transport(method, url, body, headers):
            if clock.t < 1_060:
                row = {"id": 7, "actual_status": "loading", "machine_id": 26359}
            else:
                row = {
                    "id": 7,
                    "actual_status": "running",
                    "machine_id": 26359,
                    "public_ipaddr": "203.0.113.10",
                    "ports": {"8080/tcp": [{"HostPort": "18080"}]},
                }
            return 200, json.dumps({"instances": [row]}).encode()

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=lambda *args: (_ for _ in ()).throw(AssertionError("ssh")),
        )
        provider.machine_id = "26359"
        provider._health_get = lambda url: (200, b'{"state":"worker_ready","runtime":true}')
        provider.boot("7")
        self.assertEqual(provider.worker_state, "worker_ready")
        self.assertGreaterEqual(clock.t - 1_000, 60)
        self.assertLess(clock.t - 1_000, PHASE_LIMITS["worker_ready"][1])

    def test_unreachable_health_stalls_before_the_maximum(self) -> None:
        clock = Clock()

        def transport(method, url, body, headers):
            row = {
                "id": 7,
                "actual_status": "running",
                "machine_id": 26359,
                "public_ipaddr": "203.0.113.10",
                "ports": {"8080/tcp": [{"HostPort": "18080"}]},
            }
            return 200, json.dumps({"instances": [row]}).encode()

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=lambda *args: (_ for _ in ()).throw(AssertionError("ssh")),
        )
        provider.machine_id = "26359"
        provider._health_get = lambda url: (_ for _ in ()).throw(TimeoutError("timed out"))
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "WORKER_READY_TIMEOUT")
        elapsed = clock.t - 1_000
        self.assertGreater(elapsed, PHASE_LIMITS["worker_ready"][1])
        self.assertLess(elapsed, PHASE_LIMITS["worker_ready"][1] + 10)
        self.assertLess(elapsed, PHASE_LIMITS["worker_ready"][0])

    def test_live_container_starting_keeps_the_readiness_window(self) -> None:
        clock = Clock()

        def transport(method, url, body, headers):
            row = {
                "id": 7,
                "actual_status": "running",
                "machine_id": 26359,
                "public_ipaddr": "203.0.113.10",
                "ports": {"8080/tcp": [{"HostPort": "18080"}]},
            }
            return 200, json.dumps({"instances": [row]}).encode()

        def health(url):
            if clock.t < 1_220:
                return 200, b'{"state":"container_starting","runtime":false}'
            return 200, b'{"state":"worker_ready","runtime":true}'

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=lambda *args: (_ for _ in ()).throw(AssertionError("ssh")),
        )
        provider.machine_id = "26359"
        provider._health_get = health
        provider.boot("7")
        self.assertEqual(provider.worker_state, "worker_ready")
        self.assertGreaterEqual(clock.t - 1_000, 220)
        self.assertLess(clock.t - 1_000, PHASE_LIMITS["worker_ready"][0])

    def test_startup_failed_stops_without_waiting_out_the_stall(self) -> None:
        clock = Clock()

        def transport(method, url, body, headers):
            row = {
                "id": 7,
                "actual_status": "running",
                "machine_id": 26359,
                "public_ipaddr": "203.0.113.10",
                "ports": {"8080/tcp": [{"HostPort": "18080"}]},
            }
            return 200, json.dumps({"instances": [row]}).encode()

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=clock.now,
            sleep=clock.sleep,
            ssh_run=lambda *args: (_ for _ in ()).throw(AssertionError("ssh")),
        )
        provider.machine_id = "26359"
        provider._health_get = lambda url: (200, b'{"state":"startup_failed","runtime":false,"error":"OSError"}')
        with self.assertRaises(WorkerStageError) as caught:
            provider.boot("7")
        self.assertEqual(caught.exception.code, "STARTUP_FAILED")
        self.assertLess(clock.t - 1_000, PHASE_LIMITS["worker_ready"][1])

    def test_ready_timeout_destroys_the_gpu(self) -> None:
        for code in ("WORKER_READY_TIMEOUT", "PORT_UNPUBLISHED", "STARTUP_FAILED"):
            self._assert_boot_failure_destroys(code)

    def _assert_boot_failure_destroys(self, code: str) -> None:
        import tempfile

        from media_engine.orchestrator.controller import ManualClock, MediaController
        from media_engine.providers.base import CreateResult, Quote

        class TimeoutGPU:
            name = "vast"

            def __init__(self, failure: str) -> None:
                self.failure = failure
                self.terminated: list[str] = []

            def quote(self, owner: str) -> Quote:
                return Quote(hourly_price_usd=Decimal("0.10"), provider="vast")

            def create(self, owner: str) -> CreateResult:
                return CreateResult(outcome="created", resource_id="42")

            def boot(self, resource_id: str) -> None:
                raise WorkerStageError(self.failure)

            def terminate(self, resource_id: str) -> None:
                self.terminated.append(resource_id)

            def status(self, resource_id: str) -> str:
                return "READY"

            def begin_busy(self, resource_id: str) -> None:
                return None

            def end_busy(self, resource_id: str) -> None:
                return None

            def observe(self, resource_id: str, owner: str) -> str:
                return "gone"

            def list_owned(self) -> list[dict]:
                return []

        provider = TimeoutGPU(code)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        controller = MediaController(
            str(root / "studio.sqlite3"),
            str(root / "artifacts"),
            clock=ManualClock(),
            provider=provider,
        )
        job = controller.submit_image("a newsroom", seed=20260804)
        controller.process_available()
        self.assertEqual(provider.terminated, ["42"])
        saved = controller.get_job(job.id)
        self.assertEqual(saved.public_status, "failed")
        self.assertEqual(saved.error_code, code)
        self.assertFalse(saved.artifact_id)

    def test_image_workflow_and_runtime_contain_no_secrets(self) -> None:
        root = Path(__file__).resolve().parents[1]
        names = [
            root / "deploy/worker/Dockerfile",
            root / "deploy/worker/worker_ready.py",
            root / "deploy/worker/start_worker.sh",
            root / "deploy/worker/vast_template.json",
            root / ".github/workflows/worker-image.yml",
        ]
        for path in names:
            text = path.read_text(encoding="utf-8")
            for marker in _SECRET_MARKERS:
                self.assertNotIn(marker, text, path.name)
        workflow = (root / ".github/workflows/worker-image.yml").read_text(encoding="utf-8")
        self.assertIn("contents: read", workflow)
        self.assertIn("packages: write", workflow)
        self.assertIn("secrets.GITHUB_TOKEN", workflow)
        self.assertNotIn(":latest", workflow)
        self.assertIn("sha-${{ github.sha }}", workflow)
        docker = (root / "deploy/worker/Dockerfile").read_text(encoding="utf-8")
        self.assertIn('CMD ["python3", "/workspace/worker_ready.py"]', docker)
        self.assertIn("COPY deploy/worker/start_worker.sh /workspace/start_worker.sh", docker)
        self.assertNotIn("COPY . ", docker)
