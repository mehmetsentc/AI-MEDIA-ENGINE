"""Execution preflight. Local processes and fakes only. This file never rents a GPU."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from media_engine.api.app import dispatch
from media_engine.engines.image.cache_validation import assess_inventory
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.platform.repository import PlatformBlocked
from media_engine.providers.base import WorkerStageError
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.live6_prepare import main as prepare_main
from media_engine.providers.live6_prepare import prepare_provider
from media_engine.providers.vast_image import (
    MACHINE_ID,
    MOUNT_PATH,
    REVISION,
    VOLUME_ID,
    VastImageProvider,
)
from media_engine.providers.vast_watchdog import arm_watchdog, main as watchdog_main
from media_engine.safety.spend_grant import SpendGrant
from media_engine.studio.boot import build_controller
from tests.test_live6_integration import _inventory, _manifest
from tests.test_phase2e import PNG, BoomStorage, PngEngine

ROOT = Path(__file__).resolve().parents[1]


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.kill(pid, 15)
    except OSError:
        return


class _Ledger:
    def reserved_amount(self, job_id: str) -> int:
        return 2


class _Platform:
    """Stand-in for Neon and R2. It never opens a network connection."""

    def __init__(self) -> None:
        self.repo = _Ledger()
        self.bucket = "fixture-bucket"
        self.events: list[str] = []
        self.fail_upload = False
        self.fail_settle = False

    def authorize_paid(self, user_id: str, job_id: str) -> None:
        self.events.append("authorize")

    def persist_media(self, **_kwargs):
        self.events.append("upload")
        if self.fail_upload:
            raise PlatformBlocked("ARTIFACT_UPLOAD_FAILED")
        return SimpleNamespace(sha256="abc", byte_count=len(PNG), object_key="users/nahaber/image.png")

    def settle(self, user_id: str, job_id: str, amount: int) -> None:
        self.events.append("settle")
        if self.fail_settle:
            raise RuntimeError("settlement down")

    def settle_failure(self, user_id: str, job_id: str, legitimate_cost: int = 0) -> None:
        self.events.append("refund")


class Live6PreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls: list[str] = []
        self._saved = {
            name: os.environ.get(name)
            for name in (
                "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD",
                "MEDIA_ENGINE_FORBID_R2_STAGE",
                "MEDIA_ENGINE_WATCHDOG_LAUNCH",
                "MEDIA_ENGINE_WATCHDOG_LOCAL",
                "MEDIA_ENGINE_WATCHDOG_POLL_SECONDS",
                "MEDIA_ENGINE_EXECUTION_ID",
                "MEDIA_ENGINE_EXECUTION_ISSUED_AT",
                "MEDIA_ENGINE_EXECUTION_EXPIRES_AT",
                "MEDIA_ENGINE_RUNTIME",
                "VAST_API_KEY",
                "MEDIA_ENGINE_MACHINE_ID",
                "MEDIA_ENGINE_ALLOW_COLD_STAGE",
            )
        }
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "0.50"
        os.environ["MEDIA_ENGINE_FORBID_R2_STAGE"] = "1"
        os.environ.pop("VAST_API_KEY", None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _provider(self) -> VastImageProvider:
        def transport(method, url, body, headers):
            self.calls.append(method)
            self.put_body = body
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
        provider.offer_id = "30565038"
        provider.cache_source = "warm_cache"
        provider.machine_id = str(MACHINE_ID)
        provider.gpu_model = "RTX A6000"
        provider.hourly_amount = Decimal("0.374")
        provider.disk_gb = 40
        return provider

    def _grant(self, execution_id: str = "exec-fresh") -> SpendGrant:
        return SpendGrant(
            execution_id=execution_id,
            ceiling_usd=Decimal("0.50"),
            max_instances=1,
            issued_at=1_000.0,
            expires_at=1_030.0,
            machine_id=MACHINE_ID,
        )

    def _controller(self, provider, engine=None) -> MediaController:
        slot = getattr(self, "_slot", 0) + 1
        self._slot = slot
        root = Path(self.tmp.name) / f"jobs-{slot}"
        root.mkdir()
        return MediaController(
            str(root / "meta.sqlite3"),
            str(root / "artifacts"),
            clock=ManualClock(),
            provider=provider,
            image_engine=engine or PngEngine(),
        )

    def _run(self, provider, engine=None):
        controller = self._controller(provider, engine)
        status, created = dispatch(
            controller, "POST", "/v1/images/generations",
            json.dumps({"prompt": "preflight"}).encode("utf-8"),
        )
        self.assertEqual(status, 202)
        controller.process_available()
        controller.process_available()
        return controller, created["job_id"]

    def test_launcher_binds_a_fresh_grant_and_refuses_to_rent(self) -> None:
        runtime = Path(self.tmp.name) / "runtime"
        runtime.mkdir()
        (runtime / "grant.json").write_text('{"execution_id": "stolen"}', encoding="utf-8")
        with self.assertRaises(WorkerStageError) as missing:
            prepare_provider({}, runtime=runtime, api_key="fixture-key")
        self.assertEqual(missing.exception.code, "MISSING_AUTHORIZATION")
        issued = time.time()
        environ = {
            "MEDIA_ENGINE_EXECUTION_ID": "exec-fresh",
            "MEDIA_ENGINE_EXECUTION_ISSUED_AT": str(issued),
            "MEDIA_ENGINE_EXECUTION_EXPIRES_AT": str(issued + 300),
            "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD": "0.50",
        }
        partial = dict(environ)
        partial.pop("MEDIA_ENGINE_EXECUTION_EXPIRES_AT")
        with self.assertRaises(WorkerStageError) as incomplete:
            prepare_provider(partial, runtime=runtime, api_key="fixture-key")
        self.assertEqual(incomplete.exception.code, "MISSING_AUTHORIZATION")
        provider = prepare_provider(environ, runtime=runtime, api_key="fixture-key")
        self.assertIsInstance(provider, VastImageProvider)
        self.assertEqual(provider.execution_id, "exec-fresh")
        self.assertAlmostEqual(provider.execution_deadline, issued + 300, places=3)
        self.assertEqual(provider.execution_grant.ceiling_usd, Decimal("0.50"))
        self.assertEqual(provider.execution_grant.max_instances, 1)
        with self.assertRaises(WorkerStageError) as reused:
            prepare_provider(environ, runtime=runtime, api_key="fixture-key")
        self.assertEqual(reused.exception.code, "STALE_AUTHORIZATION")
        with self.assertRaises(WorkerStageError) as second:
            provider.bind_execution(self._grant("exec-other"), execution_id="exec-other")
        self.assertEqual(second.exception.code, "STALE_AUTHORIZATION")
        os.environ["MEDIA_ENGINE_RUNTIME"] = str(runtime)
        os.environ["VAST_API_KEY"] = "fixture-key"
        for name, value in environ.items():
            os.environ[name] = value
        self.assertEqual(prepare_main(["--execute"]), 2)

    def test_studio_boot_binds_the_integrated_provider(self) -> None:
        root = Path(self.tmp.name) / "studio"
        issued = time.time()
        environ = {
            "STUDIO_MODE": "real",
            "VAST_API_KEY": "fixture-key",
            "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD": "0.50",
            "MEDIA_ENGINE_FORBID_R2_STAGE": "1",
            "MEDIA_ENGINE_EXECUTION_ID": "studio-exec",
            "MEDIA_ENGINE_EXECUTION_ISSUED_AT": str(issued),
            "MEDIA_ENGINE_EXECUTION_EXPIRES_AT": str(issued + 300),
        }
        controller = build_controller(root, "real", environ)
        self.assertIsInstance(controller.provider, VastImageProvider)
        self.assertEqual(controller.provider.execution_id, "studio-exec")
        self.assertAlmostEqual(controller.provider.execution_deadline, issued + 300, places=3)
        with self.assertRaises(WorkerStageError) as reused:
            build_controller(root, "real", environ)
        self.assertEqual(reused.exception.code, "STALE_AUTHORIZATION")

    def test_real_watchdog_starts_before_create_and_outlives_its_parent(self) -> None:
        os.environ["MEDIA_ENGINE_WATCHDOG_LAUNCH"] = "1"
        os.environ["MEDIA_ENGINE_WATCHDOG_LOCAL"] = "1"
        os.environ["MEDIA_ENGINE_WATCHDOG_POLL_SECONDS"] = "0.05"
        moment = time.time()
        provider = self._provider()
        provider._now = lambda: moment
        provider.bind_execution(SpendGrant(
            execution_id="exec-fresh",
            ceiling_usd=Decimal("0.50"),
            max_instances=1,
            issued_at=moment,
            expires_at=moment + 30,
            machine_id=MACHINE_ID,
        ), execution_id="exec-fresh")
        created = provider.create("daidoi")
        self.assertEqual(created.outcome, "created")
        self.assertEqual(provider.launch_trace, ["watchdog", "put"])
        spec = json.loads((Path(self.tmp.name) / "watchdog.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["mode"], "local")
        self.assertEqual(spec["deadline_epoch"], provider.execution_deadline)
        self.assertEqual(spec["machine_id"], MACHINE_ID)
        body = json.loads(self.put_body.decode("utf-8"))
        self.assertEqual(body["volume_info"]["volume_id"], VOLUME_ID)
        self.assertEqual(body["volume_info"]["mount_path"], MOUNT_PATH)
        self.assertFalse(body["volume_info"]["create_new"])
        self.assertTrue(_alive(provider.watchdog_pid))
        command = subprocess.check_output(
            ["ps", "-p", str(provider.watchdog_pid), "-o", "command="], text=True,
        )
        self.assertIn("vast_watchdog", command)
        self.addCleanup(_stop, provider.watchdog_pid)
        Path(spec["stop"]).write_text("1", encoding="utf-8")
        self.assertTrue(_wait_file(Path(spec["result"])))
        outcome = json.loads(Path(spec["result"]).read_text(encoding="utf-8"))
        self.assertEqual(outcome["termination_failures"], [])
        self.assertFalse(_wait_dead(provider.watchdog_pid))

        parent_root = Path(self.tmp.name) / "parent"
        parent_root.mkdir()
        instances = parent_root / "instances.json"
        instances.write_text(json.dumps([{
            "id": "local-1",
            "machine_id": MACHINE_ID,
            "label": "media-engine:preflight",
        }]), encoding="utf-8")
        child_spec = {
            "mode": "local",
            "machine_id": MACHINE_ID,
            "safe_seconds": 30,
            "deadline_epoch": time.time() + 30,
            "heartbeat": str(parent_root / "beat"),
            "stop": str(parent_root / "stop"),
            "result": str(parent_root / "result.json"),
            "instances_path": str(instances),
            "delete_log": str(parent_root / "deletes.json"),
            "poll_seconds": 0.05,
        }
        spec_path = parent_root / "watchdog.json"
        spec_path.write_text(json.dumps(child_spec), encoding="utf-8")
        pidfile = parent_root / "child.pid"
        script = parent_root / "parent.py"
        script.write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            "from media_engine.providers.vast_watchdog import arm_watchdog\n"
            "pid = arm_watchdog(Path(sys.argv[1]))\n"
            "Path(sys.argv[2]).write_text(str(pid))\n"
            "os._exit(0)\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.pop("VAST_API_KEY", None)
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, str(script), str(spec_path), str(pidfile)],
            cwd=str(ROOT),
            env=env,
            check=False,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0)
        child = int(pidfile.read_text(encoding="utf-8"))
        self.addCleanup(_stop, child)
        self.assertTrue(_alive(child))
        Path(child_spec["stop"]).write_text("1", encoding="utf-8")
        self.assertTrue(_wait_file(Path(child_spec["result"])))
        cleaned = json.loads(Path(child_spec["result"]).read_text(encoding="utf-8"))
        self.assertEqual(cleaned["stopped"], ["local-1"])
        self.assertEqual(cleaned["termination_failures"], [])
        self.assertFalse(_wait_dead(child))

    def test_watchdog_reports_a_failed_delete_and_exits(self) -> None:
        root = Path(self.tmp.name) / "fail-delete"
        root.mkdir()
        (root / "instances.json").write_text(json.dumps([{
            "id": "local-9",
            "machine_id": MACHINE_ID,
            "label": "media-engine:preflight",
        }]), encoding="utf-8")
        spec = {
            "mode": "local",
            "machine_id": MACHINE_ID,
            "safe_seconds": 1,
            "deadline_epoch": time.time() - 1,
            "heartbeat": str(root / "beat"),
            "stop": str(root / "stop"),
            "result": str(root / "result.json"),
            "instances_path": str(root / "instances.json"),
            "delete_log": str(root / "deletes.json"),
            "poll_seconds": 0.05,
            "fail_delete": True,
        }
        path = root / "watchdog.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        pid = arm_watchdog(path)
        self.addCleanup(_stop, pid)
        self.assertTrue(_wait_file(Path(spec["result"])))
        outcome = json.loads(Path(spec["result"]).read_text(encoding="utf-8"))
        self.assertEqual(outcome["termination_failures"], [{"id": "local-9", "status": 500}])
        self.assertFalse(_wait_dead(pid))
        bare = dict(spec)
        bare["mode"] = "provider"
        bare["result"] = str(root / "blocked.json")
        blocked = root / "blocked.json.spec"
        blocked.write_text(json.dumps(bare), encoding="utf-8")
        self.assertEqual(watchdog_main([str(blocked)]), 2)
        self.assertEqual(json.loads(Path(bare["result"]).read_text(encoding="utf-8"))["error"], "KEY_MISSING")

    def test_rental_restrictions_stop_before_any_request(self) -> None:
        mismatches = {
            "machine": {"machine_id": "56199"},
            "volume": {"attached_volume_id": 1},
            "gpu": {"gpu_model": "RTX 4090"},
            "price": {"hourly_amount": Decimal("0.376")},
            "cold": {"cache_source": "r2_cold_stage"},
            "disk": {"disk_gb": 140},
        }
        for name, changes in mismatches.items():
            self.calls.clear()
            provider = self._provider()
            provider.bind_execution(self._grant(name), execution_id=name)
            for key, value in changes.items():
                setattr(provider, key, value)
            provider._watchdog_popen = lambda *_args, **_kwargs: SimpleNamespace(pid=1)
            self.assertEqual(provider.create("daidoi").outcome, "definite_failure", name)
            self.assertNotIn("PUT", self.calls, name)
            self.assertEqual(provider.create("daidoi").outcome, "definite_failure", name)
            self.assertNotIn("PUT", self.calls, name)

    def test_offline_cache_rejects_a_bad_volume_and_does_not_download(self) -> None:
        missing_mount = assess_inventory({"mount": False, "revision": REVISION}, _manifest())
        self.assertEqual(missing_mount["status"], "CACHE_PATH_MISSING")
        self.assertFalse(missing_mount["mutated"])
        wrong = assess_inventory(_inventory(_manifest()), _manifest(revision="a" * 40))
        self.assertEqual(wrong["status"], "CACHE_REVISION_MISMATCH")
        incomplete = _inventory(_manifest())
        incomplete["entries"][1]["exists"] = False
        rejected = assess_inventory(incomplete, _manifest())
        self.assertEqual(rejected["status"], "CACHE_INCOMPLETE")
        self.assertFalse(rejected["complete"])
        commands: list[str] = []

        def ssh(_host, _port, command, _timeout):
            commands.append(command)
            return 0, json.dumps({"state": "COMPLETE", "mount": False, "revision": REVISION}) + "\n", ""

        provider = self._provider()
        provider._host = "203.0.113.10"
        provider._port = "22"
        provider._ssh_run = ssh
        with self.assertRaises(WorkerStageError) as caught:
            provider.confirm_cache("9")
        self.assertEqual(caught.exception.code, "MODEL_CACHE_INVALID")
        self.assertTrue(commands)
        self.assertNotIn("snapshot_download", commands[0])
        self.assertNotIn("hf_hub_download", commands[0])

    def test_cleanup_releases_one_worker_and_does_not_create_another(self) -> None:
        cases = ("connect", "cache", "load")
        for stage in cases:
            provider = FakeGPUProvider(lambda: 1_000.0)
            provider.execution_grant = self._grant(stage)
            provider.fail_stage = stage
            controller, job_id = self._run(provider)
            self.assertEqual(provider.create_calls, 1, stage)
            self.assertEqual(provider.terminate_calls, 1, stage)
            self.assertNotEqual(controller.get_job(job_id).error_code, "", stage)

        inference = FakeGPUProvider(lambda: 1_000.0)
        inference.execution_grant = self._grant("infer")
        controller, job_id = self._run(inference, PngEngine(fail=True))
        self.assertEqual(controller.get_job(job_id).error_code, "GENERATION_FAILED")
        self.assertEqual(inference.create_calls, 1)
        self.assertEqual(inference.terminate_calls, 1)

        upload = FakeGPUProvider(lambda: 1_000.0)
        upload.execution_grant = self._grant("upload")
        controller = self._controller(upload)
        platform = _Platform()
        platform.fail_upload = True
        controller.platform = platform
        dispatch(controller, "POST", "/v1/images/generations", json.dumps({"prompt": "upload"}).encode())
        controller.process_available()
        controller.process_available()
        self.assertEqual(controller.jobs.list()[0].error_code, "ARTIFACT_UPLOAD_FAILED")
        self.assertEqual(upload.create_calls, 1)
        self.assertEqual(upload.terminate_calls, 1)
        self.assertEqual(platform.events, ["authorize", "upload", "refund"])

        local_save = FakeGPUProvider(lambda: 1_000.0)
        local_save.execution_grant = self._grant("save")
        controller = self._controller(local_save)
        controller.storage = BoomStorage(str(Path(self.tmp.name) / "artifacts"))
        dispatch(controller, "POST", "/v1/images/generations", json.dumps({"prompt": "save"}).encode())
        controller.process_available()
        self.assertEqual(controller.jobs.list()[0].error_code, "ARTIFACT_TRANSFER_FAILED")
        self.assertEqual(local_save.terminate_calls, 1)
        self.assertEqual(local_save.create_calls, 1)

        settle = FakeGPUProvider(lambda: 1_000.0)
        settle.execution_grant = self._grant("settle")
        controller = self._controller(settle)
        platform = _Platform()
        platform.fail_settle = True
        controller.platform = platform
        dispatch(controller, "POST", "/v1/images/generations", json.dumps({"prompt": "settle"}).encode())
        controller.process_available()
        controller.process_available()
        self.assertEqual(controller.jobs.list()[0].error_code, "SETTLEMENT_FAILED")
        self.assertEqual(settle.create_calls, 1)
        self.assertEqual(settle.terminate_calls, 1)
        self.assertIn("refund", platform.events)

        deadline = FakeGPUProvider(lambda: 1_000.0)
        deadline.execution_grant = self._grant("deadline")
        deadline.execution_deadline = 1_000.0
        controller = self._controller(deadline)
        controller.clock.t = 1_000.0
        dispatch(controller, "POST", "/v1/images/generations", json.dumps({"prompt": "deadline"}).encode())
        controller.process_available()
        self.assertEqual(deadline.create_calls, 1)
        controller.clock.advance(0)
        controller._expire_worker()
        self.assertEqual(controller.last_shutdown_reason, "EXECUTION_DEADLINE")
        self.assertEqual(deadline.terminate_calls, 1)
        self.assertEqual(deadline.create_calls, 1)


def _wait_file(path: Path, seconds: float = 5.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if path.is_file():
            return True
        time.sleep(0.05)
    return False


def _wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not _alive(pid):
            return False
        time.sleep(0.05)
    return _alive(pid)
