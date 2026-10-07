"""Phase 2E image API. The suite never contacts Vast or starts a GPU."""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
import urllib.error
import urllib.request
import zlib
from decimal import Decimal
from pathlib import Path
from typing import Optional

from media_engine.api.app import LocalAPIServer, authorize, dispatch
from media_engine.engines.image.base import ImageEngineError
from media_engine.jobs.model import PUBLIC_PROGRESS
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.providers.base import WorkerStageError
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.vast_image import (
    MACHINE_ID,
    VOLUME_ID,
    _RUNTIME_INSTALL,
    _generate_command,
    VastImageProvider,
)
from media_engine.safety.limits import SafetyLimits, limits_from_env
from media_engine.storage.local import LocalStorage


def _png(width: int = 1024, height: int = 1024) -> bytes:
    raw = b"".join(b"\x00" + bytes((8, 16, 24)) * width for _ in range(height))
    compressed = zlib.compress(raw)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


PNG = _png()


class PngEngine:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def render(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = 0) -> bytes:
        self.calls += 1
        if self.fail:
            raise ImageEngineError("GENERATION_FAILED")
        return _png(width, height)


class BoomStorage(LocalStorage):
    def put(self, *, client_id: str, job_id: str, name: str, data: bytes) -> str:
        raise OSError("disk full")


def _http(server: LocalAPIServer, method: str, path: str, payload: Optional[dict] = None, token: str = ""):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(server.url + path, data=data, method=method)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=2) as response:
            content = response.headers.get("Content-Type", "")
            body = response.read()
            if content.startswith("image/"):
                return response.status, body
            return response.status, json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        finally:
            exc.close()


class Phase2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.db_path = str(root / "meta.sqlite3")
        self.storage_root = str(root / "artifacts")
        self.clock = ManualClock()
        self.servers: list[LocalAPIServer] = []

    def tearDown(self) -> None:
        for server in self.servers:
            server.stop()

    def _controller(self, **kwargs) -> MediaController:
        slot = getattr(self, "_slots", 0) + 1
        self._slots = slot
        root = Path(self.tmp.name) / f"slot-{slot}"
        root.mkdir()
        kwargs.setdefault("clock", self.clock)
        kwargs.setdefault("image_engine", PngEngine())
        kwargs.setdefault("provider", FakeGPUProvider(self.clock.now))
        return MediaController(str(root / "meta.sqlite3"), str(root / "artifacts"), **kwargs)

    def _post(self, controller: MediaController, payload: Optional[dict] = None):
        body = {"prompt": "a terrace at sunset"}
        if payload is not None:
            body = payload
        return dispatch(controller, "POST", "/v1/images/generations", json.dumps(body).encode("utf-8"))

    def test_auth_fails_closed_and_health_stays_open(self) -> None:
        self.assertEqual(authorize("/v1/images/generations", {}, ""), (401, {"error": "UNAUTHORIZED"}))
        self.assertEqual(authorize("/health", {}, ""), None)
        self.assertEqual(authorize("/v1/jobs/job_1", {"Authorization": "Bearer no"}, "yes"), (401, {"error": "UNAUTHORIZED"}))
        self.assertIsNone(authorize("/v1/jobs/job_1", {"Authorization": "Bearer yes"}, "yes"))
        controller = self._controller()
        server = LocalAPIServer(controller, api_key="")
        self.servers.append(server)
        server.start()
        status, health = _http(server, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["service"], "ai-media-engine")
        self.assertEqual(health["image_engine"], "qwen-image")
        self.assertEqual(health["active_workers"], 0)
        self.assertEqual(controller.provider.create_calls, 0)
        denied, body = _http(server, "POST", "/v1/images/generations", {"prompt": "x"})
        self.assertEqual(denied, 401)
        self.assertEqual(body["error"], "UNAUTHORIZED")
        self.assertNotIn("ssh", json.dumps(health))

    def test_request_validation(self) -> None:
        controller = self._controller()
        status, body = self._post(controller, {"prompt": "  "})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "PROMPT_REQUIRED")
        status, body = self._post(controller, {"prompt": "ok", "width": 512})
        self.assertEqual(body["error"], "RESOLUTION_UNSUPPORTED")
        status, body = self._post(controller, {"prompt": "ok", "seed": -1})
        self.assertEqual(body["error"], "SEED_INVALID")
        status, body = self._post(controller, {"prompt": "ok", "negative_prompt": "text"})
        self.assertEqual(body["error"], "NEGATIVE_PROMPT_UNSUPPORTED")
        self.assertEqual(controller.provider.create_calls, 0)

    def test_job_is_queued_immediately_and_completes_with_artifact(self) -> None:
        controller = self._controller()
        server = LocalAPIServer(controller, api_key="local-test")
        self.servers.append(server)
        server.start()
        status, created = _http(server, "POST", "/v1/images/generations", {"prompt": "terrace"}, token="local-test")
        self.assertEqual(status, 202)
        self.assertEqual(created["status"], "queued")
        self.assertTrue(created["job_id"].startswith("job_"))
        self.assertNotIn("vast", json.dumps(created))
        self.assertTrue(controller.wait_settled())
        status, done = _http(server, "GET", f"/v1/jobs/{created['job_id']}", token="local-test")
        self.assertEqual(status, 200)
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["error_code"], None)
        self.assertEqual(done["progress"], 1.0)
        self.assertEqual(done["width"], 1024)
        self.assertEqual(done["height"], 1024)
        self.assertIsInstance(done["seed"], int)
        self.assertEqual(len(done["sha256"]), 64)
        status, png = _http(server, "GET", f"/v1/artifacts/{done['artifact_id']}", token="local-test")
        self.assertEqual(status, 200)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(len(png), done and controller.get_job(created["job_id"]).byte_count)
        saved = controller.artifacts.get(done["artifact_id"])
        self.assertEqual(saved.engine, "image")
        self.assertEqual(saved.model, "qwen-image")
        self.assertEqual(saved.prompt, "terrace")
        self.assertEqual(saved.sha256, done["sha256"])
        event = controller.usage.for_job(created["job_id"])
        self.assertEqual(event.provider, "fake")
        self.assertEqual(event.instance_id, controller.provider.list_owned()[0]["resource_id"])
        self.assertEqual(event.hourly_price_usd, "0.50")

    def test_worker_is_reused_and_idle_shutdown_is_300_seconds(self) -> None:
        controller = self._controller()
        self._post(controller, {"prompt": "one"})
        self._post(controller, {"prompt": "two"})
        controller.process_available()
        self.assertEqual(controller.provider.create_calls, 1)
        self.assertTrue(any(line.endswith(":REUSED") for line in controller.trace))
        self.assertEqual(controller.limits.max_gpu_workers, 1)
        self.assertEqual(controller.limits.gpu_idle_shutdown_seconds, 300)
        resource_id = controller.provider.list_owned()[0]["resource_id"]
        self.clock.advance(299)
        self.assertIsNone(controller.enforce_lifecycle())
        self.assertEqual(controller.provider.status(resource_id), "READY")
        self.clock.advance(1)
        self.assertEqual(controller.enforce_lifecycle(), "GPU_IDLE_SHUTDOWN")
        self.assertEqual(controller.provider.status(resource_id), "TERMINATED")

    def test_second_worker_is_refused_and_price_ceiling_holds(self) -> None:
        provider = FakeGPUProvider(self.clock.now)
        provider.plant("gpu_held", "nahaber", status="BUSY")
        controller = self._controller(provider=provider)
        status, created = self._post(controller)
        self.assertEqual(status, 202)
        controller.process_available()
        failed = controller.get_job(created["job_id"])
        self.assertEqual(failed.public_status, "failed")
        self.assertEqual(failed.error_code, "MAX_GPU_WORKERS")
        self.assertEqual(len([row for row in provider.list_owned() if row["status"] != "TERMINATED"]), 1)
        pricey = FakeGPUProvider(self.clock.now, hourly_price=Decimal("0.61"))
        second = self._controller(provider=pricey)
        _status, priced = self._post(second, {"prompt": "too expensive"})
        second.process_available()
        self.assertEqual(second.get_job(priced["job_id"]).error_code, "PRICE_ABOVE_CEILING")
        self.assertEqual(pricey.create_calls, 0)
        self.assertEqual(SafetyLimits().max_hourly_price_usd, Decimal("0.60"))
        self.assertEqual(limits_from_env(SafetyLimits()).max_worker_lifetime_seconds, 1800)

    def test_capacity_and_ambiguous_create_do_not_retry(self) -> None:
        missing = FakeGPUProvider(self.clock.now)
        missing.unavailable = True
        controller = self._controller(provider=missing)
        self._post(controller)
        controller.process_available()
        self.assertEqual(controller.jobs.list()[0].error_code, "PROVIDER_CAPACITY_UNAVAILABLE")
        self.assertEqual(missing.create_calls, 0)
        ambiguous = FakeGPUProvider(self.clock.now, outcome="ambiguous")
        other = self._controller(provider=ambiguous)
        _status, created = self._post(other, {"prompt": "maybe"})
        other.process_available()
        other.process_available()
        self.assertEqual(other.get_job(created["job_id"]).error_code, "AMBIGUOUS_CREATE")
        self.assertEqual(ambiguous.create_calls, 1)

    def test_stage_failures_destroy_the_worker_generation_and_artifact_do_not(self) -> None:
        cases = {
            "boot": "WORKER_BOOT_FAILED",
            "timeout": "WORKER_TIMEOUT",
            "runtime": "RUNTIME_PREPARE_FAILED",
            "cache": "MODEL_CACHE_INVALID",
            "load": "MODEL_LOAD_FAILED",
        }
        for stage, code in cases.items():
            provider = FakeGPUProvider(self.clock.now)
            provider.fail_stage = stage
            controller = self._controller(provider=provider)
            _status, created = self._post(controller, {"prompt": stage})
            controller.process_available()
            self.assertEqual(controller.get_job(created["job_id"]).error_code, code, stage)
            self.assertEqual(provider.terminate_calls, 1, stage)
            self.assertFalse(controller.ledger.unresolved(), stage)
        provider = FakeGPUProvider(self.clock.now)
        controller = self._controller(provider=provider, image_engine=PngEngine(fail=True))
        _status, created = self._post(controller, {"prompt": "bad pixels"})
        controller.process_available()
        self.assertEqual(controller.get_job(created["job_id"]).error_code, "GENERATION_FAILED")
        self.assertEqual(provider.terminate_calls, 0)
        self.assertEqual(provider.status(provider.list_owned()[0]["resource_id"]), "READY")
        provider = FakeGPUProvider(self.clock.now)
        controller = self._controller(provider=provider)
        controller.storage = BoomStorage(self.storage_root)
        _status, created = self._post(controller, {"prompt": "save fails"})
        controller.process_available()
        self.assertEqual(controller.get_job(created["job_id"]).error_code, "ARTIFACT_TRANSFER_FAILED")
        self.assertEqual(provider.terminate_calls, 0)
        self.assertNotIn("disk full", controller.jobs.list()[0].error_code)

    def test_hard_lifetime_stops_reuse(self) -> None:
        limits = SafetyLimits(gpu_idle_shutdown_seconds=10_000, max_worker_lifetime_seconds=1800)
        controller = self._controller(limits=limits)
        self._post(controller, {"prompt": "first"})
        controller.process_available()
        self.clock.advance(1800)
        self._post(controller, {"prompt": "second"})
        controller.process_available()
        self.assertEqual(controller.provider.create_calls, 2)
        self.assertEqual(controller.last_shutdown_reason, "MAX_WORKER_LIFETIME")
        active = [row for row in controller.provider.list_owned() if row["status"] != "TERMINATED"]
        self.assertEqual(len(active), 1)

    def test_public_stages_are_recorded(self) -> None:
        controller = self._controller()
        _status, created = self._post(controller, {"prompt": "stages", "seed": 7})
        controller.process_available()
        seen = [line.split(":public:", 1)[1] for line in controller.trace if ":public:" in line]
        self.assertEqual(seen, [
            "planning", "provisioning", "booting", "runtime_preparing",
            "model_loading", "generating", "saving", "completed",
        ])
        self.assertEqual(PUBLIC_PROGRESS["queued"], 0.0)
        done = controller.get_job(created["job_id"])
        self.assertEqual(done.seed, 7)
        self.assertIsNone(done.error_code)

    def test_vast_quote_create_and_ownership(self) -> None:
        responses = [
            (200, json.dumps({"volumes": [{"id": VOLUME_ID, "machine_id": MACHINE_ID}]}).encode()),
            (200, json.dumps({"offers": [{
                "id": 30565038, "gpu_name": "RTX A6000", "gpu_ram": 49140,
                "machine_id": MACHINE_ID, "rentable": True,
                "dph_base": "0.36", "dph_total": "0.37", "storage_cost": "0.2666",
            }]}).encode()),
            (200, json.dumps({"success": True, "new_contract": 88}).encode()),
        ]
        calls: list[tuple[str, str, bytes]] = []

        def transport(method, url, body, headers):
            calls.append((method, url, body))
            self.assertNotIn(b"R2_", body)
            self.assertNotIn(b"fixture-key", body)
            return responses.pop(0)

        provider = VastImageProvider(api_key="fixture-key", transport=transport, now=self.clock.now, sleep=lambda _s: None)
        quote = provider.quote("nahaber")
        self.assertEqual(quote.provider, "vast")
        self.assertLessEqual(quote.hourly_price_usd, Decimal("0.60"))
        created = provider.create("nahaber")
        self.assertEqual(created.outcome, "created")
        self.assertEqual(created.resource_id, "88")
        again = provider.create("nahaber")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertEqual(sum(1 for method, _url, _body in calls if method == "PUT"), 1)
        _method, url, raw = calls[2]
        self.assertIn("/asks/30565038/", url)
        sent = json.loads(raw.decode("utf-8"))
        self.assertEqual(sent["volume_info"]["volume_id"], VOLUME_ID)
        self.assertEqual(sent["volume_info"]["mount_path"], "/models")
        self.assertFalse(sent["volume_info"]["create_new"])

    def test_ambiguous_create_requeries_once_and_foreign_destroy_is_refused(self) -> None:
        responses = [
            (200, json.dumps({"volumes": [{"id": VOLUME_ID, "machine_id": MACHINE_ID}]}).encode()),
            (200, json.dumps({"offers": [{
                "id": 11, "gpu_name": "RTX A6000", "gpu_ram": 49140, "machine_id": MACHINE_ID,
                "rentable": True, "dph_base": "0.36", "dph_total": "0.37", "storage_cost": "0.2",
            }]}).encode()),
            (500, b""),
            (200, json.dumps({"instances": [{"id": 55, "label": "media-engine:nahaber", "machine_id": MACHINE_ID}]}).encode()),
        ]
        calls: list[tuple[str, str]] = []

        def transport(method, url, body, headers):
            calls.append((method, url))
            return responses.pop(0)

        provider = VastImageProvider(api_key="fixture-key", transport=transport, now=self.clock.now, sleep=lambda _s: None)
        provider.quote("nahaber")
        created = provider.create("nahaber")
        self.assertEqual(created.resource_id, "55")
        self.assertEqual(sum(1 for method, _url in calls if method == "PUT"), 1)
        foreign_calls: list[str] = []

        def foreign(method, url, body, headers):
            foreign_calls.append(method + " " + url)
            return 200, json.dumps({"instances": [{"id": 7, "label": "other", "machine_id": MACHINE_ID}]}).encode()

        owned = VastImageProvider(api_key="fixture-key", transport=foreign, now=self.clock.now, sleep=lambda _s: None)
        with self.assertRaises(WorkerStageError) as caught:
            owned.terminate("7")
        self.assertEqual(caught.exception.code, "OWNERSHIP_MISMATCH")
        self.assertFalse(any(item.startswith("DELETE") for item in foreign_calls))
        self.assertFalse(any("/volumes/" in item and item.startswith("DELETE") for item in foreign_calls))

    def test_missing_machine_capacity_and_runtime_command_shape(self) -> None:
        responses = [
            (200, json.dumps({"volumes": [{"id": VOLUME_ID, "machine_id": MACHINE_ID}]}).encode()),
            (200, json.dumps({"offers": []}).encode()),
        ]

        def transport(method, url, body, headers):
            return responses.pop(0)

        provider = VastImageProvider(api_key="fixture-key", transport=transport, now=self.clock.now, sleep=lambda _s: None)
        with self.assertRaises(WorkerStageError) as caught:
            provider.quote("nahaber")
        self.assertEqual(caught.exception.code, "PROVIDER_CAPACITY_UNAVAILABLE")
        command = _generate_command("A terrace", 1024, 1024, 5)
        self.assertIn("HF_HUB_OFFLINE=1", command)
        self.assertNotIn("R2_", command)
        self.assertNotIn("huggingface.co", command)
        self.assertIn("/workspace/phase2d-venv/bin/python -m pip", _RUNTIME_INSTALL)
        self.assertNotIn("python3 -m pip", _RUNTIME_INSTALL)
        self.assertNotIn("--break-system-packages", _RUNTIME_INSTALL)

    def test_cache_must_be_complete_before_runtime(self) -> None:
        def ssh(host, port, command, timeout):
            if "phase2d-cache.json" in command:
                return 0, json.dumps({
                    "state": "FILLING", "repo": "Qwen/Qwen-Image", "revision": "bad",
                    "files": 1, "bytes": 1, "mount": True,
                }) + "\n", ""
            return 0, "", ""

        provider = VastImageProvider(
            api_key="fixture-key", transport=lambda *args: (200, b"{}"),
            now=self.clock.now, sleep=lambda _s: None, ssh_run=ssh,
        )
        with self.assertRaises(WorkerStageError) as caught:
            provider.confirm_cache("1")
        self.assertEqual(caught.exception.code, "MODEL_CACHE_INVALID")
