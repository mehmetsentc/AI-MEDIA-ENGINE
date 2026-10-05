"""Phase 1.2 local HTTP, text jobs, and usage rows. No network beyond loopback."""
from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from media_engine.api.app import LocalAPIServer
from media_engine.engines.binding import LOCAL_ENGINE_IDS
from media_engine.engines.text.fake import FakeTextEngine
from media_engine.jobs.model import IMAGE_GENERATE, TEXT_GENERATE
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.safety.limits import (
    ExternalProvidersDisabled,
    SafetyLimits,
    require_live_external_provider,
)


def _http(server: LocalAPIServer, method: str, path: str, payload: Optional[dict] = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(server.url + path, data=data, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        finally:
            exc.close()


class Phase12Tests(unittest.TestCase):
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
        kwargs.setdefault("clock", self.clock)
        return MediaController(self.db_path, self.storage_root, **kwargs)

    def _serve(self, controller: MediaController) -> LocalAPIServer:
        server = LocalAPIServer(controller)
        self.servers.append(server)
        server.start()
        return server

    def test_http_boot_starts_controller(self) -> None:
        controller = self._controller()
        self._serve(controller)
        self.assertTrue(controller.runner_alive)
        self.assertFalse(controller._thread.daemon)
        self.assertEqual(controller._thread.name, "media-engine-runner")

    def test_http_shutdown_stops_controller(self) -> None:
        controller = self._controller()
        server = self._serve(controller)
        server.stop()
        self.servers.remove(server)
        self.assertFalse(controller.runner_alive)

    def test_http_post_completes_automatically(self) -> None:
        controller = self._controller()
        server = self._serve(controller)
        status, created = _http(server, "POST", "/v1/jobs", {
            "client_id": "nahaber",
            "type": TEXT_GENERATE,
            "input": {"task": "news_rewrite", "prompt": "hello"},
        })
        self.assertEqual(status, 201)
        self.assertEqual(created["status"], "QUEUED")
        self.assertTrue(controller.wait_settled())
        status, done = _http(server, "GET", f"/v1/jobs/{created['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(done["status"], "SUCCEEDED")

    def test_text_generate_lifecycle_succeeds(self) -> None:
        controller = self._controller()
        server = self._serve(controller)
        body = {
            "client_id": "nahaber",
            "type": TEXT_GENERATE,
            "input": {"task": "summarize", "prompt": "same words"},
        }
        _, first = _http(server, "POST", "/v1/jobs", body)
        self.assertTrue(controller.wait_settled())
        _, second = _http(server, "POST", "/v1/jobs", body)
        self.assertTrue(controller.wait_settled())
        one = controller.get_job(first["id"])
        two = controller.get_job(second["id"])
        self.assertEqual(one.status, "SUCCEEDED")
        self.assertEqual(controller.provider.create_calls, 0)
        left = Path(unquote(urlparse(one.result_uri).path)).read_text(encoding="utf-8")
        right = Path(unquote(urlparse(two.result_uri).path)).read_text(encoding="utf-8")
        self.assertEqual(left, right)
        self.assertTrue(left.startswith("FAKETEXT1\nsummarize\n"))

    def test_nahaber_client_accepted(self) -> None:
        controller = self._controller()
        server = self._serve(controller)
        status, created = _http(server, "POST", "/v1/jobs", {
            "client_id": "nahaber",
            "type": TEXT_GENERATE,
            "input": {"task": "news_rewrite", "prompt": "wire"},
        })
        self.assertEqual(status, 201)
        self.assertEqual(created["client_id"], "nahaber")
        self.assertTrue(controller.wait_settled())

    def test_unknown_client_rejected(self) -> None:
        server = self._serve(self._controller())
        status, body = _http(server, "POST", "/v1/jobs", {
            "client_id": "other-app",
            "type": TEXT_GENERATE,
            "input": {"task": "news_rewrite", "prompt": "no"},
        })
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "UNKNOWN_CLIENT")

    def test_image_generate_regression(self) -> None:
        controller = self._controller()
        server = self._serve(controller)
        status, created = _http(server, "POST", "/v1/jobs", {
            "client_id": "film-studio",
            "type": IMAGE_GENERATE,
            "input": {"prompt": "test image"},
        })
        self.assertEqual(status, 201)
        self.assertEqual(created["status"], "QUEUED")
        self.assertTrue(controller.wait_settled())
        done = controller.get_job(created["id"])
        self.assertEqual(done.status, "SUCCEEDED")
        self.assertEqual(controller.provider.create_calls, 1)
        raw = Path(unquote(urlparse(done.result_uri).path)).read_bytes()
        self.assertTrue(raw.startswith(b"FAKEIMG1\n"))

    def test_telemetry_written_for_success(self) -> None:
        controller = self._controller()
        controller.start()
        self.addCleanup(controller.stop)
        job = controller.submit_job(
            "film-studio", TEXT_GENERATE, "count me", task="summarize",
        )
        self.assertTrue(controller.wait_settled())
        event = controller.usage.for_job(job.id)
        self.assertIsNotNone(event)
        self.assertEqual(event.client_id, "film-studio")
        self.assertEqual(event.job_type, TEXT_GENERATE)
        self.assertEqual(event.engine_id, "fake-text")
        self.assertEqual(event.status, "SUCCEEDED")
        self.assertEqual(event.attempt_count, 1)
        self.assertIsNone(event.estimated_cost_usd)
        self.assertGreaterEqual(event.duration_seconds, 0)
        self.assertTrue(event.started_at)
        self.assertTrue(event.finished_at)

    def test_telemetry_written_for_failure(self) -> None:
        controller = self._controller(text_engine=FakeTextEngine(fail=True))
        controller.start()
        self.addCleanup(controller.stop)
        job = controller.submit_job("nahaber", TEXT_GENERATE, "boom", task="summarize")
        self.assertTrue(controller.wait_settled())
        saved = controller.get_job(job.id)
        event = controller.usage.for_job(job.id)
        self.assertEqual(saved.status, "FAILED")
        self.assertEqual(saved.error_code, "GENERATION_FAILED")
        self.assertEqual(event.status, "FAILED")
        self.assertEqual(event.engine_id, "fake-text")
        self.assertEqual(event.client_id, "nahaber")
        self.assertIsNone(event.estimated_cost_usd)
        self.assertEqual(controller.provider.create_calls, 0)

    def test_external_providers_still_fail_closed(self) -> None:
        limits = SafetyLimits()
        self.assertFalse(limits.live_external_providers)
        self.assertEqual(limits.max_gpu_workers, 1)
        self.assertEqual(limits.max_job_attempts, 1)
        with self.assertRaises(ExternalProvidersDisabled):
            require_live_external_provider(limits)
        blob = " ".join(LOCAL_ENGINE_IDS.values())
        for name in ("runpod", "vast", "deepseek", "qwen"):
            self.assertNotIn(name, blob)
        server = self._serve(self._controller())
        status, created = _http(server, "POST", "/v1/jobs", {
            "client_id": "film-studio",
            "type": IMAGE_GENERATE,
            "input": {"prompt": "local only"},
        })
        self.assertEqual(status, 201)
        self.assertNotIn("provider", created)
