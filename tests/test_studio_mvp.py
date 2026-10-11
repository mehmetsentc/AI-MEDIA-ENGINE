"""Studio sign-in, credits, library, and guarded generation. No GPU is rented."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from media_engine.api.app import LocalAPIServer
from media_engine.engines.image.base import ImageEngine, ImageEngineError
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.platform.service import MemoryMedia, Platform
from media_engine.providers.fake import FakeGPUProvider
from media_engine.studio.account import history_for, job_group, submit_for_studio
from media_engine.studio.account import SessionBook
from media_engine.studio.dev_engine import DevImageEngine
from tests.test_studio import _http


LIVE_ARTIFACTS = (
    "art_5c9b6f78757947e890c16c12976f26d3",
    "art_3604b2bee4844dd5ae2200322145a084",
)


class _Boom(ImageEngine):
    def render(self, prompt: str, width: int = 1024, height: int = 1024, seed: int = 0) -> bytes:
        raise ImageEngineError("GENERATION_FAILED")


class _VastStub:
    name = "vast"
    execution_authorization = None
    create_calls = 0

    def create(self, *_args: object, **_kwargs: object) -> None:
        self.create_calls += 1


class StudioMvpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = ManualClock()
        self.servers: list[LocalAPIServer] = []

    def tearDown(self) -> None:
        for server in self.servers:
            server.stop()

    def _controller(self, engine: ImageEngine | None = None) -> MediaController:
        root = Path(self.tmp.name) / uuid.uuid4().hex
        root.mkdir()
        controller = MediaController(
            str(root / "meta.sqlite3"),
            str(root / "artifacts"),
            clock=self.clock,
            provider=FakeGPUProvider(self.clock.now),
            image_engine=engine or DevImageEngine(),
        )
        controller.platform = Platform(str(root / "platform.sqlite3"))
        controller.platform.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=0, effective_at="2026-01-01T00:00:00Z",
        )
        return controller

    def _server(self, controller: MediaController) -> tuple[LocalAPIServer, str]:
        server = LocalAPIServer(controller, api_key="studio-secret")
        self.servers.append(server)
        server.start()
        status, _html, cookie = _open(server)
        self.assertEqual(status, 200)
        return server, cookie

    def _sign(self, server: LocalAPIServer, cookie: str, client_id: str = "nahaber"):
        return _http(server, "POST", "/v1/session", {"client_id": client_id}, cookie=cookie)

    def test_sign_in_shows_the_user_balance_and_blocks_an_unknown_identity(self) -> None:
        controller = self._controller()
        server, cookie = self._server(controller)
        status, unknown = _http(server, "POST", "/v1/session", {"client_id": "someone-else"}, cookie=cookie)
        self.assertEqual(status, 400)
        self.assertEqual(unknown["error"], "UNKNOWN_CLIENT")
        status, account = self._sign(server, cookie)
        self.assertEqual(status, 200)
        self.assertEqual(account["user_id"], "nahaber")
        self.assertEqual(account["balance"], 0)
        self.assertEqual(account["image_price"], 2)
        self.assertFalse(account["can_generate"])
        self.assertEqual(account["block_reason"], "INSUFFICIENT_CREDITS")
        other = MediaController(
            str(Path(self.tmp.name) / "other.sqlite3"),
            str(Path(self.tmp.name) / "other-art"),
            clock=self.clock,
            provider=FakeGPUProvider(self.clock.now),
            image_engine=DevImageEngine(),
        )
        other.platform = Platform(str(Path(self.tmp.name) / "other-platform.sqlite3"))
        other.platform.grant("film-studio", "topup", 6, "topup_purchase", idempotency_key="film-grant")
        other_server, other_cookie = self._server(other)
        status, film = self._sign(other_server, other_cookie, "film-studio")
        self.assertEqual(film["balance"], 6)
        self.assertNotEqual(account["balance"], film["balance"])

    def test_insufficient_credits_do_not_reserve_or_create(self) -> None:
        controller = self._controller()
        server, cookie = self._server(controller)
        self._sign(server, cookie)
        status, body = _http(server, "POST", "/v1/images/generations", {
            "prompt": "A quiet courtyard", "width": 1024, "height": 1024, "idempotency_key": "studio-key-01",
        }, cookie=cookie)
        self.assertEqual(status, 402)
        self.assertEqual(body["error"], "INSUFFICIENT_CREDITS")
        self.assertEqual(body["required"], 2)
        self.assertEqual(body["balance"], 0)
        self.assertEqual(controller.jobs.list(), [])
        self.assertEqual(controller.provider.create_calls, 0)
        self.assertEqual(controller.platform.balance("nahaber"), 0)

    def test_prompt_validation_and_duplicate_click(self) -> None:
        controller = self._controller()
        controller.platform.grant("nahaber", "topup", 4, "topup_purchase", idempotency_key="grant-4")
        server, cookie = self._server(controller)
        self._sign(server, cookie)
        status, empty = _http(server, "POST", "/v1/images/generations", {
            "prompt": "   ", "idempotency_key": "studio-key-02",
        }, cookie=cookie)
        self.assertEqual(status, 400)
        self.assertEqual(empty["error"], "PROMPT_REQUIRED")
        body = {
            "prompt": "A coastal bench", "width": 1024, "height": 1024, "seed": 11,
            "idempotency_key": "studio-key-03",
        }
        status, first = _http(server, "POST", "/v1/images/generations", body, cookie=cookie)
        status, second = _http(server, "POST", "/v1/images/generations", body, cookie=cookie)
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertTrue(second["idempotent"])
        self.assertEqual(len(controller.jobs.list()), 1)
        self.assertTrue(controller.wait_settled())
        status, done = _http(server, "GET", "/v1/jobs/" + first["job_id"], cookie=cookie)
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["group"], "completed")
        status, png = _http(server, "GET", "/v1/artifacts/" + done["artifact_id"] + "?download=1", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")

    def test_pending_running_failure_and_refund(self) -> None:
        controller = self._controller()
        book = SessionBook()
        book.sign_in("cookie", "nahaber")
        controller.platform.grant("nahaber", "topup", 2, "topup_purchase", idempotency_key="grant-2")
        status, queued = submit_for_studio(controller, "nahaber", {
            "prompt": "A stone courtyard", "idempotency_key": "studio-key-04",
        }, book)
        self.assertEqual(status, 202)
        self.assertEqual(queued["status"], "queued")
        self.assertEqual(job_group("queued"), "pending")
        self.assertEqual(job_group("generating"), "running")
        job = controller.get_job(queued["job_id"])
        job.public_status = "generating"
        controller.jobs.save(job)
        running = history_for(controller, "nahaber")[0]
        self.assertEqual(running["group"], "running")
        job.public_status = "queued"
        controller.jobs.save(job)
        failed = self._controller(_Boom())
        failed.platform.grant("nahaber", "topup", 2, "topup_purchase", idempotency_key="grant-fail")
        server, cookie = self._server(failed)
        self._sign(server, cookie)
        status, created = _http(server, "POST", "/v1/images/generations", {
            "prompt": "A cliff path", "idempotency_key": "studio-key-05",
        }, cookie=cookie)
        self.assertEqual(status, 202)
        self.assertTrue(failed.wait_settled())
        status, done = _http(server, "GET", "/v1/jobs/" + created["job_id"], cookie=cookie)
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["group"], "failed")
        self.assertEqual(done["credit_status"], "refunded")
        self.assertEqual(failed.platform.balance("nahaber"), 2)
        status, jobs = _http(server, "GET", "/v1/jobs", cookie=cookie)
        self.assertEqual(jobs["jobs"][0]["credit_status"], "refunded")

    def test_library_is_private_and_live_fixtures_download(self) -> None:
        controller = self._controller()
        media = MemoryMedia()
        controller.platform.media = media
        for artifact_id, prompt in zip(LIVE_ARTIFACTS, ("courtyard", "cliff")):
            png = DevImageEngine().render(prompt)
            key = f"users/nahaber/projects/direct/scenes/image/images/{artifact_id}.png"
            media.put(key, png)
            controller.platform.repo.insert_artifact({
                "id": artifact_id,
                "user_id": "nahaber",
                "project_id": "direct",
                "scene_id": "image",
                "job_id": "job_" + artifact_id[-8:],
                "artifact_type": "images",
                "bucket": "daidoi-media",
                "object_key": key,
                "mime_type": "image/png",
                "byte_count": len(png),
                "sha256": "abc",
                "width": 1024,
                "height": 1024,
                "model": "Qwen/Qwen-Image",
                "generation_settings": json.dumps({"prompt": prompt, "seed": 1}),
            }, controller.clock.iso())
        server, cookie = self._server(controller)
        self._sign(server, cookie)
        status, library = _http(server, "GET", "/v1/library", cookie=cookie)
        self.assertEqual(status, 200)
        found = {item["artifact_id"] for item in library["artifacts"]}
        self.assertTrue(set(LIVE_ARTIFACTS).issubset(found))
        status, png = _http(server, "GET", "/v1/artifacts/" + LIVE_ARTIFACTS[0], cookie=cookie)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        status, switched = self._sign(server, cookie, "film-studio")
        self.assertEqual(switched["user_id"], "film-studio")
        status, hidden = _http(server, "GET", "/v1/library", cookie=cookie)
        self.assertEqual(hidden["artifacts"], [])
        status, denied = _http(server, "GET", "/v1/artifacts/" + LIVE_ARTIFACTS[0], cookie=cookie)
        self.assertEqual(status, 404)
        self.assertEqual(denied["error"], "JOB_NOT_FOUND")

    def test_browser_gets_no_secrets_and_cannot_rent_without_authorization(self) -> None:
        controller = self._controller()
        controller.platform.grant("nahaber", "topup", 4, "topup_purchase", idempotency_key="grant-auth")
        stub = _VastStub()
        controller.provider = stub
        server, cookie = self._server(controller)
        previous = os.environ.get("VAST_API_KEY")
        os.environ["VAST_API_KEY"] = "super-secret-vast-token"
        self.addCleanup(lambda: _restore("VAST_API_KEY", previous))
        status, html, _cookie = _open(server)
        self.assertNotIn("super-secret-vast-token", html)
        self.assertNotIn("VAST_API_KEY", html)
        self._sign(server, cookie)
        status, account = _http(server, "GET", "/v1/account", cookie=cookie)
        self.assertNotIn("super-secret-vast-token", json.dumps(account))
        self.assertEqual(account["block_reason"], "AUTHORIZATION_REQUIRED")
        self.assertFalse(account["can_generate"])
        status, blocked = _http(server, "POST", "/v1/images/generations", {
            "prompt": "A quiet courtyard", "idempotency_key": "studio-key-06",
        }, cookie=cookie)
        self.assertEqual(status, 409)
        self.assertEqual(blocked["error"], "AUTHORIZATION_REQUIRED")
        self.assertEqual(stub.create_calls, 0)
        self.assertEqual(controller.jobs.list(), [])

    def test_static_page_keeps_generation_copy_and_disables_future_tools(self) -> None:
        page = Path("src/media_engine/studio/static/studio.js").read_text(encoding="utf-8")
        html = Path("src/media_engine/studio/static/index.html").read_text(encoding="utf-8")
        self.assertIn("POLL_MS = 4000", page)
        self.assertGreaterEqual(4000, 2000)
        self.assertIn("disabled>Video", html)
        self.assertIn("disabled>Audio", html)
        self.assertIn("disabled>Voice", html)
        self.assertNotIn("machine_id", page)
        self.assertNotIn("volume", page.lower())


def _open(server: LocalAPIServer):
    import urllib.request
    request = urllib.request.Request(server.url + "/")
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status, response.read().decode("utf-8"), response.headers.get("Set-Cookie", "").split(";", 1)[0]


def _restore(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous
