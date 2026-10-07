"""Studio projects and the local image workspace. No GPU is rented."""
from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from media_engine.api.app import LocalAPIServer, authorize, dispatch
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.providers.fake import FakeGPUProvider
from media_engine.studio.dev_engine import DevImageEngine
from media_engine.studio.present import present_job

ROOT = Path(__file__).resolve().parents[1]


class StudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = ManualClock()
        self.servers: list[LocalAPIServer] = []

    def tearDown(self) -> None:
        for server in self.servers:
            server.stop()

    def _controller(self) -> MediaController:
        root = Path(self.tmp.name) / "slot"
        root.mkdir(exist_ok=True)
        return MediaController(
            str(root / "meta.sqlite3"),
            str(root / "artifacts"),
            clock=self.clock,
            provider=FakeGPUProvider(self.clock.now),
            image_engine=DevImageEngine(),
        )

    def _call(self, controller: MediaController, method: str, path: str, payload: dict | None = None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        return dispatch(controller, method, path, body)

    def test_project_scene_and_independent_image_regeneration(self) -> None:
        controller = self._controller()
        server = LocalAPIServer(controller, api_key="studio-test")
        self.servers.append(server)
        server.start()
        status, project = self._call(controller, "POST", "/v1/projects", {"name": "Hotel campaign"})
        self.assertEqual(status, 201)
        self.assertTrue(project["id"].startswith("proj_"))
        status, scene = self._call(controller, "POST", f"/v1/projects/{project['id']}/scenes", {"name": "Terrace"})
        self.assertEqual(status, 201)
        kinds = [asset["type"] for asset in scene["assets"]]
        self.assertEqual(kinds, ["text", "voice", "music", "image", "video"])
        image = next(asset for asset in scene["assets"] if asset["type"] == "image")
        text = next(asset for asset in scene["assets"] if asset["type"] == "text")
        status, image = self._call(controller, "PATCH", f"/v1/assets/{image['id']}", {
            "prompt": "A luxury Mediterranean hotel terrace at sunset",
        })
        self.assertEqual(status, 200)
        status, created = self._call(controller, "POST", "/v1/images/generations", {
            "prompt": "A luxury Mediterranean hotel terrace at sunset",
            "width": 1024,
            "height": 1024,
            "seed": 3,
        })
        self.assertEqual(status, 202)
        self.assertEqual(created["status"], "queued")
        status, image = self._call(controller, "PATCH", f"/v1/assets/{image['id']}", {"job_id": created["job_id"]})
        self.assertEqual(image["job_id"], created["job_id"])
        self.assertTrue(controller.wait_settled())
        status, done = self._call(controller, "GET", f"/v1/jobs/{created['job_id']}")
        self.assertEqual(done["status"], "completed")
        self.assertTrue(done["artifact_id"])
        status, png = _http(server, "GET", f"/v1/artifacts/{done['artifact_id']}", token="studio-test")
        self.assertEqual(status, 200)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        status, saved = self._call(controller, "GET", f"/v1/projects/{project['id']}")
        scene_now = saved["scenes"][0]
        image_now = next(asset for asset in scene_now["assets"] if asset["type"] == "image")
        text_now = next(asset for asset in scene_now["assets"] if asset["type"] == "text")
        self.assertEqual(image_now["artifact_id"], done["artifact_id"])
        self.assertEqual(image_now["phase"], "completed")
        self.assertEqual(image_now["title"], "Complete")
        self.assertEqual(text_now["job_id"], None)
        self.assertEqual(text_now["status"], "draft")
        self.assertEqual(text_now["prompt"], "")
        status, second = self._call(controller, "POST", "/v1/images/generations", {
            "prompt": "A luxury Mediterranean hotel terrace at sunset",
            "width": 1024,
            "height": 1024,
            "seed": 7,
        })
        self._call(controller, "PATCH", f"/v1/assets/{image['id']}", {"job_id": second["job_id"]})
        self.assertTrue(controller.wait_settled())
        status, again = self._call(controller, "GET", f"/v1/projects/{project['id']}")
        image_again = next(asset for asset in again["scenes"][0]["assets"] if asset["type"] == "image")
        text_again = next(asset for asset in again["scenes"][0]["assets"] if asset["type"] == "text")
        self.assertEqual(image_again["history"][0]["job_id"], created["job_id"])
        self.assertEqual(image_again["job_id"], second["job_id"])
        self.assertNotEqual(image_again["artifact_id"], done["artifact_id"])
        self.assertEqual(text_again, text_now)
        status, other = self._call(controller, "POST", f"/v1/projects/{project['id']}/scenes", {"name": "Lobby"})
        lobby = next(asset for asset in other["assets"] if asset["type"] == "image")
        self.assertIsNone(lobby["job_id"])
        self.assertIsNone(lobby["artifact_id"])

    def test_waiting_capacity_is_not_an_error_and_failures_stay_plain(self) -> None:
        waiting = present_job({"status": "failed", "error_code": "PROVIDER_CAPACITY_UNAVAILABLE", "progress": 0.1})
        self.assertEqual(waiting["phase"], "waiting_capacity")
        self.assertEqual(waiting["title"], "Waiting for available GPU capacity")
        self.assertEqual(waiting["tone"], "waiting")
        failed = present_job({"status": "failed", "error_code": "GENERATION_FAILED", "progress": 0.8})
        self.assertEqual(failed["tone"], "error")
        self.assertNotIn("Vast", failed["title"])
        page = (ROOT / "src/media_engine/studio/static/studio.js").read_text(encoding="utf-8")
        html = (ROOT / "src/media_engine/studio/static/index.html").read_text(encoding="utf-8")
        self.assertIn("Waiting for available GPU capacity", page)
        self.assertIn("PROVIDER_CAPACITY_UNAVAILABLE", page)
        self.assertIn('waiting_capacity: ["Waiting for available GPU capacity", "waiting"]', page)
        self.assertIn("Create Image", html)
        self.assertNotIn("machine_id", page)
        self.assertNotIn("volume", page.lower())

    def test_studio_shell_uses_a_server_cookie(self) -> None:
        controller = self._controller()
        server = LocalAPIServer(controller, api_key="")
        self.servers.append(server)
        server.start()
        self.assertIsNone(authorize("/health", {}, ""))
        self.assertEqual(authorize("/v1/projects", {}, "")[0], 401)
        request = urllib.request.Request(server.url + "/")
        with urllib.request.urlopen(request, timeout=2) as response:
            html = response.read().decode("utf-8")
            cookie = response.headers.get("Set-Cookie", "")
        self.assertIn("AI-MEDIA-ENGINE", html)
        self.assertIn("Quick Create", html)
        token = cookie.split(";", 1)[0]
        self.assertTrue(token.startswith("studio_session="))
        status, body = _http(server, "GET", "/studio.js")
        self.assertEqual(status, 200)
        self.assertIn("presentJob", body if isinstance(body, str) else "")
        denied, payload = _http(server, "POST", "/v1/projects", {"name": "No session"})
        self.assertEqual(denied, 401)
        status, created = _http(server, "POST", "/v1/projects", {"name": "With session"}, cookie=token)
        self.assertEqual(status, 201)
        self.assertEqual(created["name"], "With session")
        self.assertEqual(payload["error"], "UNAUTHORIZED")


def _http(server: LocalAPIServer, method: str, path: str, payload: dict | None = None,
          token: str = "", cookie: str = ""):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(server.url + path, data=data, method=method)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    if cookie:
        request.add_header("Cookie", cookie)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=2) as response:
            content = response.headers.get("Content-Type", "")
            raw = response.read()
            if content.startswith("image/"):
                return response.status, raw
            if content.startswith("text/"):
                return response.status, raw.decode("utf-8")
            return response.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        exc.close()
        try:
            return exc.code, json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return exc.code, raw.decode("utf-8", "replace")
