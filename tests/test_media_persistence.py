"""Live image bytes go to the private media bucket. Development stays local."""
from __future__ import annotations

import os
import tempfile
import unittest
import urllib.request
from pathlib import Path

from media_engine.api.app import LocalAPIServer
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.platform.pool import production_database_status
from media_engine.platform.repository import PlatformBlocked
from media_engine.platform.service import MemoryMedia, Platform
from media_engine.platform.test_grant import prepare_test_grant
from media_engine.providers.base import WorkerStageError
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.ssh_surface import POST_READINESS_SSH, REMAINING_SSH_AFTER_READY
from media_engine.providers.vast_image import VastImageProvider
from media_engine.storage.r2_media import MODEL_BUCKET, MEDIA_BUCKET, MediaStorageError, media_config
from media_engine.studio.store import StudioStore
from tests.test_phase2e import PNG, PngEngine


class MediaPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = ManualClock()
        self.servers: list[LocalAPIServer] = []

    def tearDown(self) -> None:
        for server in self.servers:
            server.stop()
        os.environ.pop("MEDIA_ENGINE_ALLOW_TEST_CREDITS", None)

    def test_live_image_uses_the_media_bucket_and_studio_can_read_it(self) -> None:
        controller, platform = self._live()
        store = StudioStore(controller.jobs.db_path)
        project = store.create_project("Film", controller.clock.iso())
        scene = store.create_scene(project["id"], "Opening", controller.clock.iso())
        asset = next(item for item in scene["assets"] if item["type"] == "image")
        job = controller.submit_image("a terrace at sunset")
        store.update_asset(asset["id"], controller.clock.iso(), job_id=job.id)
        controller.process_available()
        done = controller.get_job(job.id)
        self.assertEqual(done.public_status, "completed")
        self.assertTrue(str(done.result_uri).startswith("r2://" + MEDIA_BUCKET + "/users/nahaber/projects/"))
        self.assertIn("/scenes/" + scene["id"] + "/images/", done.result_uri)
        self.assertNotIn(MODEL_BUCKET, done.result_uri)
        self.assertNotIn("hf-cache", done.result_uri)
        self.assertEqual(list(Path(controller.storage.root).rglob("*.png")), [])
        self.assertEqual(controller.artifact_bytes(done.artifact_id), PNG)
        linked = store.get_asset(asset["id"])
        self.assertEqual(linked["artifact_id"], done.artifact_id)
        self.assertEqual(platform.balance("nahaber"), 3)
        platform.settle("nahaber", job.id, 2)
        platform.settle_failure("nahaber", job.id, legitimate_cost=2)
        self.assertEqual(platform.balance("nahaber"), 3)
        server = LocalAPIServer(controller, api_key="local-test")
        self.servers.append(server)
        server.start()
        status, body = _get(server.url + "/v1/artifacts/" + done.artifact_id, "local-test")
        self.assertEqual(status, 200)
        self.assertEqual(body[:8], b"\x89PNG\r\n\x1a\n")
        self.assertNotIn(b"R2_SECRET", body)
        self.assertNotIn(b"R2_ACCESS_KEY", body)
        token = platform.issue_signed_url(done.result_uri.split("/", 3)[-1])
        self.assertNotIn("R2_", token)
        self.assertTrue(platform.resolve_signed_url(token).startswith("users/nahaber/"))

    def test_development_image_stays_on_local_storage(self) -> None:
        controller = self._controller()
        job = controller.submit_image("local preview")
        controller.process_available()
        done = controller.get_job(job.id)
        self.assertEqual(done.public_status, "completed")
        self.assertFalse(str(done.result_uri).startswith("r2://"))
        self.assertTrue(Path(controller.artifacts.get(done.artifact_id).path).is_file())

    def test_upload_failure_does_not_complete_and_metadata_failure_is_recoverable(self) -> None:
        controller, platform = self._live()
        platform.media.fail_next = True
        failed = controller.submit_image("upload fails")
        controller.process_available()
        self.assertEqual(controller.get_job(failed.id).public_status, "failed")
        self.assertNotEqual(controller.get_job(failed.id).public_status, "completed")
        self.assertEqual(platform.media.objects, {})
        self.assertEqual(platform.balance("nahaber"), 5)
        platform.repo.insert_artifact = _reject_metadata
        orphaned = controller.submit_image("metadata fails")
        controller.process_available()
        self.assertEqual(controller.get_job(orphaned.id).error_code, "ARTIFACT_METADATA_FAILED")
        self.assertEqual(len(platform.media.objects), 1)
        key = next(iter(platform.media.objects))
        self.assertEqual(platform.reconcile(key), "reconciled")
        self.assertEqual(platform.balance("nahaber"), 5)

    def test_test_grant_is_idempotent_and_off_by_default(self) -> None:
        platform = Platform(str(Path(self.tmp.name) / "grant.sqlite3"))
        with self.assertRaises(PlatformBlocked) as blocked:
            prepare_test_grant(
                platform, user_id="nahaber", credits=4, price_credits=2, idempotency_key="final-image-1",
            )
        self.assertEqual(blocked.exception.code, "TEST_CREDITS_DISABLED")
        os.environ["MEDIA_ENGINE_ALLOW_TEST_CREDITS"] = "1"
        first = prepare_test_grant(
            platform, user_id="nahaber", credits=4, price_credits=2, idempotency_key="final-image-1",
        )
        prepare_test_grant(
            platform, user_id="nahaber", credits=4, price_credits=2, idempotency_key="final-image-1",
        )
        self.assertEqual(first, "test-final-image-1")
        self.assertEqual(platform.balance("nahaber"), 4)
        import sys
        from media_engine.platform.test_grant import main
        previous = sys.argv
        sys.argv = [
            "test_grant", "--db", "postgres://production/media", "--user", "nahaber",
            "--credits", "1", "--price-credits", "1", "--key", "final-image-1",
        ]
        try:
            with self.assertRaises(SystemExit) as caught:
                main()
        finally:
            sys.argv = previous
        self.assertEqual(str(caught.exception), "TEST_GRANT_REFUSES_PRODUCTION_DSN")

    def test_production_database_does_not_fall_back_to_sqlite(self) -> None:
        path = Path(self.tmp.name) / "silent.sqlite3"
        self.assertEqual(production_database_status(""), "POSTGRES_CONFIGURATION_REQUIRED")
        with self.assertRaises(RuntimeError) as caught:
            production_database_status("sqlite:///" + str(path))
        self.assertEqual(str(caught.exception), "POSTGRES_CONFIGURATION_REQUIRED")
        self.assertFalse(path.exists())

    def test_media_bucket_is_separate_and_post_ready_ssh_can_attach(self) -> None:
        with self.assertRaises(MediaStorageError):
            media_config({
                "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
                "R2_BUCKET": MODEL_BUCKET,
                "MEDIA_R2_BUCKET": MODEL_BUCKET,
                "R2_ACCESS_KEY_ID": "access",
                "R2_SECRET_ACCESS_KEY": "secret",
            })
        config = media_config({
            "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
            "R2_BUCKET": MODEL_BUCKET,
            "MEDIA_R2_BUCKET": MEDIA_BUCKET,
            "R2_ACCESS_KEY_ID": "access",
            "R2_SECRET_ACCESS_KEY": "secret",
        })
        self.assertEqual(config.bucket, MEDIA_BUCKET)
        self.assertNotIn("secret", repr(config))
        calls = []

        def transport(method, url, body, headers):
            calls.append(url)
            return 200, b'{"instances":[]}'

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, ssh_run=None,
            sleep=lambda _seconds: None, now=lambda: 1_000.0,
        )
        provider._resource_id = "9"
        provider.cache_source = "warm_cache"
        with self.assertRaises(WorkerStageError):
            provider.confirm_cache("9")
        self.assertTrue(calls)
        self.assertEqual(POST_READINESS_SSH, "SAFE_FOR_ONE_FINAL_TEST")
        self.assertIn("generate", REMAINING_SSH_AFTER_READY)
        self.assertIn("fetch_png", REMAINING_SSH_AFTER_READY)

    def _live(self) -> tuple[MediaController, Platform]:
        controller = self._controller()
        platform = Platform(str(Path(self.tmp.name) / "platform.sqlite3"), media=MemoryMedia())
        platform.add_price(
            version="test-rule", service="image", model="qwen-image", quality="test",
            unit="image", credit_price=2, internal_cost_units=1, effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 5, "topup_purchase", idempotency_key="live-grant")
        controller.platform = platform
        return controller, platform

    def _controller(self) -> MediaController:
        root = Path(self.tmp.name) / "jobs"
        root.mkdir(exist_ok=True)
        return MediaController(
            str(root / "meta.sqlite3"),
            str(root / "artifacts"),
            clock=self.clock,
            provider=FakeGPUProvider(self.clock.now),
            image_engine=PngEngine(),
        )


def _reject_metadata(_row, _now) -> None:
    raise PlatformBlocked("ARTIFACT_METADATA_FAILED")


def _get(url: str, token: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, method="GET")
    request.add_header("Authorization", "Bearer " + token)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=2) as response:
        return response.status, response.read()
