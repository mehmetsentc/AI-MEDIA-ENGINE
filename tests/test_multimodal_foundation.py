"""Offline multimodal foundation. No weights are downloaded and no GPU is rented."""
from __future__ import annotations

import tempfile
import unittest
import urllib.request
from decimal import Decimal
from pathlib import Path

from media_engine.api.app import LocalAPIServer
from media_engine.engines.binding import engine_id_for
from media_engine.jobs.model import IMAGE_GENERATE, SUPPORTED_JOB_TYPES, UnsupportedJobType
from media_engine.models.adapters import AdapterError, adapter_for
from media_engine.models.cost import estimate
from media_engine.models.lifecycle import plan_load, recover_worker
from media_engine.models.registry import CATALOG, READINESS, primary_for
from media_engine.models.schema import catalog, submit_capability
from media_engine.models.uploads import UploadError, UploadStore
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.platform.service import Platform
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.vast_watchdog import HEARTBEAT_STALE_SECONDS
from media_engine.studio.account import SessionBook
from media_engine.studio.dev_engine import DevImageEngine
from tests.test_studio import _http


class _VastStub:
    name = "vast"
    execution_authorization = None
    create_calls = 0

    def create(self, *_args: object, **_kwargs: object) -> None:
        self.create_calls += 1


class MultimodalFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = ManualClock()
        self.servers: list[LocalAPIServer] = []
        self.addCleanup(self._stop)

    def _stop(self) -> None:
        for server in self.servers:
            server.stop()

    def _controller(self) -> MediaController:
        root = Path(self.tmp.name) / "slot"
        root.mkdir(exist_ok=True)
        controller = MediaController(
            str(root / "jobs.sqlite3"),
            str(root / "artifacts"),
            clock=self.clock,
            provider=FakeGPUProvider(self.clock.now),
            image_engine=DevImageEngine(),
        )
        controller.platform = Platform(str(root / "platform.sqlite3"))
        controller.platform.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=0, effective_at="2026-01-01T00:00:00Z",
        )
        return controller

    def test_registry_pins_reviewed_models_and_blocks_noncommercial_weights(self) -> None:
        self.assertNotIn("PRODUCTION_READY", {spec.readiness for spec in CATALOG})
        image = primary_for("image.generate")
        self.assertEqual(image.model_id, "Qwen/Qwen-Image")
        self.assertEqual(image.revision, "75e0b4be04f60ec59a75f475837eced720f823b6")
        self.assertEqual(image.license_name, "apache-2.0")
        self.assertEqual(image.readiness, "GPU_VERIFIED")
        self.assertTrue(image.executable)
        self.assertEqual(primary_for("audio.tts").model_id, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        self.assertEqual(primary_for("audio.tts").readiness, "OFFLINE_TESTED")
        self.assertFalse(primary_for("audio.tts").executable)
        blocked = {spec.model_id for spec in CATALOG if spec.commercial_use == "blocked"}
        self.assertIn("SWivid/F5-TTS", blocked)
        self.assertIn("facebook/nllb-200-distilled-600M", blocked)
        self.assertIn("facebook/musicgen-medium", blocked)
        self.assertIn("Lightricks/LTX-Video", blocked)
        self.assertIsNone(primary_for("video.image_to_video"))
        self.assertTrue(READINESS.issuperset({spec.readiness for spec in CATALOG}))

    def test_unverified_kinds_do_not_enter_the_image_job_runner(self) -> None:
        self.assertEqual(SUPPORTED_JOB_TYPES, frozenset({IMAGE_GENERATE, "TEXT_GENERATE"}))
        with self.assertRaises(UnsupportedJobType):
            engine_id_for("audio.tts")
        controller = self._controller()
        book = SessionBook()
        status, body = submit_capability(controller, "nahaber", "audio.tts", {
            "text": "A short line", "language": "English", "voice": "Ryan",
        }, book)
        self.assertEqual(status, 409)
        self.assertEqual(body["error"], "CAPABILITY_UNAVAILABLE")
        self.assertEqual(controller.jobs.list(), [])
        self.assertEqual(controller.provider.create_calls, 0)
        with self.assertRaises(AdapterError) as refused:
            adapter_for("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice").run({})
        self.assertEqual(refused.exception.code, "CAPABILITY_UNAVAILABLE")

    def test_voice_clone_requires_owner_consent_and_still_does_not_generate(self) -> None:
        controller = self._controller()
        book = SessionBook()
        status, missing = submit_capability(controller, "nahaber", "audio.voice_clone", {
            "text": "Hello from the owner", "language": "English", "consent_id": "consent_missing",
        }, book)
        self.assertEqual(status, 403)
        self.assertEqual(missing["error"], "CONSENT_REQUIRED")
        from media_engine.models.schema import record_consent
        status, consent = record_consent(controller, "nahaber", {
            "speaker_name": "Owner",
            "statement": "I own this voice and authorize this studio account to clone it.",
            "subject_is_owner": True,
            "third_party_impersonation": False,
        })
        self.assertEqual(status, 201)
        status, impersonation = record_consent(controller, "nahaber", {
            "speaker_name": "Someone else",
            "statement": "Clone a celebrity voice without permission today.",
            "subject_is_owner": True,
            "third_party_impersonation": True,
        })
        self.assertEqual(impersonation["error"], "IMPERSONATION_REFUSED")
        status, hidden = submit_capability(controller, "film-studio", "audio.voice_clone", {
            "text": "Hello from the owner", "language": "English", "consent_id": consent["consent_id"],
        }, book)
        self.assertEqual(status, 403)
        status, blocked = submit_capability(controller, "nahaber", "audio.voice_clone", {
            "text": "Hello from the owner", "language": "English", "consent_id": consent["consent_id"],
        }, book)
        self.assertEqual(blocked["error"], "CAPABILITY_UNAVAILABLE")
        self.assertEqual(controller.jobs.list(), [])

    def test_uploads_are_private_and_size_limited(self) -> None:
        store = UploadStore(Path(self.tmp.name) / "uploads")
        saved = store.accept(
            "nahaber", filename="../clip.wav", header=b"RIFF....WAVE", declared_bytes=1200,
            kind="audio", duration_seconds=3,
        )
        self.assertEqual(saved["filename"], "clip.wav")
        self.assertIsNone(store.owned("film-studio", saved["upload_id"]))
        self.assertFalse(store.delete("film-studio", saved["upload_id"]))
        self.assertTrue(store.delete("nahaber", saved["upload_id"]))
        with self.assertRaises(UploadError) as too_big:
            store.accept("nahaber", filename="big.wav", header=b"RIFF", declared_bytes=26 * 1024 * 1024, kind="audio", duration_seconds=1)
        self.assertEqual(too_big.exception.code, "UPLOAD_TOO_LARGE")
        with self.assertRaises(UploadError):
            store.accept("nahaber", filename="notes.txt", header=b"hello", declared_bytes=5, kind="audio", duration_seconds=1)

    def test_image_price_stays_fixed_and_video_is_estimated_separately(self) -> None:
        image = estimate("qwen-image-1024", hourly_usd=Decimal("10"), image_price=2)
        self.assertEqual(image["credits"], 2)
        self.assertEqual(image["priced_by"], "existing_image_price")
        video = estimate("wan-t2v-5s", hourly_usd=Decimal("0.66"), image_price=2)
        self.assertEqual(video["priced_by"], "gpu_time")
        self.assertNotEqual(video["credits"], 2)
        self.assertIsNone(estimate("unpriced", hourly_usd=Decimal("0.66"))["credits"])

    def test_gpu_plan_reuses_one_worker_and_keeps_the_watchdog(self) -> None:
        image = primary_for("image.generate")
        disk = image.storage_bytes + (8 * 1024 * 1024 * 1024)
        self.assertEqual(plan_load(
            image, resident=None, free_disk_bytes=disk, active_workers=0, execution_authorized=False,
        )["error"], "AUTHORIZATION_REQUIRED")
        self.assertEqual(plan_load(
            image, resident=None, free_disk_bytes=10, active_workers=0, execution_authorized=True,
        )["error"], "DISK_CAPACITY")
        self.assertEqual(plan_load(
            image, resident=None, free_disk_bytes=disk, active_workers=2, execution_authorized=True,
        )["error"], "SECOND_GPU_REFUSED")
        loaded = plan_load(
            image, resident=image.model_id, free_disk_bytes=disk, active_workers=1, execution_authorized=True,
        )
        self.assertEqual(loaded["action"], "REUSE")
        waiting = plan_load(
            image, resident="other-model", free_disk_bytes=disk, active_workers=1,
            execution_authorized=True, busy=True,
        )
        self.assertEqual(waiting["error"], "MODEL_BUSY")
        voice = primary_for("audio.tts")
        self.assertEqual(plan_load(
            voice, resident=None, free_disk_bytes=disk, active_workers=0, execution_authorized=True,
        )["error"], "CAPABILITY_UNAVAILABLE")
        recovery = recover_worker(image.model_id)
        self.assertEqual(recovery["stale_after_seconds"], HEARTBEAT_STALE_SECONDS)
        self.assertEqual(recovery["action"], "UNLOAD_AND_DELETE")
        self.assertEqual(HEARTBEAT_STALE_SECONDS, 300.0)

    def test_image_generation_still_refuses_without_credits_or_authorization(self) -> None:
        controller = self._controller()
        book = SessionBook()
        status, poor = submit_capability(controller, "nahaber", "image.generate", {
            "prompt": "A quiet courtyard", "idempotency_key": "studio-key-11",
        }, book)
        self.assertEqual(status, 402)
        self.assertEqual(poor["error"], "INSUFFICIENT_CREDITS")
        self.assertEqual(controller.jobs.list(), [])
        controller.provider = _VastStub()
        controller.platform.grant("nahaber", "topup", 4, "topup_purchase", idempotency_key="local-grant")
        status, blocked = submit_capability(controller, "nahaber", "image.generate", {
            "prompt": "A quiet courtyard", "idempotency_key": "studio-key-12",
        }, book)
        self.assertEqual(blocked["error"], "AUTHORIZATION_REQUIRED")
        self.assertEqual(controller.provider.create_calls, 0)
        self.assertEqual(controller.jobs.list(), [])

    def test_adapter_does_not_invent_bytes_without_a_pipeline(self) -> None:
        with self.assertRaises(AdapterError):
            adapter_for("Qwen/Qwen-Image").run({"prompt": "A courtyard", "seed": 3})
        png = adapter_for("Qwen/Qwen-Image").run(
            {"prompt": "A courtyard", "seed": 3}, pipeline=DevImageEngine().render,
        )
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_signed_in_catalog_and_studio_forms_stay_closed(self) -> None:
        controller = self._controller()
        server = LocalAPIServer(controller, api_key="studio-secret")
        self.servers.append(server)
        server.start()
        request = urllib.request.Request(server.url + "/")
        with urllib.request.urlopen(request, timeout=2) as response:
            cookie = response.headers.get("Set-Cookie", "").split(";", 1)[0]
        _http(server, "POST", "/v1/session", {"client_id": "nahaber"}, cookie=cookie)
        status, body = _http(server, "GET", "/v1/capabilities", cookie=cookie)
        self.assertEqual(status, 200)
        kinds = {item["kind"]: item for item in body["capabilities"]}
        self.assertTrue(kinds["image.generate"]["executable"])
        self.assertFalse(kinds["audio.tts"]["executable"])
        self.assertFalse(kinds["video.text_to_video"]["executable"])
        self.assertFalse(body["pipelines"][0]["executable"])
        status, denied = _http(server, "POST", "/v1/capabilities/audio.tts", {
            "text": "A short line", "language": "English", "voice": "Ryan",
        }, cookie=cookie)
        self.assertEqual(status, 409)
        self.assertEqual(denied["error"], "CAPABILITY_UNAVAILABLE")
        self.assertEqual(controller.jobs.list(), [])
        page = Path("src/media_engine/studio/static/index.html").read_text(encoding="utf-8")
        script = Path("src/media_engine/studio/static/studio.js").read_text(encoding="utf-8")
        self.assertIn("disabled>Video", page)
        self.assertIn("disabled>Audio", page)
        self.assertIn("disabled>Voice", page)
        self.assertIn("Generate Image", page)
        self.assertIn("Record voice authorization", page)
        self.assertIn("Unavailable until the model is GPU-verified", page)
        self.assertIn("Generate video", page)
        self.assertIn("Generate music", page)
        self.assertNotIn("machine_id", script)
        self.assertNotIn("volume", script.lower())


if __name__ == "__main__":
    unittest.main()
