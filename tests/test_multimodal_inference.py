"""Offline multimodal inference. Runtimes are injected. No weights and no GPU rental."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from media_engine.models.audio import encode_wav
from media_engine.models.cost import CREDIT_RULES
from media_engine.models.dependencies import dependency_status
from media_engine.models.errors import AdapterError
from media_engine.models.invoke import TRANSLATION_QUALITY_VERIFIED, evaluate_translation, Invocation
from media_engine.models.registry import primary_for
from media_engine.models.runtimes import load_wan, load_whisper
from media_engine.models.schema import record_consent, submit_capability
from media_engine.models.uploads import UploadStore
from media_engine.models.video import write_mp4
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.platform.service import Platform
from media_engine.providers.fake import FakeGPUProvider
from media_engine.studio.account import SessionBook, artifact_owner, history_for
from media_engine.studio.dev_engine import DevImageEngine


class _Word:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class _Segment:
    def __init__(self):
        self.start = 0.0
        self.end = 1.2
        self.text = " hello courtyard"
        self.words = [_Word("hello", 0.0, 0.4), _Word("courtyard", 0.4, 1.2)]


class _Info:
    language = "en"
    language_probability = 0.94


class WhisperRuntime:
    def __init__(self, controller=None, fail=False):
        self.controller = controller
        self.fail = fail

    def transcribe(self, audio, word_timestamps=False):
        if self.fail:
            raise RuntimeError("decoder failed")
        if self.controller is not None:
            self.controller.multimodal_cancel.set()
        if not word_timestamps:
            raise AssertionError("word timestamps")
        return [_Segment()], _Info()


class VoiceRuntime:
    def generate_custom_voice(self, *, text, language, speaker):
        if not text or language != "English" or speaker != "Ryan":
            raise RuntimeError("bad voice request")
        return [0.0, 0.2, -0.2, 0.1], 24000

    def generate_voice_clone(self, *, text, language, ref_audio, ref_text):
        if not str(ref_audio).endswith(".wav"):
            raise RuntimeError("reference path")
        return [0.1, -0.1, 0.1, -0.1], 24000


class TranslatorRuntime:
    def translate(self, messages):
        content = messages[0]["content"]
        if "Hello" in content:
            return '{"translation": "你好"}'
        return '{"translation": "translated line"}'


class EditRuntime:
    def __call__(self, *, image, prompt):
        return DevImageEngine().render(prompt)


class MusicRuntime:
    def generate(self, *, prompt, duration_seconds, style):
        if not prompt or not style:
            raise RuntimeError("music input")
        return [0.05] * (24000 * duration_seconds), 24000


class VideoRuntime:
    def __call__(self, *, prompt, width, height, num_frames):
        if width != 832 or height != 480 or num_frames != 81:
            raise RuntimeError("frame contract")
        return write_mp4(b"encoded-frames")


class _Vast:
    name = "vast"
    execution_authorization = None
    create_calls = 0

    def create(self, *_args, **_kwargs):
        self.create_calls += 1


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = ManualClock()
        root = Path(self.tmp.name)
        self.controller = MediaController(
            str(root / "jobs.sqlite3"), str(root / "artifacts"), clock=self.clock,
            provider=FakeGPUProvider(self.clock.now), image_engine=DevImageEngine(),
        )
        self.controller.platform = Platform(str(root / "platform.sqlite3"))
        self.controller.platform.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=0, effective_at="2026-01-01T00:00:00Z",
        )
        for service, model, credits in CREDIT_RULES.values():
            self.controller.platform.add_price(
                version="mm-" + service, service=service, model=model, quality="standard",
                unit="job", credit_price=credits, internal_cost_units=0,
                effective_at="2026-01-01T00:00:00Z",
            )
        self.controller.platform.grant(
            "nahaber", "topup", 20, "topup_purchase", idempotency_key="mm-grant",
        )
        self.controller.multimodal_runtimes = {}
        self.book = SessionBook()
        self.controller.start()
        self.addCleanup(self.controller.stop)

    def test_transcription_persists_timestamps_and_refunds_failures(self) -> None:
        self.assertFalse(primary_for("audio.transcribe").executable)
        self.assertEqual(primary_for("audio.transcribe").readiness, "OFFLINE_TESTED")
        self.controller.multimodal_runtimes["audio.transcribe"] = WhisperRuntime()
        upload = self._wav("nahaber")
        status, created = self._submit("audio.transcribe", {
            "upload_id": upload, "idempotency_key": "transcribe-01",
        })
        self.assertEqual(status, 202)
        self.assertTrue(self.controller.wait_settled())
        job = self.controller.get_job(created["job_id"])
        self.assertEqual(job.public_status, "completed")
        data = json.loads(self.controller.artifact_bytes(job.artifact_id).decode("utf-8"))
        self.assertEqual(data["language"], "en")
        self.assertEqual(data["segments"][0]["words"][0]["word"], "hello")
        self.assertIn("hello courtyard", data["srt"])
        self.assertTrue(data["vtt"].startswith("WEBVTT"))
        self.assertEqual(artifact_owner(self.controller, job.artifact_id), "nahaber")
        self.assertIn("users/nahaber/", job.result_uri)
        self.assertEqual(self.controller.platform.balance("nahaber"), 19)
        self.assertEqual(self.controller.provider.create_calls, 0)
        self.controller.multimodal_runtimes["audio.transcribe"] = WhisperRuntime(fail=True)
        before = self.controller.platform.balance("nahaber")
        status, failed = self._submit("audio.transcribe", {
            "upload_id": upload, "idempotency_key": "transcribe-02",
        })
        self.assertEqual(status, 202)
        self.assertTrue(self.controller.wait_settled())
        done = self.controller.get_job(failed["job_id"])
        self.assertEqual(done.public_status, "failed")
        self.assertEqual(self.controller.platform.balance("nahaber"), before)
        rows = history_for(self.controller, "nahaber")
        failed_row = next(item for item in rows if item["job_id"] == failed["job_id"])
        self.assertEqual(failed_row["credit_status"], "refunded")
        done_row = next(item for item in rows if item["job_id"] == created["job_id"])
        self.assertEqual(done_row["capability"], "audio.transcribe")

    def test_tts_and_voice_clone_require_a_runtime_and_consent(self) -> None:
        self.controller.multimodal_runtimes["audio.tts"] = VoiceRuntime()
        self.controller.multimodal_runtimes["audio.voice_clone"] = VoiceRuntime()
        status, poor = self._submit("audio.tts", {
            "text": "A short line", "language": "English", "voice": "Ryan",
            "idempotency_key": "tts-poor-01",
        }, user="film-studio")
        self.assertEqual(status, 402)
        self.assertEqual(poor["error"], "INSUFFICIENT_CREDITS")
        self.assertEqual(self.controller.jobs.list(), [])
        status, bad_speaker = self._submit("audio.tts", {
            "text": "A short line", "language": "English", "voice": "Celebrity",
            "idempotency_key": "tts-bad-01",
        })
        self.assertEqual(bad_speaker["error"], "SPEAKER_UNSUPPORTED")
        status, spoken = self._submit("audio.tts", {
            "text": "A short line", "language": "English", "voice": "Ryan",
            "idempotency_key": "tts-ok-01",
        })
        self.assertEqual(status, 202)
        self.assertTrue(self.controller.wait_settled())
        job = self.controller.get_job(spoken["job_id"])
        wav = self.controller.artifact_bytes(job.artifact_id)
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav[:16])
        status, missing = self._submit("audio.voice_clone", {
            "text": "Hello from the owner", "language": "English", "consent_id": "consent_missing",
            "idempotency_key": "clone-missing",
        })
        self.assertEqual(missing["error"], "CONSENT_REQUIRED")
        status, consent = record_consent(self.controller, "nahaber", {
            "speaker_name": "Owner",
            "statement": "I own this voice and authorize this studio account to clone it.",
            "subject_is_owner": True,
            "third_party_impersonation": False,
        })
        reference = self._wav("nahaber", seconds=2)
        status, cloned = self._submit("audio.voice_clone", {
            "text": "Hello from the owner", "language": "English",
            "consent_id": consent["consent_id"], "reference_upload_id": reference,
            "reference_text": "reference line", "idempotency_key": "clone-ok-01",
        })
        self.assertEqual(status, 202)
        self.assertTrue(self.controller.wait_settled())
        clone_job = self.controller.get_job(cloned["job_id"])
        self.assertEqual(clone_job.public_status, "completed")
        self.assertTrue(self.controller.artifact_bytes(clone_job.artifact_id).startswith(b"RIFF"))
        status, stolen = self._submit("audio.voice_clone", {
            "text": "Hello from the owner", "language": "English",
            "consent_id": consent["consent_id"], "reference_upload_id": reference,
            "idempotency_key": "clone-other",
        }, user="film-studio")
        self.assertEqual(stolen["error"], "CONSENT_REQUIRED")
        self.assertEqual(self.controller.provider.create_calls, 0)

    def test_translation_fixtures_keep_subtitle_timing_and_are_not_gpu_verified(self) -> None:
        self.assertFalse(TRANSLATION_QUALITY_VERIFIED)
        self.assertNotEqual(primary_for("audio.translate").readiness, "GPU_VERIFIED")
        runtime = TranslatorRuntime()
        report = evaluate_translation(runtime, Invocation(10**12, None))
        self.assertFalse(report["gpu_verified"])
        self.assertTrue(report["results"][0]["passed"])
        self.controller.multimodal_runtimes["audio.translate"] = runtime
        srt = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"
        status, created = self._submit("audio.translate", {
            "text": srt, "source_language": "English", "target_language": "Chinese",
            "idempotency_key": "translate-01",
        })
        self.assertEqual(status, 202)
        self.assertTrue(self.controller.wait_settled())
        job = self.controller.get_job(created["job_id"])
        body = json.loads(self.controller.artifact_bytes(job.artifact_id).decode("utf-8"))
        self.assertIn("00:00:01,000 --> 00:00:02,000", body["translation"])
        self.assertIn("你好", body["translation"])
        self.assertFalse(body["quality_verified"])

    def test_image_edit_is_private_and_will_not_share_the_qwen_image_model(self) -> None:
        self.controller.multimodal_runtimes["image.edit"] = EditRuntime()
        png = DevImageEngine().render("source")
        store = self.controller.upload_store if hasattr(self.controller, "upload_store") else None
        from media_engine.models.schema import _upload_store
        store = _upload_store(self.controller)
        saved = store.accept(
            "nahaber", filename="source.png", header=png[:16], declared_bytes=len(png),
            kind="image", body=png,
        )
        self.controller.model_gate = {"resident": "Qwen/Qwen-Image"}
        status, busy = self._submit("image.edit", {
            "prompt": "Make the sky brighter", "source_upload_id": saved["upload_id"],
            "idempotency_key": "edit-busy-01",
        })
        self.assertEqual(status, 202)
        self.assertTrue(self.controller.wait_settled())
        self.assertEqual(self.controller.get_job(busy["job_id"]).error_code, "MODEL_BUSY")
        self.assertEqual(self.controller.model_gate["resident"], "Qwen/Qwen-Image")
        self.controller.model_gate = {"resident": None}
        status, edited = self._submit("image.edit", {
            "prompt": "Make the sky brighter", "source_upload_id": saved["upload_id"],
            "idempotency_key": "edit-ok-01",
        })
        self.assertTrue(self.controller.wait_settled())
        job = self.controller.get_job(edited["job_id"])
        self.assertTrue(self.controller.artifact_bytes(job.artifact_id).startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIsNone(store.owned("film-studio", saved["upload_id"]))
        self.assertEqual(self.controller.provider.create_calls, 0)

    def test_music_and_video_validate_output_and_a_timeout_refunds(self) -> None:
        self.controller.multimodal_runtimes["music.generate"] = MusicRuntime()
        self.controller.multimodal_runtimes["video.text_to_video"] = VideoRuntime()
        status, music = self._submit("music.generate", {
            "prompt": "A quiet piano", "style": "ambient", "duration_seconds": 1,
            "idempotency_key": "music-01",
        })
        self.assertEqual(status, 202)
        self.assertTrue(self.controller.wait_settled())
        track = self.controller.artifact_bytes(self.controller.get_job(music["job_id"]).artifact_id)
        self.assertTrue(track.startswith(b"RIFF"))
        status, rejected = self._submit("video.text_to_video", {
            "prompt": "A courtyard at dusk", "aspect_ratio": "4:3", "duration_seconds": 5,
            "idempotency_key": "video-aspect",
        })
        self.assertEqual(rejected["error"], "ASPECT_UNSUPPORTED")
        before = self.controller.platform.balance("nahaber")
        self.controller.multimodal_timeout = 0
        status, timed = self._submit("video.text_to_video", {
            "prompt": "A courtyard at dusk", "aspect_ratio": "16:9", "duration_seconds": 5,
            "idempotency_key": "video-timeout",
        })
        self.assertEqual(status, 202)
        self.assertTrue(self.controller.wait_settled())
        self.assertEqual(self.controller.get_job(timed["job_id"]).error_code, "TIMEOUT")
        self.assertEqual(self.controller.platform.balance("nahaber"), before)
        self.controller.multimodal_timeout = 30
        status, video = self._submit("video.text_to_video", {
            "prompt": "A courtyard at dusk", "aspect_ratio": "16:9", "duration_seconds": 5,
            "idempotency_key": "video-ok",
        })
        self.assertTrue(self.controller.wait_settled())
        movie = self.controller.artifact_bytes(self.controller.get_job(video["job_id"]).artifact_id)
        self.assertEqual(movie[4:8], b"ftyp")
        self.assertIsNone(primary_for("video.image_to_video"))
        self.assertEqual(self.controller.provider.create_calls, 0)

    def test_cancellation_refunds_and_vast_cannot_start_the_local_path(self) -> None:
        self.controller.multimodal_runtimes["audio.transcribe"] = WhisperRuntime(self.controller)
        upload = self._wav("nahaber")
        before = self.controller.platform.balance("nahaber")
        status, cancelled = self._submit("audio.transcribe", {
            "upload_id": upload, "idempotency_key": "cancel-01",
        })
        self.assertTrue(self.controller.wait_settled())
        self.assertEqual(self.controller.get_job(cancelled["job_id"]).error_code, "CANCELLED")
        self.assertEqual(self.controller.platform.balance("nahaber"), before)
        self.controller.provider = _Vast()
        self.controller.multimodal_runtimes["audio.tts"] = VoiceRuntime()
        status, blocked = self._submit("audio.tts", {
            "text": "A short line", "language": "English", "voice": "Ryan",
            "idempotency_key": "vast-tts",
        })
        self.assertEqual(status, 409)
        self.assertEqual(blocked["error"], "CAPABILITY_UNAVAILABLE")
        self.assertEqual(self.controller.provider.create_calls, 0)
        self.assertEqual(self.controller.platform.image_price(), 2)

    def test_loaders_refuse_hub_ids_and_missing_weights(self) -> None:
        with self.assertRaises(AdapterError) as missing:
            load_whisper("Systran/faster-whisper-large-v3")
        self.assertEqual(missing.exception.code, "WEIGHTS_NOT_PRESENT")
        with self.assertRaises(AdapterError) as video:
            load_wan(str(Path(self.tmp.name) / "missing-wan"))
        self.assertEqual(video.exception.code, "WEIGHTS_NOT_PRESENT")
        status = dependency_status()
        self.assertIn("faster_whisper", status)
        self.assertIsInstance(status["faster_whisper"], bool)
        empty = self.controller.multimodal_runtimes
        self.controller.multimodal_runtimes = {}
        before = [job.id for job in self.controller.jobs.list()]
        status, unavailable = self._submit("music.generate", {
            "prompt": "A quiet piano", "style": "ambient", "duration_seconds": 1,
            "idempotency_key": "music-closed",
        })
        self.assertEqual(unavailable["error"], "CAPABILITY_UNAVAILABLE")
        self.assertEqual([job.id for job in self.controller.jobs.list()], before)
        self.controller.multimodal_runtimes = empty

    def _submit(self, kind, payload, user="nahaber"):
        return submit_capability(self.controller, user, kind, payload, self.book)

    def _wav(self, user: str, seconds: int = 1) -> str:
        from media_engine.models.schema import _upload_store
        data = encode_wav([0.1, -0.1] * (8000 * seconds), 16000)
        saved = _upload_store(self.controller).accept(
            user, filename="clip.wav", header=data[:16], declared_bytes=len(data),
            kind="audio", duration_seconds=seconds, body=data,
        )
        return saved["upload_id"]


if __name__ == "__main__":
    unittest.main()
