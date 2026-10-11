"""Speech cache preflight. No weight download, no GPU rental, no volume change."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT
from media_engine.infra.volume_audit import VOLUME_ID
from media_engine.models.lifecycle import SCRATCH_BYTES
from media_engine.models.registry import primary_for
from media_engine.models.speech_inventory import TARGETS
from media_engine.models.speech_preflight import SPEECH_PINS, speech_preflight
from media_engine.models.speech_stage import (
    ResidentGate,
    SpeechPrepError,
    _R2RangeSource,
    file_bytes,
    object_key,
    stage_from_r2,
    stage_tree,
    staged_disk_bytes,
)
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.session_heartbeat import HEARTBEAT_INTERVAL_SECONDS
from media_engine.providers.vast_watchdog import HEARTBEAT_STALE_SECONDS

WHISPER = "Systran/faster-whisper-large-v3"
TTS = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
ROOT = Path(__file__).resolve().parents[1]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Source:
    def __init__(self, files: dict[str, bytes], clock: _Clock | None = None) -> None:
        self.files = files
        self.clock = clock
        self.reads = 0

    def read(self, relative: str, offset: int, length: int) -> bytes:
        self.reads += 1
        if self.clock is not None:
            self.clock.now += 2
        return self.files[relative][offset:offset + length]


def _spec(body: bytes, *, path: str = "model.bin", revision: str = "a" * 40) -> dict:
    return {
        "model_id": "test/tiny",
        "revision": revision,
        "files": (
            {"path": path, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "sha_source": "content"},
        ),
    }


class SpeechPreflightTests(unittest.TestCase):
    def test_inventory_matches_the_pinned_revisions_and_is_smaller_than_hub_storage(self) -> None:
        by_id = {item["model_id"]: item for item in TARGETS}
        self.assertEqual(by_id[WHISPER]["revision"], primary_for("audio.transcribe").revision)
        self.assertEqual(by_id[TTS]["revision"], primary_for("audio.tts").revision)
        for model_id, hub in ((WHISPER, 3_145_762_253), (TTS, 4_617_449_338)):
            self.assertEqual(by_id[model_id]["hub_used_storage"], hub)
            self.assertEqual(primary_for(by_id[model_id]["capability"]).storage_bytes, hub)
            self.assertLess(file_bytes(model_id), hub)
            for item in by_id[model_id]["files"]:
                self.assertEqual(len(item["sha256"]), 64)
                self.assertNotIn("..", Path(item["path"]).parts)
        self.assertEqual(file_bytes(WHISPER), 3_090_839_273)
        self.assertEqual(file_bytes(TTS), 4_520_218_951)
        self.assertEqual(staged_disk_bytes(), file_bytes(WHISPER) + file_bytes(TTS) + SCRATCH_BYTES)
        self.assertFalse(primary_for("audio.transcribe").executable)
        self.assertFalse(primary_for("audio.tts").executable)
        self.assertTrue(primary_for("image.generate").executable)

    def test_object_keys_stay_out_of_the_image_cache(self) -> None:
        key = object_key(WHISPER, primary_for("audio.transcribe").revision, "model.bin")
        self.assertTrue(key.startswith("speech/"))
        self.assertNotIn("hf-cache/", key)
        self.assertNotIn("phase2d-cache.json", key)
        with self.assertRaises(SpeechPrepError) as raised:
            object_key(WHISPER, "0" * 40, "model.bin")
        self.assertEqual(raised.exception.code, "REVISION_MISMATCH")

    def test_stage_resumes_checks_the_checksum_and_times_the_copy(self) -> None:
        body = b"abcdefgh"
        spec = _spec(body)
        clock = _Clock()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            partial = dest / "model.bin.partial"
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(body[:4])
            report = stage_tree(
                spec, dest, _Source({"model.bin": body}, clock),
                free_disk_bytes=len(body) + SCRATCH_BYTES, clock=clock, chunk_bytes=4,
            )
            self.assertTrue(report["resumed"])
            self.assertEqual((dest / "model.bin").read_bytes(), body)
            self.assertEqual((dest / "REVISION").read_text(encoding="utf-8").strip(), spec["revision"])
            self.assertGreater(report["stage_seconds"], 0)
            again = stage_tree(
                spec, dest, _Source({"model.bin": body}),
                free_disk_bytes=0, clock=_Clock(),
            )
            self.assertTrue(again["already_complete"])
            self.assertEqual(again["stage_seconds"], 0)

    def test_corrupt_checksum_budget_stop_disk_and_volume_do_not_publish(self) -> None:
        body = b"abcdefgh"
        spec = _spec(body)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            source = _Source({"model.bin": b"xxxxefgh"})
            with self.assertRaises(SpeechPrepError) as raised:
                stage_tree(
                    spec, dest, source,
                    free_disk_bytes=len(body) + SCRATCH_BYTES, clock=_Clock(), chunk_bytes=4,
                )
            self.assertEqual(raised.exception.code, "CACHE_CORRUPT")
            self.assertFalse((dest / "model.bin").exists())
            self.assertFalse((dest / "REVISION").exists())

            clock = _Clock()
            source = _Source({"model.bin": body}, clock)
            with self.assertRaises(SpeechPrepError) as stopped:
                stage_tree(
                    spec, dest, source,
                    free_disk_bytes=len(body) + SCRATCH_BYTES, clock=clock,
                    spending_deadline=1, chunk_bytes=4,
                )
            self.assertEqual(stopped.exception.code, "BUDGET_STOP")
            self.assertTrue((dest / "model.bin.partial").is_file())
            self.assertFalse((dest / "REVISION").exists())
            kept = (dest / "model.bin.partial").read_bytes()

            with self.assertRaises(SpeechPrepError) as disk:
                stage_tree(spec, dest / "other", _Source({"model.bin": body}), free_disk_bytes=0, clock=_Clock())
            self.assertEqual(disk.exception.code, "DISK_CAPACITY")
            with self.assertRaises(SpeechPrepError) as volume:
                stage_tree(
                    spec, dest / "volume", _Source({"model.bin": body}),
                    free_disk_bytes=len(body) + SCRATCH_BYTES, clock=_Clock(), volume_id=VOLUME_ID,
                )
            self.assertEqual(volume.exception.code, "VOLUME_FORBIDDEN")
            self.assertEqual((dest / "model.bin.partial").read_bytes(), kept)

    def test_one_resident_model_reuses_and_refuses_a_busy_switch(self) -> None:
        gate = ResidentGate()
        clock = _Clock()
        calls = {"n": 0}

        def loader() -> None:
            calls["n"] += 1
            clock.now += 3

        first = load_model_result(gate, WHISPER, loader, clock)
        self.assertEqual(first["action"], "LOAD")
        self.assertEqual(first["load_seconds"], 3)
        second = load_model_result(gate, WHISPER, loader, clock)
        self.assertEqual(second["action"], "REUSE")
        self.assertEqual(second["load_seconds"], 0)
        self.assertEqual(calls["n"], 1)
        busy = ResidentGate()
        self.assertEqual(busy.begin(WHISPER), "LOAD")
        self.assertEqual(busy.plan(TTS)["error"], "MODEL_BUSY")
        self.assertEqual(busy.model_id, WHISPER)
        with self.assertRaises(SpeechPrepError) as held:
            busy.unload()
        self.assertEqual(held.exception.code, "MODEL_BUSY")

    def test_r2_transfer_is_refused_until_approved_and_a_full_body_cannot_append(self) -> None:
        client = _Client()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SpeechPrepError) as raised:
                stage_from_r2(client, WHISPER, Path(tmp), free_disk_bytes=0, clock=_Clock())
            self.assertEqual(raised.exception.code, "WEIGHTS_NOT_APPROVED")
            self.assertEqual(client.calls, 0)
        source = _R2RangeSource(client, WHISPER, primary_for("audio.transcribe").revision)
        self.assertEqual(source.read("config.json", 0, 4), b"abcd")
        with self.assertRaises(SpeechPrepError) as ranged:
            source.read("config.json", 2, 2)
        self.assertEqual(ranged.exception.code, "RANGE_REQUIRED")

    def test_preflight_reports_missing_cache_without_inventing_time_or_cost(self) -> None:
        dockerfile = (ROOT / "deploy" / "worker" / "Dockerfile").read_text(encoding="utf-8")
        requirements = (ROOT / "deploy" / "worker" / "speech" / "requirements.txt").read_text(encoding="utf-8")
        speech_docker = (ROOT / "deploy" / "worker" / "speech" / "Dockerfile").read_text(encoding="utf-8")
        report = speech_preflight(
            remote_entries={WHISPER: [], TTS: []},
            free_disk_bytes=staged_disk_bytes(),
            image_manifest={
                "state": "COMPLETE",
                "repo_id": "Qwen/Qwen-Image",
                "revision": REVISION,
                "expected_file_count": QWEN_FILE_COUNT,
                "expected_total_bytes": QWEN_CACHE_MIN_BYTES,
                "files": [{}] * QWEN_FILE_COUNT,
            },
            dockerfile_text=dockerfile,
        )
        whisper = report["models"][0]
        self.assertEqual(whisper["r2"]["status"], "ABSENT")
        self.assertEqual(whisper["r2"]["download_bytes"], file_bytes(WHISPER))
        self.assertIn("model.bin", whisper["r2"]["missing"])
        self.assertEqual(report["models"][1]["r2"]["download_bytes"], file_bytes(TTS))
        self.assertEqual(report["image_cache"]["status"], "COMPLETE")
        self.assertEqual(report["image_cache"]["revision"], REVISION)
        self.assertTrue(report["image_worker"]["compatible"])
        self.assertEqual(report["timing"]["transfer_time"], "unmeasured")
        self.assertIsNone(report["timing"]["transfer_seconds"])
        self.assertIsNone(report["cost"]["session_usd"])
        self.assertEqual(report["cost"]["session_usd_reason"], "TRANSFER_AND_INFERENCE_UNMEASURED")
        self.assertTrue(report["authorization"]["issuer_accepts_speech_models"])
        self.assertFalse(report["authorization"]["paid_authorization_issued"])
        self.assertIsNone(report["authorization"]["recorded_worker_digest"])
        self.assertEqual(report["authorization"]["refusal_without_digest"], "SPEECH_IMAGE_NOT_BUILT")
        self.assertEqual(report["authorization"]["constraints"]["forbidden_volume_id"], VOLUME_ID)
        self.assertEqual(report["watchdog"]["heartbeat_stale_seconds"], 300.0)
        self.assertEqual(report["watchdog"]["heartbeat_interval_seconds"], 20.0)
        self.assertEqual(HEARTBEAT_STALE_SECONDS, 300.0)
        self.assertEqual(HEARTBEAT_INTERVAL_SECONDS, 20.0)
        self.assertEqual(report["disk"]["status"], "ENOUGH")
        self.assertEqual(report["dependencies"]["worker_image_build"], "NOT_RUN")
        self.assertNotIn("faster-whisper", dockerfile)
        self.assertNotIn("qwen-tts", dockerfile)
        for pin in SPEECH_PINS:
            self.assertIn(pin, requirements)
        self.assertIn("requirements.txt", speech_docker)
        self.assertIn("meta.version('qwen-tts')=='0.1.1'", speech_docker)


def load_model_result(gate, model_id, loader, clock):
    from media_engine.models.speech_stage import load_model
    return load_model(gate, model_id, loader, clock)


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def _request(self, method, key, body, extra=None, **kwargs):
        self.calls += 1
        if extra and extra.get("range", "").startswith("bytes=0-"):
            return 200, {}, b"abcd"
        return 200, {}, b"cd"
