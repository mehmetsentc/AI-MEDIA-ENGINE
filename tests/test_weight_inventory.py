"""Which models may be staged, and why the pinned pair must stay pinned."""
from __future__ import annotations

import unittest

from media_engine.models.lifecycle import SCRATCH_BYTES
from media_engine.models.speech_inventory import EXTRA_TARGETS, TARGETS
from media_engine.models.speech_stage import (
    SpeechPrepError,
    all_targets,
    file_bytes,
    object_key,
    staged_disk_bytes,
    target,
    targets,
)
from media_engine.providers.execution_authorization import _speech_pins

WHISPER = "Systran/faster-whisper-large-v3"
TTS = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


class PinnedPairTests(unittest.TestCase):
    def test_targets_carries_the_pinned_pair_and_nothing_else(self) -> None:
        """Two safety values are derived from TARGETS and both break if it grows.

        The authorization pins exactly one model per capability, so a second
        audio.tts entry reads as model substitution and refuses the session.
        The staging disk budget is the sum of every tree in TARGETS, so an extra
        entry demands disk that a run never uses. Anything else stageable
        belongs in EXTRA_TARGETS.
        """
        self.assertEqual([item["model_id"] for item in targets()], [WHISPER, TTS])
        for capability in ("audio.transcribe", "audio.tts"):
            matches = [item for item in TARGETS if item["capability"] == capability]
            self.assertEqual(len(matches), 1, capability)

    def test_the_authorization_can_still_pin_one_model_per_capability(self) -> None:
        pins = _speech_pins()
        self.assertEqual([model_id for model_id, _revision in pins], [WHISPER, TTS])

    def test_the_disk_budget_covers_the_pair_and_scratch_only(self) -> None:
        self.assertEqual(
            staged_disk_bytes(), file_bytes(WHISPER) + file_bytes(TTS) + SCRATCH_BYTES,
        )


class ExtraTargetTests(unittest.TestCase):
    def test_every_extra_model_is_stageable_and_keyed(self) -> None:
        self.assertTrue(EXTRA_TARGETS)
        for item in EXTRA_TARGETS:
            model_id, revision = item["model_id"], item["revision"]
            self.assertIs(target(model_id), item)
            key = object_key(model_id, revision, "config.json")
            self.assertTrue(key.startswith("speech/%s/%s/" % (model_id, revision)))
            self.assertIn(item, all_targets())
            self.assertNotIn(item, targets())

    def test_an_extra_model_cannot_be_keyed_under_a_different_revision(self) -> None:
        item = EXTRA_TARGETS[0]
        with self.assertRaises(SpeechPrepError) as refused:
            object_key(item["model_id"], "0" * 40, "config.json")
        self.assertEqual(refused.exception.code, "REVISION_MISMATCH")

    def test_every_checksum_records_how_it_was_obtained(self) -> None:
        """A hub-published LFS id and a hash of bytes we held are not the same
        claim, so the inventory says which one each file carries."""
        allowed = {"lfs", "content", "downloaded"}
        for item in all_targets():
            self.assertTrue(item["files"])
            for entry in item["files"]:
                self.assertIn(entry["sha_source"], allowed, entry["path"])
                self.assertEqual(len(entry["sha256"]), 64, entry["path"])
                self.assertGreater(entry["bytes"], 0, entry["path"])

    def test_the_extra_inventories_were_hashed_from_bytes_we_held(self) -> None:
        for item in EXTRA_TARGETS:
            sources = {entry["sha_source"] for entry in item["files"]}
            self.assertEqual(sources, {"downloaded"}, item["model_id"])

    def test_no_model_id_appears_in_both_lists(self) -> None:
        pinned = {item["model_id"] for item in TARGETS}
        extra = {item["model_id"] for item in EXTRA_TARGETS}
        self.assertEqual(pinned & extra, set())


if __name__ == "__main__":
    unittest.main()
