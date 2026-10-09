"""Offline cache fixtures. Nothing here downloads or writes a model file."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from media_engine.engines.image.cache_validation import (
    CACHE_COMPLETE,
    CACHE_INCOMPLETE,
    CACHE_MANIFEST_INVALID,
    CACHE_PATH_MISSING,
    CACHE_REVISION_MISMATCH,
    CACHE_UNVERIFIED,
    REVISION,
    qwen_contract,
    validate_cache,
)
from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT, snapshot_dir


def _manifest(files: list[dict], **extra) -> dict:
    payload = {
        "state": "COMPLETE",
        "repo_id": "Qwen/Qwen-Image",
        "revision": REVISION,
        "files": files,
    }
    payload.update(extra)
    return payload


class CacheValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mount = Path(self.tmp.name) / "models"

    def test_missing_path_and_unverified_mount(self) -> None:
        missing = validate_cache(self.mount)
        self.assertEqual(missing["status"], CACHE_PATH_MISSING)
        self.assertFalse(missing["complete"])
        self.mount.mkdir()
        unverified = validate_cache(self.mount)
        self.assertEqual(unverified["status"], CACHE_UNVERIFIED)
        self.assertFalse(unverified["mutated"])

    def test_invalid_manifest_and_revision_mismatch(self) -> None:
        self.mount.mkdir()
        (self.mount / "phase2d-cache.json").write_text("{", encoding="utf-8")
        invalid = validate_cache(self.mount)
        self.assertEqual(invalid["status"], CACHE_MANIFEST_INVALID)
        mismatch = validate_cache(self.mount, {"revision": "other", "files": [{"path": "a", "bytes": 1}]})
        self.assertEqual(mismatch["status"], CACHE_REVISION_MISMATCH)
        self.assertFalse(mismatch["complete"])

    def test_incomplete_truncated_and_hash_mismatch(self) -> None:
        self.mount.mkdir()
        snapshot = snapshot_dir(self.mount, REVISION)
        snapshot.mkdir(parents=True)
        (snapshot / "weights.bin").write_bytes(b"ab")
        blob = snapshot / "blobs" / "not-the-digest"
        blob.parent.mkdir()
        blob.write_bytes(b"")
        link = snapshot / "linked.bin"
        link.symlink_to("blobs/not-the-digest")
        report = validate_cache(self.mount, _manifest([
            {"path": "weights.bin", "bytes": 4},
            {"path": "linked.bin", "bytes": 0, "sha256": "abc123"},
            {"path": "gone.bin", "bytes": 1},
        ]))
        self.assertEqual(report["status"], CACHE_INCOMPLETE)
        self.assertIn("gone.bin", report["missing"])
        self.assertIn("weights.bin", report["truncated"])
        self.assertIn("linked.bin", report["hash_mismatches"])
        self.assertFalse(report["complete"])
        self.assertFalse((snapshot / "gone.bin").exists())

    def test_complete_fixture_matches_sizes_and_does_not_write(self) -> None:
        self.mount.mkdir()
        snapshot = snapshot_dir(self.mount, REVISION)
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(b"{}")
        target = snapshot / "blobs" / "sha"
        target.parent.mkdir()
        target.write_bytes(b"blob")
        link = snapshot / "model.safetensors"
        link.symlink_to("blobs/sha")
        before = sorted(path.relative_to(self.mount).as_posix() for path in self.mount.rglob("*"))
        report = validate_cache(self.mount, _manifest(
            [
                {"path": "config.json", "bytes": 2},
                {"path": "model.safetensors", "bytes": 4, "blob_sha256": "sha"},
            ],
            expected_file_count=2,
            expected_total_bytes=6,
        ))
        after = sorted(path.relative_to(self.mount).as_posix() for path in self.mount.rglob("*"))
        self.assertEqual(report["status"], CACHE_COMPLETE)
        self.assertTrue(report["complete"])
        self.assertEqual(report["hashes_checked"], 1)
        self.assertEqual(before, after)
        self.assertFalse(report["mutated"])

    def test_bad_path_and_count_disagreement_are_invalid(self) -> None:
        self.mount.mkdir()
        escaped = validate_cache(self.mount, _manifest([{"path": "../secret", "bytes": 1}]))
        self.assertEqual(escaped["status"], CACHE_MANIFEST_INVALID)
        declared = validate_cache(self.mount, _manifest(
            [{"path": "a", "bytes": 1}], expected_file_count=2,
        ))
        self.assertEqual(declared["status"], CACHE_MANIFEST_INVALID)

    def test_qwen_contract_requires_the_known_revision_and_size(self) -> None:
        files = [{"path": f"f{index}", "bytes": 1} for index in range(QWEN_FILE_COUNT)]
        self.assertFalse(qwen_contract(_manifest(files)))
        self.assertTrue(qwen_contract(_manifest(
            files,
            expected_file_count=QWEN_FILE_COUNT,
            expected_total_bytes=QWEN_CACHE_MIN_BYTES,
        )))
        raw = json.dumps({"remote_volume_verified": False})
        self.assertNotIn("CACHE_COMPLETE", raw)
