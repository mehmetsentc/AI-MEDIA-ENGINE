"""Persistent cache contract. No GPU, no volume, no model download."""
from __future__ import annotations

import inspect
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.model_cache import (
    COMPLETE,
    EMPTY,
    FILL,
    FILLING,
    GENERATE,
    INVALID,
    MOUNT_PATH,
    QWEN_CACHE_MIN_BYTES,
    REPO_ID,
    VOLUME_SIZE_GB,
    ModelCacheNotReady,
    assess_cache,
    fill_model_cache,
    hf_cache,
    mark_filling,
    prepare_generation,
    repo_dirname,
    snapshot_dir,
    write_manifest,
)
from media_engine.engines.image import qwen_remote
from media_engine.providers.phase2d import (
    APPROVED_GPU_MODEL,
    MAX_CREATE_ATTEMPTS,
    classify_runtime_failure,
    generation_job,
    policy_fingerprint,
)
from media_engine.usage import UsageEvent, UsageStore, cost_observation


def _complete(root: Path, payload: bytes = b"abc") -> Path:
    revision = "rev1"
    blob_dir = hf_cache(root) / repo_dirname() / "blobs"
    blob_dir.mkdir(parents=True)
    (blob_dir / "blob").write_bytes(payload)
    snapshot = snapshot_dir(root, revision)
    snapshot.mkdir(parents=True)
    (snapshot / "weights.safetensors").symlink_to(Path("../../blobs/blob"))
    from media_engine.engines.image.model_cache import mark_complete
    mark_complete(
        root,
        revision=revision,
        files=[{"path": "weights.safetensors", "bytes": len(payload)}],
        completed_at="2026-10-06T00:00:00Z",
    )
    return root


class ModelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mount = Path(self.tmp.name) / "models"

    def test_complete_cache_is_accepted_and_loaded_offline(self) -> None:
        mount = _complete(self.mount)
        self.assertEqual(assess_cache(mount, min_bytes=3), COMPLETE)
        calls = []

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            return "/snapshot"

        local = prepare_generation(mount, download, min_bytes=3)
        self.assertEqual(local, "/snapshot")
        self.assertEqual(calls[0][0], (REPO_ID,))
        self.assertTrue(calls[0][1]["local_files_only"])
        self.assertEqual(calls[0][1]["cache_dir"], str(hf_cache(mount)))

    def test_empty_filling_and_invalid_are_rejected(self) -> None:
        self.assertEqual(assess_cache(self.mount, min_bytes=1), EMPTY)
        with self.assertRaises(ModelCacheNotReady):
            prepare_generation(self.mount, lambda *a, **k: None, min_bytes=1)
        mark_filling(self.mount)
        self.assertEqual(assess_cache(self.mount, min_bytes=1), FILLING)
        write_manifest(self.mount, {"state": INVALID, "repo_id": REPO_ID})
        self.assertEqual(assess_cache(self.mount, min_bytes=1), INVALID)

    def test_missing_blob_snapshot_and_wrong_repo_are_rejected(self) -> None:
        mount = _complete(self.mount)
        blob = hf_cache(mount) / repo_dirname() / "blobs" / "blob"
        blob.unlink()
        self.assertEqual(assess_cache(mount, min_bytes=1), INVALID)
        blob.write_bytes(b"abc")
        snapshot = snapshot_dir(mount, "rev1")
        for child in snapshot.iterdir():
            child.unlink()
        snapshot.rmdir()
        self.assertEqual(assess_cache(mount, min_bytes=1), INVALID)
        mount2 = Path(self.tmp.name) / "other"
        _complete(mount2)
        write_manifest(mount2, {
            "state": COMPLETE,
            "repo_id": "other/model",
            "revision": "rev1",
            "expected_file_count": 1,
            "files": [{"path": "weights.safetensors", "bytes": 3}],
        })
        self.assertEqual(assess_cache(mount2, min_bytes=1), INVALID)

    def test_generation_does_not_download_and_fill_is_separate(self) -> None:
        calls = []
        mount = _complete(self.mount)

        def download(*args, **kwargs):
            calls.append(kwargs["local_files_only"])
            return "ok"

        prepare_generation(self.mount, download, min_bytes=3)
        self.assertEqual(calls, [True])
        fill_model_cache(
            Path(self.tmp.name) / "fill",
            download,
            revision="rev1",
            files=[{"path": "weights.safetensors", "bytes": 3}],
            completed_at="2026-10-06T00:00:00Z",
        )
        self.assertEqual(calls, [True, False])
        source = inspect.getsource(qwen_remote.main)
        self.assertNotIn("snapshot_download(", source)
        self.assertNotIn("fill_model_cache", source)
        self.assertIn("prepare_generation", source)
        job = generation_job()
        self.assertEqual(job["operation"], GENERATE)
        self.assertNotEqual(job["operation"], FILL)
        self.assertEqual(job["cache_mount"], MOUNT_PATH)
        self.assertNotIn("create_volume", inspect.getsource(prepare_generation))

    def test_undersized_complete_cache_fails_the_production_floor(self) -> None:
        mount = _complete(self.mount)
        self.assertEqual(QWEN_CACHE_MIN_BYTES, 57_704_594_653)
        self.assertEqual(assess_cache(mount), INVALID)

    def test_cache_miss_is_classified_without_a_gpu_create(self) -> None:
        self.assertEqual(
            classify_runtime_failure(1, b"", b"", {"error": "MODEL_CACHE_NOT_READY"}),
            "MODEL_CACHE_NOT_READY",
        )
        self.assertEqual(APPROVED_GPU_MODEL, "L40")
        self.assertEqual(MAX_CREATE_ATTEMPTS, 1)
        self.assertEqual(policy_fingerprint(), "b716f03932af9c6621a47aff552765061b0fc64aafe8a2b47f37a792f7ab390d")
        self.assertEqual(VOLUME_SIZE_GB, 120)
        self.assertNotIn("PUT", inspect.getsource(fill_model_cache))

    def test_credit_gap_is_recorded_without_changing_the_formula(self) -> None:
        calculated = "0.12715040432229455"
        observed = cost_observation("9.982849501", "9.8124853636", calculated)
        self.assertEqual(observed["calculated_runtime_cost"], calculated)
        delta = Decimal("9.982849501") - Decimal("9.8124853636")
        self.assertEqual(Decimal(observed["credit_delta"]), delta)
        self.assertEqual(Decimal(observed["unexplained_cost_delta"]), delta - Decimal(calculated))
        store = UsageStore(str(Path(self.tmp.name) / "usage.sqlite3"))
        store.record(UsageEvent(
            job_id="phase2d-cache",
            client_id="phase2d",
            job_type="PHASE2D_IMAGE",
            engine_id="qwen-image",
            started_at="2026-10-06T13:23:21Z",
            finished_at="2026-10-06T13:43:33Z",
            duration_seconds=1,
            attempt_count=1,
            estimated_cost_usd=calculated,
            status="TERMINATED",
            credit_before=observed["credit_before"],
            credit_after=observed["credit_after"],
            credit_delta=observed["credit_delta"],
            calculated_runtime_cost=observed["calculated_runtime_cost"],
            unexplained_cost_delta=observed["unexplained_cost_delta"],
        ))
        saved = store.for_job("phase2d-cache")
        self.assertIsNotNone(saved)
        self.assertEqual(saved.unexplained_cost_delta, observed["unexplained_cost_delta"])
