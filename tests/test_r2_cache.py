"""R2 model store. No live network, no secrets, no GPU."""
from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from media_engine.engines.image.model_cache import (
    COMPLETE,
    EMPTY,
    FILLING,
    MOUNT_PATH,
    ModelCacheNotReady,
    assess_cache,
    hf_cache,
    mark_complete,
    repo_dirname,
    snapshot_dir,
)
from media_engine.engines.image import qwen_remote
from media_engine.engines.image import model_cache, r2_cache
from media_engine.engines.image.r2_cache import (
    MemoryObjectStore,
    R2Client,
    _authorization,
    config_from_env,
    load_from_r2,
    sync_complete_cache,
    upload_cache,
)
from media_engine.providers.phase2d import generation_job, policy_fingerprint


def _complete(root: Path) -> None:
    revision = "rev1"
    blob_dir = hf_cache(root) / repo_dirname() / "blobs"
    blob_dir.mkdir(parents=True)
    (blob_dir / "blob").write_bytes(b"abc")
    snapshot = snapshot_dir(root, revision)
    snapshot.mkdir(parents=True)
    (snapshot / "weights.safetensors").symlink_to(Path("../../blobs/blob"))
    mark_complete(
        root,
        revision=revision,
        files=[{"path": "weights.safetensors", "bytes": 3}],
        completed_at="2026-10-06T00:00:00Z",
    )


class R2CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mount = Path(self.tmp.name) / "models"

    def test_complete_cache_syncs_structure_and_loads_offline(self) -> None:
        source = Path(self.tmp.name) / "source"
        _complete(source)
        store = MemoryObjectStore()
        upload_cache(store, source)
        calls = []

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            return str(snapshot_dir(self.mount, "rev1"))

        local = load_from_r2(store, self.mount, download, min_bytes=3)
        self.assertEqual(assess_cache(self.mount, min_bytes=3), COMPLETE)
        link = snapshot_dir(self.mount, "rev1") / "weights.safetensors"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.read_bytes(), b"abc")
        self.assertTrue(calls[0][1]["local_files_only"])
        self.assertEqual(local, str(snapshot_dir(self.mount, "rev1")))
        self.assertEqual(store.objects.__class__, dict)

    def test_missing_manifest_refuses_generation_without_download(self) -> None:
        store = MemoryObjectStore()
        calls = []
        with self.assertRaises(ModelCacheNotReady) as raised:
            load_from_r2(store, self.mount, lambda *a, **k: calls.append(1), min_bytes=3)
        self.assertEqual(raised.exception.state, EMPTY)
        self.assertEqual(calls, [])
        self.assertEqual(store.lists, 0)

    def test_incomplete_manifest_is_not_fetched(self) -> None:
        store = MemoryObjectStore()
        store.put("phase2d-cache.json", json.dumps({"state": FILLING}).encode(), {})
        store.put("hf-cache/secret-weights", b"nope", {})
        with self.assertRaises(ModelCacheNotReady) as raised:
            sync_complete_cache(store, self.mount, min_bytes=3)
        self.assertEqual(raised.exception.state, FILLING)
        self.assertEqual(store.lists, 0)
        self.assertFalse((self.mount / "hf-cache").exists())

    def test_upload_is_separate_from_generation(self) -> None:
        source = inspect.getsource(qwen_remote.main)
        self.assertNotIn("upload_cache", source)
        self.assertNotIn("snapshot_download(", source)
        self.assertIn('MODEL_CACHE_PROVIDER") == "r2"', source)
        self.assertEqual(generation_job()["cache_mount"], MOUNT_PATH)
        self.assertNotIn("R2_SECRET_ACCESS_KEY", json.dumps(generation_job()))

    def test_config_omits_secret_and_refuses_partial_env(self) -> None:
        secret = "r2-secret-value"
        with self.assertRaises(ModelCacheNotReady):
            config_from_env({"MODEL_CACHE_PROVIDER": "r2", "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com"})
        config = config_from_env({
            "MODEL_CACHE_PROVIDER": "r2",
            "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
            "R2_BUCKET": "models",
            "R2_ACCESS_KEY_ID": "access",
            "R2_SECRET_ACCESS_KEY": secret,
        })
        self.assertNotIn(secret, repr(config))
        self.assertNotIn(secret, repr(R2Client(config)))

    def test_signed_request_matches_aws_vector_and_hides_secret(self) -> None:
        secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        headers = {
            "host": "examplebucket.s3.amazonaws.com",
            "range": "bytes=0-9",
            "x-amz-content-sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "x-amz-date": "20130524T000000Z",
        }
        header = _authorization(
            "GET", "/test.txt", "", headers, headers["x-amz-content-sha256"],
            "20130524T000000Z", "AKIAIOSFODNN7EXAMPLE", secret, region="us-east-1",
        )
        self.assertEqual(
            header,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
            "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
            "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41",
        )
        seen = []

        def opener(request, timeout):
            seen.append(request)
            raise type("Missing", (Exception,), {"code": 404})()

        client = R2Client(config_from_env({
            "MODEL_CACHE_PROVIDER": "r2",
            "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
            "R2_BUCKET": "models",
            "R2_ACCESS_KEY_ID": "access",
            "R2_SECRET_ACCESS_KEY": secret,
        }), opener=opener)
        self.assertIsNone(client.get("phase2d-cache.json"))
        self.assertNotIn(secret, seen[0].full_url)
        self.assertEqual(policy_fingerprint(), "b716f03932af9c6621a47aff552765061b0fc64aafe8a2b47f37a792f7ab390d")

    def test_worker_copies_import_without_the_package(self) -> None:
        workspace = Path(self.tmp.name) / "workspace"
        workspace.mkdir()
        shutil.copy(Path(r2_cache.__file__), workspace / "r2_cache.py")
        shutil.copy(Path(model_cache.__file__), workspace / "model_cache.py")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, "-c", "import r2_cache; print(r2_cache.MANIFEST_NAME)"],
            cwd=workspace,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "phase2d-cache.json")
