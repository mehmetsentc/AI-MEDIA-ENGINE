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
    PART_SIZE,
    MemoryObjectStore,
    R2Client,
    R2StorageError,
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


class _Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def _client(opener, part_size: int = 4) -> R2Client:
    return R2Client(config_from_env({
        "MODEL_CACHE_PROVIDER": "r2",
        "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
        "R2_BUCKET": "models",
        "R2_ACCESS_KEY_ID": "access",
        "R2_SECRET_ACCESS_KEY": "secret",
    }), opener=opener, part_size=part_size)


class MultipartUploadTests(unittest.TestCase):
    def test_small_file_stays_a_single_put(self) -> None:
        seen = []

        def opener(request, timeout):
            seen.append((request.method, request.full_url, request.data))
            return _Response(200, {}, b"")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.bin"
            path.write_bytes(b"abc")
            _client(opener).put_file("hf-cache/small.bin", path)
        self.assertEqual(seen[0][0], "PUT")
        self.assertNotIn("uploads", seen[0][1])
        self.assertEqual(seen[0][2], b"abc")
        self.assertEqual(len(seen), 1)

    def test_large_file_streams_ordered_parts_and_completes(self) -> None:
        seen = []

        def opener(request, timeout):
            body = request.data or b""
            seen.append((request.method, request.full_url, body))
            if request.method == "POST" and "uploads=" in request.full_url:
                xml = b"<InitiateMultipartUploadResult><UploadId>up-1</UploadId></InitiateMultipartUploadResult>"
                return _Response(200, {}, xml)
            if request.method == "PUT":
                number = request.full_url.split("partNumber=", 1)[1].split("&", 1)[0]
                return _Response(200, {"ETag": f'"etag-{number}"'}, b"")
            return _Response(200, {}, b"<CompleteMultipartUploadResult></CompleteMultipartUploadResult>")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"abcdefghij")
            _client(opener, part_size=4).put_file("hf-cache/weights.bin", path)
        put_bodies = [body for method, _url, body in seen if method == "PUT"]
        self.assertLessEqual(max(len(body) for body in put_bodies), 4)
        self.assertEqual(b"".join(put_bodies), b"abcdefghij")
        parts = [url for method, url, _body in seen if method == "PUT"]
        self.assertEqual(
            [url.split("partNumber=", 1)[1].split("&", 1)[0] for url in parts],
            ["1", "2", "3"],
        )
        complete = seen[-1][2]
        self.assertIn(b"<PartNumber>1</PartNumber><ETag>\"etag-1\"</ETag>", complete)
        self.assertLess(complete.find(b"<PartNumber>1</PartNumber>"), complete.find(b"<PartNumber>2</PartNumber>"))
        self.assertLess(complete.find(b"<PartNumber>2</PartNumber>"), complete.find(b"<PartNumber>3</PartNumber>"))
        self.assertEqual(PART_SIZE, 8 * 1024 * 1024)

    def test_failed_part_aborts_and_does_not_complete(self) -> None:
        seen = []

        def opener(request, timeout):
            seen.append((request.method, request.full_url))
            if request.method == "POST" and "uploads=" in request.full_url:
                return _Response(200, {}, b"<InitiateMultipartUploadResult><UploadId>up-1</UploadId></InitiateMultipartUploadResult>")
            if request.method == "PUT" and "partNumber=2" in request.full_url:
                raise TimeoutError("timed out")
            if request.method == "PUT":
                return _Response(200, {"ETag": '"etag-1"'}, b"")
            return _Response(204, {}, b"")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.bin"
            path.write_bytes(b"abcdefgh")
            with self.assertRaises(R2StorageError):
                _client(opener, part_size=4).put_file("hf-cache/weights.bin", path)
        methods = [method for method, url in seen]
        self.assertIn("DELETE", methods)
        self.assertTrue(any(method == "DELETE" and "uploadId=up-1" in url for method, url in seen))
        self.assertFalse(any(method == "POST" and "uploadId=" in url for method, url in seen))

    def test_manifest_is_last_and_absent_when_a_model_object_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mount = Path(tmp) / "models"
            _complete(mount)
            store = MemoryObjectStore()

            def fail_blob(key, body, metadata=None):
                if key.endswith("/blob"):
                    raise R2StorageError(0)
                MemoryObjectStore.put(store, key, body, metadata)

            store.put = fail_blob
            with self.assertRaises(R2StorageError):
                upload_cache(store, mount)
            self.assertNotIn("phase2d-cache.json", store.objects)
            self.assertFalse(any(payload.get("state") == COMPLETE for payload, _meta in [
                (json.loads(body.decode()), meta) for body, meta in store.objects.values() if body.startswith(b"{")
            ]))
            ordered = MemoryObjectStore()
            upload_cache(ordered, mount)
            self.assertEqual(list(ordered.objects)[-1], "phase2d-cache.json")
