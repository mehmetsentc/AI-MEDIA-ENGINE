"""R2 model store. No live network, no secrets, no GPU."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
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
from media_engine.engines.image.qwen_remote import _load_workspace_module
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


class _StreamResponse:
    def __init__(self, body) -> None:
        self.status = 200
        self.headers: dict[str, str] = {}
        self._body = body

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


class _Response:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self._body = body

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0 or n >= len(self._body):
            data = self._body
            self._body = b""
            return data
        data = self._body[:n]
        self._body = self._body[n:]
        return data

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def _worker_module(directory: Path):
    workspace = directory / "workspace"
    workspace.mkdir(parents=True)
    shutil.copy(Path(model_cache.__file__), workspace / "model_cache.py")
    shutil.copy(Path(r2_cache.__file__), workspace / "r2_cache.py")
    name = "phase2d_r2_cache_test"
    sys.modules.pop(name, None)
    return _load_workspace_module(name, str(workspace / "r2_cache.py"))


def _client(opener, part_size: int = 4) -> R2Client:
    return R2Client(config_from_env({
        "MODEL_CACHE_PROVIDER": "r2",
        "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
        "R2_BUCKET": "models",
        "R2_ACCESS_KEY_ID": "access",
        "R2_SECRET_ACCESS_KEY": "secret",
    }), opener=opener, part_size=part_size)


def _manifest(state: str = COMPLETE) -> bytes:
    return json.dumps({
        "state": state,
        "repo_id": "Qwen/Qwen-Image",
        "revision": "rev1",
        "expected_file_count": 1,
        "expected_total_bytes": 3,
        "completed_at": "2026-10-06T00:00:00Z",
        "validation_status": state,
        "files": [{"path": "weights.safetensors", "bytes": 3}],
    }).encode()


_LIST_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents><Key>hf-cache/models--Qwen--Qwen-Image/blobs/blob</Key></Contents>
  <Contents><Key>hf-cache/models--Qwen--Qwen-Image/snapshots/rev1/weights.safetensors</Key></Contents>
</ListBucketResult>"""


class WorkerSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mount = Path(self.tmp.name) / "models"

    def test_unregistered_importlib_load_raises_the_live_attribute_error(self) -> None:
        workspace = Path(self.tmp.name) / "bare"
        workspace.mkdir()
        shutil.copy(Path(model_cache.__file__), workspace / "model_cache.py")
        shutil.copy(Path(r2_cache.__file__), workspace / "r2_cache.py")
        spec = importlib.util.spec_from_file_location("phase2d_r2_cache_bare", workspace / "r2_cache.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with self.assertRaises(AttributeError) as raised:
            spec.loader.exec_module(module)
        self.assertIsNone(sys.modules.get("phase2d_r2_cache_bare"))
        self.assertIn("__dict__", str(raised.exception))

    def test_worker_loader_syncs_through_r2client_without_huggingface(self) -> None:
        worker = _worker_module(Path(self.tmp.name))
        self.addCleanup(lambda: sys.modules.pop("phase2d_r2_cache_test", None))
        calls = []

        def opener(request, timeout):
            url = request.full_url
            calls.append(url)
            if "list-type=" in url:
                return _Response(200, {}, _LIST_XML)
            if url.endswith("phase2d-cache.json"):
                return _Response(200, {}, _manifest())
            if url.endswith("/blob"):
                return _Response(200, {}, b"abc")
            if url.endswith("weights.safetensors"):
                return _Response(200, {"x-amz-meta-hf-symlink": "../../blobs/blob"}, b"")
            return _Response(404, {}, b"")

        client = worker.R2Client(worker.config_from_env({
            "MODEL_CACHE_PROVIDER": "r2",
            "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
            "R2_BUCKET": "models",
            "R2_ACCESS_KEY_ID": "access",
            "R2_SECRET_ACCESS_KEY": "secret",
        }), opener=opener)
        body, meta = client.get("phase2d-cache.json")
        self.assertIsInstance(body, bytes)
        self.assertIsInstance(meta, dict)
        self.assertEqual(json.loads(body.decode())["state"], COMPLETE)
        self.assertIn("list_keys", worker.R2Client.__dict__)
        keys = client.list_keys("hf-cache/")
        self.assertEqual(keys, [
            "hf-cache/models--Qwen--Qwen-Image/blobs/blob",
            "hf-cache/models--Qwen--Qwen-Image/snapshots/rev1/weights.safetensors",
        ])
        calls.clear()
        downloads = []
        local = worker.load_from_r2(
            client, self.mount, lambda *args, **kwargs: downloads.append((args, kwargs)) or "local",
            min_bytes=3,
        )
        blob = self.mount / "hf-cache/models--Qwen--Qwen-Image/blobs/blob"
        link = self.mount / "hf-cache/models--Qwen--Qwen-Image/snapshots/rev1/weights.safetensors"
        self.assertFalse(blob.is_symlink())
        self.assertEqual(blob.read_bytes(), b"abc")
        self.assertTrue(link.is_symlink())
        self.assertEqual(os.readlink(link), "../../blobs/blob")
        self.assertEqual(link.read_bytes(), b"abc")
        self.assertTrue(any("list-type=" in url for url in calls))
        self.assertEqual(downloads[0][1]["local_files_only"], True)
        self.assertEqual(local, "local")
        source = inspect.getsource(worker.sync_complete_cache)
        self.assertNotIn("snapshot_download", source)
        self.assertNotIn("huggingface", source.lower())
        self.assertIn("list_keys", source)

    def test_incomplete_r2client_cache_does_not_list_or_download(self) -> None:
        worker = _worker_module(Path(self.tmp.name) / "incomplete")
        self.addCleanup(lambda: sys.modules.pop("phase2d_r2_cache_test", None))
        calls = []

        def opener(request, timeout):
            calls.append(request.full_url)
            return _Response(200, {}, _manifest(FILLING))

        client = worker.R2Client(worker.config_from_env({
            "MODEL_CACHE_PROVIDER": "r2",
            "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
            "R2_BUCKET": "models",
            "R2_ACCESS_KEY_ID": "access",
            "R2_SECRET_ACCESS_KEY": "secret",
        }), opener=opener)
        with self.assertRaises(worker.ModelCacheNotReady) as raised:
            worker.sync_complete_cache(client, self.mount, min_bytes=3)
        self.assertEqual(raised.exception.state, FILLING)
        self.assertFalse(any("list-type=" in url for url in calls))
        self.assertFalse((self.mount / "hf-cache").exists())


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


class ParallelDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mount = Path(self.tmp.name) / "models"
        self.previous_chunk = r2_cache.STREAM_CHUNK
        r2_cache.STREAM_CHUNK = 3
        self.addCleanup(self._restore_chunk)

    def _restore_chunk(self) -> None:
        r2_cache.STREAM_CHUNK = self.previous_chunk

    def _manifest(self) -> bytes:
        return json.dumps({
            "state": COMPLETE,
            "repo_id": "Qwen/Qwen-Image",
            "revision": "rev1",
            "expected_file_count": 1,
            "expected_total_bytes": 8,
            "completed_at": "2026-10-06T00:00:00Z",
            "validation_status": COMPLETE,
            "files": [{"path": "weights.safetensors", "bytes": 8}],
        }).encode()

    def _list_xml(self) -> bytes:
        rows = []
        for name in ("b0", "b1", "b2", "b3", "b4"):
            key = f"hf-cache/models--Qwen--Qwen-Image/blobs/{name}"
            rows.append(f"<Contents><Key>{key}</Key><Size>8</Size></Contents>")
        link = "hf-cache/models--Qwen--Qwen-Image/snapshots/rev1/weights.safetensors"
        rows.append(f"<Contents><Key>{link}</Key><Size>0</Size></Contents>")
        body = "".join(rows)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            "<IsTruncated>false</IsTruncated>"
            f"{body}</ListBucketResult>"
        ).encode()

    def test_bounded_parallel_streams_rename_skip_and_restore_symlinks(self) -> None:
        self.assertEqual(r2_cache.DOWNLOAD_CONCURRENCY, 4)
        payload = b"abcdefgh"
        active = {"n": 0, "max": 0}
        lock = threading.Lock()
        release = threading.Event()
        calls: list[str] = []
        progress: list[dict] = []
        blob_root = self.mount / "hf-cache/models--Qwen--Qwen-Image/blobs"
        link = self.mount / "hf-cache/models--Qwen--Qwen-Image/snapshots/rev1/weights.safetensors"

        class _Chunk:
            def __init__(self, name: str) -> None:
                self.name = name
                self.offset = 0
                self.reads = 0

            def read(self, n: int = -1) -> bytes:
                self.reads += 1
                if self.reads == 1:
                    with lock:
                        active["n"] += 1
                        active["max"] = max(active["max"], active["n"])
                        if active["n"] >= 4:
                            release.set()
                    self.assert_wait()
                    with lock:
                        active["n"] -= 1
                if self.reads == 2:
                    final = blob_root / self.name
                    partial = final.with_name(final.name + ".partial")
                    if final.exists():
                        raise AssertionError("final file appeared before the last chunk")
                    if not partial.exists():
                        raise AssertionError("partial file missing during stream")
                if self.offset >= len(payload):
                    return b""
                take = len(payload) if n is None or n < 0 else n
                data = payload[self.offset:self.offset + take]
                self.offset += len(data)
                return data

            def assert_wait(self) -> None:
                if not release.wait(3):
                    raise AssertionError("downloads were not concurrent")

        def opener(request, timeout):
            self.assertEqual(timeout, 60)
            url = request.full_url
            calls.append(url)
            if "list-type=" in url:
                return _Response(200, {}, self._list_xml())
            if url.endswith("phase2d-cache.json"):
                return _Response(200, {}, self._manifest())
            if "/blobs/" in url:
                return _StreamResponse(_Chunk(url.rsplit("/", 1)[-1]))
            if url.endswith("weights.safetensors"):
                if not (blob_root / "b0").is_file():
                    raise AssertionError("symlink fetched before blob file existed")
                return _Response(200, {"x-amz-meta-hf-symlink": "../../blobs/b0"}, b"")
            return _Response(404, {}, b"")

        progress_path = Path(self.tmp.name) / "r2-sync-progress.json"
        sync_complete_cache(
            _client(opener),
            self.mount,
            min_bytes=8,
            on_progress=progress.append,
            progress_path=progress_path,
        )
        self.assertEqual(active["max"], 4)
        self.assertEqual((blob_root / "b0").read_bytes(), payload)
        self.assertFalse((blob_root / "b0").is_symlink())
        self.assertTrue(link.is_symlink())
        self.assertEqual(os.readlink(link), "../../blobs/b0")
        self.assertEqual(link.read_bytes(), payload)
        self.assertFalse(any(path.name.endswith(".partial") for path in self.mount.rglob("*")))
        self.assertEqual(assess_cache(self.mount, min_bytes=8), COMPLETE)
        self.assertEqual(max(item["objects_done"] for item in progress), 6)
        self.assertTrue(any(item["bytes_done"] == 40 and item["bytes_total"] == 40 for item in progress))
        self.assertTrue(all("MB_per_sec" in item for item in progress))
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["objects_done"], 6)
        self.assertEqual(saved["bytes_done"], 40)
        self.assertEqual(saved["bytes_total"], 40)
        self.assertEqual(saved["current"], [])
        self.assertIn("MB_per_sec", saved)
        calls.clear()
        sync_complete_cache(_client(opener), self.mount, min_bytes=8)
        self.assertFalse(any("/blobs/" in url for url in calls))
        self.assertTrue(link.is_symlink())

    def test_failed_stream_removes_the_partial_file(self) -> None:
        class _Boom:
            def __init__(self) -> None:
                self.reads = 0

            def read(self, n: int = -1) -> bytes:
                self.reads += 1
                if self.reads > 1:
                    raise RuntimeError("stream broke")
                return b"abc"

        def opener(request, timeout):
            url = request.full_url
            if "list-type=" in url:
                key = "hf-cache/models--Qwen--Qwen-Image/blobs/b0"
                xml = (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<ListBucketResult><IsTruncated>false</IsTruncated>'
                    f"<Contents><Key>{key}</Key><Size>8</Size></Contents>"
                    "</ListBucketResult>"
                ).encode()
                return _Response(200, {}, xml)
            if url.endswith("phase2d-cache.json"):
                return _Response(200, {}, self._manifest())
            if url.endswith("/b0"):
                return _StreamResponse(_Boom())
            return _Response(404, {}, b"")

        with self.assertRaises(R2StorageError):
            sync_complete_cache(_client(opener), self.mount, min_bytes=8)
        blob = self.mount / "hf-cache/models--Qwen--Qwen-Image/blobs/b0"
        self.assertFalse(blob.exists())
        self.assertFalse(blob.with_name("b0.partial").exists())
        self.assertNotEqual(assess_cache(self.mount, min_bytes=8), COMPLETE)
