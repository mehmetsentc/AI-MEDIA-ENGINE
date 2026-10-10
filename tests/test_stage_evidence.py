"""Staging failure evidence. No provider calls leave the process."""
from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from media_engine.engines.image.model_cache import COMPLETE, MANIFEST_NAME, ModelCacheNotReady
from media_engine.engines.image.qwen import MODEL_ID
from media_engine.engines.image.r2_cache import R2StorageError, reliable_transfer_enabled, sync_complete_cache
from media_engine.providers.base import WorkerStageError
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.stage_evidence import (
    CAPTURE_TIMEOUT_SECONDS,
    parse_stage_bundle,
    preserve_stage_evidence,
    redact_stage_text,
    release_after_evidence,
)
from media_engine.providers.stage_preflight import assess_manifest_body, local_stage_problems
from media_engine.providers.vast_image import VastImageProvider, _R2_SYNC


def _env(**overrides) -> dict[str, str]:
    values = {
        "MODEL_CACHE_PROVIDER": "r2",
        "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
        "R2_BUCKET": "ai-media-engine-models",
        "R2_ACCESS_KEY_ID": "fixture-access",
        "R2_SECRET_ACCESS_KEY": "fixture-secret",
    }
    values.update(overrides)
    return values


def _manifest() -> bytes:
    return json.dumps({
        "state": COMPLETE,
        "repo_id": MODEL_ID,
        "revision": REVISION,
        "files": [{"path": "weights.safetensors", "bytes": 4}],
    }).encode()


class _FailingStore:
    def __init__(self, status: int = 403) -> None:
        self.status = status
        self.prefixes: list[str] = []
        self.downloads = 0

    def get(self, key: str):
        if key == MANIFEST_NAME and self.status != 200:
            raise R2StorageError(self.status)
        if key == MANIFEST_NAME:
            return _manifest(), {}
        return None

    def list_entries(self, prefix: str):
        self.prefixes.append(prefix)
        return [("hf-cache/models--Qwen--Qwen-Image/blobs/blob", 4)]

    def download_file(self, key: str, destination: Path, should_stop=None) -> dict:
        del key, destination, should_stop
        self.downloads += 1
        raise R2StorageError(403)


class ConfigTests(unittest.TestCase):
    def test_missing_worker_environment_is_named(self) -> None:
        problems = local_stage_problems(_env(R2_SECRET_ACCESS_KEY=""), _R2_SYNC)
        self.assertIn("R2_ENV_INCOMPLETE", problems)

    def test_wrong_bucket_and_plaintext_endpoint_are_named(self) -> None:
        problems = local_stage_problems(
            _env(R2_BUCKET="elsewhere", R2_ENDPOINT="http://example.invalid"),
            _R2_SYNC,
        )
        self.assertEqual(problems, ["R2_ENDPOINT", "R2_BUCKET"])

    def test_enabled_reliable_transfer_is_refused_before_rental(self) -> None:
        problems = local_stage_problems(_env(MEDIA_ENGINE_R2_RELIABLE_TRANSFER="1"), _R2_SYNC)
        self.assertEqual(problems, ["RELIABLE_TRANSFER_NOT_SHIPPED"])

    def test_manifest_revision_must_match_the_pin(self) -> None:
        body = json.loads(_manifest().decode())
        body["revision"] = "not-the-pin"
        self.assertEqual(assess_manifest_body(json.dumps(body).encode()), "MANIFEST_REVISION")
        self.assertEqual(assess_manifest_body(_manifest()), "")

    def test_worker_copy_does_not_import_the_package_when_resume_is_off(self) -> None:
        root = Path(tempfile.mkdtemp())
        source = Path(reliable_transfer_enabled.__code__.co_filename)
        (root / "r2_cache.py").write_bytes(source.read_bytes())
        sibling = source.with_name("model_cache.py")
        (root / "model_cache.py").write_bytes(sibling.read_bytes())
        script = (
            "import builtins, os\n"
            "os.environ.pop('MEDIA_ENGINE_R2_RELIABLE_TRANSFER', None)\n"
            "real = builtins.__import__\n"
            "def guard(name, *args, **kwargs):\n"
            "    if str(name).startswith('media_engine'):\n"
            "        raise ImportError(name)\n"
            "    return real(name, *args, **kwargs)\n"
            "builtins.__import__ = guard\n"
            "import r2_cache\n"
            "print(r2_cache.reliable_transfer_enabled())\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env.pop("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")

    def test_worker_downloads_the_first_file_without_the_package(self) -> None:
        root = Path(tempfile.mkdtemp())
        source = Path(reliable_transfer_enabled.__code__.co_filename)
        (root / "r2_cache.py").write_bytes(source.read_bytes())
        (root / "model_cache.py").write_bytes(source.with_name("model_cache.py").read_bytes())
        script = r"""
import builtins, json, os
from pathlib import Path
os.environ.pop("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", None)
real = builtins.__import__
def guard(name, *args, **kwargs):
    if str(name).startswith("media_engine"):
        raise ImportError(name)
    return real(name, *args, **kwargs)
builtins.__import__ = guard
import r2_cache
counts = {"enabled": 0, "plain": 0, "resume": 0}
original = r2_cache.reliable_transfer_enabled
def counted():
    counts["enabled"] += 1
    value = original()
    print("enabled", value)
    return value
r2_cache.reliable_transfer_enabled = counted
revision = "75e0b4be04f60ec59a75f475837eced720f823b6"
key = "hf-cache/models--Qwen--Qwen-Image/snapshots/" + revision + "/weights.safetensors"
manifest = json.dumps({
    "state": "COMPLETE",
    "repo_id": "Qwen/Qwen-Image",
    "revision": revision,
    "expected_file_count": 1,
    "files": [{"path": "weights.safetensors", "bytes": 4}],
}).encode()
class Store:
    def get(self, name):
        if name == r2_cache.MANIFEST_NAME:
            return manifest, {}
        return None
    def list_entries(self, prefix):
        print("prefix", prefix)
        return [(key, 4)]
    def download_file(self, object_key, destination, should_stop=None):
        counts["plain"] += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"data")
        return {}
    def download_reliable(self, *args, **kwargs):
        counts["resume"] += 1
        raise RuntimeError("resume")
result = r2_cache.sync_complete_cache(Store(), Path("mount"), min_bytes=1)
print("result", result)
print("counts", counts["enabled"], counts["plain"], counts["resume"])
"""
        (root / "run_sync.py").write_text(script, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env.pop("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", None)
        completed = subprocess.run(
            [sys.executable, str(root / "run_sync.py")],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("enabled False", completed.stdout)
        self.assertIn("prefix hf-cache/", completed.stdout)
        self.assertIn("result COMPLETE", completed.stdout)
        self.assertIn("counts 1 1 0", completed.stdout)


class SyncFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mount = Path(self.tmp.name) / "models"
        prior = os.environ.get("MEDIA_ENGINE_R2_RELIABLE_TRANSFER")
        os.environ.pop("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", None)
        self.addCleanup(lambda: os.environ.__setitem__("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", prior) if prior else None)

    def test_authentication_failure_stops_before_a_download(self) -> None:
        store = _FailingStore(403)
        progress = Path(self.tmp.name) / "progress.json"
        with self.assertRaises(R2StorageError) as raised:
            sync_complete_cache(store, self.mount, min_bytes=1, progress_path=progress)
        self.assertEqual(raised.exception.status, 403)
        self.assertFalse(progress.exists())
        self.assertEqual(store.downloads, 0)

    def test_listing_uses_the_cache_prefix(self) -> None:
        store = _FailingStore(200)
        progress = Path(self.tmp.name) / "progress.json"
        with self.assertRaises(R2StorageError):
            sync_complete_cache(store, self.mount, min_bytes=1, progress_path=progress)
        self.assertEqual(store.prefixes, ["hf-cache/"])

    def test_zero_byte_progress_is_written_before_the_download_fails(self) -> None:
        store = _FailingStore(200)
        progress = Path(self.tmp.name) / "progress.json"
        with self.assertRaises(R2StorageError):
            sync_complete_cache(store, self.mount, min_bytes=1, progress_path=progress)
        record = json.loads(progress.read_text(encoding="utf-8"))
        self.assertEqual(record["bytes_done"], 0)
        self.assertEqual(store.downloads, 1)

    def test_missing_resume_module_is_a_cache_error(self) -> None:
        os.environ["MEDIA_ENGINE_R2_RELIABLE_TRANSFER"] = "1"
        real = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "media_engine.engines.image.r2_reliability":
                raise ImportError(name)
            return real(name, *args, **kwargs)

        builtins.__import__ = guard
        try:
            with self.assertRaises(ModelCacheNotReady) as raised:
                reliable_transfer_enabled()
        finally:
            builtins.__import__ = real
            os.environ.pop("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", None)
        self.assertEqual(raised.exception.state, "RELIABILITY_MODULE_MISSING")


class EvidenceTests(unittest.TestCase):
    def test_stderr_and_signed_urls_are_redacted(self) -> None:
        text = "failed https://bucket.example/blob?X-Amz-Signature=abcdef&X-Amz-Credential=secretkey token=fixture-secret"
        cleaned = redact_stage_text(text, ("fixture-secret",))
        self.assertNotIn("abcdef", cleaned)
        self.assertNotIn("secretkey", cleaned)
        self.assertNotIn("fixture-secret", cleaned)
        self.assertIn("[redacted]", cleaned)

    def test_nonzero_exit_keeps_the_worker_log(self) -> None:
        bundle = (
            "---EXIT---\n1\n---PROGRESS---\n"
            '{"MB_per_sec": 0, "bytes_done": 0}\n'
            '---REPORT---\n{"detail": "ImportError"}\n'
            "---LOG---\nImportError: No module named media_engine\n"
        )
        parsed = parse_stage_bundle(bundle)
        self.assertEqual(parsed["exit_code"], "1")
        root = Path(tempfile.mkdtemp())
        record = preserve_stage_evidence(
            destination=root / "stage-evidence.json",
            fetch=lambda: (bundle, "stderr fixture-secret"),
            secrets=("fixture-secret",),
            exit_code="",
            bytes_done=None,
            started_at=10.0,
            ended_at=12.0,
            timeout_seconds=2,
        )
        self.assertEqual(record["exit_code"], "1")
        self.assertEqual(record["bytes_transferred"], 0)
        self.assertIn("ImportError", record["cache_sync_log"])
        self.assertNotIn("fixture-secret", record["stderr"])
        self.assertEqual(record["stage_started_at"], 10.0)
        saved = json.loads((root / "stage-evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["report"], '{"detail": "ImportError"}')

    def test_fetch_failure_is_recorded(self) -> None:
        root = Path(tempfile.mkdtemp())

        def fetch():
            raise OSError("ssh closed")

        record = preserve_stage_evidence(
            destination=root / "stage-evidence.json",
            fetch=fetch,
            timeout_seconds=2,
        )
        self.assertEqual(record["capture_error"], "OSError")
        self.assertTrue((root / "stage-evidence.json").is_file())

    def test_capture_timeout_returns(self) -> None:
        root = Path(tempfile.mkdtemp())

        def fetch():
            time.sleep(2)
            return "---EXIT---\n1\n", ""

        started = time.monotonic()
        record = preserve_stage_evidence(
            destination=root / "stage-evidence.json",
            fetch=fetch,
            timeout_seconds=0.05,
        )
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(record["capture_error"], "CAPTURE_TIMEOUT")

    def test_termination_runs_when_capture_fails_or_times_out(self) -> None:
        calls = []

        def boom():
            raise OSError("down")

        failed = release_after_evidence(boom, lambda: calls.append("delete"), timeout_seconds=1)
        self.assertEqual(calls, ["delete"])
        self.assertEqual(failed["capture_error"], "OSError")
        calls.clear()

        def slow():
            time.sleep(2)
            return {"cache_sync_log": "late"}

        started = time.monotonic()
        record = release_after_evidence(slow, lambda: calls.append("delete"), timeout_seconds=0.05)
        self.assertEqual(calls, ["delete"])
        self.assertEqual(record["capture_error"], "CAPTURE_TIMEOUT")
        self.assertLess(time.monotonic() - started, 1)

    def test_hung_capture_releases_the_gpu_at_the_eight_second_limit(self) -> None:
        self.assertEqual(CAPTURE_TIMEOUT_SECONDS, 8)
        calls = []

        def hang():
            time.sleep(30)
            return {"cache_sync_log": "late"}

        started = time.monotonic()
        record = release_after_evidence(hang, lambda: calls.append("delete"))
        elapsed = time.monotonic() - started
        self.assertEqual(calls, ["delete"])
        self.assertEqual(record["capture_error"], "CAPTURE_TIMEOUT")
        self.assertGreaterEqual(elapsed, 7.5)
        self.assertLess(elapsed, 9.0)

    def test_provider_keeps_a_nonzero_sync_log_before_the_error(self) -> None:
        provider = VastImageProvider(
            api_key="fixture-key", runtime=Path(tempfile.mkdtemp()), now=lambda: 5_000.0,
        )
        provider._sleep = lambda _seconds: None
        provider._host = "worker"
        provider._port = "22"

        def ssh(_host, _port, command, _timeout):
            if "---EXIT---" in command:
                stderr = "Authorization: Bearer fixture-key"
                return 0, (
                    "---EXIT---\n7\n---PROGRESS---\n"
                    '{"MB_per_sec": 0.0, "bytes_done": 0}\n'
                    "---REPORT---\n{}\n---LOG---\nboom\n"
                ), stderr
            if "cache-sync.exit" in command:
                return 0, "7\n", ""
            if "r2-sync-progress" in command:
                return 0, '{"MB_per_sec": 0.0, "bytes_done": 0, "bytes_total": 4}\n', ""
            return 0, "STARTED\n", ""

        provider._ssh_run = ssh
        with self.assertRaises(WorkerStageError) as raised:
            provider._watch_cache_sync()
        self.assertEqual(raised.exception.code, "MODEL_CACHE_INVALID")
        saved = json.loads((provider.runtime / "stage-evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["exit_code"], "7")
        self.assertEqual(saved["bytes_transferred"], 0)
        self.assertIn("boom", saved["cache_sync_log"])
        self.assertNotIn("fixture-key", saved["stderr"])
        self.assertIn("[redacted]", saved["stderr"])
