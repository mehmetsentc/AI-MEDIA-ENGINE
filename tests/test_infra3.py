"""Live-session readiness. No provider create and no production write."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.model_cache import mark_complete, snapshot_dir
from media_engine.engines.image.qwen import MODEL_ID
from media_engine.engines.image.qwen_remote import (
    PINNED_QWEN_REVISION,
    cache_is_ready,
    resolve_worker_mount,
    run_session,
)
from media_engine.infra.live_session import (
    SUCCESSFUL_ADA_MACHINE_ID,
    authorization_template,
    cache_strategy,
    warm_session_cost,
)
from media_engine.infra.manager import GpuManager
from media_engine.infra.pipeline import IntegratedSession
from media_engine.infra.store import ManagerStore
from media_engine.infra.vast_read import VastReadOnly
from media_engine.infra.warm import WarmLease, WarmWorker
from media_engine.platform.service import Platform
from media_engine.providers.phase2d import r2_worker_env
from media_engine.providers.vast_image import MACHINE_ID, VOLUME_ID, VastImageProvider, _CACHE_CHECK


def _worker_tree(root: Path, revision: str, body: bytes = b"aa") -> None:
    snapshot = snapshot_dir(root, revision)
    snapshot.mkdir(parents=True)
    (snapshot / "a.txt").write_bytes(body)
    mark_complete(
        root, revision=revision, files=[{"path": "a.txt", "bytes": len(body)}],
        completed_at="2026-10-11T00:00:00Z",
    )


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Loader:
    def __init__(self) -> None:
        self.loads = 0
        self.order: list[str] = []

    def load(self, revision: str) -> int:
        del revision
        self.loads += 1
        return 48000

    def generate(self, job: dict) -> dict:
        self.order.append(job["id"])
        return {"id": job["id"]}


class CacheConnectionTests(unittest.TestCase):
    def test_worker_layout_hit_skips_download_and_a_local_revision_cache_does_not(self) -> None:
        root = Path(tempfile.mkdtemp())
        worker = root / "models"
        _worker_tree(worker, PINNED_QWEN_REVISION)
        self.assertTrue(cache_is_ready(worker, min_bytes=2))
        other = root / "other"
        _worker_tree(other, "not-the-pin")
        self.assertFalse(cache_is_ready(other, min_bytes=2))
        self.assertEqual(resolve_worker_mount({"cache_mount": "/models"}), Path("/models"))

        downloads = []

        def sync(mount: Path) -> None:
            downloads.append(mount)
            _worker_tree(mount, PINNED_QWEN_REVISION)

        loaded = []

        def load(mount: Path) -> dict:
            loaded.append(mount)
            return {"mount": str(mount)}

        def generate(pipe: dict, job: dict) -> dict:
            return {"pipe": pipe, "id": job["id"]}

        empty = root / "empty-models"
        empty.mkdir()
        first = run_session(
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            mount=empty, cache_ready=lambda path: cache_is_ready(path, min_bytes=2),
            sync=sync, load=load, generate=generate,
        )
        second = run_session(
            [{"id": "d"}, {"id": "e"}, {"id": "f"}],
            mount=worker, cache_ready=lambda path: cache_is_ready(path, min_bytes=2),
            sync=sync, load=load, generate=generate,
        )
        self.assertEqual(first["downloads"], 1)
        self.assertEqual(first["loads"], 1)
        self.assertEqual(len(first["outputs"]), 3)
        self.assertEqual(second["downloads"], 0)
        self.assertEqual(second["loads"], 1)
        self.assertEqual(len(downloads), 1)

    def test_remote_complete_report_skips_staging_and_a_miss_does_not(self) -> None:
        from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT, REPO_ID
        hit = json.dumps({
            "state": "COMPLETE",
            "repo": REPO_ID,
            "revision": PINNED_QWEN_REVISION,
            "files": QWEN_FILE_COUNT,
            "bytes": QWEN_CACHE_MIN_BYTES,
            "mount": False,
            "entries": [],
        })
        staged: list[str] = []
        seen: list[str] = []

        def ssh(host, port, command, timeout):
            del host, port, timeout
            seen.append(command)
            if staged:
                return 0, hit, ""
            return 0, json.dumps({"state": "MISSING", "mount": False, "entries": []}), ""

        provider = VastImageProvider(api_key="fixture-key", transport=lambda *args: b"", runtime=Path(tempfile.mkdtemp()))
        provider.cache_source = "r2_cold_stage"
        provider._ssh_run = ssh
        provider._host = "worker"
        provider._port = "22"
        provider._stage_r2 = lambda: staged.append("r2")
        provider.confirm_cache("1")
        self.assertEqual(staged, ["r2"])
        provider.cache_source = "r2_cold_stage"
        provider.confirm_cache("1")
        self.assertEqual(staged, ["r2"])
        self.assertEqual(provider.cache_source, "worker_cache_hit")
        self.assertIn("MEDIA_ENGINE_MODEL_CACHE", _CACHE_CHECK)
        self.assertEqual(provider.create_calls if hasattr(provider, "create_calls") else 0, 0)
        self.assertTrue(seen)

    def test_worker_env_forwards_only_the_gpu_mount(self) -> None:
        base = {
            "MODEL_CACHE_PROVIDER": "r2",
            "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
            "R2_BUCKET": "models",
            "R2_ACCESS_KEY_ID": "access",
            "R2_SECRET_ACCESS_KEY": "secret",
        }
        laptop = r2_worker_env({**base, "MEDIA_ENGINE_MODEL_CACHE": "/Users/example/cache"})
        worker = r2_worker_env({**base, "MEDIA_ENGINE_MODEL_CACHE": "/models"})
        self.assertNotIn("/Users/example/cache", laptop.decode())
        self.assertIn("MEDIA_ENGINE_MODEL_CACHE=/models", worker.decode())
        self.assertIn(PINNED_QWEN_REVISION, worker.decode())


class SessionPlanTests(unittest.TestCase):
    def test_volume_cannot_serve_the_ada_machine_and_deletion_requires_another_download(self) -> None:
        plan = cache_strategy()
        self.assertEqual(plan["choice"], "r2_once_then_retained_instance")
        self.assertTrue(plan["download_after_deletion"])
        self.assertFalse(plan["volume_same_machine"])
        self.assertFalse(plan["volume_large_enough"])
        self.assertNotEqual(MACHINE_ID, SUCCESSFUL_ADA_MACHINE_ID)
        self.assertEqual(VOLUME_ID, 54653022)
        cost = warm_session_cost()
        self.assertEqual(cost["images"], 3)
        self.assertEqual(cost["load_seconds"], 600)
        self.assertEqual(cost["deletion_reserve_seconds"], 840)
        self.assertGreater(Decimal(cost["total_usd"]), Decimal(cost["disk_usd"]))
        self.assertLess(Decimal(cost["total_usd"]), Decimal("5.00"))
        template = authorization_template()
        self.assertFalse(template["issued"])
        self.assertFalse(template["consumed"])
        self.assertFalse(template["allows_paid_create"])
        self.assertIn("live8-run-1011a", template["rejected_execution_ids"])
        self.assertEqual(template["images"], 3)
        self.assertEqual(template["model_loads"], 1)
        self.assertEqual(template["max_creates"], 1)
        self.assertEqual(MODEL_ID, "Qwen/Qwen-Image")

    def test_three_queued_jobs_load_once_and_do_not_rent(self) -> None:
        clock = _Clock(100)
        root = Path(tempfile.mkdtemp())
        platform = Platform(str(root / "platform.sqlite3"), now=lambda: datetime(2026, 10, 11, tzinfo=timezone.utc))
        platform.set_pool_priority(["topup"])
        platform.add_price(
            version="v1", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=1,
            effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 8, "topup_purchase", idempotency_key="infra3-grant")
        loader = _Loader()
        worker = WarmWorker(
            revision="rev",
            lease=WarmLease(revision="rev", expires_at=50_000, paid=False),
            loader=loader,
            now=clock,
        )
        manager = GpuManager(ManagerStore(root / "gpu.sqlite3"), now=clock)
        session = IntegratedSession(
            manager=manager, worker=worker, cache=_Cache(root / "cache"),
            platform=platform, source=VastReadOnly(lambda *args: (200, b'{"offers":[]}')),
            policy_name="1h", started_at=0, now=clock,
        )
        for name in ("job-1", "job-2", "job-3"):
            session.submit("nahaber", name)
        session.submit("nahaber", "job-1")
        result = session.run("nahaber", "rev", [{"path": "a.txt", "bytes": 2}], _Body())
        self.assertEqual(result["jobs"], ["job-1", "job-2", "job-3"])
        self.assertEqual(loader.loads, 1)
        self.assertEqual(loader.order, ["job-1", "job-2", "job-3"])
        self.assertEqual(platform.balance("nahaber"), 2)
        self.assertEqual(manager.provider_creates, 0)
        self.assertFalse(manager.evaluate_capacity("1h", {"execution_id": "live8-run-1011a", "expires_at": 9_000})["allowed"])
        worker.crash()
        worker.restart()
        self.assertEqual(worker.loads, 2)
        self.assertEqual(manager.provider_creates, 0)
        manager.reconcile([{"id": "77", "machine_id": 10, "label": "media-engine:run", "dph_total": "0.55"}])
        self.assertEqual(manager.request_shutdown(
            [{"id": "77", "machine_id": 10, "label": "media-engine:run"}],
            heartbeat_age=301, deadline=None,
        ), ["77"])


class _Body:
    def read(self, name: str) -> bytes:
        del name
        return b"aa"


class _Cache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.staged = False

    def lookup(self, revision: str) -> str:
        del revision
        return "HIT" if self.staged else "MISS"

    def stage(self, revision: str, source, files: list[dict]) -> str:
        del revision, files
        source.read("a.txt")
        self.staged = True
        return "STAGED"


if __name__ == "__main__":
    unittest.main()
