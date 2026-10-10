"""Integration of the GPU manager with the existing queue, cache, and Vast read path."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from media_engine.infra.cache import PINNED_REVISION, RevisionCache
from media_engine.infra.dashboard import snapshot_for
from media_engine.infra.manager import GpuManager
from media_engine.infra.ownership import SharedOwnership
from media_engine.infra.pipeline import PipelineError, IntegratedSession
from media_engine.infra.policies import session_contract
from media_engine.infra.store import ManagerStore
from media_engine.infra.vast_read import ReadOnlyRefusal, VastReadOnly
from media_engine.infra.warm import WarmLease, WarmWorker
from media_engine.infra.worker_cache import persistent_cache_hit
from media_engine.platform.repository import SqliteRepository
from media_engine.platform.service import Platform
from media_engine.providers.vast_image import BUNDLES, VastImageProvider


def _offer(offer_id: int, machine_id: int, **overrides) -> dict:
    row = {
        "id": offer_id,
        "machine_id": machine_id,
        "gpu_name": "RTX 6000Ada",
        "gpu_ram": 49140,
        "num_gpus": 1,
        "cuda_max_good": 13.0,
        "compute_cap": 890,
        "disk_space": 250,
        "inet_down": 10000,
        "reliability2": 0.99,
        "verification": "verified",
        "rentable": True,
        "geolocation": "California, US",
        "dph_base": "0.50",
        "dph_total": "0.55",
        "storage_cost": "0.20",
    }
    row.update(overrides)
    return row


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Source:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.reads: list[str] = []

    def read(self, name: str) -> bytes:
        self.reads.append(name)
        return self.body


class _Loader:
    def __init__(self) -> None:
        self.order: list[str] = []

    def load(self, revision: str) -> int:
        del revision
        return 48000

    def generate(self, job: dict) -> dict:
        self.order.append(job["id"])
        return {"id": job["id"]}


class _Market:
    def __init__(self, offers: list, instances: list, *, fail: bool = False) -> None:
        self.offers = offers
        self.instances = instances
        self.fail = fail
        self.methods: list[str] = []

    def __call__(self, method: str, url: str, body: dict | None) -> tuple[int, bytes]:
        self.methods.append(method)
        if method not in {"GET", "POST"}:
            raise AssertionError(method)
        if self.fail and method == "POST":
            return 503, b""
        if method == "POST":
            return 200, json.dumps({"offers": self.offers}).encode()
        return 200, json.dumps({"instances": self.instances, "next_token": None}).encode()


class InventoryTests(unittest.TestCase):
    def test_mocked_bundle_search_ranks_and_a_missing_offer_is_not_rentable(self) -> None:
        clock = _Clock(1_000)
        market = _Market(
            [_offer(2, 20, inet_down=1000), _offer(1, 10, inet_down=20000)],
            [],
        )
        source = VastReadOnly(market)
        store = ManagerStore(Path(tempfile.mkdtemp()) / "gpu.sqlite3")
        manager = GpuManager(store, now=clock)
        ranked = manager.pull(source, ceiling=Decimal("5.00"))
        self.assertEqual([item.offer_id for item in ranked], [1, 2])
        self.assertTrue(all(call.startswith("POST") or call.startswith("GET") for call in source.calls))
        self.assertNotIn("PUT", market.methods)
        gone = source.machine_offers(10)
        missing, reason = manager.verify_pair(gone, 99, 99, ceiling=Decimal("5.00"))
        self.assertIsNone(missing)
        self.assertEqual(reason, "OFFER_UNAVAILABLE")
        with self.assertRaises(ReadOnlyRefusal):
            source._call("PUT", "https://console.vast.ai/api/v0/asks/1/", None)
        decision = manager.evaluate_capacity("8h", None)
        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["reason"], "PAID_ACTIONS_DISABLED")

    def test_provider_error_backs_off_and_ambiguous_create_survives_restart(self) -> None:
        clock = _Clock(2_000)
        market = _Market([], [], fail=True)
        store = ManagerStore(Path(tempfile.mkdtemp()) / "gpu.sqlite3")
        manager = GpuManager(store, now=clock)
        manager.pull(VastReadOnly(market), ceiling=Decimal("5.00"))
        self.assertEqual(manager.alerts[-1]["code"], "DISCOVERY_FAILED")
        self.assertGreater(manager.next_pull_at, clock.value)
        manager.note_ambiguous_create()
        restarted = GpuManager(store, now=clock)
        decision = restarted.evaluate_capacity("1h", {"execution_id": "future-run", "expires_at": 9_000})
        self.assertEqual(decision["reason"], "AMBIGUOUS_CREATE")
        self.assertFalse(restarted.store.authorizes_rental())
        self.assertEqual(restarted.provider_creates, 0)

    def test_watchdog_stops_a_reconciled_instance(self) -> None:
        clock = _Clock(3_000)
        instance = {"id": "77", "machine_id": 10, "label": "media-engine:run", "dph_total": "0.55"}
        market = _Market([], [instance])
        store = ManagerStore(Path(tempfile.mkdtemp()) / "gpu.sqlite3")
        manager = GpuManager(store, now=clock)
        manager.pull(VastReadOnly(market), ceiling=Decimal("5.00"))
        self.assertEqual(manager.instance_id, "77")
        targets = manager.request_shutdown([instance], heartbeat_age=301, deadline=None)
        self.assertEqual(targets, ["77"])
        self.assertEqual(manager.state, "STOPPED")
        self.assertEqual(manager.provider_creates, 0)


class QueueAndCacheTests(unittest.TestCase):
    def _session(self, now: float = 100.0, started: float = 0.0):
        clock = _Clock(now)
        root = Path(tempfile.mkdtemp())
        platform = Platform(
            str(root / "platform.sqlite3"),
            now=lambda: datetime(2026, 10, 11, tzinfo=timezone.utc),
        )
        platform.set_pool_priority(["topup"])
        platform.add_price(
            version="v1", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=1,
            effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 6, "topup_purchase", idempotency_key="grant-infra2")
        cache = RevisionCache(
            root / "cache", revision="rev", expected_files=1, expected_bytes=2, enforce_pin=False,
        )
        worker = WarmWorker(
            revision="rev",
            lease=WarmLease(revision="rev", expires_at=now + 10_000, paid=False),
            loader=_Loader(),
            now=clock,
        )
        manager = GpuManager(ManagerStore(root / "gpu.sqlite3"), now=clock)
        session = IntegratedSession(
            manager=manager, worker=worker, cache=cache, platform=platform,
            source=VastReadOnly(_Market([], [])), policy_name="1h", started_at=started, now=clock,
        )
        return session, platform, cache

    def test_sequential_jobs_settle_once_and_a_duplicate_does_not_reserve_again(self) -> None:
        session, platform, cache = self._session()
        source = _Source(b"aa")
        files = [{"path": "a.txt", "bytes": 2}]
        session.submit("nahaber", "job-1")
        session.submit("nahaber", "job-1")
        self.assertEqual(platform.balance("nahaber"), 4)
        session.submit("nahaber", "job-2")
        first = session.run("nahaber", "rev", files, source)
        second = session.run("nahaber", "rev", files, source)
        self.assertEqual(first["jobs"], ["job-1", "job-2"])
        self.assertEqual(first["cache"], "STAGED")
        self.assertEqual(second["cache"], "HIT")
        self.assertEqual(source.reads, ["a.txt"])
        self.assertEqual(session.worker.loader.order, ["job-1", "job-2"])
        self.assertEqual(platform.balance("nahaber"), 2)
        self.assertEqual(session.manager.provider_creates, 0)
        self.assertEqual(cache.lookup("rev"), "HIT")

    def test_failed_job_refunds_once_and_budget_expiry_rejects_another(self) -> None:
        session, platform, _cache = self._session()
        session.submit("nahaber", "job-9")
        self.assertEqual(session.fail("nahaber", "job-9"), "refunded")
        self.assertEqual(session.fail("nahaber", "job-9"), "settled")
        self.assertEqual(platform.balance("nahaber"), 6)
        expired, _platform, _cache = self._session(now=5_000, started=0)
        with self.assertRaises(PipelineError) as raised:
            expired.submit("nahaber", "job-10")
        self.assertEqual(raised.exception.code, "BUDGET_EXHAUSTED")

    def test_two_workers_cannot_claim_one_job_and_live8_auth_is_not_reused(self) -> None:
        root = Path(tempfile.mkdtemp()) / "own.sqlite3"
        repo = SqliteRepository(str(root))
        now = "2026-10-11T00:00:00Z"
        repo.ensure_user("nahaber", now)
        repo.enqueue(
            user_id="nahaber", job_id="job-1", idempotency_key="job:job-1",
            pricing_version="v1", max_concurrent=1, now=now,
        )
        ownership = SharedOwnership(repo)
        found: list[str | None] = []

        def claim() -> None:
            found.append(ownership.claim("worker", now, "2026-10-11T00:01:00Z"))

        threads = [threading.Thread(target=claim) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(item or "" for item in found), ["", "job-1"])
        self.assertFalse(ownership.authorizes_rental())
        self.assertFalse(ownership.note_authorization("live8-run-1011a"))
        self.assertFalse(ownership.note_authorization("future-run"))
        postgres = Path("src/media_engine/platform/postgres_repo.py").read_text(encoding="utf-8")
        self.assertIn("FOR UPDATE SKIP LOCKED", postgres)
        self.assertNotIn("open_production_stores", Path("src/media_engine/infra/pipeline.py").read_text(encoding="utf-8"))

    def test_persistent_cache_hit_skips_the_worker_download(self) -> None:
        root = Path(tempfile.mkdtemp()) / "models"
        cache = RevisionCache(root, revision=PINNED_REVISION, expected_files=1, expected_bytes=2)
        cache.stage(PINNED_REVISION, _Source(b"aa"), [{"path": "a.txt", "bytes": 2}])
        self.assertTrue(persistent_cache_hit(str(root)))
        calls = []

        def transport(method, url, body, headers):
            calls.append(method)
            return b""

        provider = VastImageProvider(api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()))
        provider.cache_source = "r2_cold_stage"
        previous = __import__("os").environ.get("MEDIA_ENGINE_MODEL_CACHE")
        __import__("os").environ["MEDIA_ENGINE_MODEL_CACHE"] = str(root)
        try:
            provider.confirm_cache("1")
        finally:
            if previous is None:
                __import__("os").environ.pop("MEDIA_ENGINE_MODEL_CACHE", None)
            else:
                __import__("os").environ["MEDIA_ENGINE_MODEL_CACHE"] = previous
        self.assertEqual(provider.cache_validation, "COMPLETE")
        self.assertEqual(provider.cache_source, "persistent_revision_cache")
        self.assertEqual(calls, [])
        self.assertNotEqual(BUNDLES, "")

    def test_admin_snapshot_lists_disabled_session_budgets(self) -> None:
        contracts = [session_contract(name) for name in ("1h", "8h", "10h", "24h")]
        self.assertTrue(all(item["paid_execution"] is False for item in contracts))
        self.assertEqual([item["hours"] for item in contracts], [1, 8, 10, 24])
        clock = _Clock(10)
        manager = GpuManager(ManagerStore(Path(tempfile.mkdtemp()) / "gpu.sqlite3"), now=clock)
        view = snapshot_for(manager)
        self.assertFalse(view["paid_actions_enabled"])
        self.assertEqual(len(view["session_budgets"]), 4)
        self.assertEqual(view["billing_exposure"]["provider_creates"], 0)
        self.assertNotIn("api_key", json.dumps(view))


if __name__ == "__main__":
    unittest.main()
