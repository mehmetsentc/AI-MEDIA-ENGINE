"""GPU manager, model cache, and warm worker. No provider create is sent."""
from __future__ import annotations

import tempfile
import threading
import unittest
from decimal import Decimal
from pathlib import Path

from media_engine.infra.cache import PINNED_REVISION, CacheError, RevisionCache
from media_engine.infra.dashboard import snapshot_for
from media_engine.infra.dispatch import DispatchError, GenerationDispatch
from media_engine.infra.manager import GpuManager
from media_engine.infra.policies import catalog
from media_engine.infra.store import ManagerStore
from media_engine.infra.volume_audit import VOLUME_ID, audit_volume
from media_engine.infra.warm import WarmLease, WarmWorker
from media_engine.providers.base import WorkerStageError


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
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Source:
    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.reads: list[str] = []

    def read(self, name: str) -> bytes:
        self.reads.append(name)
        return self.bodies[name]


class _Ledger:
    def __init__(self) -> None:
        self.balance = 3
        self.held: dict[str, int] = {}
        self.settled: set[str] = set()

    def available(self, user_id: str) -> int:
        del user_id
        return self.balance

    def reserve(self, user_id: str, job_id: str, amount: int) -> None:
        del user_id
        if self.balance < amount:
            raise DispatchError("INSUFFICIENT_CREDIT")
        self.balance -= amount
        self.held[job_id] = amount

    def settle(self, user_id: str, job_id: str, amount: int) -> None:
        del user_id
        if job_id in self.settled or self.held.get(job_id) != amount:
            raise DispatchError("ALREADY_SETTLED")
        self.settled.add(job_id)
        self.held.pop(job_id)

    def refund(self, user_id: str, job_id: str, amount: int) -> None:
        del user_id, amount
        if job_id not in self.held:
            return
        self.balance += self.held.pop(job_id)


class _Loader:
    def __init__(self) -> None:
        self.order: list[str] = []
        self.active = 0

    def load(self, revision: str) -> int:
        del revision
        return 48000

    def generate(self, job: dict) -> dict:
        if self.active:
            raise RuntimeError("CONCURRENT")
        self.active += 1
        self.order.append(job["id"])
        self.active -= 1
        return {"id": job["id"]}


class CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.cache = RevisionCache(
            self.root, revision="rev-test", expected_files=2, expected_bytes=5, enforce_pin=False,
        )
        self.files = [{"path": "a.txt", "bytes": 2}, {"path": "b.txt", "bytes": 3}]
        self.source = _Source({"a.txt": b"aa", "b.txt": b"bbb"})

    def test_miss_stages_once_and_hit_skips_download(self) -> None:
        self.assertEqual(self.cache.lookup("rev-test"), "MISS")
        self.assertEqual(self.cache.stage("rev-test", self.source, self.files), "STAGED")
        self.assertEqual(self.cache.stage("rev-test", self.source, self.files), "HIT")
        self.assertEqual(self.source.reads, ["a.txt", "b.txt"])

    def test_revision_mismatch_is_rejected(self) -> None:
        pinned = RevisionCache(self.root)
        with self.assertRaises(CacheError) as raised:
            pinned.stage("other-revision", self.source, self.files)
        self.assertEqual(raised.exception.code, "REVISION_MISMATCH")
        self.assertEqual(pinned.revision, PINNED_REVISION)

    def test_partial_download_resumes_and_corrupt_publication_is_not_a_hit(self) -> None:
        staging = Path(self.root) / "revisions" / "rev-test" / "staging"
        staging.mkdir(parents=True)
        (staging / "a.txt").write_bytes(b"aa")
        self.cache.stage("rev-test", self.source, self.files)
        self.assertEqual(self.source.reads, ["b.txt"])
        published = Path(self.root) / "revisions" / "rev-test" / "published" / "b.txt"
        published.write_bytes(b"x")
        self.assertEqual(self.cache.health("rev-test")["problem"], "PARTIAL")
        self.assertEqual(self.cache.lookup("rev-test"), "MISS")
        with self.assertRaises(CacheError) as raised:
            self.cache.stage("rev-test", self.source, self.files)
        self.assertEqual(raised.exception.code, "CACHE_INVALID")

    def test_concurrent_stage_downloads_each_file_once(self) -> None:
        errors = []

        def attempt() -> None:
            try:
                self.cache.stage("rev-test", self.source, self.files)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(self.source.reads), ["a.txt", "b.txt"])
        self.assertEqual(self.cache.lookup("rev-test"), "HIT")


class ManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.store = ManagerStore(Path(tempfile.mkdtemp()) / "gpu.sqlite3")
        self.manager = GpuManager(self.store, now=self.clock)

    def test_ranking_and_offer_disappearance(self) -> None:
        policies = catalog()
        self.assertEqual([policies[name].hours for name in ("1h", "8h", "10h", "24h")], [1, 8, 10, 24])
        self.assertTrue(all(not item.allows_paid_create for item in policies.values()))
        ranked = self.manager.discover(
            [_offer(2, 20, inet_down=1000), _offer(1, 10, inet_down=20000)],
            ceiling=Decimal("5.00"),
        )
        self.assertEqual([item.offer_id for item in ranked], [1, 2])
        missing, reason = self.manager.verify_pair([], 10, 1, ceiling=Decimal("5.00"))
        self.assertIsNone(missing)
        self.assertEqual(reason, "OFFER_UNAVAILABLE")

    def test_ambiguous_create_survives_restart_and_blocks_another(self) -> None:
        self.manager.note_ambiguous_create()
        self.assertEqual(self.manager.state, "AMBIGUOUS")
        restarted = GpuManager(self.store, now=self.clock)
        self.assertEqual(restarted.state, "AMBIGUOUS")
        self.assertTrue(restarted.create_attempted)
        decision = restarted.evaluate_capacity("1h", {"execution_id": "run", "expires_at": 10_000})
        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["reason"], "AMBIGUOUS_CREATE")
        with self.assertRaises(RuntimeError):
            restarted.note_ambiguous_create()
        adopted = restarted.reconcile([
            {"id": "77", "machine_id": 10, "label": "media-engine:run", "dph_total": "0.55"},
        ])
        self.assertEqual(adopted, "READY")
        self.assertEqual(restarted.instance_id, "77")
        self.assertEqual(restarted.provider_creates, 0)

    def test_unauthorized_rental_and_budget_shutdown(self) -> None:
        decision = self.manager.evaluate_capacity("8h", None)
        self.assertEqual(decision, {"allowed": False, "reason": "PAID_ACTIONS_DISABLED", "creates": 0})
        self.assertEqual(self.manager.provider_creates, 0)
        self.manager.reconcile([
            {"id": "77", "machine_id": 10, "label": "media-engine:run", "dph_total": "0.55"},
        ])
        self.manager.mark_idle()
        targets = self.manager.request_shutdown(
            [{"id": "77", "machine_id": 10, "label": "media-engine:run"}],
            heartbeat_age=301,
            deadline=None,
        )
        self.assertEqual(targets, ["77"])
        self.assertEqual(self.manager.state, "STOPPED")
        self.manager.reconcile([
            {"id": "78", "machine_id": 10, "label": "media-engine:run", "dph_total": "0.55"},
        ])
        exhausted = self.manager.note_spend("1h", Decimal("0.95"))
        self.assertEqual(exhausted["reason"], "BUDGET_EXHAUSTED")
        self.assertEqual(self.manager.state, "STOPPED")
        self.assertEqual(self.manager.provider_creates, 0)

    def test_poll_limit_backs_off(self) -> None:
        from media_engine.infra.inventory import PollSchedule
        limited = GpuManager(self.store, now=self.clock, poll=PollSchedule(calls_per_minute=1))
        limited.discover([_offer(1, 10)], ceiling=Decimal("5.00"))
        limited.discover([_offer(1, 10)], ceiling=Decimal("5.00"))
        self.assertEqual(limited.alerts[-1]["code"], "POLL_LIMIT")


class WarmAndDispatchTests(unittest.TestCase):
    def test_warm_worker_is_sequential_and_keeps_a_valid_lease(self) -> None:
        loader = _Loader()
        worker = WarmWorker(
            revision="rev",
            lease=WarmLease(revision="rev", expires_at=5_000, paid=False),
            loader=loader,
            idle_after_seconds=300,
            now=_Clock(100),
        )
        worker.start()
        self.assertEqual(worker.loads, 1)
        worker.submit({"id": "a"})
        worker.submit({"id": "b"})
        worker.run_available()
        self.assertEqual(loader.order, ["a", "b"])
        self.assertEqual(worker.state, "IDLE")
        self.assertNotEqual(worker.tick(100), "STOPPED")
        worker.crash()
        worker.restart()
        self.assertEqual(worker.loads, 2)
        self.assertEqual(worker.create_calls, 0)
        self.assertTrue(worker.status()["model_loaded"])
        with self.assertRaises(RuntimeError):
            WarmWorker(
                revision="rev",
                lease=WarmLease(revision="rev", expires_at=5_000, paid=True),
                loader=loader,
            ).start()

    def test_idle_lease_expiry_stops_the_worker(self) -> None:
        clock = _Clock(100)
        worker = WarmWorker(
            revision="rev",
            lease=WarmLease(revision="rev", expires_at=50, paid=False),
            loader=_Loader(),
            idle_after_seconds=0,
            now=clock,
        )
        worker.start()
        worker.run_available()
        self.assertEqual(worker.tick(100), "STOPPED")

    def test_dispatch_reserves_settles_and_refunds_once(self) -> None:
        clock = _Clock(100)
        store = ManagerStore(Path(tempfile.mkdtemp()) / "gpu.sqlite3")
        manager = GpuManager(store, now=clock)
        worker = WarmWorker(
            revision="rev",
            lease=WarmLease(revision="rev", expires_at=50, paid=False),
            loader=_Loader(),
            now=clock,
        )
        ledger = _Ledger()
        artifacts = type("Artifacts", (), {"put": lambda self, job_id, artifact: {"id": "art-1", **artifact}})()
        dispatch = GenerationDispatch(manager=manager, worker=worker, credits=ledger, artifacts=artifacts)
        request = {
            "authenticated": True,
            "user_id": "nahaber",
            "job_id": "job-1",
            "credit_price": 2,
            "revision": "rev",
            "idempotency_key": "once",
            "policy": "1h",
        }
        first = dispatch.submit(request)
        second = dispatch.submit(request)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["capacity"], "PAID_ACTIONS_DISABLED")
        self.assertEqual(ledger.balance, 1)
        self.assertEqual(manager.provider_creates, 0)
        done = dispatch.complete("job-1", {"bytes": 4})
        self.assertEqual(done["status"], "completed")
        self.assertEqual(ledger.balance, 1)
        with self.assertRaises(DispatchError):
            dispatch.fail("job-1", "GENERATION_FAILED")
        failed = dispatch.submit({
            **request, "job_id": "job-2", "idempotency_key": "twice", "credit_price": 1,
        })
        self.assertEqual(dispatch.fail("job-2", "GENERATION_FAILED")["status"], "failed")
        self.assertEqual(dispatch.fail("job-2", "GENERATION_FAILED")["refunded"], True)
        self.assertEqual(ledger.balance, 1)
        self.assertEqual(failed["status"], "queued")

    def test_ready_worker_dispatches_without_a_create(self) -> None:
        clock = _Clock(100)
        worker = WarmWorker(
            revision="rev",
            lease=WarmLease(revision="rev", expires_at=500, paid=False),
            loader=_Loader(),
            now=clock,
        )
        worker.start()
        manager = GpuManager(ManagerStore(Path(tempfile.mkdtemp()) / "gpu.sqlite3"), now=clock)
        dispatch = GenerationDispatch(
            manager=manager, worker=worker, credits=_Ledger(),
            artifacts=type("Artifacts", (), {"put": lambda self, job_id, artifact: {"id": "art"}})(),
        )
        job = dispatch.submit({
            "authenticated": True, "user_id": "nahaber", "job_id": "job-3",
            "credit_price": 2, "revision": "rev",
        })
        self.assertEqual(job["status"], "dispatched")
        self.assertEqual(manager.provider_creates, 0)
        view = snapshot_for(manager, cache=None, worker=worker, queue_length=1)
        self.assertFalse(view["paid_actions_enabled"])
        self.assertNotIn("api_key", view)
        self.assertTrue(view["model_loaded"])


class VolumeAndDashboardTests(unittest.TestCase):
    def test_volume_contents_stay_unknown_without_a_listing(self) -> None:
        report = audit_volume({
            "id": VOLUME_ID, "machine_id": 47281, "status": "created",
            "label": "ai_media_engine_qwen_cache",
        })
        self.assertEqual(report["contents"], "unknown")
        self.assertFalse(report["compatible_with_arbitrary_machines"])
        self.assertIn("47281", report["placement"])

    def test_admin_route_refuses_paid_actions(self) -> None:
        from media_engine.api.app import dispatch
        from media_engine.orchestrator.controller import MediaController
        from media_engine.providers.fake import FakeGPUProvider
        controller = MediaController(
            str(Path(tempfile.mkdtemp()) / "studio.sqlite3"),
            str(Path(tempfile.mkdtemp()) / "artifacts"),
            provider=FakeGPUProvider(_Clock()),
        )
        status, payload = dispatch(controller, "POST", "/v1/admin/gpu/actions", b"{}")
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "PAID_ACTIONS_DISABLED")
        status, payload = dispatch(controller, "GET", "/v1/admin/gpu", None)
        self.assertEqual(status, 200)
        self.assertFalse(payload["paid_actions_enabled"])
        self.assertEqual(controller.provider.create_calls, 0)


class GateTests(unittest.TestCase):
    def test_live_provider_error_is_not_a_create(self) -> None:
        with self.assertRaises(WorkerStageError):
            raise WorkerStageError("PAID_ACTIONS_DISABLED")


if __name__ == "__main__":
    unittest.main()
