"""LIVE-7 alternative GPU contract. Clocks, offers, and transfers are local."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.cold_stage import StageStopped, accept_manifest, run_cold_stage
from media_engine.engines.image.model_cache import REPO_ID
from media_engine.engines.image.qwen import FIRST_PROMPT
from media_engine.jobs.model import JobError, JobStatus
from media_engine.orchestrator.controller import ManualClock, MediaController
from media_engine.platform.service import MemoryMedia, Platform
from media_engine.providers.base import WorkerStageError
from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.live6_prepare import claim_execution_id
from media_engine.providers.live7_contract import (
    FORBIDDEN_VOLUME_ID,
    LIVE7_MACHINE_ID,
    REVISION,
    accept_live7_grant,
    image_contract,
    make_live7_grant,
    persistence_audit,
    static_compatibility,
)
from media_engine.providers.live7_prepare import grant_from_environ
from media_engine.providers.vast_image import MACHINE_ID, VOLUME_ID, VastImageProvider
from media_engine.providers.vast_watchdog import run_until_stopped
from media_engine.safety.execution_budget import (
    LIVE6_RESERVE_USD,
    LIVE7_CEILING_USD,
    LIVE7_MAX_HOURLY_USD,
    LIVE7_MINIMUM_SESSION_SECONDS,
    billable_seconds,
    live7_billable_seconds,
    open_budget,
    open_live7_budget,
    retain_deadline,
    termination_reserve_usd,
)
from media_engine.safety.limits import SafetyLimits
from media_engine.safety.spend_grant import SpendGrant
from media_engine.studio.__main__ import supervisor_action
from tests.test_phase2e import PngEngine

ROOT = Path(__file__).resolve().parents[1]
QUOTE = Decimal("0.8648")


class Clock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


class MemorySource:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.reads: list[tuple[str, int]] = []

    def read(self, key: str, offset: int, length: int) -> bytes:
        self.reads.append((key, offset))
        blob = self.files[key]
        return blob[offset:offset + length]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(files: dict[str, bytes]) -> dict:
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "files": [
            {"path": name, "bytes": len(data), "sha256": _sha(data)}
            for name, data in files.items()
        ],
    }


def _offer(**overrides) -> dict:
    row = {
        "id": 44120922,
        "machine_id": LIVE7_MACHINE_ID,
        "gpu_name": "RTX 5880 Ada",
        "gpu_ram": 49140,
        "num_gpus": 1,
        "rentable": True,
        "disk_space": 632.8,
        "cuda_max_good": 13.2,
        "compute_cap": 890,
        "dph_base": "0.84",
        "storage_cost": "0.12",
        "dph_total": "0.86",
        "reliability2": 0.99,
    }
    row.update(overrides)
    return row


class Live7BudgetTests(unittest.TestCase):
    def test_live6_window_is_unchanged(self) -> None:
        hourly = Decimal("0.3748148148148148511111111111")
        self.assertEqual(billable_seconds(hourly), 3841)
        self.assertEqual(LIVE6_RESERVE_USD, Decimal("0.10"))
        with self.assertRaises(WorkerStageError) as raised:
            open_budget(
                execution_id="a", grant_execution_id="a", grant_ceiling=Decimal("1.50"),
                hourly=hourly, wall_now=1, monotonic_now=1,
            )
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")

    def test_live7_reserve_scales_and_fits_the_session(self) -> None:
        reserve = termination_reserve_usd(QUOTE)
        self.assertGreater(reserve, Decimal("0.20"))
        self.assertLess(reserve, Decimal("0.21"))
        seconds = live7_billable_seconds(QUOTE)
        self.assertIsNotNone(seconds)
        assert seconds is not None
        self.assertGreaterEqual(seconds, LIVE7_MINIMUM_SESSION_SECONDS)
        planning = LIVE7_CEILING_USD - reserve
        cost = (QUOTE * Decimal(seconds)) / Decimal(3600)
        self.assertLessEqual(cost, planning)
        self.assertEqual(planning + reserve, LIVE7_CEILING_USD)
        self.assertIsNone(live7_billable_seconds(Decimal("0.96")))
        self.assertIsNone(live7_billable_seconds(LIVE7_MAX_HOURLY_USD + Decimal("0.01")))

    def test_deadline_is_frozen_at_create_and_is_not_the_grant_expiry(self) -> None:
        budget = open_live7_budget(
            execution_id="live7-1", grant_execution_id="live7-1",
            grant_ceiling=LIVE7_CEILING_USD, hourly=QUOTE,
            wall_now=10_000, monotonic_now=50,
        )
        self.assertEqual(budget.billing_start_event, "create_request")
        self.assertEqual(budget.spending_deadline_wall, 10_000 + budget.allowed_seconds)
        self.assertNotEqual(budget.spending_deadline_wall, 10_030)
        self.assertEqual(
            retain_deadline(budget.as_mapping(), budget.spending_deadline_wall + 500, "live7-1"),
            budget.spending_deadline_wall,
        )
        with self.assertRaises(WorkerStageError):
            open_live7_budget(
                execution_id="live7-1", grant_execution_id="live7-1",
                grant_ceiling=Decimal("0.50"), hourly=QUOTE, wall_now=1, monotonic_now=1,
            )
        host = type("Host", (), {"execution_budget": budget, "provider": None})()
        host.provider = host
        host.spending_stop = budget.stop
        self.assertIsNone(supervisor_action(
            host, wall_now=budget.spending_deadline_wall - 1,
            monotonic_now=budget.spending_deadline_monotonic - 1, finished=False,
        ))
        self.assertEqual(supervisor_action(
            host, wall_now=budget.spending_deadline_wall,
            monotonic_now=0, finished=False,
        ), "SPENDING_DEADLINE")

    def test_one_dollar_fifty_is_not_rewritten(self) -> None:
        from media_engine.providers.vast_image import active_operation_budget
        previous = os.environ.get("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD")
        profile = os.environ.get("MEDIA_ENGINE_EXECUTION_PROFILE")
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "1.50"
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live7"
        try:
            self.assertEqual(active_operation_budget(), Decimal("1.50"))
            os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live6"
            with self.assertRaises(WorkerStageError):
                active_operation_budget()
        finally:
            if previous is None:
                os.environ.pop("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", None)
            else:
                os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = previous
            if profile is None:
                os.environ.pop("MEDIA_ENGINE_EXECUTION_PROFILE", None)
            else:
                os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = profile


class Live7GrantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._saved = dict(os.environ)
        os.environ["MEDIA_ENGINE_ALLOW_COLD_STAGE"] = "1"
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live7"
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "1.50"
        os.environ.pop("MEDIA_ENGINE_FORBID_R2_STAGE", None)
        os.environ.pop("MEDIA_ENGINE_MACHINE_ID", None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._saved)

    def _provider(self) -> VastImageProvider:
        puts: list[dict] = []

        def transport(method, url, body, headers):
            del url, headers
            if method == "PUT":
                puts.append(json.loads(body.decode("utf-8")))
                return 200, b'{"success": true, "new_contract": 77}'
            if method == "POST":
                return 200, json.dumps({"offers": [_offer(), _offer(id=1, machine_id=26359, gpu_name="RTX A6000")]}).encode()
            return 200, b'{"instances": []}'

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(self.tmp.name),
            now=lambda: 5_000.0,
        )
        provider._monotonic = lambda: 10.0
        provider.puts = puts
        return provider

    def _arm(self, provider: VastImageProvider, execution_id: str = "live7-fresh") -> None:
        grant = make_live7_grant(execution_id=execution_id, issued_at=1_000, expires_at=20_000)
        provider.bind_execution(grant, execution_id=execution_id)
        provider.offer_id = "44120922"
        provider.gpu_model = "RTX 5880 Ada"
        provider.gpu_ram_mib = 49140
        provider.cuda_max_good = 13.2
        provider.compute_cap = 890
        provider.hourly_amount = QUOTE

        class _Proc:
            pid = 44

        provider._watchdog_popen = lambda *_args, **_kwargs: _Proc()

    def test_exact_machine_creates_without_the_warm_volume(self) -> None:
        os.environ["MEDIA_ENGINE_WATCHDOG_LAUNCH"] = "1"
        os.environ["STUDIO_GPU_BUDGET_USD"] = "1.50"
        provider = self._provider()
        self._arm(provider)
        created = provider.create("daidoi")
        self.assertEqual(created.outcome, "created")
        self.assertEqual(provider.launch_trace, ["watchdog", "put"])
        self.assertEqual(len(provider.puts), 1)
        self.assertEqual(provider.puts[0]["disk"], 140)
        self.assertNotIn("volume_info", provider.puts[0])
        spec = json.loads((Path(self.tmp.name) / "watchdog.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["machine_id"], LIVE7_MACHINE_ID)
        self.assertEqual(spec["ceiling_usd"], "1.50")
        self.assertEqual(spec["billing_start_event"], "create_request")
        self.assertNotEqual(spec["spending_deadline_wall"], spec["authorization_expires_at"])
        self.assertEqual(spec["spending_deadline_wall"], 5_000.0 + spec["allowed_seconds"])
        self.assertEqual(spec["deadline_epoch"], spec["spending_deadline_wall"])
        again = provider.create("daidoi")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertEqual(len(provider.puts), 1)

    def test_rejections_happen_before_put(self) -> None:
        os.environ["MEDIA_ENGINE_WATCHDOG_LAUNCH"] = "1"
        cases = (
            ("machine_id", "26359", "MACHINE_SCOPE"),
            ("gpu_model", "RTX A6000", "GPU_MISMATCH"),
            ("cuda_max_good", 12.5, "CUDA_MISMATCH"),
            ("gpu_ram_mib", 24576, "VRAM_MISMATCH"),
            ("hourly_amount", Decimal("0.96"), "PRICE_CHANGED"),
            ("attached_volume_id", VOLUME_ID, "VOLUME_MISMATCH"),
        )
        for attr, value, code in cases:
            provider = self._provider()
            self._arm(provider, execution_id="case-" + attr)
            setattr(provider, attr, value)
            self.assertEqual(provider.create("daidoi").outcome, "definite_failure")
            self.assertEqual(provider.live6_stop, code)
            self.assertEqual(provider.puts, [])

    def test_grant_reuse_and_cross_profile_are_refused(self) -> None:
        grant = make_live7_grant(execution_id="once", issued_at=1, expires_at=20_000)
        accept_live7_grant(grant)
        path = Path(self.tmp.name) / "live7-executions"
        claim_execution_id(path, grant.execution_id)
        with self.assertRaises(WorkerStageError) as raised:
            claim_execution_id(path, grant.execution_id)
        self.assertEqual(raised.exception.code, "STALE_AUTHORIZATION")
        provider = self._provider()
        provider.bind_execution(grant, execution_id="once")
        with self.assertRaises(WorkerStageError):
            provider.bind_execution(grant, execution_id="once")
        live6 = SpendGrant(
            execution_id="live6", ceiling_usd=Decimal("0.50"), max_instances=1,
            issued_at=1, expires_at=50, machine_id=LIVE7_MACHINE_ID,
        )
        with self.assertRaises(WorkerStageError) as live6_refused:
            provider.bind_execution(live6, execution_id="live6")
        self.assertEqual(live6_refused.exception.code, "STALE_AUTHORIZATION")
        from dataclasses import replace
        wrong_machine = replace(grant, execution_id="wrong-machine", machine_id=MACHINE_ID)
        wrong_provider = VastImageProvider(api_key="fixture-key", now=lambda: 10.0, runtime=Path(self.tmp.name) / "b")
        with self.assertRaises(WorkerStageError) as wrong:
            wrong_provider.bind_execution(wrong_machine, execution_id="wrong-machine")
        self.assertEqual(wrong.exception.code, "MACHINE_SCOPE")
        fresh = VastImageProvider(api_key="fixture-key", now=lambda: 10.0, runtime=Path(self.tmp.name) / "c")
        live6_ok = SpendGrant(
            execution_id="warm", ceiling_usd=Decimal("0.50"), max_instances=1,
            issued_at=1, expires_at=50, machine_id=MACHINE_ID,
        )
        fresh.bind_execution(live6_ok, execution_id="warm")
        with self.assertRaises(WorkerStageError) as crossed:
            fresh.bind_execution(make_live7_grant(execution_id="other", issued_at=1, expires_at=50), execution_id="other")
        self.assertEqual(crossed.exception.code, "STALE_AUTHORIZATION")

    def test_quote_ignores_every_other_machine(self) -> None:
        os.environ["MEDIA_ENGINE_WATCHDOG_LAUNCH"] = "1"
        provider = self._provider()
        self._arm(provider, execution_id="quoted")
        provider.offer_id = ""
        queries = []
        original = provider._transport

        def transport(method, url, body, headers):
            if method == "POST":
                queries.append(json.loads(body.decode("utf-8")))
            return original(method, url, body, headers)

        provider._transport = transport
        created = provider.create("daidoi")
        self.assertEqual(created.outcome, "created")
        self.assertEqual(queries[0]["machine_id"], {"eq": LIVE7_MACHINE_ID})
        self.assertEqual(provider.machine_id, str(LIVE7_MACHINE_ID))
        self.assertNotIn("volume_info", provider.puts[0])
        self.assertNotEqual(provider.puts[0].get("volume_id"), FORBIDDEN_VOLUME_ID)

    def test_environ_grant_refuses_a_different_machine_and_execute(self) -> None:
        environ = {
            "MEDIA_ENGINE_EXECUTION_PROFILE": "live7",
            "MEDIA_ENGINE_EXECUTION_ID": "env-1",
            "MEDIA_ENGINE_EXECUTION_ISSUED_AT": "10",
            "MEDIA_ENGINE_EXECUTION_EXPIRES_AT": "99",
            "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD": "1.50",
            "MEDIA_ENGINE_ALLOW_COLD_STAGE": "1",
            "MEDIA_ENGINE_MACHINE_ID": "26359",
        }
        with self.assertRaises(WorkerStageError) as raised:
            grant_from_environ(environ)
        self.assertEqual(raised.exception.code, "MACHINE_SCOPE")
        from media_engine.providers.live7_prepare import main
        self.assertEqual(main(["--execute"]), 2)


class ColdStageTests(unittest.TestCase):
    def test_transfer_cases_stay_offline(self) -> None:
        files = {"weights.bin": b"qwen-weights", "config.json": b"{}"}
        manifest = _manifest(files)
        dest = Path(tempfile.mkdtemp())
        clock = Clock(0)
        source = MemorySource(files)
        done = run_cold_stage(
            manifest, dest, source, clock,
            transfer_deadline=100, spending_deadline=100, tail_seconds=10,
        )
        self.assertEqual(done["status"], "COMPLETE")
        self.assertEqual(done["bytes"], len(files["weights.bin"]) + len(files["config.json"]))
        self.assertTrue(done["marker"])
        self.assertGreater(done["hashes_checked"], 0)
        reads = len(source.reads)
        again = run_cold_stage(
            manifest, dest, source, clock,
            transfer_deadline=100, spending_deadline=100, tail_seconds=10,
        )
        self.assertTrue(again["already_complete"])
        self.assertEqual(len(source.reads), reads)

        slow_dest = Path(tempfile.mkdtemp())
        slow_clock = Clock(0)
        slow_files = {"weights.bin": b"abcdefgh"}

        class Slow(MemorySource):
            def read(self, key, offset, length):
                self.reads.append((key, offset))
                slow_clock.t += 1
                return self.files[key][offset:offset + 1]

        slow = Slow(slow_files)
        run_cold_stage(
            _manifest(slow_files), slow_dest, slow, slow_clock,
            transfer_deadline=100, spending_deadline=100, tail_seconds=5, stall_seconds=30,
        )
        self.assertEqual((slow_dest / "weights.bin").read_bytes(), b"abcdefgh")

        stall_clock = Clock(0)

        class Stalled(MemorySource):
            def read(self, key, offset, length):
                stall_clock.t += 40
                return b""

        with self.assertRaises(StageStopped) as stalled:
            run_cold_stage(
                _manifest(files), Path(tempfile.mkdtemp()), Stalled(files), stall_clock,
                transfer_deadline=1_000, spending_deadline=1_000, tail_seconds=1, stall_seconds=30,
            )
        self.assertEqual(stalled.exception.code, "CACHE_STALL")

        resume_dest = Path(tempfile.mkdtemp())
        target = files["weights.bin"]
        partial = resume_dest / "weights.bin.partial"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(target[:4])
        resume_source = MemorySource({"weights.bin": target})
        only = {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "files": [{"path": "weights.bin", "bytes": len(target), "sha256": _sha(target)}],
        }
        report = run_cold_stage(
            only, resume_dest, resume_source, Clock(0),
            transfer_deadline=50, spending_deadline=50, tail_seconds=1,
        )
        self.assertTrue(report["resumed"])
        self.assertEqual(resume_source.reads[0][1], 4)
        self.assertEqual((resume_dest / "weights.bin").read_bytes(), target)

        bad = MemorySource({"weights.bin": b"\x00" * len(target)})
        corrupt_dest = Path(tempfile.mkdtemp())
        with self.assertRaises(StageStopped) as corrupt:
            run_cold_stage(
                only, corrupt_dest, bad, Clock(0),
                transfer_deadline=50, spending_deadline=50, tail_seconds=1,
            )
        self.assertEqual(corrupt.exception.code, "CACHE_CORRUPT")
        self.assertEqual(list(corrupt_dest.rglob("*.partial")), [])

        with self.assertRaises(StageStopped) as missing:
            run_cold_stage(
                only, Path(tempfile.mkdtemp()), MemorySource({}), Clock(0),
                transfer_deadline=50, spending_deadline=50, tail_seconds=1,
            )
        self.assertEqual(missing.exception.code, "CACHE_INCOMPLETE")

        with self.assertRaises(StageStopped) as timeout:
            run_cold_stage(
                only, Path(tempfile.mkdtemp()), MemorySource(files), Clock(10),
                transfer_deadline=10, spending_deadline=100, tail_seconds=1,
            )
        self.assertEqual(timeout.exception.code, "TRANSFER_DEADLINE")

        with self.assertRaises(StageStopped) as budget:
            run_cold_stage(
                only, Path(tempfile.mkdtemp()), MemorySource({"weights.bin": target}), Clock(80),
                transfer_deadline=200, spending_deadline=100, tail_seconds=30,
            )
        self.assertEqual(budget.exception.code, "BUDGET_STOP")

        substituted = dict(manifest, repo_id="other/model")
        with self.assertRaises(StageStopped) as replaced:
            accept_manifest(substituted, enforce_qwen=False)
        self.assertEqual(replaced.exception.code, "MODEL_SUBSTITUTION")
        with self.assertRaises(StageStopped) as inventory:
            accept_manifest(manifest, enforce_qwen=True)
        self.assertEqual(inventory.exception.code, "CACHE_INCOMPLETE")
        os.environ["R2_SECRET_ACCESS_KEY"] = "super-secret-value"
        try:
            rendered = json.dumps(done)
        finally:
            os.environ.pop("R2_SECRET_ACCESS_KEY", None)
        self.assertNotIn("super-secret-value", rendered)


class Live7JobAndAuditTests(unittest.TestCase):
    def test_image_contract_is_not_submitted_and_a_second_job_is_refused(self) -> None:
        contract = image_contract()
        self.assertFalse(contract["submitted"])
        self.assertEqual(contract["prompt"], FIRST_PROMPT)
        self.assertEqual((contract["width"], contract["height"], contract["seed"], contract["count"]), (1024, 1024, 20260804, 1))
        root = Path(tempfile.mkdtemp())
        provider = FakeGPUProvider(time.time)
        provider.single_image_only = True
        provider.image_claimed = False
        controller = MediaController(
            str(root / "jobs.sqlite3"), str(root / "artifacts"),
            clock=ManualClock(), provider=provider, image_engine=PngEngine(),
            limits=SafetyLimits(max_job_attempts=1),
        )
        first = controller.submit_image(FIRST_PROMPT, seed=20260804)
        with self.assertRaises(JobError):
            controller.submit_image(FIRST_PROMPT, seed=20260804)
        first.attempt_count = 1
        first.status = JobStatus.QUEUED
        controller.jobs.save(first)
        controller._run_one(first.id)
        self.assertEqual(controller.jobs.get(first.id).error_code, "ATTEMPT_LIMIT")

    def test_cleanup_codes_and_credit_settlement_once(self) -> None:
        grant = make_live7_grant(execution_id="audit", issued_at=1, expires_at=50)
        host = type("Host", (), {})()
        host.provider = type("Provider", (), {"execution_grant": grant})()
        host._bound_execution = lambda: True
        for code in ("MODEL_LOAD_FAILED", "GENERATION_FAILED", "ARTIFACT_UPLOAD_FAILED", "SETTLEMENT_FAILED", "CACHE_TIMEOUT"):
            self.assertTrue(MediaController._destroy_for(host, code))
        root = Path(tempfile.mkdtemp())
        platform = Platform(str(root / "platform.sqlite3"), media=MemoryMedia())
        platform.add_price(
            version="live7", service="image", model="qwen-image", quality="test",
            unit="image", credit_price=2, internal_cost_units=1, effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 5, "topup_purchase", idempotency_key="live7-grant")
        self.assertEqual(platform.balance("nahaber"), 5)
        platform.authorize_paid("nahaber", "job-1")
        self.assertEqual(platform.balance("nahaber"), 3)
        platform.settle("nahaber", "job-1", 2)
        platform.settle("nahaber", "job-1", 2)
        self.assertEqual(platform.balance("nahaber"), 3)
        audit = persistence_audit()
        self.assertTrue(audit["neon_integrated"])
        self.assertFalse(audit["sqlite_production_fallback"])
        self.assertEqual(audit["e2e_blocker"], "GPU_NOT_AUTHORIZED")
        self.assertFalse(audit["real_inference_is_full_e2e"])
        report = static_compatibility(ROOT)
        self.assertFalse(report["gpu_validated"])
        self.assertTrue(report["worker_digest_matches"])
        self.assertTrue(report["bf16_code_allows_family"])
        self.assertTrue(report["torch_pinned"])

    def test_watchdog_keeps_the_deadline_and_reports_a_failed_delete(self) -> None:
        root = Path(tempfile.mkdtemp())
        rows = [{"id": "gpu-1", "machine_id": LIVE7_MACHINE_ID, "label": "media-engine:live7"}]
        deleted = []

        def fetch():
            return rows

        def delete(instance_id):
            deleted.append(instance_id)
            return 500

        spec = {
            "machine_id": LIVE7_MACHINE_ID,
            "safe_seconds": 10,
            "spending_deadline_wall": 100.0,
            "deadline_epoch": 50.0,
            "authorization_expires_at": 40.0,
            "heartbeat": str(root / "beat"),
            "stop": str(root / "stop"),
            "poll_seconds": 0.01,
        }
        outcome = run_until_stopped(spec, fetch=fetch, delete=delete, now=lambda: 100.0, sleep=lambda _seconds: None)
        self.assertEqual(outcome["deadline"], 100.0)
        self.assertEqual(outcome["stopped"], ["gpu-1"])
        self.assertEqual(outcome["termination_failures"], [{"id": "gpu-1", "status": 500}])
        self.assertEqual(retain_deadline({
            "execution_id": "live7-1",
            "spending_deadline_wall": 100.0,
        }, 500.0, "live7-1"), 100.0)


class WatchdogParentTests(unittest.TestCase):
    def test_watchdog_outlives_the_parent_on_the_pinned_machine(self) -> None:
        root = Path(tempfile.mkdtemp())
        instances = root / "instances.json"
        instances.write_text(json.dumps([{
            "id": "local-7",
            "machine_id": LIVE7_MACHINE_ID,
            "label": "media-engine:live7",
        }]), encoding="utf-8")
        spec = {
            "mode": "local",
            "machine_id": LIVE7_MACHINE_ID,
            "safe_seconds": 30,
            "execution_id": "parent-live7",
            "spending_deadline_wall": 9_000_000_000.0,
            "authorization_expires_at": 8_000_000_000.0,
            "deadline_epoch": 8_000_000_000.0,
            "heartbeat": str(root / "beat"),
            "stop": str(root / "stop"),
            "result": str(root / "result.json"),
            "instances_path": str(instances),
            "delete_log": str(root / "deletes.json"),
            "poll_seconds": 0.05,
        }
        spec_path = root / "watchdog.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        pidfile = root / "child.pid"
        script = root / "parent.py"
        script.write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            "from media_engine.providers.vast_watchdog import arm_watchdog\n"
            "pid = arm_watchdog(Path(sys.argv[1]))\n"
            "Path(sys.argv[2]).write_text(str(pid))\n"
            "os._exit(0)\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.pop("VAST_API_KEY", None)
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, str(script), str(spec_path), str(pidfile)],
            cwd=str(ROOT), env=env, check=False, timeout=10,
        )
        self.assertEqual(completed.returncode, 0)
        child = int(pidfile.read_text(encoding="utf-8"))
        self.addCleanup(lambda: os.kill(child, 15) if _alive(child) else None)
        self.assertTrue(_alive(child))
        Path(spec["stop"]).write_text("1", encoding="utf-8")
        deadline = time.time() + 5
        while time.time() < deadline and not Path(spec["result"]).is_file():
            time.sleep(0.05)
        outcome = json.loads(Path(spec["result"]).read_text(encoding="utf-8"))
        self.assertEqual(outcome["stopped"], ["local-7"])
        self.assertEqual(outcome["deadline"], 9_000_000_000.0)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
