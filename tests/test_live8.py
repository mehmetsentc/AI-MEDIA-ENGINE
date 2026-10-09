"""LIVE-8 dynamic selection. Paid provider calls stay mocked."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.qwen import MODEL_ID
from media_engine.platform.service import Platform
from media_engine.providers.base import WorkerStageError
from media_engine.providers.execution_authorization import (
    accept_record,
    authorization_from_json,
    issue_authorization,
    select_for_authorization,
)
from media_engine.providers.gpu_discovery import assess_offers, rejected, shortlist
from media_engine.providers.live6_prepare import claim_execution_id
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.vast_image import VastImageProvider
from media_engine.safety.execution_budget import open_authorized_budget

ROOT = Path(__file__).resolve().parents[1]
SHA = "402cbfb0f2636acde3f44234e88e489de1d8afdd"


def _offer(offer_id: int, machine_id: int, **overrides) -> dict:
    row = {
        "id": offer_id,
        "machine_id": machine_id,
        "gpu_name": "RTX 6000 Ada",
        "gpu_ram": 49140,
        "num_gpus": 1,
        "rentable": True,
        "disk_space": 400,
        "cuda_max_good": 13.0,
        "compute_cap": 890,
        "dph_base": "0.70",
        "storage_cost": "0.10",
        "dph_total": "0.72",
        "inet_down": 8000,
        "reliability2": 0.99,
        "verification": "verified",
        "geolocation": "California, US",
    }
    row.update(overrides)
    return row


class _Popen:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.pid = 5151

    def poll(self):
        return None

    def wait(self, timeout=None):
        del timeout
        return 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class DiscoveryTests(unittest.TestCase):
    def test_shortlist_ranks_compatible_hosts_and_rejects_the_rest(self) -> None:
        rows = [
            _offer(1002, 502, gpu_name="RTX 5880 Ada", dph_base="0.48", dph_total="0.50", inet_down=2000, reliability2=0.995),
            _offer(1001, 501),
            _offer(9, 9, gpu_name="RTX 4090", gpu_ram=49140),
            _offer(8, 8, gpu_name="GTX 1080", gpu_ram=49140),
            _offer(7, 7, gpu_ram=20000),
            _offer(6, 6, disk_space=80),
            _offer(5, 5, cuda_max_good=11.8),
            _offer(4, 4, compute_cap=750),
            _offer(3, 3, rentable=False),
            _offer(2, 2, inet_down=50),
            _offer(1, 1, dph_base="1.20", dph_total="1.20"),
            _offer(11, 11, reliability2=0),
        ]
        assessed = assess_offers(rows, observed_at=5_000)
        ranked = shortlist(assessed)
        self.assertEqual([item.offer_id for item in ranked], [1001, 1002])
        self.assertLess(ranked[0].transfer_seconds, 2400)
        self.assertGreater(ranked[0].session_cost_usd, 0)
        self.assertGreaterEqual(ranked[0].inference_room_seconds, 1380)
        self.assertEqual(ranked[0].observed_at, 5_000)
        codes = {item.offer_id: item.rejection for item in rejected(assessed)}
        self.assertEqual(codes[9], "VRAM_UNUSABLE")
        self.assertEqual(codes[8], "GPU_MISMATCH")
        self.assertEqual(codes[7], "VRAM_MISMATCH")
        self.assertEqual(codes[6], "DISK_INSUFFICIENT")
        self.assertEqual(codes[5], "CUDA_MISMATCH")
        self.assertEqual(codes[4], "COMPUTE_MISMATCH")
        self.assertEqual(codes[3], "NOT_RENTABLE")
        self.assertEqual(codes[2], "TRANSFER_RISK")
        self.assertEqual(codes[1], "PRICE_CHANGED")
        self.assertEqual(codes[11], "RELIABILITY_UNKNOWN")

    def test_missing_bandwidth_is_not_treated_as_a_fast_link(self) -> None:
        assessed = assess_offers([_offer(12, 12, inet_down=0)], observed_at=5_000)
        self.assertEqual(assessed[0].rejection, "TRANSFER_RISK")


class AuthorizationTests(unittest.TestCase):
    def _auth(self, **overrides):
        fields = dict(
            authorization_id="live8-auth-a",
            execution_id="live8-run-a",
            code_sha=SHA,
            allowed_machine_ids=(501,),
            allowed_offer_ids=(1001,),
            selection_policy="exact_offers",
            max_hourly_usd=Decimal("0.95"),
            max_total_usd=Decimal("1.50"),
            max_instances=1,
            max_inference_attempts=1,
            max_image_count=1,
            max_disk_gb=140,
            model=MODEL_ID,
            revision=REVISION,
            issued_at=1_000,
            expires_at=20_000,
        )
        fields.update(overrides)
        return issue_authorization(**fields)

    def test_record_is_immutable_and_single_use(self) -> None:
        authorization = self._auth()
        with self.assertRaises(FrozenInstanceError):
            authorization.max_instances = 2
        with self.assertRaises(WorkerStageError) as raised:
            self._auth(max_instances=2)
        self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")
        with self.assertRaises(WorkerStageError) as raised:
            self._auth(max_total_usd=Decimal("1.51"))
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")
        root = Path(tempfile.mkdtemp())
        claim_execution_id(root / "live8-authorizations", authorization.authorization_id)
        with self.assertRaises(WorkerStageError) as raised:
            claim_execution_id(root / "live8-authorizations", authorization.authorization_id)
        self.assertEqual(raised.exception.code, "STALE_AUTHORIZATION")
        restored = authorization_from_json(json.dumps(authorization.to_dict()))
        self.assertEqual(restored, authorization)

    def test_expired_record_and_wrong_sha_are_refused(self) -> None:
        authorization = self._auth()
        with self.assertRaises(WorkerStageError) as raised:
            accept_record(authorization, now=20_000, code_sha=SHA)
        self.assertEqual(raised.exception.code, "EXPIRED_AUTHORIZATION")
        with self.assertRaises(WorkerStageError) as raised:
            accept_record(authorization, now=5_000, code_sha="a" * 40)
        self.assertEqual(raised.exception.code, "STALE_AUTHORIZATION")

    def test_previous_offer_is_not_implied(self) -> None:
        authorization = self._auth()
        old = assess_offers([_offer(53055111, 150718)], observed_at=5_000)
        chosen, reason = select_for_authorization(authorization, old, now=5_000)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "OFFER_UNAVAILABLE")

    def test_pool_stays_inside_the_approved_ids(self) -> None:
        authorization = self._auth(
            selection_policy="approved_pool",
            allowed_machine_ids=(501, 502),
            allowed_offer_ids=(1001, 1002),
        )
        rows = assess_offers([
            _offer(1002, 502, gpu_name="RTX 5880 Ada", dph_base="0.48", dph_total="0.50", inet_down=2000),
            _offer(1003, 503, inet_down=12000, reliability2=0.999, dph_base="0.40", dph_total="0.42"),
            _offer(1001, 501),
        ], observed_at=5_000)
        chosen, reason = select_for_authorization(authorization, rows, now=5_000)
        self.assertEqual(reason, "")
        self.assertEqual(chosen.offer_id, 1001)
        remaining = [item for item in rows if item.offer_id != 1001]
        fallback, fallback_reason = select_for_authorization(authorization, remaining, now=5_000)
        self.assertEqual(fallback_reason, "")
        self.assertEqual(fallback.offer_id, 1002)

    def test_unauthorized_machine_or_offer_is_named(self) -> None:
        authorization = self._auth()
        wrong_machine = assess_offers([_offer(1001, 999)], observed_at=5_000)
        chosen, reason = select_for_authorization(authorization, wrong_machine, now=5_000)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "MACHINE_SCOPE")
        wrong_offer = assess_offers([_offer(1003, 501)], observed_at=5_000)
        chosen, reason = select_for_authorization(authorization, wrong_offer, now=5_000)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "OFFER_REJECTED")


class CreateFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cold = os.environ.get("MEDIA_ENGINE_ALLOW_COLD_STAGE")
        self._forbid = os.environ.get("MEDIA_ENGINE_FORBID_R2_STAGE")
        os.environ["MEDIA_ENGINE_ALLOW_COLD_STAGE"] = "1"
        os.environ.pop("MEDIA_ENGINE_FORBID_R2_STAGE", None)

    def tearDown(self) -> None:
        if self._cold is None:
            os.environ.pop("MEDIA_ENGINE_ALLOW_COLD_STAGE", None)
        else:
            os.environ["MEDIA_ENGINE_ALLOW_COLD_STAGE"] = self._cold
        if self._forbid is None:
            os.environ.pop("MEDIA_ENGINE_FORBID_R2_STAGE", None)
        else:
            os.environ["MEDIA_ENGINE_FORBID_R2_STAGE"] = self._forbid

    def _auth(self, **overrides):
        fields = dict(
            authorization_id="live8-auth-a",
            execution_id="live8-run-a",
            code_sha=SHA,
            allowed_machine_ids=(501,),
            allowed_offer_ids=(1001,),
            selection_policy="exact_offers",
            max_hourly_usd=Decimal("0.95"),
            max_total_usd=Decimal("1.50"),
            max_instances=1,
            max_inference_attempts=1,
            max_image_count=1,
            max_disk_gb=140,
            model=MODEL_ID,
            revision=REVISION,
            issued_at=1_000,
            expires_at=20_000,
        )
        fields.update(overrides)
        return issue_authorization(**fields)

    def _provider(self, rows, instances=None, put=(200, b'{"success": true, "new_contract": 77}')):
        listed = [] if instances is None else instances
        puts = []
        gets = []

        def transport(method, url, body, headers):
            del headers
            if method == "PUT":
                puts.append((url, json.loads(body.decode("utf-8"))))
                return put
            if method == "POST":
                return 200, json.dumps({"offers": [dict(row) for row in rows]}).encode()
            if method == "GET":
                gets.append(url)
                if isinstance(listed, Exception):
                    return 410, b'{"success": false, "error": "deprecated_endpoint"}'
                return 200, json.dumps({
                    "success": True, "instances": [dict(row) for row in listed], "next_token": None,
                }).encode()
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._monotonic = lambda: 10.0
        provider._watchdog_popen = _Popen
        provider.puts = puts
        provider.gets = gets
        return provider

    def test_create_binds_the_selected_offer_once(self) -> None:
        provider = self._provider([_offer(1001, 501), _offer(1003, 503)])
        provider.bind_authorization(self._auth(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "created")
        self.assertEqual(result.resource_id, "77")
        self.assertEqual(len(provider.puts), 1)
        url, body = provider.puts[0]
        self.assertTrue(url.endswith("/asks/1001/"))
        self.assertEqual(body["label"], "media-engine:live8-run-a")
        self.assertNotIn("volume_info", body)
        self.assertEqual(body["disk"], 140)
        self.assertEqual(provider.execution_grant.machine_id, 501)
        self.assertEqual(provider.execution_grant.offer_id, 1001)
        spec = json.loads((provider.runtime / "watchdog.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["machine_id"], 501)
        intent = json.loads((provider.runtime / "create-intent.json").read_text(encoding="utf-8"))
        self.assertEqual(intent["state"], "created")
        self.assertEqual(intent["instance_id"], "77")
        self.assertEqual(intent["offer_id"], 1001)
        again = provider.create("studio")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertEqual(len(provider.puts), 1)

    def test_offer_disappearing_does_not_switch_offers(self) -> None:
        rows = [_offer(1001, 501), _offer(1003, 503, inet_down=12000, dph_base="0.40", dph_total="0.42")]
        provider = self._provider(rows)
        provider.bind_authorization(self._auth(), code_sha=SHA)
        provider.quote("studio")
        self.assertEqual(provider.offer_id, "1001")
        rows[:] = [rows[1]]
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "OFFER_UNAVAILABLE")
        self.assertEqual(provider.puts, [])

    def test_price_increase_and_non_rentable_stop_before_create(self) -> None:
        priced = _offer(1001, 501)
        provider = self._provider([priced])
        provider.bind_authorization(self._auth(), code_sha=SHA)
        priced["dph_base"] = "1.10"
        priced["dph_total"] = "1.10"
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "PRICE_CHANGED")
        self.assertEqual(provider.puts, [])
        priced["dph_base"] = "0.70"
        priced["dph_total"] = "0.72"
        priced["rentable"] = False
        second = self._provider([priced])
        second.bind_authorization(self._auth(authorization_id="live8-auth-b", execution_id="live8-run-b"), code_sha=SHA)
        result = second.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(second.live6_stop, "NOT_RENTABLE")
        self.assertEqual(second.puts, [])

    def test_unauthorized_and_expired_authorization_do_not_create(self) -> None:
        provider = self._provider([_offer(1003, 501)])
        provider.bind_authorization(self._auth(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(provider.live6_stop, "OFFER_REJECTED")
        self.assertEqual(provider.puts, [])
        self.assertEqual(result.outcome, "definite_failure")
        late = self._provider([_offer(1001, 501)])
        late.bind_authorization(self._auth(authorization_id="live8-auth-c", execution_id="live8-run-c"), code_sha=SHA)
        late._now = lambda: 30_000.0
        result = late.create("studio")
        self.assertEqual(late.live6_stop, "EXPIRED_AUTHORIZATION")
        self.assertEqual(late.puts, [])

    def test_pool_uses_only_a_remaining_approved_offer(self) -> None:
        provider = self._provider([
            _offer(1003, 503, inet_down=12000, dph_base="0.40", dph_total="0.42"),
            _offer(1002, 502, gpu_name="RTX 5880 Ada", dph_base="0.48", dph_total="0.50", inet_down=2000),
        ])
        provider.bind_authorization(self._auth(
            selection_policy="approved_pool",
            allowed_machine_ids=(501, 502),
            allowed_offer_ids=(1001, 1002),
        ), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "created")
        self.assertTrue(provider.puts[0][0].endswith("/asks/1002/"))
        self.assertEqual(provider.execution_grant.machine_id, 502)
        self.assertEqual(len(provider.puts), 1)

    def test_ambiguous_create_adopts_the_labeled_instance_only(self) -> None:
        appeared = []

        def transport(method, url, body, headers):
            del headers
            if method == "PUT":
                appeared.append({
                    "id": "88",
                    "machine_id": 501,
                    "label": "media-engine:live8-run-a",
                })
                return 500, b""
            if method == "POST":
                return 200, json.dumps({"offers": [_offer(1001, 501)]}).encode()
            if method == "GET":
                return 200, json.dumps({
                    "success": True, "instances": list(appeared), "next_token": None,
                }).encode()
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._monotonic = lambda: 10.0
        provider._watchdog_popen = _Popen
        provider.puts = appeared
        provider.bind_authorization(self._auth(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "created")
        self.assertEqual(result.resource_id, "88")
        self.assertEqual(len(appeared), 1)
        blank = self._provider(
            [_offer(1001, 501)], instances=[], put=(500, b""),
        )
        blank.bind_authorization(self._auth(authorization_id="live8-auth-d", execution_id="live8-run-d"), code_sha=SHA)
        missed = blank.create("studio")
        self.assertEqual(missed.outcome, "ambiguous")
        self.assertEqual(blank.live6_stop, "UNRESOLVED_BILLING")
        self.assertEqual(len(blank.puts), 1)
        retry = VastImageProvider(
            api_key="fixture-key",
            transport=blank._transport,
            runtime=blank.runtime,
            now=lambda: 5_000.0,
        )
        retry._watchdog_popen = _Popen
        retry.puts = blank.puts
        retry.bind_authorization(self._auth(authorization_id="live8-auth-d", execution_id="live8-run-d"), code_sha=SHA)
        again = retry.create("studio")
        self.assertEqual(again.outcome, "ambiguous")
        self.assertEqual(retry.live6_stop, "UNRESOLVED_BILLING")
        self.assertEqual(len(blank.puts), 1)

    def test_instance_list_failure_blocks_create(self) -> None:
        provider = self._provider([_offer(1001, 501)], instances=WorkerStageError("LIST_DEPRECATED"))
        provider.bind_authorization(self._auth(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "LIST_DEPRECATED")
        self.assertEqual(provider.puts, [])

    def test_search_failure_is_not_an_empty_market(self) -> None:
        def transport(method, url, body, headers):
            del url, body, headers
            if method == "POST":
                return 500, b""
            return 200, b'{"success": true, "instances": [], "next_token": null}'

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._watchdog_popen = _Popen
        provider.bind_authorization(self._auth(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "PROVIDER_CAPACITY_UNAVAILABLE")


class BudgetAndWatchdogTests(unittest.TestCase):
    def test_deadline_stays_inside_the_authorization_ceiling(self) -> None:
        budget = open_authorized_budget(
            execution_id="live8-run-a",
            grant_execution_id="live8-run-a",
            grant_ceiling=Decimal("1.50"),
            hourly=Decimal("0.72"),
            wall_now=5_000,
            monotonic_now=10,
            observed_starts=[5_000],
        )
        self.assertEqual(budget.ceiling_usd, Decimal("1.50"))
        self.assertFalse(budget.stop(5_000, 10))
        self.assertTrue(budget.stop(budget.spending_deadline_wall, budget.spending_deadline_monotonic))
        with self.assertRaises(WorkerStageError) as raised:
            open_authorized_budget(
                execution_id="live8-run-a",
                grant_execution_id="live8-run-a",
                grant_ceiling=Decimal("1.51"),
                hourly=Decimal("0.72"),
                wall_now=5_000,
                monotonic_now=10,
            )
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")

    def test_watchdog_outlives_the_parent(self) -> None:
        root = Path(tempfile.mkdtemp())
        instances = root / "instances.json"
        instances.write_text(json.dumps([{
            "id": "local-8",
            "machine_id": 501,
            "label": "media-engine:live8-run-a",
        }]), encoding="utf-8")
        spec = {
            "mode": "local",
            "machine_id": 501,
            "safe_seconds": 30,
            "execution_id": "live8-run-a",
            "spending_deadline_wall": 9_000_000_000.0,
            "authorization_expires_at": 8_000_000_000.0,
            "deadline_epoch": 8_000_000_000.0,
            "heartbeat": str(root / "beat"),
            "stop": str(root / "stop"),
            "result": str(root / "result.json"),
            "instances_path": str(instances),
            "delete_log": str(root / "deletes.json"),
            "instance_file": str(root / "watchdog.instance"),
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
        self.assertEqual(outcome["stopped"], ["local-8"])
        self.assertEqual(outcome["confirmed_absent"], ["local-8"])


class CreditTests(unittest.TestCase):
    def test_reservation_is_idempotent_and_refunds_once(self) -> None:
        platform = Platform(str(Path(tempfile.mkdtemp()) / "platform.sqlite3"))
        platform.repo.set_config("pool_priority", "purchased")
        platform.grant("nahaber", "purchased", 3, "promo_grant", idempotency_key="seed-live8")
        platform.repo.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="",
            unit="image", credit_price=2, internal_cost_units=0, effective_at="2020-01-01T00:00:00Z",
        )
        first = platform.authorize_paid("nahaber", "job-live8")
        second = platform.authorize_paid("nahaber", "job-live8")
        self.assertEqual(first, second)
        self.assertEqual(platform.balance("nahaber"), 1)
        platform.settle_failure("nahaber", "job-live8")
        platform.settle_failure("nahaber", "job-live8")
        self.assertEqual(platform.balance("nahaber"), 3)


if __name__ == "__main__":
    unittest.main()
