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
from media_engine.orchestrator.controller import MediaController
from media_engine.providers.base import Quote, WorkerStageError
from media_engine.providers.execution_authorization import (
    accept_record,
    authorization_from_json,
    issue_authorization,
    select_for_authorization,
)
from media_engine.providers.gpu_discovery import assess_offers, rejected, shortlist
from media_engine.providers.live6_prepare import claim_execution_id
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.vast_image import VastImageProvider, active_operation_budget
from media_engine.safety.execution_budget import (
    LIVE7_RESERVE_SECONDS,
    LIVE8_JIT_CEILING_USD,
    open_authorized_budget,
    termination_reserve_usd,
)

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

    def test_truncated_authorization_sha_is_refused(self) -> None:
        truncated = "0af07ddb2b374d7c0c81c9a6e748de6f68ef06"
        executed = "0af07ddb2b374d7c0c81c9a6e6a748de6f68ef06"
        self.assertEqual(len(truncated), 38)
        self.assertEqual(len(executed), 40)
        with self.assertRaises(WorkerStageError) as raised:
            self._auth(code_sha=truncated)
        self.assertEqual(raised.exception.code, "STALE_AUTHORIZATION")
        issued = self._auth(code_sha=executed)
        with self.assertRaises(WorkerStageError) as raised:
            accept_record(issued, now=5_000, code_sha=SHA)
        self.assertEqual(raised.exception.code, "STALE_AUTHORIZATION")
        accept_record(issued, now=5_000, code_sha=executed)

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

    def test_just_in_time_needs_an_explicit_pair_set(self) -> None:
        with self.assertRaises(WorkerStageError) as raised:
            self._auth(selection_policy="just_in_time")
        self.assertEqual(raised.exception.code, "MACHINE_SCOPE")
        with self.assertRaises(WorkerStageError) as raised:
            self._auth(allowed_pairs=((501, 1001),))
        self.assertEqual(raised.exception.code, "SELECTION_POLICY")
        with self.assertRaises(WorkerStageError) as raised:
            self._auth(
                selection_policy="just_in_time",
                allowed_pairs=((501, 1001),),
                selection_constraints={"max_hourly_usd": "0.95"},
            )
        self.assertEqual(raised.exception.code, "SELECTION_POLICY")

    def test_cross_pair_is_outside_the_just_in_time_set(self) -> None:
        authorization = self._auth(
            selection_policy="just_in_time",
            allowed_machine_ids=(501, 502),
            allowed_offer_ids=(1001, 1002),
            allowed_pairs=((501, 1001), (502, 1002)),
        )
        crossed = assess_offers([_offer(1002, 501, inet_down=20000)], observed_at=5_000)
        chosen, reason = select_for_authorization(authorization, crossed, now=5_000)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "OFFER_REJECTED")


class CreateFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cold = os.environ.get("MEDIA_ENGINE_ALLOW_COLD_STAGE")
        self._forbid = os.environ.get("MEDIA_ENGINE_FORBID_R2_STAGE")
        self._saved_stage = {
            name: os.environ.get(name)
            for name in (
                "MODEL_CACHE_PROVIDER",
                "R2_ENDPOINT",
                "R2_BUCKET",
                "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY",
                "MEDIA_ENGINE_R2_RELIABLE_TRANSFER",
            )
        }
        os.environ["MEDIA_ENGINE_ALLOW_COLD_STAGE"] = "1"
        os.environ.pop("MEDIA_ENGINE_FORBID_R2_STAGE", None)
        os.environ["MODEL_CACHE_PROVIDER"] = "r2"
        os.environ["R2_ENDPOINT"] = "https://example.r2.cloudflarestorage.com"
        os.environ["R2_BUCKET"] = "ai-media-engine-models"
        os.environ["R2_ACCESS_KEY_ID"] = "fixture-access"
        os.environ["R2_SECRET_ACCESS_KEY"] = "fixture-secret"
        os.environ.pop("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", None)

    def tearDown(self) -> None:
        if self._cold is None:
            os.environ.pop("MEDIA_ENGINE_ALLOW_COLD_STAGE", None)
        else:
            os.environ["MEDIA_ENGINE_ALLOW_COLD_STAGE"] = self._cold
        if self._forbid is None:
            os.environ.pop("MEDIA_ENGINE_FORBID_R2_STAGE", None)
        else:
            os.environ["MEDIA_ENGINE_FORBID_R2_STAGE"] = self._forbid
        for name, value in self._saved_stage.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

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
        provider._manifest_ready = lambda: True
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

    def test_incomplete_stage_config_does_not_create(self) -> None:
        os.environ["R2_BUCKET"] = "other-bucket"
        provider = self._provider([_offer(1001, 501)])
        provider.bind_authorization(self._auth(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "STAGE_CONFIG_INVALID")
        self.assertEqual(provider.puts, [])
        self.assertIn("R2_BUCKET", provider.stage_problems)

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
        provider._manifest_ready = lambda: True
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

    def test_pre_create_failure_refunds_without_reporting_a_create(self) -> None:
        platform = Platform(str(Path(tempfile.mkdtemp()) / "platform.sqlite3"))
        platform.repo.set_config("pool_priority", "purchased")
        platform.grant("nahaber", "purchased", 3, "promo_grant", idempotency_key="seed-live8-jit")
        platform.repo.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="",
            unit="image", credit_price=2, internal_cost_units=0, effective_at="2020-01-01T00:00:00Z",
        )
        provider = self._provider([])
        provider.bind_authorization(self._auth(), code_sha=SHA)
        root = Path(tempfile.mkdtemp())
        controller = MediaController(str(root / "jobs.sqlite3"), str(root / "storage"))
        controller.provider = provider
        controller.platform = platform
        job = controller.submit_image("A cinematic editorial photograph of a newsroom.")
        controller._create_worker(job, Quote(hourly_price_usd=Decimal("0.72"), provider="vast"))
        self.assertEqual(job.error_code, "OFFER_UNAVAILABLE")
        self.assertEqual(platform.balance("nahaber"), 3)
        self.assertEqual(controller.ledger.unresolved(), [])
        self.assertEqual(provider.puts, [])
        self.assertFalse(provider.create_request_sent)
        self.assertIsNone(provider.execution_grant)
        self.assertEqual(provider.create_classification, "offer_unavailable_before_create")
        self.assertFalse(provider.billing_exposure)
        record = controller.ledger.list()[0]
        self.assertFalse(record.create_attempted)
        self.assertEqual(record.last_error, "OFFER_UNAVAILABLE")

    def test_provider_rejection_confirms_there_is_no_instance(self) -> None:
        provider = self._provider(
            [_offer(1001, 501)],
            put=(400, b'{"success": false, "error": "no_such_ask"}'),
        )
        provider.bind_authorization(self._auth(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(len(provider.puts), 1)
        self.assertTrue(provider.create_request_sent)
        self.assertEqual(provider.create_classification, "provider_create_rejected")
        self.assertFalse(provider.billing_exposure)
        observation = json.loads((provider.runtime / "create-observation.json").read_text(encoding="utf-8"))
        self.assertEqual(observation["classification"], "provider_create_rejected")
        self.assertEqual(observation["provider_response"]["status"], 400)
        self.assertEqual(observation["provider_response"]["outcome"], "definite_failure")
        self.assertIsNotNone(observation["create_request_at"])
        self.assertFalse(observation["billing_exposure"])
        retry = VastImageProvider(
            api_key="fixture-key",
            transport=provider._transport,
            runtime=provider.runtime,
            now=lambda: 5_000.0,
        )
        retry._watchdog_popen = _Popen
        retry._manifest_ready = lambda: True
        retry.puts = provider.puts
        retry.bind_authorization(self._auth(), code_sha=SHA)
        again = retry.create("studio")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertFalse(retry.create_request_sent)
        self.assertFalse(retry.billing_exposure)
        self.assertEqual(len(provider.puts), 1)

    def test_approved_pool_does_not_switch_after_the_final_check_fails(self) -> None:
        posts = []

        def transport(method, url, body, headers):
            del url, headers
            if method == "PUT":
                raise AssertionError("create request")
            if method == "POST":
                query = json.loads(body.decode("utf-8"))
                posts.append(query)
                if "machine_id" in query:
                    return 200, json.dumps({"offers": [_offer(1001, 501, rentable=False)]}).encode()
                return 200, json.dumps({"offers": [
                    _offer(1001, 501),
                    _offer(1002, 502, gpu_name="RTX 5880 Ada", inet_down=2000),
                ]}).encode()
            if method == "GET":
                return 200, b'{"success": true, "instances": [], "next_token": null}'
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._watchdog_popen = _Popen
        provider._manifest_ready = lambda: True
        provider.bind_authorization(self._auth(
            selection_policy="approved_pool",
            allowed_machine_ids=(501, 502),
            allowed_offer_ids=(1001, 1002),
        ), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "NOT_RENTABLE")
        self.assertFalse(provider.create_request_sent)
        self.assertIsNone(provider.execution_grant)
        self.assertEqual(provider.launch_trace, [])
        self.assertFalse((provider.runtime / "watchdog.json").is_file())
        self.assertFalse(any("machine_id" in query and query["machine_id"].get("eq") == 502 for query in posts))


class JustInTimeTests(unittest.TestCase):
    def setUp(self) -> None:
        CreateFlowTests.setUp(self)

    def tearDown(self) -> None:
        CreateFlowTests.tearDown(self)

    def _auth(self, **overrides):
        return CreateFlowTests._auth(self, **overrides)

    def _jit(self, **overrides):
        fields = dict(
            selection_policy="just_in_time",
            allowed_machine_ids=(501, 502),
            allowed_offer_ids=(1001, 1002),
            allowed_pairs=((501, 1001), (502, 1002)),
        )
        fields.update(overrides)
        return self._auth(**fields)

    def _timed(self):
        state = {"now": 5_000.0}

        def now() -> float:
            state["now"] += 1
            return state["now"]

        return now

    def test_missing_primary_uses_only_the_other_authorized_pair(self) -> None:
        puts = []
        box = {}
        now = self._timed()

        def transport(method, url, body, headers):
            del headers
            if method == "PUT":
                spec = json.loads((box["provider"].runtime / "watchdog.json").read_text(encoding="utf-8"))
                self.assertEqual(spec["machine_id"], 502)
                puts.append((url, json.loads(body.decode("utf-8"))))
                return 200, b'{"success": true, "new_contract": 77}'
            if method == "POST":
                query = json.loads(body.decode("utf-8"))
                machine = (query.get("machine_id") or {}).get("eq")
                if machine == 501:
                    return 200, json.dumps({"offers": []}).encode()
                if machine == 502:
                    return 200, json.dumps({"offers": [
                        _offer(1002, 502, gpu_name="RTX 5880 Ada", inet_down=2000),
                    ]}).encode()
                return 200, json.dumps({"offers": [
                    _offer(1003, 503, inet_down=20000, dph_base="0.40", dph_total="0.42"),
                ]}).encode()
            if method == "GET":
                return 200, b'{"success": true, "instances": [], "next_token": null}'
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()), now=now,
        )
        box["provider"] = provider
        provider._watchdog_popen = _Popen
        provider._manifest_ready = lambda: True
        provider.bind_authorization(self._jit(), code_sha=SHA)
        self.assertIsNone(provider.execution_grant)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "created")
        self.assertEqual(result.resource_id, "77")
        self.assertEqual(len(puts), 1)
        self.assertTrue(puts[0][0].endswith("/asks/1002/"))
        self.assertEqual(puts[0][1]["disk"], 140)
        self.assertNotIn("volume_info", puts[0][1])
        self.assertEqual(provider.execution_grant.machine_id, 502)
        self.assertEqual(provider.execution_grant.offer_id, 1002)
        self.assertEqual(provider.launch_trace, ["watchdog", "put"])
        self.assertEqual(provider.create_classification, "instance_created")
        observation = json.loads((provider.runtime / "create-observation.json").read_text(encoding="utf-8"))
        self.assertEqual(observation["selected_machine_id"], 502)
        self.assertEqual(observation["selected_offer_id"], 1002)
        self.assertLess(observation["offer_observed_at"], observation["final_verification_at"])
        self.assertLess(observation["final_verification_at"], observation["create_request_at"])
        self.assertEqual(observation["provider_response"]["outcome"], "created")
        again = provider.create("studio")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertEqual(len(puts), 1)

    def test_unauthorized_offer_does_not_win_the_ranking(self) -> None:
        puts = []

        def transport(method, url, body, headers):
            del body, headers
            if method == "PUT":
                puts.append(url)
                return 200, b'{"success": true, "new_contract": 77}'
            if method == "POST":
                return 200, json.dumps({"offers": [
                    _offer(1003, 503, inet_down=20000, dph_base="0.40", dph_total="0.42", reliability2=0.999),
                    _offer(1001, 501, inet_down=3000),
                    _offer(1002, 502, gpu_name="RTX 5880 Ada", inet_down=2000),
                ]}).encode()
            if method == "GET":
                return 200, b'{"success": true, "instances": [], "next_token": null}'
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._watchdog_popen = _Popen
        provider._manifest_ready = lambda: True
        provider.bind_authorization(self._jit(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "created")
        self.assertEqual(len(puts), 1)
        self.assertTrue(str(puts[0]).endswith("/asks/1001/"))
        self.assertEqual(provider.execution_grant.offer_id, 1001)

    def test_final_price_increase_stops_before_the_grant_and_the_request(self) -> None:
        seen = {"count": 0}

        def transport(method, url, body, headers):
            del url, headers
            if method == "PUT":
                raise AssertionError("create request")
            if method == "POST":
                seen["count"] += 1
                if seen["count"] == 1:
                    return 200, json.dumps({"offers": [_offer(1001, 501)]}).encode()
                return 200, json.dumps({"offers": [
                    _offer(1001, 501, dph_base="1.10", dph_total="1.10"),
                ]}).encode()
            if method == "GET":
                return 200, b'{"success": true, "instances": [], "next_token": null}'
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._watchdog_popen = _Popen
        provider._manifest_ready = lambda: True
        provider.bind_authorization(self._jit(
            allowed_machine_ids=(501,),
            allowed_offer_ids=(1001,),
            allowed_pairs=((501, 1001),),
        ), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "PRICE_CHANGED")
        self.assertFalse(provider.create_request_sent)
        self.assertIsNone(provider.execution_grant)
        self.assertEqual(provider.launch_trace, [])
        self.assertFalse((provider.runtime / "watchdog.json").is_file())
        self.assertEqual(provider.create_classification, "offer_unavailable_before_create")
        again = provider.create("studio")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertEqual(seen["count"], 2)

    def test_offer_becoming_non_rentable_tries_the_next_pair_before_create(self) -> None:
        puts = []
        reads = {"501": 0}

        def transport(method, url, body, headers):
            del headers
            if method == "PUT":
                puts.append(url)
                return 200, b'{"success": true, "new_contract": 78}'
            if method == "POST":
                query = json.loads(body.decode("utf-8"))
                machine = (query.get("machine_id") or {}).get("eq")
                if machine == 501:
                    reads["501"] += 1
                    rentable = reads["501"] == 1
                    return 200, json.dumps({"offers": [_offer(1001, 501, rentable=rentable)]}).encode()
                if machine == 502:
                    return 200, json.dumps({"offers": [
                        _offer(1002, 502, gpu_name="RTX 5880 Ada", inet_down=2000),
                    ]}).encode()
                return 200, b'{"offers": []}'
            if method == "GET":
                return 200, b'{"success": true, "instances": [], "next_token": null}'
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._watchdog_popen = _Popen
        provider._manifest_ready = lambda: True
        provider.bind_authorization(self._jit(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "created")
        self.assertEqual(len(puts), 1)
        self.assertTrue(puts[0].endswith("/asks/1002/"))
        self.assertEqual(provider.execution_grant.machine_id, 502)
        self.assertGreaterEqual(reads["501"], 2)
        again = provider.create("studio")
        self.assertEqual(len(puts), 1)

    def test_gpu_family_constraint_narrows_the_authorized_pairs(self) -> None:
        puts = []

        def transport(method, url, body, headers):
            del url, headers
            if method == "PUT":
                puts.append(1)
                return 200, b'{"success": true, "new_contract": 79}'
            if method == "POST":
                return 200, json.dumps({"offers": [
                    _offer(1001, 501, inet_down=12000),
                    _offer(1002, 502, gpu_name="RTX 5880 Ada", inet_down=2000),
                ]}).encode()
            if method == "GET":
                return 200, b'{"success": true, "instances": [], "next_token": null}'
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._watchdog_popen = _Popen
        provider._manifest_ready = lambda: True
        provider.bind_authorization(self._jit(
            selection_constraints={"gpu_families": ["5880ada"]},
        ), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "created")
        self.assertEqual(len(puts), 1)
        self.assertEqual(provider.execution_grant.offer_id, 1002)
        self.assertEqual(provider.execution_grant.gpu_family, "5880ada")


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


class FiveDollarBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        CreateFlowTests.setUp(self)
        self._saved_profile = os.environ.get("MEDIA_ENGINE_EXECUTION_PROFILE")
        self._saved_budget = os.environ.get("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD")
        self._saved_studio = os.environ.get("STUDIO_GPU_BUDGET_USD")

    def tearDown(self) -> None:
        self._restore("MEDIA_ENGINE_EXECUTION_PROFILE", self._saved_profile)
        self._restore("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", self._saved_budget)
        self._restore("STUDIO_GPU_BUDGET_USD", self._saved_studio)
        CreateFlowTests.tearDown(self)

    def _restore(self, name: str, value: str | None) -> None:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _auth(self, **overrides):
        return CreateFlowTests._auth(self, **overrides)

    def _five(self, **overrides):
        fields = dict(
            selection_policy="just_in_time",
            allowed_machine_ids=(501, 502),
            allowed_offer_ids=(1001, 1002),
            allowed_pairs=((501, 1001), (502, 1002)),
            max_total_usd=Decimal("5.00"),
        )
        fields.update(overrides)
        return self._auth(**fields)

    def _enable_five(self) -> None:
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live8"
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "5.00"

    def test_just_in_time_five_dollar_authorization_opens_the_same_target(self) -> None:
        authorization = self._five()
        accept_record(authorization, now=5_000, code_sha=SHA)
        self.assertEqual(authorization.max_total_usd, LIVE8_JIT_CEILING_USD)
        budget = open_authorized_budget(
            execution_id=authorization.execution_id,
            grant_execution_id=authorization.execution_id,
            grant_ceiling=authorization.max_total_usd,
            hourly=Decimal("0.72"),
            wall_now=5_000,
            monotonic_now=10,
            observed_starts=[5_000],
        )
        reserve = termination_reserve_usd(Decimal("0.72"))
        self.assertEqual(LIVE7_RESERVE_SECONDS, 840)
        self.assertEqual(budget.ceiling_usd, Decimal("5.00"))
        self.assertEqual(budget.reserve_usd, reserve)
        self.assertEqual(budget.planning_usd + budget.reserve_usd, Decimal("5.00"))
        self.assertEqual(budget.spending_deadline_wall, 5_000 + budget.allowed_seconds)
        self.assertFalse(budget.stop(5_000, 10))
        self.assertTrue(budget.stop(budget.spending_deadline_wall, budget.spending_deadline_monotonic))
        with self.assertRaises(WorkerStageError) as raised:
            open_authorized_budget(
                execution_id=authorization.execution_id,
                grant_execution_id=authorization.execution_id,
                grant_ceiling=Decimal("1.51"),
                hourly=Decimal("0.72"),
                wall_now=5_000,
                monotonic_now=10,
            )
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")
        legacy = open_authorized_budget(
            execution_id="live8-run-a",
            grant_execution_id="live8-run-a",
            grant_ceiling=Decimal("1.50"),
            hourly=Decimal("0.72"),
            wall_now=5_000,
            monotonic_now=10,
        )
        self.assertEqual(legacy.ceiling_usd, Decimal("1.50"))

    def test_five_dollars_requires_the_live8_profile_and_does_not_fall_back(self) -> None:
        os.environ.pop("MEDIA_ENGINE_EXECUTION_PROFILE", None)
        os.environ.pop("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", None)
        self.assertEqual(active_operation_budget(), Decimal("1.00"))
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "5.00"
        with self.assertRaises(WorkerStageError) as raised:
            active_operation_budget()
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live7"
        with self.assertRaises(WorkerStageError) as raised:
            active_operation_budget()
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live8"
        self.assertEqual(active_operation_budget(), Decimal("5.00"))
        os.environ["STUDIO_GPU_BUDGET_USD"] = "1.50"
        self.assertEqual(active_operation_budget(), Decimal("1.50"))
        os.environ["STUDIO_GPU_BUDGET_USD"] = "9.00"
        self.assertEqual(active_operation_budget(), Decimal("5.00"))
        os.environ.pop("MEDIA_ENGINE_EXECUTION_PROFILE", None)
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "9.00"
        self.assertEqual(active_operation_budget(), Decimal("1.00"))
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "1.50"
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live8"
        self.assertEqual(active_operation_budget(), Decimal("1.50"))

    def test_other_policies_and_invalid_totals_reject_five_dollars(self) -> None:
        with self.assertRaises(WorkerStageError) as raised:
            self._auth(max_total_usd=Decimal("5.00"))
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")
        with self.assertRaises(WorkerStageError) as raised:
            self._five(max_total_usd=Decimal("5.01"))
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")
        with self.assertRaises(WorkerStageError) as raised:
            self._five(max_hourly_usd=Decimal("0.96"))
        self.assertEqual(raised.exception.code, "PRICE_CHANGED")
        with self.assertRaises(WorkerStageError) as raised:
            self._five(max_instances=2)
        self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")
        with self.assertRaises(WorkerStageError) as raised:
            self._five(max_inference_attempts=2)
        self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")

    def test_sha_expiry_reuse_and_unauthorized_pair_are_refused(self) -> None:
        authorization = self._five()
        with self.assertRaises(WorkerStageError) as raised:
            accept_record(authorization, now=5_000, code_sha="a" * 40)
        self.assertEqual(raised.exception.code, "STALE_AUTHORIZATION")
        with self.assertRaises(WorkerStageError) as raised:
            accept_record(authorization, now=20_000, code_sha=SHA)
        self.assertEqual(raised.exception.code, "EXPIRED_AUTHORIZATION")
        root = Path(tempfile.mkdtemp())
        claim_execution_id(root / "live8-authorizations", authorization.authorization_id)
        with self.assertRaises(WorkerStageError) as raised:
            claim_execution_id(root / "live8-authorizations", authorization.authorization_id)
        self.assertEqual(raised.exception.code, "STALE_AUTHORIZATION")
        outside = assess_offers([_offer(1003, 501)], observed_at=5_000)
        chosen, reason = select_for_authorization(authorization, outside, now=5_000)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "OFFER_REJECTED")

    def test_matching_configuration_creates_once_at_the_five_dollar_target(self) -> None:
        self._enable_five()
        os.environ["STUDIO_GPU_BUDGET_USD"] = "5.00"
        puts = []

        def transport(method, url, body, headers):
            del body, headers
            if method == "PUT":
                puts.append(url)
                return 200, b'{"success": true, "new_contract": 91}'
            if method == "POST":
                return 200, json.dumps({"offers": [
                    _offer(1003, 503, inet_down=20000, dph_base="0.40", dph_total="0.42"),
                    _offer(1001, 501),
                ]}).encode()
            if method == "GET":
                return 200, b'{"success": true, "instances": [], "next_token": null}'
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._watchdog_popen = _Popen
        provider._manifest_ready = lambda: True
        provider.bind_authorization(self._five(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "created")
        self.assertEqual(len(puts), 1)
        self.assertTrue(puts[0].endswith("/asks/1001/"))
        self.assertEqual(provider.execution_grant.ceiling_usd, Decimal("5.00"))
        self.assertEqual(provider.execution_grant.max_instances, 1)
        spec = json.loads((provider.runtime / "watchdog.json").read_text(encoding="utf-8"))
        self.assertEqual(Decimal(spec["ceiling_usd"]), Decimal("5.00"))
        self.assertEqual(Decimal(spec["reserve_usd"]), termination_reserve_usd(provider.hourly_amount))
        self.assertEqual(spec["machine_id"], 501)
        self.assertEqual(provider.launch_trace, ["watchdog", "put"])
        again = provider.create("studio")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertEqual(len(puts), 1)

    def test_five_dollar_process_value_does_not_authorize_another_profile(self) -> None:
        self._enable_five()
        calls = []

        def transport(method, url, body, headers):
            del url, body, headers
            calls.append(method)
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        with self.assertRaises(WorkerStageError) as raised:
            provider.quote("studio")
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "BUDGET_MISMATCH")
        self.assertEqual(calls, [])
        self.assertFalse(provider.create_request_sent)
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live7"
        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        result = provider.create("studio")
        self.assertEqual(provider.live6_stop, "BUDGET_MISMATCH")
        self.assertEqual(calls, [])

    def test_five_dollar_authorization_without_matching_config_does_not_create(self) -> None:
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live8"
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "1.50"
        provider = CreateFlowTests._provider(self, [_offer(1001, 501)])
        provider.bind_authorization(self._five(), code_sha=SHA)
        result = provider.create("studio")
        self.assertEqual(result.outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "BUDGET_MISMATCH")
        self.assertEqual(provider.puts, [])
        self.assertIsNone(provider.execution_grant)
        self.assertFalse(provider.create_request_sent)

    def test_unconfirmed_deletion_keeps_the_billing_exposure(self) -> None:
        listed = [{"id": "77", "machine_id": 501, "label": "media-engine:live8-run-a"}]

        def transport(method, url, body, headers):
            del url, body, headers
            if method == "DELETE":
                return 200, b"{}"
            if method == "GET":
                return 200, json.dumps({
                    "success": True, "instances": listed, "next_token": None,
                }).encode()
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider.machine_id = "501"
        provider.billing_exposure = True
        with self.assertRaises(WorkerStageError) as raised:
            provider.terminate("77")
        self.assertEqual(raised.exception.code, "DELETE_UNCONFIRMED")
        self.assertTrue(provider.billing_exposure)


if __name__ == "__main__":
    unittest.main()
