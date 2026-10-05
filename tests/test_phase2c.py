"""Phase 2C policy approval. Fake transport only. No live rental."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.request
from decimal import Decimal
from pathlib import Path

from media_engine.providers.offers import GpuOffer
from media_engine.providers.phase2c import (
    Phase2CStop,
    accept_offer,
    fingerprint_for,
    format_policy,
    policy_fields,
    policy_fingerprint,
    run_phase2c,
)
from media_engine.resources.ledger import ResourceRecord, ResourceState
from media_engine.usage import UsageStore


def _env(**overrides) -> dict:
    env = {
        "VAST_API_KEY": "unit-test-vast-secret",
        "LIVE_EXTERNAL_PROVIDERS": "true",
        "VAST_PROVISIONING": "1",
        "VAST_HUMAN_POLICY_APPROVAL": "approve-policy:" + policy_fingerprint(),
    }
    env.update(overrides)
    return env


def _row(**overrides) -> dict:
    row = {
        "id": 200,
        "gpu_name": "RTX 5090",
        "num_gpus": 1,
        "gpu_ram": 32607,
        "dph_total": "0.20",
        "reliability2": "0.99",
        "dlperf": "80",
        "rentable": True,
    }
    row.update(overrides)
    return row


def _gpu(**kwargs) -> GpuOffer:
    values = dict(
        provider="vast", offer_id="200", gpu_model="RTX 5090", gpu_count=1,
        vram_gb=Decimal("32"), hourly_price_usd=Decimal("0.20"),
        reliability=Decimal("0.99"), performance=Decimal("80"), availability="rentable",
    )
    values.update(kwargs)
    return GpuOffer(**values)


class _Clock:
    def __init__(self) -> None:
        self.t = 1_700_000_000.0

    def __call__(self) -> float:
        return self.t


class Phase2CTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "meta.sqlite3")
        self.clock = _Clock()
        self.calls: list[tuple[str, str]] = []
        self._urlopen = urllib.request.urlopen
        urllib.request.urlopen = self._block

    def tearDown(self) -> None:
        urllib.request.urlopen = self._urlopen

    def _block(self, *args, **kwargs):
        raise AssertionError("network")

    def _run(self, transport, env=None) -> tuple[int, str]:
        stdout = io.StringIO()
        code = run_phase2c(
            _env() if env is None else env,
            db_path=self.db_path,
            search_transport=transport,
            lifecycle_transport=transport,
            now=self.clock,
            stdout=stdout,
        )
        text = stdout.getvalue()
        self.assertNotIn("unit-test-vast-secret", text)
        return code, text

    def test_policy_fingerprint_is_deterministic_and_changes(self) -> None:
        self.assertEqual(policy_fingerprint(), policy_fingerprint())
        changed = dict(policy_fields())
        changed["max_hourly_price_usd"] = "0.41"
        self.assertNotEqual(fingerprint_for(changed), policy_fingerprint())
        text = format_policy()
        self.assertIn("PHASE 2C POLICY", text)
        self.assertIn("gpu_model: RTX 5090", text)
        self.assertIn("min_reliability: 0.98", text)
        self.assertIn("POLICY FINGERPRINT: " + policy_fingerprint(), text)
        self.assertIn("STATUS: HUMAN_POLICY_APPROVAL_REQUIRED", text)
        self.assertNotIn("approve:any", text)

    def test_missing_or_wrong_approval_blocks_before_discovery(self) -> None:
        def transport(*args):
            raise AssertionError("transport")

        code, text = self._run(transport, _env(VAST_HUMAN_POLICY_APPROVAL=""))
        self.assertEqual(code, 2)
        self.assertIn("HUMAN_POLICY_APPROVAL_REQUIRED", text)
        code, text = self._run(transport, _env(VAST_HUMAN_POLICY_APPROVAL="approve-policy:deadbeef"))
        self.assertEqual(code, 2)
        self.assertIn("APPROVAL_MISMATCH", text)
        code, text = self._run(transport, _env(VAST_HUMAN_POLICY_APPROVAL="approve:any"))
        self.assertEqual(code, 2)
        self.assertIn("APPROVAL_MISMATCH", text)
        code, text = self._run(transport, _env(VAST_HUMAN_POLICY_APPROVAL="approve-policy:any"))
        self.assertEqual(code, 2)
        self.assertIn("APPROVAL_MISMATCH", text)

    def test_policy_rejections(self) -> None:
        with self.assertRaises(Phase2CStop) as caught:
            accept_offer(_gpu(hourly_price_usd=Decimal("0.41")))
        self.assertEqual(caught.exception.code, "PRICE_ABOVE_CEILING")
        with self.assertRaises(Phase2CStop) as caught:
            accept_offer(_gpu(hourly_price_usd=Decimal("0.43")))
        self.assertEqual(caught.exception.code, "COST_ABOVE_CEILING")
        with self.assertRaises(Phase2CStop) as caught:
            accept_offer(_gpu(reliability=Decimal("0.97")))
        self.assertEqual(caught.exception.code, "OFFER_MISMATCH")
        with self.assertRaises(Phase2CStop) as caught:
            accept_offer(_gpu(gpu_model="RTX 3090"))
        self.assertEqual(caught.exception.code, "OFFER_MISMATCH")
        with self.assertRaises(Phase2CStop) as caught:
            accept_offer(_gpu(vram_gb=Decimal("8")))
        self.assertEqual(caught.exception.code, "OFFER_MISMATCH")
        for row in (
            _row(dph_total="0.41"),
            _row(reliability2="0.97"),
            _row(gpu_name="RTX 3090"),
            _row(gpu_ram=8192),
            _row(dph_total="0.43"),
        ):
            self.calls.clear()
            code, text = self._run(self._offers(row))
            self.assertEqual(code, 3)
            self.assertIn("OFFER_GONE", text)
            self.assertNotIn("PUT", [call[0] for call in self.calls])

    def test_jit_selects_one_offer_and_creates_it_once(self) -> None:
        rows = [
            _row(id=100, dph_total="0.30", dlperf="10"),
            _row(id=200, dph_total="0.20", dlperf="80"),
        ]
        code, text = self._run(self._offers(*rows, create_body=b"{}"))
        puts = [url for method, url in self.calls if method == "PUT"]
        posts = [url for method, url in self.calls if method == "POST"]
        self.assertEqual(code, 4)
        self.assertIn("AMBIGUOUS", text)
        self.assertEqual(len(posts), 1)
        self.assertEqual(puts, ["https://console.vast.ai/api/v0/asks/200/"])

    def test_offer_gone_does_not_fallback_or_create_again(self) -> None:
        code, text = self._run(self._offers(_row(id=100, gpu_name="RTX 3090"), _row(id=300, gpu_ram=8192)))
        self.assertEqual(code, 3)
        self.assertIn("OFFER_GONE", text)
        self.assertNotIn("PUT", [call[0] for call in self.calls])
        self.assertEqual([method for method, _url in self.calls].count("POST"), 1)

    def test_unresolved_resource_blocks_before_create(self) -> None:
        from media_engine.resources.ledger import ResourceLedger
        ResourceLedger(self.db_path).upsert(ResourceRecord(
            resource_id="pending", provider="vast", owner_id="phase2c",
            state=ResourceState.AMBIGUOUS, create_attempted=True, hourly_price_usd="0.20",
            created_at=0, updated_at=0, last_error="AMBIGUOUS_CREATE",
        ))

        def transport(*args):
            raise AssertionError("transport")

        code, text = self._run(transport)
        self.assertEqual(code, 2)
        self.assertIn("UNRESOLVED_RESOURCE", text)

    def test_cleanup_after_confirmed_owned_create(self) -> None:
        code, text = self._run(self._machine(fail_first_delete=True))
        self.assertEqual(code, 0)
        self.assertIn("TERMINATED", text)
        methods = [method for method, _url in self.calls]
        self.assertEqual(methods.count("POST"), 1)
        self.assertEqual(methods.count("PUT"), 1)
        self.assertEqual(methods.count("DELETE"), 2)
        saved = json.loads(Path(self.db_path).with_suffix(".phase2c.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["offer_id"], "200")
        self.assertEqual(saved["owner_label"], "media-engine:phase2c")
        self.assertEqual(saved["cost_basis"], "estimated_from_runtime")
        self.assertIsNone(UsageStore(self.db_path).for_job("phase2c-555").actual_cost_usd)

    def test_ownership_verification_before_destroy(self) -> None:
        code, text = self._run(self._machine(label="media-engine:other-app"))
        self.assertEqual(code, 5)
        self.assertIn("FOREIGN", text)
        self.assertNotIn("DELETE", [method for method, _url in self.calls])
        self.assertEqual([method for method, _url in self.calls].count("PUT"), 1)

    def test_max_lifetime_guard(self) -> None:
        def transport(method, url, body, headers):
            self.calls.append((method, url))
            if method == "POST":
                return json.dumps({"offers": [_row()]}).encode("utf-8")
            if method == "PUT":
                self.clock.t += 601
                return b'{"new_contract": 555}'
            if method == "DELETE":
                self.destroyed = True
                return b"{}"
            if getattr(self, "destroyed", False):
                return b'{"instances": []}'
            return b'{"instances": [{"id": 555, "label": "media-engine:phase2c"}]}'

        code, text = self._run(transport)
        self.assertEqual(code, 6)
        self.assertIn("LIFETIME_EXCEEDED", text)
        self.assertEqual([method for method, _url in self.calls].count("PUT"), 1)
        self.assertIn("DELETE", [method for method, _url in self.calls])

    def _offers(self, *rows: dict, create_body: bytes = b'{"new_contract": 555}'):
        def transport(method, url, body, headers):
            self.calls.append((method, url))
            if method == "POST":
                return json.dumps({"offers": list(rows)}).encode("utf-8")
            if method == "PUT":
                return create_body
            return b'{"instances": []}'

        return transport

    def _machine(self, *, label: str = "media-engine:phase2c", fail_first_delete: bool = False):
        state = {"destroyed": False, "deletes": 0}

        def transport(method, url, body, headers):
            self.calls.append((method, url))
            self.assertNotIn(b"unit-test-vast-secret", body)
            if method == "POST":
                return json.dumps({"offers": [_row()]}).encode("utf-8")
            if method == "PUT":
                self.assertTrue(url.endswith("/asks/200/"))
                self.assertIn(b"media-engine:phase2c", body)
                return b'{"new_contract": 555}'
            if method == "DELETE":
                state["deletes"] += 1
                if fail_first_delete and state["deletes"] == 1:
                    raise RuntimeError("destroy failed")
                state["destroyed"] = True
                return b"{}"
            if state["destroyed"]:
                return b'{"instances": []}'
            return json.dumps({"instances": [{"id": 555, "label": label}]}).encode("utf-8")

        return transport
