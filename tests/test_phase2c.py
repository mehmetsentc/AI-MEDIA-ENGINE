"""Phase 2C guards. Fake transport only. No live rental."""
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
    APPROVED_OFFER_ID,
    Phase2CStop,
    choose_approved,
    run_phase2c,
)
from media_engine.usage import UsageStore


def _env(**overrides) -> dict:
    env = {
        "VAST_API_KEY": "unit-test-vast-secret",
        "LIVE_EXTERNAL_PROVIDERS": "true",
        "VAST_PROVISIONING": "1",
        "VAST_APPROVED_OFFER_ID": APPROVED_OFFER_ID,
        "VAST_HUMAN_APPROVAL": "approve:" + APPROVED_OFFER_ID,
    }
    env.update(overrides)
    return env


def _row(**overrides) -> dict:
    row = {
        "id": int(APPROVED_OFFER_ID),
        "gpu_name": "RTX 5090",
        "num_gpus": 1,
        "gpu_ram": 32607,
        "dph_total": "0.3888888888888889",
        "reliability2": "0.9959727",
        "dlperf": "140",
        "rentable": True,
    }
    row.update(overrides)
    return row


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

    def test_exact_approved_offer_enforcement(self) -> None:
        def transport(*args):
            raise AssertionError("transport")

        code, text = self._run(transport, _env(VAST_APPROVED_OFFER_ID="111"))
        self.assertEqual(code, 2)
        self.assertIn("OFFER_NOT_APPROVED", text)
        code, text = self._run(self._offers(_row(gpu_name="RTX 3090")))
        self.assertEqual(code, 3)
        self.assertIn("OFFER_MISMATCH", text)
        self.assertNotIn("PUT", [call[0] for call in self.calls])

    def test_price_increase_rejection(self) -> None:
        code, text = self._run(self._offers(_row(dph_total="0.41")))
        self.assertEqual(code, 3)
        self.assertIn("PRICE_ABOVE_CEILING", text)
        self.assertNotIn("PUT", [call[0] for call in self.calls])

    def test_no_fallback_offer(self) -> None:
        other = _row(id=111, gpu_name="RTX 5090")
        with self.assertRaises(Phase2CStop) as caught:
            choose_approved([GpuOffer(
                provider="vast", offer_id="111", gpu_model="RTX 5090", gpu_count=1,
                vram_gb=Decimal("32"), hourly_price_usd=Decimal("0.10"),
                reliability=Decimal("0.99"),
            )])
        self.assertEqual(caught.exception.code, "NO_FALLBACK_OFFER")
        code, text = self._run(self._offers(other))
        self.assertEqual(code, 3)
        self.assertIn("OFFER_GONE", text)
        self.assertNotIn("PUT", [call[0] for call in self.calls])

    def test_one_create_attempt(self) -> None:
        code, text = self._run(self._machine(create_body=b"{}"))
        self.assertEqual(code, 4)
        self.assertIn("AMBIGUOUS", text)
        self.assertEqual([call[0] for call in self.calls].count("PUT"), 1)
        self.assertNotIn("DELETE", [call[0] for call in self.calls])

    def test_cleanup_after_confirmed_owned_create(self) -> None:
        code, text = self._run(self._machine(fail_first_delete=True))
        self.assertEqual(code, 0)
        self.assertIn("TERMINATED", text)
        methods = [call[0] for call in self.calls]
        self.assertEqual(methods.count("PUT"), 1)
        self.assertEqual(methods.count("DELETE"), 2)
        saved = json.loads(Path(self.db_path).with_suffix(".phase2c.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["offer_id"], APPROVED_OFFER_ID)
        self.assertEqual(saved["owner_label"], "media-engine:phase2c")
        self.assertEqual(saved["cost_basis"], "estimated_from_runtime")
        self.assertIsNone(UsageStore(self.db_path).for_job("phase2c-555").actual_cost_usd)

    def test_ownership_verification_before_destroy(self) -> None:
        code, text = self._run(self._machine(label="media-engine:other-app"))
        self.assertEqual(code, 5)
        self.assertIn("FOREIGN", text)
        self.assertNotIn("DELETE", [call[0] for call in self.calls])
        self.assertEqual([call[0] for call in self.calls].count("PUT"), 1)

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
        self.assertEqual([call[0] for call in self.calls].count("PUT"), 1)
        self.assertIn("DELETE", [call[0] for call in self.calls])
        self.assertTrue(all(call[1].endswith("/asks/43994880/") for call in self.calls if call[0] == "PUT"))

    def _offers(self, row: dict):
        def transport(method, url, body, headers):
            self.calls.append((method, url))
            self.assertNotEqual(method, "PUT")
            return json.dumps({"offers": [row]}).encode("utf-8")

        return transport

    def _machine(self, *, create_body: bytes = b'{"new_contract": 555}', label: str = "media-engine:phase2c",
                 fail_first_delete: bool = False):
        state = {"destroyed": False, "deletes": 0}

        def transport(method, url, body, headers):
            self.calls.append((method, url))
            self.assertNotIn(b"unit-test-vast-secret", body)
            if method == "POST":
                return json.dumps({"offers": [_row()]}).encode("utf-8")
            if method == "PUT":
                self.assertTrue(url.endswith("/asks/43994880/"))
                self.assertIn(b"media-engine:phase2c", body)
                self.assertIn(b"ubuntu:22.04", body)
                return create_body
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
