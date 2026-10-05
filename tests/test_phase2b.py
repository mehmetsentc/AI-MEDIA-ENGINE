"""Phase 2B planner and gated Vast lifecycle. No network and no rental."""
from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
from decimal import Decimal
from pathlib import Path

from media_engine.providers.offers import GpuOffer
from media_engine.providers.planner import (
    ResourceRequirements,
    max_estimated_gpu_cost,
    plan,
)
from media_engine.providers.vast import SEARCH_URL, VastDiscovery
from media_engine.providers.vast_lifecycle import ProvisionBlocked, VastLifecycle, approval_token
from media_engine.providers.vast_preflight import main
from media_engine.resources.ledger import ResourceRecord, ResourceState
from media_engine.safety.limits import ExternalProvidersDisabled, SafetyLimits


def _offer(offer_id: str = "1", **kwargs) -> GpuOffer:
    values = dict(
        provider="vast",
        offer_id=offer_id,
        gpu_model="Example GPU",
        gpu_count=1,
        vram_gb=Decimal("24"),
        hourly_price_usd=Decimal("0.20"),
        reliability=Decimal("0.99"),
        performance=Decimal("40"),
    )
    values.update(kwargs)
    return GpuOffer(**values)


def _requirements(**kwargs) -> ResourceRequirements:
    values = dict(
        min_vram_gb=Decimal("16"),
        gpu_count=1,
        max_hourly_price_usd=Decimal("0.50"),
        min_reliability=Decimal("0.90"),
        max_job_seconds=1800,
    )
    values.update(kwargs)
    return ResourceRequirements(**values)


def _row(offer_id: int, **kwargs) -> dict:
    row = {
        "id": offer_id,
        "gpu_name": "Example GPU",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "dph_total": "0.20",
        "reliability2": "0.99",
        "dlperf": "40",
        "rentable": True,
    }
    row.update(kwargs)
    return row


class Phase2BTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = str(Path(self.tmp.name) / "meta.sqlite3")
        self._urlopen = urllib.request.urlopen
        urllib.request.urlopen = self._block_network

    def tearDown(self) -> None:
        urllib.request.urlopen = self._urlopen

    def _block_network(self, *args, **kwargs):
        raise AssertionError("network")

    def test_larger_discovery_result_handling(self) -> None:
        rows = [_row(30), _row(2), _row(10)]
        seen = {}

        def transport(method, url, body, headers):
            seen["query"] = json.loads(body.decode("utf-8"))
            self.assertEqual(url, SEARCH_URL)
            return json.dumps({"offers": rows}).encode("utf-8")

        offers = VastDiscovery(transport=transport).search(
            _requirements().to_search(), api_key="unit-test-vast-secret", read_only=True, limit=25,
        )
        self.assertEqual(seen["query"]["limit"], 25)
        self.assertNotIn("order", seen["query"])
        self.assertEqual([offer.offer_id for offer in offers], ["2", "10", "30"])

    def test_optional_model_name_filtering(self) -> None:
        seen = {}

        def transport(method, url, body, headers):
            seen["query"] = json.loads(body.decode("utf-8"))
            return json.dumps({"offers": [
                _row(1, gpu_name="Example GPU"),
                _row(2, gpu_name="Other GPU"),
            ]}).encode("utf-8")

        offers = VastDiscovery(transport=transport).search(
            _requirements().to_search(gpu_model="EXAMPLE_GPU"),
            api_key="unit-test-vast-secret",
            read_only=True,
        )
        self.assertEqual(seen["query"]["gpu_name"], {"eq": "EXAMPLE_GPU"})
        self.assertEqual([offer.offer_id for offer in offers], ["1"])
        root = Path(__file__).resolve().parents[1] / "src" / "media_engine" / "providers"
        blob = "\n".join(
            path.read_text(encoding="utf-8")
            for path in root.glob("*.py")
            if path.name != "phase2c.py"
        )
        for name in ("3090", "4090", "5090", "A6000", "L40S"):
            self.assertNotIn(name, blob)

    def test_planner_rejects_insufficient_vram(self) -> None:
        result = plan([_offer(vram_gb=Decimal("8"))], _requirements())
        self.assertIsNone(result.selected)
        self.assertEqual(result.eligible, ())
        self.assertIn("vram", result.explanation)

    def test_planner_rejects_reliability_below_threshold(self) -> None:
        result = plan([_offer(reliability=Decimal("0.50"))], _requirements())
        self.assertIsNone(result.selected)
        self.assertIn("reliability", result.explanation)

    def test_planner_rejects_price_above_ceiling(self) -> None:
        result = plan([_offer(hourly_price_usd=Decimal("1.25"))], _requirements())
        self.assertIsNone(result.selected)
        self.assertIn("price", result.explanation)

    def test_planner_rejects_explicitly_incompatible_gpu(self) -> None:
        cheap = _offer("1", gpu_model="Excluded GPU", hourly_price_usd=Decimal("0.05"), performance=Decimal("100"))
        kept = _offer("2", gpu_model="Useful GPU", hourly_price_usd=Decimal("0.40"), performance=Decimal("10"))
        result = plan([cheap, kept], _requirements(excluded_gpu_models=("EXCLUDED_GPU",)))
        self.assertEqual(result.selected.offer_id, "2")
        only = plan([cheap], _requirements(excluded_gpu_models=("Excluded GPU",)))
        self.assertIsNone(only.selected)
        self.assertIn("incompatible", only.explanation)

    def test_deterministic_ranking(self) -> None:
        first = _offer("9", hourly_price_usd=Decimal("0.10"), performance=Decimal("5"))
        second = _offer("2", hourly_price_usd=Decimal("0.20"), performance=Decimal("10"))
        left = plan([second, first], _requirements())
        right = plan([first, second], _requirements())
        self.assertEqual(left.selected.offer_id, "9")
        self.assertEqual([offer.offer_id for offer in left.shortlist], [offer.offer_id for offer in right.shortlist])

    def test_performance_price_heuristic(self) -> None:
        weak = _offer("1", hourly_price_usd=Decimal("0.05"), performance=Decimal("1"))
        strong = _offer("2", hourly_price_usd=Decimal("0.40"), performance=Decimal("40"))
        too_small = _offer("3", vram_gb=Decimal("4"), hourly_price_usd=Decimal("0.01"), performance=Decimal("100"))
        result = plan([weak, too_small, strong], _requirements())
        self.assertEqual(result.selected.offer_id, "2")
        self.assertNotIn("3", [offer.offer_id for offer in result.eligible])
        self.assertIn("not measured inference speed", result.explanation)

    def test_max_estimated_cost_calculation(self) -> None:
        self.assertEqual(max_estimated_gpu_cost(Decimal("0.36"), 1800), Decimal("0.18"))
        result = plan([_offer(hourly_price_usd=Decimal("0.36"))], _requirements(max_job_seconds=1800))
        self.assertEqual(result.max_estimated_cost_usd, Decimal("0.18"))

    def test_human_approval_required_before_create(self) -> None:
        calls = []

        def transport(method, url, body, headers):
            calls.append((method, url))
            return json.dumps({"offers": [_row(7, dph_total="0.36", dlperf="40")]}).encode("utf-8")

        import io
        stdout = io.StringIO()
        code = main(environ={
            "VAST_API_KEY": "unit-test-vast-secret",
            "VAST_READ_ONLY_DISCOVERY": "1",
            "VAST_MIN_VRAM_GB": "16",
            "VAST_GPU_COUNT": "1",
            "VAST_MAX_HOURLY_PRICE_USD": "0.50",
            "VAST_MIN_RELIABILITY": "0.90",
            "VAST_MAX_JOB_SECONDS": "1800",
        }, transport=transport, stdout=stdout)
        text = stdout.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("PROVIDER: vast", text)
        self.assertIn("OFFER ID: 7", text)
        self.assertIn("MAX ESTIMATED GPU COST: $0.18", text)
        self.assertIn("STATUS: HUMAN_APPROVAL_REQUIRED", text)
        self.assertEqual(calls, [("POST", SEARCH_URL)])
        self.assertNotIn("unit-test-vast-secret", text)
        source = Path(__file__).resolve().parents[1].joinpath(
            "src/media_engine/providers/vast_preflight.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("/asks/", source)
        self.assertNotIn("VastLifecycle", source)

    def test_missing_approval_blocks_create(self) -> None:
        calls = []
        lifecycle = self._lifecycle(calls, live=True)
        with self.assertRaises(ProvisionBlocked) as caught:
            self._create(lifecycle, human_approval="")
        self.assertEqual(caught.exception.code, "HUMAN_APPROVAL_REQUIRED")
        self.assertEqual(calls, [])

    def test_live_external_providers_false_blocks_create(self) -> None:
        calls = []
        lifecycle = self._lifecycle(calls, live=False)
        with self.assertRaises(ExternalProvidersDisabled):
            self._create(lifecycle, human_approval=approval_token("1"))
        self.assertEqual(calls, [])

    def test_missing_api_key_blocks_create(self) -> None:
        calls = []
        lifecycle = self._lifecycle(calls, live=True)
        with self.assertRaises(ProvisionBlocked) as caught:
            self._create(lifecycle, api_key="", human_approval=approval_token("1"))
        self.assertEqual(caught.exception.code, "VAST_API_KEY_MISSING")
        self.assertEqual(calls, [])

    def test_unresolved_resource_blocks_create(self) -> None:
        calls = []
        lifecycle = self._lifecycle(calls, live=True)
        lifecycle.ledger.upsert(ResourceRecord(
            resource_id="pending-1", provider="vast", owner_id="film-studio",
            state=ResourceState.AMBIGUOUS, create_attempted=True, hourly_price_usd="0.20",
            created_at=0, updated_at=0, last_error="AMBIGUOUS_CREATE",
        ))
        with self.assertRaises(ProvisionBlocked) as caught:
            self._create(lifecycle, human_approval=approval_token("1"))
        self.assertEqual(caught.exception.code, "UNRESOLVED_RESOURCE")
        self.assertEqual(calls, [])

    def test_foreign_resource_is_never_terminated(self) -> None:
        calls = []

        def transport(method, url, body, headers):
            calls.append(method)
            self.assertNotIn("unit-test-vast-secret", body.decode("utf-8"))
            return json.dumps({"instances": [{"id": 7, "label": "media-engine:other-app"}]}).encode("utf-8")

        lifecycle = VastLifecycle(
            self.db_path,
            limits=SafetyLimits(live_external_providers=True),
            transport=transport,
        )
        with self.assertRaises(ProvisionBlocked) as caught:
            lifecycle.terminate(
                "7", "film-studio", api_key="unit-test-vast-secret", provision=True,
                human_approval="approve-terminate:7",
            )
        self.assertEqual(caught.exception.code, "OWNERSHIP_MISMATCH")
        self.assertEqual(calls, ["GET"])

    def test_ambiguous_create_is_never_retried(self) -> None:
        calls = []
        lifecycle = self._lifecycle(calls, live=True, body=b"{}")
        first = self._create(lifecycle, human_approval=approval_token("1"))
        second_blocked = False
        try:
            self._create(lifecycle, human_approval=approval_token("1"))
        except ProvisionBlocked as exc:
            second_blocked = exc.code == "UNRESOLVED_RESOURCE"
        self.assertEqual(first.outcome, "ambiguous")
        self.assertTrue(second_blocked)
        self.assertEqual(len(calls), 1)
        self.assertEqual(lifecycle.ledger.unresolved()[0].state, ResourceState.AMBIGUOUS)

    def test_max_gpu_workers_remains_enforced(self) -> None:
        calls = []
        lifecycle = self._lifecycle(calls, live=True)
        lifecycle.ledger.upsert(ResourceRecord(
            resource_id="9", provider="vast", owner_id="film-studio", state=ResourceState.READY,
            create_attempted=True, hourly_price_usd="0.20", created_at=0, updated_at=0,
        ))
        self.assertEqual(SafetyLimits().max_gpu_workers, 1)
        with self.assertRaises(ProvisionBlocked) as caught:
            self._create(lifecycle, human_approval=approval_token("1"))
        self.assertEqual(caught.exception.code, "MAX_GPU_WORKERS")
        self.assertEqual(calls, [])

    def test_mutation_uses_fake_transport_only(self) -> None:
        calls = []
        lifecycle = self._lifecycle(calls, live=True, body=b'{"new_contract": 44}')
        created = self._create(lifecycle, human_approval=approval_token("1"))
        self.assertEqual(created.outcome, "created")
        self.assertEqual(created.resource_id, "44")
        self.assertEqual(calls[0][0], "PUT")
        self.assertEqual(calls[0][1], "https://console.vast.ai/api/v0/asks/1/")
        self.assertIn(b"media-engine:film-studio", calls[0][2])
        self.assertNotIn(b"unit-test-vast-secret", calls[0][2])

    def _lifecycle(self, calls: list, *, live: bool, body: bytes = b"{}") -> VastLifecycle:
        def transport(method, url, payload, headers):
            calls.append((method, url, payload))
            self.assertTrue(headers["Authorization"].endswith("unit-test-vast-secret"))
            return body

        return VastLifecycle(
            self.db_path,
            limits=SafetyLimits(live_external_providers=live),
            transport=transport,
        )

    def _create(self, lifecycle: VastLifecycle, **kwargs):
        values = dict(
            offers=[_offer()],
            requirements=_requirements(),
            offer_id="1",
            owner="film-studio",
            api_key="unit-test-vast-secret",
            provision=True,
            human_approval="",
            image="fixture-image",
            disk_gb=8,
        )
        values.update(kwargs)
        return lifecycle.create(**values)
