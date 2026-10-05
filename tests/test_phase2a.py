"""Phase 2A read-only Vast discovery. No network and no provisioning."""
from __future__ import annotations

import io
import json
import logging
import unittest
import urllib.request
from decimal import Decimal
from pathlib import Path

from media_engine.providers.fake import FakeGPUProvider
from media_engine.providers.offers import GpuRequirements
from media_engine.providers.vast import (
    SEARCH_METHOD,
    SEARCH_URL,
    DiscoveryDisabled,
    MissingApiKey,
    VastDiscovery,
)
from media_engine.providers.vast_discover import main
from media_engine.safety.limits import ExternalProvidersDisabled, SafetyLimits, require_live_external_provider

_SECRET = "unit-test-vast-secret"
_OFFER = {
    "id": 42,
    "gpu_name": "Example GPU",
    "num_gpus": 2,
    "gpu_ram": 24576,
    "dph_total": "0.40",
    "reliability2": "0.98",
    "dlperf": "100.5",
    "cuda_max_good": 12.4,
    "disk_space": "150",
    "geolocation": "Example, EX",
    "rentable": True,
    "public_ipaddr": "203.0.113.10",
}


def _requirements() -> GpuRequirements:
    return GpuRequirements(
        min_vram_gb=Decimal("16"),
        max_hourly_price_usd=Decimal("0.50"),
        min_reliability=Decimal("0.90"),
        gpu_count=2,
    )


class Phase2ATests(unittest.TestCase):
    def test_vast_response_normalization(self) -> None:
        calls: list[tuple] = []

        def transport(method, url, body, headers):
            calls.append((method, url, body, headers))
            return json.dumps({"offers": [_OFFER]}).encode("utf-8")

        offers = VastDiscovery(transport=transport).search(
            _requirements(), api_key=_SECRET, read_only=True,
        )
        self.assertEqual(len(offers), 1)
        offer = offers[0]
        self.assertEqual(offer.provider, "vast")
        self.assertEqual(offer.offer_id, "42")
        self.assertEqual(offer.gpu_model, "Example GPU")
        self.assertEqual(offer.gpu_count, 2)
        self.assertEqual(offer.vram_gb, Decimal("24"))
        self.assertEqual(offer.hourly_price_usd, Decimal("0.40"))
        self.assertEqual(offer.reliability, Decimal("0.98"))
        self.assertEqual(offer.performance, Decimal("100.5"))
        self.assertEqual(offer.cuda, "12.4")
        self.assertEqual(offer.disk_gb, Decimal("150"))
        self.assertEqual(offer.location, "Example, EX")
        self.assertEqual(offer.availability, "rentable")
        self.assertNotIn("public_ipaddr", offer.as_dict())
        self.assertEqual(calls[0][0], SEARCH_METHOD)
        self.assertEqual(calls[0][1], SEARCH_URL)

    def test_generic_gpu_requirements(self) -> None:
        seen: dict = {}

        def transport(method, url, body, headers):
            seen["query"] = json.loads(body.decode("utf-8"))
            cheap = dict(_OFFER)
            expensive = dict(_OFFER, id=99, dph_total="9.00", gpu_name="Other GPU")
            return json.dumps({"offers": [expensive, cheap]}).encode("utf-8")

        offers = VastDiscovery(transport=transport).search(
            _requirements(), api_key=_SECRET, read_only=True,
        )
        query = seen["query"]
        self.assertEqual(query["num_gpus"], {"eq": 2})
        self.assertEqual(query["gpu_ram"], {"gte": 16384})
        self.assertEqual(query["dph_total"], {"lte": 0.5})
        self.assertEqual(query["reliability"], {"gte": 0.9})
        self.assertNotIn("gpu_name", query)
        self.assertNotIn("geolocation", query)
        self.assertEqual([offer.offer_id for offer in offers], ["42"])

    def test_missing_api_key_fails_closed(self) -> None:
        def transport(*args):
            raise AssertionError("transport called")

        with self.assertRaises(MissingApiKey):
            VastDiscovery(transport=transport).search(
                GpuRequirements(), api_key="", read_only=True,
            )
        self.assertEqual(main(environ={"VAST_READ_ONLY_DISCOVERY": "1"}, transport=transport), 2)

    def test_live_discovery_disabled_by_default(self) -> None:
        limits = SafetyLimits()
        self.assertFalse(limits.live_external_providers)
        with self.assertRaises(ExternalProvidersDisabled):
            require_live_external_provider(limits)

        def transport(*args):
            raise AssertionError("transport called")

        with self.assertRaises(DiscoveryDisabled):
            VastDiscovery(transport=transport).search(
                GpuRequirements(), api_key=_SECRET, read_only=False,
            )
        self.assertEqual(main(environ={"VAST_API_KEY": _SECRET}, transport=transport), 2)
        self.assertEqual(main(environ={}, transport=transport), 2)

    def test_fake_vast_discovery_works_without_network(self) -> None:
        def transport(method, url, body, headers):
            return json.dumps({"offers": [_OFFER]}).encode("utf-8")

        original = urllib.request.urlopen

        def blocked(*args, **kwargs):
            raise AssertionError("network")

        urllib.request.urlopen = blocked
        try:
            offers = VastDiscovery(transport=transport).search(
                GpuRequirements(), api_key=_SECRET, read_only=True,
            )
        finally:
            urllib.request.urlopen = original
        self.assertEqual(offers[0].offer_id, "42")

    def test_no_mutation_provision_operation(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "media_engine" / "providers"
        text = (root / "vast.py").read_text(encoding="utf-8")
        text += (root / "vast_discover.py").read_text(encoding="utf-8")
        for token in ("/asks/", "/instances", "serverless", "4090", "5090"):
            self.assertNotIn(token, text)
        names = set(dir(VastDiscovery))
        for name in ("create", "terminate", "destroy", "rent", "start", "stop"):
            self.assertNotIn(name, names)
        self.assertEqual(SEARCH_URL, "https://console.vast.ai/api/v0/bundles/")
        self.assertEqual(SEARCH_METHOD, "POST")

    def test_vast_secrets_are_not_persisted_or_logged(self) -> None:
        records: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record.getMessage())
        logging.getLogger().addHandler(handler)
        try:
            def transport(method, url, body, headers):
                self.assertNotIn(_SECRET, body.decode("utf-8"))
                self.assertTrue(headers["Authorization"].endswith(_SECRET))
                return json.dumps({"offers": [_OFFER]}).encode("utf-8")

            discovery = VastDiscovery(transport=transport)
            offers = discovery.search(GpuRequirements(), api_key=_SECRET, read_only=True)
            rendered = json.dumps([offer.as_dict() for offer in offers])
            self.assertNotIn(_SECRET, rendered)
            self.assertNotIn(_SECRET, repr(discovery.__dict__))
            stdout = io.StringIO()
            code = main(
                environ={"VAST_API_KEY": _SECRET, "VAST_READ_ONLY_DISCOVERY": "1"},
                transport=transport,
                stdout=stdout,
            )
            self.assertEqual(code, 0)
            self.assertNotIn(_SECRET, stdout.getvalue())
            self.assertFalse(any(_SECRET in line for line in records))
        finally:
            logging.getLogger().removeHandler(handler)

    def test_fake_gpu_provider_regression(self) -> None:
        provider = FakeGPUProvider(lambda: 0)
        quote = provider.quote("film-studio")
        created = provider.create("film-studio")
        self.assertEqual(quote.provider, "fake")
        self.assertEqual(created.outcome, "created")
        self.assertEqual(provider.create_calls, 1)
        self.assertTrue(created.resource_id)
