"""The Vast provider for media capabilities. Nothing here reaches the network.

What these pin is what a mistake would cost: the wrong image, a moving tag,
the image engine's volume, a second rental, or renting before pricing.
"""
from __future__ import annotations

import json
import unittest
from decimal import Decimal
from pathlib import Path

from media_engine.providers.base import CreateResult, WorkerStageError
from media_engine.providers.offers import GpuOffer
from media_engine.providers.vast_speech import (
    VastSpeechProvider,
    owner_terminator,
    speech_image_reference,
)

IMAGE = "ghcr.io/example/worker@sha256:" + "ab" * 32


def _offer(offer_id="123", price="0.25", vram="24"):
    return GpuOffer(
        provider="vast", offer_id=offer_id, gpu_model="RTX 4090", gpu_count=1,
        vram_gb=Decimal(vram), hourly_price_usd=Decimal(price),
        reliability=Decimal("0.99"), disk_gb=Decimal("200"),
    )


class _Discovery:
    def __init__(self, offers=None) -> None:
        self.offers = offers if offers is not None else [_offer()]
        self.calls = []

    def search(self, requirements, *, api_key="", read_only=False, limit=5):
        self.calls.append({"read_only": read_only, "limit": limit})
        return list(self.offers)


class _Lifecycle:
    def __init__(self, outcome="created") -> None:
        self.outcome = outcome
        self.created: list[dict] = []
        self.terminated: list[dict] = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return CreateResult(outcome=self.outcome, resource_id="777")

    def observe(self, resource_id, owner, *, api_key="", provision=False):
        return "present"

    def terminate(self, resource_id, owner, *, api_key="", provision=False, human_approval=""):
        self.terminated.append({"id": resource_id, "owner": owner, "approval": human_approval})


def _provider(discovery=None, lifecycle=None, price="0.60"):
    return VastSpeechProvider(
        api_key="k", image=IMAGE, lifecycle=lifecycle or _Lifecycle(),
        discovery=discovery or _Discovery(), max_hourly_usd=Decimal(price),
    )


class ImageTests(unittest.TestCase):
    def test_a_moving_tag_is_refused(self):
        """A tag can point somewhere else tomorrow; a digest cannot."""
        with self.assertRaises(WorkerStageError) as caught:
            VastSpeechProvider(
                api_key="k", image="ghcr.io/example/worker:latest",
                lifecycle=_Lifecycle(), discovery=_Discovery(),
                max_hourly_usd=Decimal("0.60"),
            )
        self.assertEqual(caught.exception.code, "WORKLOAD_IMAGE_REQUIRED")

    def test_the_repository_ships_a_digest_pinned_speech_image(self):
        reference = speech_image_reference()
        self.assertIn("@sha256:", reference)
        self.assertIn("ai-media-engine-speech-worker", reference)

    def test_an_empty_api_key_is_refused(self):
        with self.assertRaises(WorkerStageError):
            VastSpeechProvider(
                api_key="  ", image=IMAGE, lifecycle=_Lifecycle(),
                discovery=_Discovery(), max_hourly_usd=Decimal("0.60"),
            )


class QuoteTests(unittest.TestCase):
    def test_the_search_is_read_only(self):
        discovery = _Discovery()
        _provider(discovery).quote("nahaber")
        self.assertTrue(discovery.calls[0]["read_only"])

    def test_the_cheapest_eligible_offer_is_priced(self):
        discovery = _Discovery([_offer("1", "0.55"), _offer("2", "0.19"), _offer("3", "0.33")])
        quote = _provider(discovery).quote("nahaber")
        self.assertEqual(quote.hourly_price_usd, Decimal("0.19"))
        self.assertEqual(quote.provider, "vast")

    def test_an_offer_above_the_ceiling_is_not_eligible(self):
        discovery = _Discovery([_offer("1", "5.00")])
        with self.assertRaises(WorkerStageError) as caught:
            _provider(discovery, price="0.60").quote("nahaber")
        self.assertEqual(caught.exception.code, "NO_ELIGIBLE_OFFER")

    def test_a_card_too_small_for_the_models_is_not_eligible(self):
        discovery = _Discovery([_offer("1", "0.10", vram="8")])
        with self.assertRaises(WorkerStageError) as caught:
            _provider(discovery).quote("nahaber")
        self.assertEqual(caught.exception.code, "NO_ELIGIBLE_OFFER")


class CreateTests(unittest.TestCase):
    def test_renting_before_pricing_is_refused(self):
        """The ceiling lives in the quote. Skipping it skips the ceiling."""
        provider = _provider()
        with self.assertRaises(WorkerStageError) as caught:
            provider.create("nahaber")
        self.assertEqual(caught.exception.code, "OFFER_NOT_SELECTED")

    def test_a_second_create_is_refused(self):
        lifecycle = _Lifecycle()
        provider = _provider(lifecycle=lifecycle)
        provider.quote("nahaber")
        provider.create("nahaber")
        with self.assertRaises(WorkerStageError) as caught:
            provider.create("nahaber")
        self.assertEqual(caught.exception.code, "INSTANCE_LIMIT")
        self.assertEqual(len(lifecycle.created), 1, "a second machine was requested")

    def test_the_create_carries_the_pinned_image_and_no_volume(self):
        """The image engine's warm volume must not be mounted by this path."""
        lifecycle = _Lifecycle()
        provider = _provider(lifecycle=lifecycle)
        provider.quote("nahaber")
        provider.create("nahaber")
        sent = lifecycle.created[0]
        self.assertEqual(sent["image"], IMAGE)
        self.assertTrue(sent["provision"])
        self.assertEqual(sent["human_approval"], "approve:123")
        self.assertGreaterEqual(sent["disk_gb"], 60)
        body = json.dumps(sent, default=str)
        self.assertNotIn("volume", body.lower())
        self.assertNotIn("54653022", body)


class TerminateTests(unittest.TestCase):
    def test_terminate_without_an_owner_is_refused(self):
        provider = _provider()
        with self.assertRaises(WorkerStageError) as caught:
            provider.terminate("777")
        self.assertEqual(caught.exception.code, "OWNER_REQUIRED")

    def test_the_bound_terminator_proves_ownership(self):
        lifecycle = _Lifecycle()
        provider = _provider(lifecycle=lifecycle)
        bound = owner_terminator(provider, "nahaber")
        bound.quote("nahaber")
        bound.create("nahaber")
        bound.terminate("777")
        self.assertEqual(lifecycle.terminated[0]["owner"], "nahaber")
        self.assertEqual(lifecycle.terminated[0]["approval"], "approve-terminate:777")
        self.assertEqual(bound.create_calls, 1)
        self.assertEqual(bound.terminate_calls, 1)

    def test_the_rehearsal_still_refuses_this_provider(self):
        """The unpaid path must keep refusing a real Vast provider."""
        from media_engine.models.speech_execution import run_speech_rehearsal
        from media_engine.models.speech_stage import SpeechPrepError

        with self.assertRaises(SpeechPrepError) as caught:
            run_speech_rehearsal(
                None, now=0, code_sha="", recorded_digest=None, provider=_provider(),
                platform=None, user_id="nahaber", whisper_runtime=None, tts_runtime=None,
                claim_path=Path("/tmp/none"), machine_id=1, offer_id=1, instance_id="1",
                clock=None,
            )
        self.assertEqual(caught.exception.code, "GPU_RENTAL_REFUSED")


if __name__ == "__main__":
    unittest.main()
