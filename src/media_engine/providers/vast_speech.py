"""A Vast provider for the media capabilities, built on the generic lifecycle.

``VastImageProvider`` cannot serve this path. It is nailed to the image
engine: it pulls that engine's runtime image, and it attaches volume
``VOLUME_ID`` - the warm Qwen-Image cache that ``speech_stage._refuse_volume``
exists to keep this path away from. Using it for a speech run would mount a
volume this workload has no business touching and boot the wrong image.

So this adapts the pieces that were already general and unused:

* ``VastDiscovery`` - read-only offer search, no mutation
* ``VastLifecycle`` - create/observe/terminate for an arbitrary image, with no
  volume, a price ceiling, ``provision=True`` and a per-action approval token

What it adds is the ``GPUProvider`` shape the execution path expects, the
pinned speech image, and the counters that path asserts on.

The class is called ``vast`` on purpose. ``run_speech_rehearsal`` refuses a
provider by that name, and it should keep refusing: the rehearsal is the
unpaid path. The paid path has its own gate in ``remote_execution``.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Optional

from media_engine.providers.base import (
    OBSERVE_GONE,
    OBSERVE_UNKNOWN,
    CreateResult,
    GPUProvider,
    Quote,
    WorkerStageError,
)
from media_engine.providers.planner import ResourceRequirements, plan
from media_engine.providers.vast_lifecycle import (
    ProvisionBlocked,
    VastLifecycle,
    approval_token,
)

#: One model is resident at a time. The largest documented peak among the
#: pinned speech models is faster-whisper's 4525 MiB; the TTS checkpoints are
#: about 4.3GB of bf16 weights. 16GB leaves room for activations without
#: paying for a card the workload cannot use.
MIN_VRAM_GB = Decimal("16")

#: The five staged models are 20.9GB, plus the image and scratch.
DISK_GB = 60


class VastSpeechProvider(GPUProvider):
    name = "vast"

    def __init__(
        self,
        *,
        api_key: str,
        image: str,
        lifecycle: VastLifecycle,
        discovery,
        max_hourly_usd: Decimal,
        disk_gb: int = DISK_GB,
        min_vram_gb: Decimal = MIN_VRAM_GB,
        max_job_seconds: Optional[int] = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise WorkerStageError("PROVIDER_CREATE_FAILED")
        if not isinstance(image, str) or "@sha256:" not in image:
            # A tag can move; a digest cannot. Renting a machine to run an
            # image that changed under us is not something to discover from
            # the output.
            raise WorkerStageError("WORKLOAD_IMAGE_REQUIRED")
        self._api_key = api_key.strip()
        self.image = image
        self.lifecycle = lifecycle
        self.discovery = discovery
        self.disk_gb = int(disk_gb)
        self.requirements = ResourceRequirements(
            min_vram_gb=Decimal(min_vram_gb),
            gpu_count=1,
            max_hourly_price_usd=Decimal(max_hourly_usd),
            min_reliability=Decimal("0.95"),
            max_job_seconds=max_job_seconds,
        )
        self.offers: list = []
        self.selected = None
        self.hourly_amount = Decimal("0")
        self.create_calls = 0
        self.terminate_calls = 0
        self.last_instance_id = ""

    # --- GPUProvider ------------------------------------------------------

    def quote(self, owner: str) -> Quote:
        """Search and choose, without renting anything."""
        # read_only=True is the search's own refusal to mutate anything.
        self.offers = list(self.discovery.search(
            self.requirements.to_search(), api_key=self._api_key, read_only=True, limit=10,
        ))
        chosen = plan(self.offers, self.requirements)
        if chosen.selected is None:
            raise WorkerStageError("NO_ELIGIBLE_OFFER")
        self.selected = chosen.selected
        self.hourly_amount = Decimal(chosen.selected.hourly_price_usd)
        return Quote(hourly_price_usd=self.hourly_amount, provider=self.name)

    def create(self, owner: str) -> CreateResult:
        if self.selected is None:
            # Renting before pricing is how a ceiling gets skipped.
            raise WorkerStageError("OFFER_NOT_SELECTED")
        if self.create_calls:
            # The authorization allows one. A second attempt is a second bill.
            raise WorkerStageError("INSTANCE_LIMIT")
        self.create_calls += 1
        offer_id = str(self.selected.offer_id)
        try:
            result = self.lifecycle.create(
                offers=self.offers,
                requirements=self.requirements,
                offer_id=offer_id,
                owner=owner,
                api_key=self._api_key,
                provision=True,
                human_approval=approval_token(offer_id),
                image=self.image,
                disk_gb=self.disk_gb,
            )
        except ProvisionBlocked as exc:
            raise WorkerStageError(exc.code) from exc
        if result.outcome == "created" and result.resource_id:
            self.last_instance_id = str(result.resource_id)
        return result

    def observe(self, resource_id: str, owner: str) -> str:
        try:
            return self.lifecycle.observe(
                resource_id, owner, api_key=self._api_key, provision=True,
            )
        except ProvisionBlocked:
            return OBSERVE_UNKNOWN

    def terminate(self, resource_id: str) -> None:
        raise WorkerStageError("OWNER_REQUIRED")

    def terminate_owned(self, resource_id: str, owner: str) -> None:
        """Vast needs the owner to prove the instance is ours before deleting."""
        self.terminate_calls += 1
        self.lifecycle.terminate(
            resource_id, owner, api_key=self._api_key, provision=True,
            human_approval="approve-terminate:" + str(resource_id),
        )

    def status(self, resource_id: str) -> str:
        return OBSERVE_UNKNOWN

    def list_owned(self) -> list[dict]:
        return []


def speech_image_reference(ref_path: Optional[Path] = None) -> str:
    """The pinned, digest-addressed speech worker image."""
    from media_engine.models.speech_image import (
        assess_speech_image_ref,
        load_image_ref,
        speech_image_ref_path,
    )

    document = load_image_ref(Path(ref_path) if ref_path else speech_image_ref_path())
    assessed = assess_speech_image_ref(document)
    reference = str(assessed.get("pinned_reference") or "")
    if "@sha256:" not in reference:
        raise WorkerStageError("WORKLOAD_IMAGE_REQUIRED")
    return reference


def owner_terminator(provider: VastSpeechProvider, owner: str):
    """Bind the owner so run_media_jobs can call provider.terminate(id).

    The execution path deletes with the resource id alone, while Vast insists
    the caller prove ownership first. Binding the owner here keeps that proof
    without teaching the execution path about one vendor.
    """

    class _Bound:
        name = provider.name

        def __init__(self) -> None:
            self.create_calls = 0
            self.terminate_calls = 0

        def quote(self, who: str) -> Quote:
            return provider.quote(who)

        def create(self, who: str) -> CreateResult:
            result = provider.create(who)
            self.create_calls = provider.create_calls
            return result

        def terminate(self, resource_id: str) -> None:
            provider.terminate_owned(resource_id, owner)
            self.terminate_calls = provider.terminate_calls

        def observe(self, resource_id: str, who: str) -> str:
            return provider.observe(resource_id, who)

    return _Bound()
