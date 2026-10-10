"""Three-image session plan. Nothing here issues an authorization or rents a GPU."""
from __future__ import annotations

from decimal import Decimal, ROUND_UP

from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT
from media_engine.engines.image.qwen_remote import PINNED_QWEN_REVISION
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.vast_image import MACHINE_ID, VOLUME_ID
from media_engine.safety.execution_budget import (
    LIVE7_INFERENCE_SECONDS,
    LIVE7_LOAD_SECONDS,
    LIVE7_PROVISION_SECONDS,
    LIVE7_RESERVE_SECONDS,
    LIVE7_UPLOAD_SECONDS,
)

# The instance that produced the first real image. It is not the volume's machine.
SUCCESSFUL_ADA_MACHINE_ID = 137576
VOLUME_DISK_GB = 80
STAGING_DISK_GB = 140
# Last observed all-in hourly on that Ada rental, including the 140 GB disk.
OBSERVED_HOURLY_USD = Decimal("0.6370370370370371")
OBSERVED_GPU_HOURLY_USD = Decimal("0.5333333333333333")
# A previous full copy took 527 seconds. The planner's link estimate is lower.
MEASURED_TRANSFER_SECONDS = 527
IDLE_SHUTDOWN_SECONDS = 300


def cache_requirements() -> dict:
    return {
        "revision": PINNED_QWEN_REVISION,
        "files": QWEN_FILE_COUNT,
        "bytes": QWEN_CACHE_MIN_BYTES,
        "staging_disk_gb": STAGING_DISK_GB,
        "volume_disk_gb": VOLUME_DISK_GB,
        "volume_id": VOLUME_ID,
        "volume_machine_id": MACHINE_ID,
        "successful_machine_id": SUCCESSFUL_ADA_MACHINE_ID,
        "revision_matches_contract": PINNED_QWEN_REVISION == REVISION,
    }


def cache_strategy() -> dict:
    """Cheapest way to avoid a second download inside one three-image session.

    After the instance is deleted, the model is gone. Volume 54653022 cannot
    replace that disk: it is bound to machine 47281, the successful Ada run
    used machine 137576, and 80 GB is below the 140 GB staging disk.
    """
    facts = cache_requirements()
    volume_same_machine = facts["volume_machine_id"] == facts["successful_machine_id"]
    volume_large_enough = facts["volume_disk_gb"] >= facts["staging_disk_gb"]
    return {
        "choice": "r2_once_then_retained_instance",
        "download_after_deletion": True,
        "loads_per_session": 1,
        "downloads_per_session": 1,
        "vast_volume": "incompatible",
        "volume_same_machine": volume_same_machine,
        "volume_large_enough": volume_large_enough,
        "machine_local_disk_survives_deletion": False,
        "prebuilt_image": "not_in_pinned_worker_image",
        "paid_storage_required_now": False,
    }


def warm_session_cost(
    *,
    hourly: Decimal = OBSERVED_HOURLY_USD,
    gpu_hourly: Decimal = OBSERVED_GPU_HOURLY_USD,
    images: int = 3,
    transfer_seconds: int = MEASURED_TRANSFER_SECONDS,
    idle_seconds: int = IDLE_SHUTDOWN_SECONDS,
) -> dict:
    """One provision, one copy, one load, N inferences, idle, then the deletion reserve."""
    if images < 1:
        raise ValueError("images")
    rate = Decimal(hourly)
    compute = Decimal(gpu_hourly)
    disk_hourly = rate - compute
    seconds = (
        LIVE7_PROVISION_SECONDS
        + int(transfer_seconds)
        + LIVE7_LOAD_SECONDS
        + images * (LIVE7_INFERENCE_SECONDS + LIVE7_UPLOAD_SECONDS)
        + int(idle_seconds)
        + LIVE7_RESERVE_SECONDS
    )
    total = (rate * Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.000001"), rounding=ROUND_UP)
    disk = (disk_hourly * Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.000001"), rounding=ROUND_UP)
    return {
        "images": images,
        "seconds": seconds,
        "hourly_usd": str(rate),
        "disk_hourly_usd": str(disk_hourly),
        "disk_usd": str(disk),
        "provision_seconds": LIVE7_PROVISION_SECONDS,
        "transfer_seconds": int(transfer_seconds),
        "load_seconds": LIVE7_LOAD_SECONDS,
        "inference_seconds": images * LIVE7_INFERENCE_SECONDS,
        "upload_seconds": images * LIVE7_UPLOAD_SECONDS,
        "idle_seconds": int(idle_seconds),
        "deletion_reserve_seconds": LIVE7_RESERVE_SECONDS,
        "total_usd": str(total),
        "paid_actions_enabled": False,
    }


def authorization_template() -> dict:
    """A blank for a future test. Calling this does not issue or consume one."""
    cost = warm_session_cost()
    return {
        "issued": False,
        "consumed": False,
        "paid_actions_enabled": False,
        "allows_paid_create": False,
        "execution_id": "",
        "authorization_id": "",
        "rejected_execution_ids": ["live8-run-1011a", "live8-auth-1011a"],
        "profile": "live-session",
        "images": 3,
        "sequential": True,
        "model_loads": 1,
        "max_creates": 1,
        "model": "Qwen/Qwen-Image",
        "revision": PINNED_QWEN_REVISION,
        "width": 1024,
        "height": 1024,
        "hourly_ceiling_usd": "0.95",
        "session_target_usd": "5.00",
        "estimated_usd": cost["total_usd"],
        "disk_gb": STAGING_DISK_GB,
        "deletion_reserve_seconds": LIVE7_RESERVE_SECONDS,
        "idle_shutdown_seconds": IDLE_SHUTDOWN_SECONDS,
        "forbidden_volume_id": VOLUME_ID,
        "volume_machine_id": MACHINE_ID,
        "cache": cache_strategy()["choice"],
        "download_after_deletion": True,
        "watchdog": "independent",
        "verified_deletion": True,
    }
