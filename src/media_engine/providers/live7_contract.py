"""LIVE-7 machine, image, and persistence contract. This module never rents a GPU."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.qwen import FIRST_HEIGHT, FIRST_PROMPT, FIRST_SEED, FIRST_WIDTH, MODEL_ID
from media_engine.providers.base import WorkerStageError
from media_engine.safety.execution_budget import (
    LIVE7_CEILING_USD,
    LIVE7_MAX_HOURLY_USD,
    LIVE7_REFERENCE_HOURLY_USD,
    live7_billable_seconds,
    live7_transfer_room,
)
from media_engine.safety.spend_grant import SpendGrant

LIVE7_MACHINE_ID = 142658
LIVE7_GPU_FAMILY = "5880ada"
LIVE7_GPU_NAME = "RTX 5880 Ada"
LIVE7_MIN_VRAM_MIB = 49140
LIVE7_MAX_VRAM_MIB = 52000
LIVE7_DISK_GB = 140
LIVE7_CUDA_MIN = Decimal("12.8")
LIVE7_COMPUTE_CAP = 890
FORBIDDEN_VOLUME_ID = 54653022
PINNED_DIGEST = "sha256:be2c96a09947b15ff1fd8fb1d2aaa4692dcc9c06083ef1486674ac4fbc4a907c"
REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"


def gpu_family_token(name: str) -> str:
    compact = str(name).casefold().replace(" ", "").replace("_", "").replace("-", "")
    if "5880ada" in compact:
        return "5880ada"
    return compact


def make_live7_grant(*, execution_id: str, issued_at: float, expires_at: float) -> SpendGrant:
    """Build the only grant shape this profile accepts. The machine is not taken from the environment."""
    if not execution_id or expires_at <= issued_at:
        raise WorkerStageError("EXPIRED_AUTHORIZATION")
    seconds = live7_billable_seconds(LIVE7_MAX_HOURLY_USD)
    transfer = 0 if seconds is None else live7_transfer_room(seconds)
    if seconds is None or transfer < 2400:
        raise WorkerStageError("BUDGET_MISMATCH")
    return SpendGrant(
        execution_id=execution_id,
        ceiling_usd=LIVE7_CEILING_USD,
        max_instances=1,
        issued_at=float(issued_at),
        expires_at=float(expires_at),
        machine_id=LIVE7_MACHINE_ID,
        profile="live7",
        gpu_family=LIVE7_GPU_FAMILY,
        min_vram_mib=LIVE7_MIN_VRAM_MIB,
        max_hourly_usd=LIVE7_MAX_HOURLY_USD,
        disk_gb=LIVE7_DISK_GB,
        cold_stage=True,
        max_transfer_seconds=transfer,
        max_execution_seconds=seconds,
        cuda_min=str(LIVE7_CUDA_MIN),
        forbidden_volume_id=FORBIDDEN_VOLUME_ID,
    )


def accept_live7_grant(grant: SpendGrant) -> None:
    """Refuse a LIVE-6 grant, a reused shape, or any field that is not the pinned contract."""
    if not isinstance(grant, SpendGrant) or grant.profile != "live7":
        raise WorkerStageError("BUDGET_MISMATCH")
    if grant.ceiling_usd != LIVE7_CEILING_USD or grant.max_instances != 1:
        raise WorkerStageError("BUDGET_MISMATCH")
    if grant.machine_id != LIVE7_MACHINE_ID:
        raise WorkerStageError("MACHINE_SCOPE")
    if grant.gpu_family != LIVE7_GPU_FAMILY:
        raise WorkerStageError("GPU_MISMATCH")
    if grant.min_vram_mib != LIVE7_MIN_VRAM_MIB:
        raise WorkerStageError("VRAM_MISMATCH")
    if grant.max_hourly_usd != LIVE7_MAX_HOURLY_USD:
        raise WorkerStageError("PRICE_CHANGED")
    if grant.disk_gb != LIVE7_DISK_GB or grant.cold_stage is not True:
        raise WorkerStageError("BUDGET_MISMATCH")
    if grant.forbidden_volume_id != FORBIDDEN_VOLUME_ID:
        raise WorkerStageError("VOLUME_MISMATCH")
    if grant.cuda_min != str(LIVE7_CUDA_MIN):
        raise WorkerStageError("CUDA_MISMATCH")
    if grant.expires_at <= grant.issued_at:
        raise WorkerStageError("EXPIRED_AUTHORIZATION")
    expected = live7_billable_seconds(LIVE7_MAX_HOURLY_USD)
    if expected is None or grant.max_execution_seconds != expected:
        raise WorkerStageError("BUDGET_MISMATCH")
    if grant.max_transfer_seconds != live7_transfer_room(expected):
        raise WorkerStageError("BUDGET_MISMATCH")


def accept_offer(row: dict, hourly: Decimal) -> str:
    """Return an empty string when this offer may be used. Otherwise a refusal code."""
    try:
        machine = int(row.get("machine_id") or 0)
        ram = float(row.get("gpu_ram") or 0)
        free = float(row.get("disk_space") or 0)
        cuda = float(row.get("cuda_max_good") or 0)
        cap = int(row.get("compute_cap") or 0)
        gpus = int(row.get("num_gpus") or 0)
    except (TypeError, ValueError):
        return "OFFER_REJECTED"
    if machine != LIVE7_MACHINE_ID:
        return "MACHINE_SCOPE"
    if row.get("rentable") is not True or gpus != 1:
        return "OFFER_REJECTED"
    if gpu_family_token(str(row.get("gpu_name") or "")) != LIVE7_GPU_FAMILY:
        return "GPU_MISMATCH"
    if ram < LIVE7_MIN_VRAM_MIB or ram > LIVE7_MAX_VRAM_MIB:
        return "VRAM_MISMATCH"
    if cap != LIVE7_COMPUTE_CAP:
        return "COMPUTE_MISMATCH"
    if Decimal(str(cuda)) < LIVE7_CUDA_MIN:
        return "CUDA_MISMATCH"
    if free < LIVE7_DISK_GB:
        return "OFFER_REJECTED"
    if hourly <= 0 or hourly > LIVE7_MAX_HOURLY_USD:
        return "PRICE_CHANGED"
    return ""


def image_contract() -> dict:
    """The one future image. Calling this does not submit a job."""
    return {
        "submitted": False,
        "model": MODEL_ID,
        "prompt": FIRST_PROMPT,
        "width": FIRST_WIDTH,
        "height": FIRST_HEIGHT,
        "seed": FIRST_SEED,
        "count": 1,
        "revision": REVISION,
    }


def static_compatibility(root: Path | None = None) -> dict:
    """Static image and GPU checks. gpu_validated stays false without a real run."""
    base = Path(__file__).resolve().parents[3] if root is None else Path(root)
    image_path = base / "deploy" / "worker" / "image_ref.json"
    docker_path = base / "deploy" / "worker" / "Dockerfile"
    pinned = ""
    try:
        payload = json.loads(image_path.read_text(encoding="utf-8"))
        pinned = str(payload.get("pinned_reference") or "")
    except (OSError, json.JSONDecodeError):
        pinned = ""
    docker = ""
    try:
        docker = docker_path.read_text(encoding="utf-8")
    except OSError:
        docker = ""
    from media_engine.providers.vast_image import supports_bf16
    digest_ok = pinned.endswith(PINNED_DIGEST)
    return {
        "gpu_family": LIVE7_GPU_NAME,
        "compute_capability": "8.9",
        "cuda_runtime": "12.8",
        "cuda_host_minimum": str(LIVE7_CUDA_MIN),
        "bf16_code_allows_family": supports_bf16(LIVE7_GPU_NAME),
        "worker_image": pinned,
        "worker_digest_matches": digest_ok,
        "torch_pinned": "pytorch:2.11.0-cuda12.8" in docker,
        "diffusers_pinned": "diffusers==0.41.0" in docker,
        "transformers_pinned": "transformers==5.19.0" in docker,
        "gpu_validated": False,
        "reference_hourly_usd": str(LIVE7_REFERENCE_HOURLY_USD),
        "maximum_hourly_usd": str(LIVE7_MAX_HOURLY_USD),
        "unverified": [
            "No worker process loaded Qwen-Image on an RTX 5880 Ada",
            "The offer's driver and cuda_max_good must be read again before rental",
            "Peak VRAM for this model load has not been measured",
        ],
    }


def persistence_audit() -> dict:
    """What the committed studio path actually stores. Neon is not wired here."""
    boot = Path(__file__).resolve().parents[1] / "studio" / "boot.py"
    text = boot.read_text(encoding="utf-8")
    uses_sqlite = 'platform.sqlite3' in text and "postgres" not in text.casefold()
    return {
        "job_ledger": "sqlite",
        "credit_reservation": "Platform.authorize_paid",
        "credit_settlement": "Platform.settle",
        "credit_refund": "Platform.settle_failure",
        "studio_platform": "sqlite" if uses_sqlite else "unknown",
        "neon_integrated": False,
        "r2_media": "open_media_store or UnavailableMedia",
        "model_bucket": "ai-media-engine-models",
        "media_bucket": "daidoi-media",
        "e2e_blocker": "NEON_PATH_MISSING",
        "real_inference_is_full_e2e": False,
    }
