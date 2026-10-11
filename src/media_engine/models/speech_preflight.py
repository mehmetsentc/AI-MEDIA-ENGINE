"""Read-only speech GPU preflight. It does not rent, download, or write storage."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT, REPO_ID
from media_engine.infra.volume_audit import VOLUME_ID
from media_engine.models.dependencies import dependency_status
from media_engine.models.lifecycle import SCRATCH_BYTES
from media_engine.models.speech_stage import (
    file_bytes,
    remote_gap,
    staged_disk_bytes,
    targets,
    verify_tree,
)
from media_engine.providers.execution_authorization import _FIELDS
from media_engine.providers.live7_contract import (
    FORBIDDEN_VOLUME_ID,
    LIVE7_CUDA_MIN,
    LIVE7_DISK_GB,
    LIVE7_GPU_NAME,
    REVISION,
)
from media_engine.providers.session_heartbeat import HEARTBEAT_INTERVAL_SECONDS
from media_engine.providers.vast_watchdog import HEARTBEAT_STALE_SECONDS
from media_engine.safety.execution_budget import (
    LIVE7_CEILING_USD,
    LIVE7_MAX_HOURLY_USD,
    LIVE7_REFERENCE_HOURLY_USD,
    LIVE7_RESERVE_SECONDS,
    termination_reserve_usd,
)

IMAGE_WORKER_BASE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"
IMAGE_PINS = ("diffusers==0.41.0", "transformers==5.19.0", "accelerate==1.15.0", "safetensors==0.8.0")
SPEECH_PINS = (
    "faster-whisper==1.2.1",
    "ctranslate2==4.8.2",
    "qwen-tts==0.1.1",
    "transformers==4.57.3",
    "accelerate==1.12.0",
    "torchaudio==2.11.0",
    "safetensors==0.8.0",
)
# qwen-tts 0.1.1 requires these exact versions. They conflict with the image worker.
SPEECH_CONFLICTS = ("transformers==4.57.3", "accelerate==1.12.0")


def speech_preflight(
    *,
    remote_entries: dict[str, list[tuple[str, int]] | None] | None = None,
    local_roots: dict[str, Path] | None = None,
    free_disk_bytes: int | None = None,
    volume_id: int | None = None,
    image_manifest: dict | None = None,
    dockerfile_text: str | None = None,
    measured_transfer_seconds: float | None = None,
    measured_inference_seconds: dict | None = None,
) -> dict:
    """Assemble the report. None means that input was not observed."""
    models = []
    for spec in targets():
        remote = None if remote_entries is None else remote_entries.get(spec["model_id"])
        gap = remote_gap(spec["model_id"], remote)
        local = None
        if local_roots and spec["model_id"] in local_roots:
            local = verify_tree(local_roots[spec["model_id"]], spec["model_id"])
        models.append({
            "model_id": spec["model_id"],
            "revision": spec["revision"],
            "capability": spec["capability"],
            "hub_used_storage": spec["hub_used_storage"],
            "transfer_bytes": file_bytes(spec["model_id"]),
            "r2": gap,
            "local": local,
            "weight_checksums": "git-lfs-oid-not-recomputed",
            "gpu_verified": False,
        })
    return {
        "models": models,
        "disk": _disk(free_disk_bytes),
        "volume": _volume(volume_id),
        "image_cache": assess_image_manifest(image_manifest),
        "image_worker": image_worker_compatible(dockerfile_text if dockerfile_text is not None else _image_dockerfile()),
        "dependencies": dependency_report(),
        "gpu": gpu_requirements(),
        "timing": {
            "transfer_seconds": measured_transfer_seconds,
            "transfer_time": "measured" if measured_transfer_seconds is not None else "unmeasured",
            "inference_seconds": measured_inference_seconds or {},
            "inference_time": "unmeasured" if not measured_inference_seconds else "measured",
        },
        "cost": cost_estimate(measured_transfer_seconds, measured_inference_seconds),
        "authorization": authorization_requirements(),
        "watchdog": {
            "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
            "heartbeat_stale_seconds": HEARTBEAT_STALE_SECONDS,
            "spending_deadline": "set at create and not extended by a fresh heartbeat",
            "deletion_reserve_seconds": LIVE7_RESERVE_SECONDS,
        },
    }


def assess_image_manifest(payload: dict | None) -> dict:
    if payload is None:
        return {"status": "NOT_OBSERVED"}
    files = payload.get("files")
    total = payload.get("expected_total_bytes")
    complete = (
        payload.get("state") == "COMPLETE"
        and payload.get("repo_id") == REPO_ID
        and payload.get("revision") == REVISION
        and payload.get("expected_file_count") == QWEN_FILE_COUNT
        and total == QWEN_CACHE_MIN_BYTES
        and isinstance(files, list)
        and len(files) == QWEN_FILE_COUNT
    )
    return {
        "status": "COMPLETE" if complete else "INVALID",
        "repo_id": payload.get("repo_id"),
        "revision": payload.get("revision"),
        "expected_file_count": payload.get("expected_file_count"),
        "expected_total_bytes": total,
    }


def image_worker_compatible(text: str) -> dict:
    problems = []
    if IMAGE_WORKER_BASE not in text:
        problems.append("BASE_IMAGE_CHANGED")
    for pin in IMAGE_PINS:
        if pin not in text:
            problems.append("IMAGE_PINS_CHANGED")
            break
    for token in ("faster-whisper", "qwen-tts", "qwen_tts"):
        if token in text:
            problems.append("SPEECH_PACKAGE_IN_IMAGE_WORKER")
            break
    return {"compatible": not problems, "problems": problems, "base": IMAGE_WORKER_BASE}


def dependency_report() -> dict:
    return {
        "speech_pins": list(SPEECH_PINS),
        "image_worker_conflicts": list(SPEECH_CONFLICTS),
        "worker_image_build": "NOT_RUN",
        "ctranslate2_cuda_wheel": "UNVERIFIED",
        "ctranslate2_note": (
            "PyPI ctranslate2 4.8.2 publishes a manylinux wheel whose filename does not name CUDA. "
            "A GPU session must observe ctranslate2 CUDA support before accepting a transcript."
        ),
        "qwen_tts_note": (
            "qwen-tts 0.1.1 requires transformers 4.57.3 and accelerate 1.12.0. "
            "The image worker pins transformers 5.19.0 and accelerate 1.15.0, so the speech image is separate."
        ),
        "local_interpreter": dependency_status(),
    }


def gpu_requirements() -> dict:
    return {
        "family": "6000ada",
        "name": LIVE7_GPU_NAME,
        "cuda_min": str(LIVE7_CUDA_MIN),
        "one_resident_model": True,
        "whisper_documented_fp16_mib": 4525,
        "whisper_vram_remeasured": False,
        "tts_vram_mib": None,
        "tts_fit": "unmeasured",
        "candidate_reason": "The image engine already ran on an RTX 6000 Ada. Speech VRAM is not measured.",
    }


def cost_estimate(transfer_seconds: float | None, inference_seconds: dict | None) -> dict:
    reserve = termination_reserve_usd(LIVE7_REFERENCE_HOURLY_USD)
    measured = transfer_seconds is not None and bool(inference_seconds)
    return {
        "hourly_usd_reference": format(LIVE7_REFERENCE_HOURLY_USD, "f"),
        "hourly_usd_max": format(LIVE7_MAX_HOURLY_USD, "f"),
        "ceiling_usd": format(LIVE7_CEILING_USD, "f"),
        "deletion_reserve_seconds": LIVE7_RESERVE_SECONDS,
        "deletion_reserve_usd_at_reference": format(reserve, "f"),
        "session_usd": None,
        "session_usd_reason": "TRANSFER_AND_INFERENCE_UNMEASURED" if not measured else "MEASURED_INPUT_PRESENT",
        "assumptions": [
            "The hourly reference is the recorded LIVE-7 image rate, not a new Vast quote.",
            "No bytes-per-second measurement exists for these two models.",
            "A session dollar total is omitted instead of being divided from an assumed rate.",
            "R2 storage dollars were not quoted. The report gives bytes only.",
        ],
    }


def authorization_requirements() -> dict:
    return {
        "issuer_accepts_speech_models": False,
        "refusal": "MODEL_SUBSTITUTION",
        "accepted_model": "Qwen/Qwen-Image",
        "accepted_revision": REVISION,
        "required_fields": list(_FIELDS),
        "constraints": {
            "max_instances": 1,
            "max_creates": 1,
            "max_gpu_instances": 1,
            "max_model_stages": 1,
            "max_model_loads": 1,
            "max_hourly_usd_at_most": format(LIVE7_MAX_HOURLY_USD, "f"),
            "max_total_usd_at_most": format(LIVE7_CEILING_USD, "f"),
            "max_disk_gb": LIVE7_DISK_GB,
            "forbidden_volume_id": FORBIDDEN_VOLUME_ID,
            "one_use": True,
            "code_sha": "the commit that will actually run",
        },
        "missing": "A speech authorization must name these revisions without weakening the Qwen issuer.",
    }


def _disk(free_disk_bytes: int | None) -> dict:
    needed = staged_disk_bytes()
    report = {
        "scratch_bytes": SCRATCH_BYTES,
        "staged_bytes": needed - SCRATCH_BYTES,
        "required_bytes": needed,
        "contract_disk_gb": LIVE7_DISK_GB,
        "free_disk_bytes": free_disk_bytes,
    }
    if free_disk_bytes is None:
        report["status"] = "NOT_OBSERVED"
    elif int(free_disk_bytes) < needed:
        report["status"] = "DISK_CAPACITY"
    else:
        report["status"] = "ENOUGH"
    return report


def _volume(volume_id: int | None) -> dict:
    if volume_id is None:
        return {"id": VOLUME_ID, "attached_in_this_report": False, "change": "forbidden"}
    return {
        "id": int(volume_id),
        "forbidden": int(volume_id) == VOLUME_ID,
        "change": "forbidden",
    }


def _image_dockerfile() -> str:
    path = Path(__file__).resolve().parents[3] / "deploy" / "worker" / "Dockerfile"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def reference_hourly() -> Decimal:
    return LIVE7_REFERENCE_HOURLY_USD
