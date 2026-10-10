"""Read-only report for volume 54653022. This module never attaches or changes it."""
from __future__ import annotations

VOLUME_ID = 54653022
VOLUME_MACHINE_ID = 47281
VOLUME_LABEL = "ai_media_engine_qwen_cache"


def audit_volume(row: dict | None) -> dict:
    """Contents stay unknown unless a file listing is already present. Do not mount the volume."""
    if not isinstance(row, dict):
        return {
            "id": VOLUME_ID,
            "observed": False,
            "contents": "unknown",
            "compatible_with_arbitrary_machines": False,
            "placement": _placement(),
        }
    identity = int(row.get("id") or 0)
    machine = int(row.get("machine_id") or 0)
    files = row.get("files")
    contents = "unknown"
    if isinstance(files, list):
        contents = "listed"
    return {
        "id": identity,
        "observed": identity == VOLUME_ID,
        "machine_id": machine,
        "expected_machine_id": VOLUME_MACHINE_ID,
        "on_expected_machine": machine == VOLUME_MACHINE_ID,
        "status": row.get("status"),
        "label": row.get("label"),
        "contents": contents,
        "compatible_with_arbitrary_machines": False,
        "placement": _placement(),
    }


def _placement() -> str:
    return (
        "Vast volumes attach only to the machine that created them. "
        "Volume 54653022 stays on machine 47281 and is not a cache for other GPUs. "
        "Instance disk disappears when the instance is deleted. "
        "The reusable cache is a revision-addressed directory outside the instance, "
        "filled from the private R2 bucket ai-media-engine-models only on a miss."
    )
