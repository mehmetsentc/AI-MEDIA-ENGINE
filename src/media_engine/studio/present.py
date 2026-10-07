"""User-facing job language. Provider internals stay out of the studio."""
from __future__ import annotations

from typing import Optional

CAPACITY = "PROVIDER_CAPACITY_UNAVAILABLE"

_PHASES = {
    "queued": ("Preparing", "progress"),
    "planning": ("Preparing", "progress"),
    "waiting_capacity": ("Waiting for available GPU capacity", "waiting"),
    "provisioning": ("Starting AI worker", "progress"),
    "booting": ("Starting AI worker", "progress"),
    "runtime_preparing": ("Starting AI worker", "progress"),
    "model_loading": ("Loading image model", "progress"),
    "generating": ("Generating image", "progress"),
    "saving": ("Saving result", "progress"),
    "completed": ("Complete", "done"),
    "failed": ("Could not create the image", "error"),
    "draft": ("Ready", "idle"),
}


def present_job(job: Optional[dict]) -> dict:
    """Map an image job onto a studio phase. Capacity wait is not a failure."""
    if not job:
        title, tone = _PHASES["draft"]
        return {"phase": "draft", "title": title, "tone": tone, "progress": 0}
    status = str(job.get("status") or "")
    code = job.get("error_code")
    if code == CAPACITY or status == "waiting_capacity":
        status = "waiting_capacity"
    title, tone = _PHASES.get(status, ("Creating image", "progress"))
    if status == "failed" and code == "CANCELLED":
        title, tone = "Cancelled", "idle"
    return {
        "phase": status if status in _PHASES else "queued",
        "title": title,
        "tone": tone,
        "progress": job.get("progress") or 0,
    }
