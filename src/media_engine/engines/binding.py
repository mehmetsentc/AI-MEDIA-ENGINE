"""Local engine ids. The HTTP contract names a job type, not a vendor."""
from __future__ import annotations

from media_engine.jobs.model import IMAGE_GENERATE, TEXT_GENERATE, UnsupportedJobType

LOCAL_ENGINE_IDS = {
    IMAGE_GENERATE: "fake-image",
    TEXT_GENERATE: "fake-text",
}


def engine_id_for(job_type: str) -> str:
    try:
        return LOCAL_ENGINE_IDS[job_type]
    except KeyError as exc:
        raise UnsupportedJobType(job_type) from exc
