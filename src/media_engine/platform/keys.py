"""Object keys for private user media. The original filename is not the identity."""
from __future__ import annotations

import os
import re

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")
_EXT = re.compile(r"^[a-z0-9]{1,8}$")
_KINDS = frozenset({"images", "voice", "music", "audio", "video"})

MEDIA_BUCKET = "ai-media-engine-media"


def media_bucket() -> str:
    """Configured media bucket. This does not create the bucket."""
    return os.environ.get("MEDIA_ENGINE_MEDIA_BUCKET", MEDIA_BUCKET).strip() or MEDIA_BUCKET


def storage_limit(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(name)
    return value


def object_key(
    *,
    user_id: str,
    project_id: str,
    artifact_id: str,
    ext: str,
    scene_id: str = "",
    kind: str = "images",
) -> str:
    for value in (user_id, project_id, artifact_id):
        if _ID.fullmatch(value) is None:
            raise ValueError("storage id")
    if _EXT.fullmatch(ext) is None:
        raise ValueError("storage ext")
    if kind == "exports":
        return f"users/{user_id}/projects/{project_id}/exports/{artifact_id}.{ext}"
    if kind not in _KINDS or _ID.fullmatch(scene_id) is None:
        raise ValueError("storage kind")
    return (
        f"users/{user_id}/projects/{project_id}/scenes/{scene_id}/"
        f"{kind}/{artifact_id}.{ext}"
    )
