"""Tell the Qwen worker whether the persistent revision cache is already valid."""
from __future__ import annotations

import os

from media_engine.infra.cache import PINNED_REVISION, RevisionCache


def persistent_cache_hit(root: str | None = None, revision: str = PINNED_REVISION) -> bool:
    """A hit skips another download. Instance disk is not this path unless configured."""
    path = root if root is not None else os.environ.get("MEDIA_ENGINE_MODEL_CACHE", "")
    if not str(path or "").strip():
        return False
    return RevisionCache(path).lookup(revision) == "HIT"
