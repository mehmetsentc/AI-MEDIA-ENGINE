"""Worker-layout cache decision. A control-plane directory is not the GPU."""
from __future__ import annotations

from pathlib import Path

from media_engine.engines.image.qwen_remote import cache_is_ready


def persistent_cache_hit(root: str | None = None, *, min_bytes: int | None = None) -> bool:
    """Assess the Qwen worker layout. This does not skip a remote check by itself."""
    if not str(root or "").strip():
        return False
    return cache_is_ready(Path(root), min_bytes=min_bytes)
