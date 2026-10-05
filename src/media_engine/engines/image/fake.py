"""Deterministic fixture writer. This is not an image model."""
from __future__ import annotations

import hashlib

from media_engine.engines.image.base import ImageEngine, ImageEngineError


class FakeImageEngine(ImageEngine):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def render(self, prompt: str) -> bytes:
        self.calls += 1
        if self.fail:
            raise ImageEngineError("GENERATION_FAILED")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"FAKEIMG1\n{digest}\n".encode("ascii")
