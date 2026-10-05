"""Deterministic text fixture. This is not a language model."""
from __future__ import annotations

import hashlib

from media_engine.engines.text.base import TextEngine, TextEngineError


class FakeTextEngine(TextEngine):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def render(self, *, task: str, prompt: str) -> str:
        self.calls += 1
        if self.fail:
            raise TextEngineError("GENERATION_FAILED")
        digest = hashlib.sha256(f"{task}\n{prompt}".encode("utf-8")).hexdigest()
        return f"FAKETEXT1\n{task}\n{digest}\n"
