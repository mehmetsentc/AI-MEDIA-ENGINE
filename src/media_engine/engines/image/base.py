"""Image generation contract. Implementations return bytes; storage writes them."""
from __future__ import annotations

import abc


class ImageEngineError(Exception):
    code = "GENERATION_FAILED"


class ImageEngine(abc.ABC):
    @abc.abstractmethod
    def render(self, prompt: str) -> bytes:
        raise NotImplementedError
