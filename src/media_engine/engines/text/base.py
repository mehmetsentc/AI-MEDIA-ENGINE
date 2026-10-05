"""Text generation contract. Implementations return text, not a model call."""
from __future__ import annotations

import abc


class TextEngineError(Exception):
    code = "GENERATION_FAILED"


class TextEngine(abc.ABC):
    @abc.abstractmethod
    def render(self, *, task: str, prompt: str) -> str:
        raise NotImplementedError
