"""Capability adapters. Unverified models refuse before any weight download."""
from __future__ import annotations

from media_engine.engines.image.qwen import QwenImageEngine
from media_engine.models.registry import ModelSpec, get_model


class AdapterError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RefusingAdapter:
    def __init__(self, spec: ModelSpec) -> None:
        self.model_id = spec.model_id

    def run(self, payload: dict) -> bytes:
        raise AdapterError("CAPABILITY_UNAVAILABLE")


class QwenImageAdapter:
    """The live image path stays on the controller. This adapter is the offline contract."""

    model_id = "Qwen/Qwen-Image"

    def run(self, payload: dict, pipeline=None) -> bytes:
        if pipeline is None:
            raise AdapterError("PIPELINE_NOT_BOUND")
        engine = QwenImageEngine(pipeline)
        return engine.render(
            str(payload.get("prompt") or ""),
            width=int(payload.get("width", 1024)),
            height=int(payload.get("height", 1024)),
            seed=int(payload.get("seed", 0)),
        )


def adapter_for(model_id: str):
    spec = get_model(model_id)
    if spec.model_id == QwenImageAdapter.model_id and spec.executable:
        return QwenImageAdapter()
    return RefusingAdapter(spec)
