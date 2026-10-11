"""Capability adapters. Unverified GPU paths stay closed. Injected runtimes execute."""
from __future__ import annotations

from media_engine.engines.image.qwen import QwenImageEngine
from media_engine.models.errors import AdapterError
from media_engine.models.registry import ModelSpec, get_model

__all__ = ["AdapterError", "QwenImageAdapter", "RefusingAdapter", "adapter_for"]


class RefusingAdapter:
    def __init__(self, spec: ModelSpec) -> None:
        self.model_id = spec.model_id

    def run(self, payload: dict, runtime=None) -> bytes:
        raise AdapterError("CAPABILITY_UNAVAILABLE")


class QwenImageAdapter:
    """The live image path stays on the controller. This adapter is the offline contract."""

    model_id = "Qwen/Qwen-Image"

    def run(self, payload: dict, pipeline=None, runtime=None) -> bytes:
        pipeline = pipeline or runtime
        if pipeline is None:
            raise AdapterError("PIPELINE_NOT_BOUND")
        engine = QwenImageEngine(pipeline)
        return engine.render(
            str(payload.get("prompt") or ""),
            width=int(payload.get("width", 1024)),
            height=int(payload.get("height", 1024)),
            seed=int(payload.get("seed", 0)),
        )


class BoundAdapter:
    def __init__(self, spec: ModelSpec, kind: str) -> None:
        self.model_id = spec.model_id
        self.kind = kind

    def run(self, payload: dict, runtime=None):
        if runtime is None:
            raise AdapterError("CAPABILITY_UNAVAILABLE")
        from media_engine.models.invoke import Invocation
        import time
        ctx = Invocation(time.monotonic() + 30, None)
        from media_engine.models import invoke
        call = {
            "audio.transcribe": lambda: invoke.invoke_transcribe(runtime, payload["wav"], ctx),
            "audio.tts": lambda: invoke.invoke_tts(runtime, payload, ctx),
            "audio.translate": lambda: invoke.invoke_translation(runtime, payload, ctx),
            "music.generate": lambda: invoke.invoke_music(runtime, payload, ctx),
            "video.text_to_video": lambda: invoke.invoke_video(runtime, payload, ctx),
        }[self.kind]
        return call()


_KINDS = {
    "Systran/faster-whisper-large-v3": "audio.transcribe",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "audio.tts",
    "Qwen/Qwen2.5-1.5B-Instruct": "audio.translate",
    "ACE-Step/ACE-Step-v1-3.5B": "music.generate",
    "Wan-AI/Wan2.1-T2V-1.3B": "video.text_to_video",
    "Qwen/Qwen-Image-Edit": "image.edit",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "audio.voice_clone",
}


def adapter_for(model_id: str):
    spec = get_model(model_id)
    if spec.commercial_use != "allowed" or spec.role != "primary":
        return RefusingAdapter(spec)
    if spec.model_id == QwenImageAdapter.model_id:
        return QwenImageAdapter()
    if spec.model_id in _KINDS:
        return BoundAdapter(spec, _KINDS[spec.model_id])
    return RefusingAdapter(spec)
