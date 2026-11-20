"""Bind the real local model loaders to a controller.

Nothing here downloads anything. Weights have to be on disk already, one
directory per model id, and the loaders in runtimes.py refuse a hub id. Loading
is deferred until a job actually asks for a capability: these models are
gigabytes and the controller keeps one resident at a time, so eager loading
would spend memory nothing can use.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from media_engine.models import runtimes
from media_engine.models.errors import AdapterError
from media_engine.models.registry import get_model, primary_for, primary_for_language

ENV_ENABLED = "MEDIA_ENGINE_LOCAL_MODELS"
ENV_WEIGHTS_DIR = "MEDIA_ENGINE_WEIGHTS_DIR"
ENV_TIMEOUT = "MEDIA_ENGINE_MULTIMODAL_TIMEOUT_SECONDS"

#: model id -> loader, each taking that model's own weights directory.
#: Keyed by model rather than capability because one capability can be served
#: by more than one model: Turkish speech needs a different checkpoint than
#: English, and each has its own weights and its own loader.
MODEL_LOADERS: dict[str, Callable[[str], object]] = {
    "Systran/faster-whisper-large-v3": runtimes.load_whisper,
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": runtimes.load_qwen_tts,
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": lambda directory: runtimes.load_qwen_tts(directory, clone=True),
    "freyavoice/Freya-TTS": runtimes.load_freya,
    "Qwen/Qwen2.5-1.5B-Instruct": runtimes.load_translator,
    "Qwen/Qwen-Image-Edit": runtimes.load_image_edit,
    "ACE-Step/ACE-Step-v1-3.5B": runtimes.load_music,
    "Wan-AI/Wan2.1-T2V-1.3B": runtimes.load_wan,
}


def capabilities() -> tuple[str, ...]:
    """Capabilities some loadable model can serve."""
    found: list[str] = []
    for model_id in MODEL_LOADERS:
        try:
            capability = get_model(model_id).capability
        except KeyError:
            continue
        if capability not in found:
            found.append(capability)
    return tuple(found)


def weights_dir_for(root: Path, capability: str, language: str | None = None) -> Path:
    """Where the model serving this capability keeps its files.

    With a language, the model that speaks it decides the directory, so Turkish
    speech does not get pointed at the English checkpoint's weights.
    """
    spec = primary_for_language(capability, language) if language else primary_for(capability)
    if spec is None:
        raise AdapterError("CAPABILITY_UNAVAILABLE")
    return Path(root) / spec.model_id


class Deferred:
    """Stands in for a model until a job actually calls it.

    submit_local() asks for a capability's runtime just to decide whether the
    capability is available, before it has even checked the caller's credits.
    Loading there would spend minutes and gigabytes on a request that may be
    refused a line later, so the weights are read on the first real attribute
    access instead.
    """

    def __init__(self, capability: str, load: Callable[[], object]) -> None:
        self.capability = capability
        self._load = load
        self._real: Optional[object] = None

    def loaded(self) -> bool:
        return self._real is not None

    def resolve(self) -> object:
        if self._real is None:
            self._real = self._load()
        return self._real

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.resolve(), name)

    def __call__(self, *args, **kwargs):
        # video.text_to_video calls its runtime directly rather than a method.
        return self.resolve()(*args, **kwargs)


class LazyRuntimes(dict):
    """Serves a real local runtime per capability, loading the weights late.

    A dict so an injected runtime still wins: a caller or a test can assign one
    directly and it is handed back untouched. Only a missing key reaches a
    loader.
    """

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.resident: Optional[str] = None

    def __bool__(self) -> bool:
        """Always true, even while empty.

        bound_runtime() reads this table as `getattr(...) or {}`, so an empty
        dict would be discarded as if no table were attached and every
        capability would report itself unavailable. Emptiness here means
        "nothing loaded yet", not "nothing on offer".
        """
        return True

    def for_model(self, capability, model_id, default=None):
        """The runtime for one named model of this capability."""
        injected = super().get(capability)
        if injected is not None:
            return injected
        key = str(model_id)
        present = super().get(key)
        if present is not None:
            return present
        loader = MODEL_LOADERS.get(key)
        if loader is None:
            return default
        # One model at a time. The controller's own gate already refuses a
        # second capability while one is resident, so keeping the previous
        # model would hold gigabytes nothing can reach.
        if self.resident is not None and self.resident != key:
            super().pop(self.resident, None)
            self.resident = None
        directory = str(Path(self.root) / key)
        runtime = Deferred(key, lambda: loader(directory))
        self[key] = runtime
        self.resident = key
        return runtime

    def get(self, capability, default=None):  # type: ignore[override]
        injected = super().get(capability)
        if injected is not None:
            return injected
        spec = primary_for(str(capability))
        if spec is None:
            return default
        found = self.for_model(capability, spec.model_id, default)
        return default if found is None else found


def attach_local_runtimes(controller, source: Optional[dict] = None) -> bool:
    """Give the controller real local runtimes. Off unless explicitly asked for.

    Returns whether it attached, so a caller can say so instead of guessing.
    """
    env = os.environ if source is None else source
    enabled = str(env.get(ENV_ENABLED, "")).strip().casefold()
    if enabled not in {"1", "true", "yes", "on"}:
        return False
    raw = str(env.get(ENV_WEIGHTS_DIR, "")).strip()
    if not raw:
        raise ValueError(ENV_WEIGHTS_DIR)
    root = Path(raw)
    if not root.is_dir():
        raise ValueError(ENV_WEIGHTS_DIR)
    controller.multimodal_runtimes = LazyRuntimes(root)
    timeout = str(env.get(ENV_TIMEOUT, "")).strip()
    if timeout:
        try:
            seconds = float(timeout)
        except ValueError as exc:
            raise ValueError(ENV_TIMEOUT) from exc
        if seconds <= 0:
            raise ValueError(ENV_TIMEOUT)
        controller.multimodal_timeout = seconds
    return True
