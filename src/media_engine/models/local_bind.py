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
from media_engine.models.registry import primary_for

ENV_ENABLED = "MEDIA_ENGINE_LOCAL_MODELS"
ENV_WEIGHTS_DIR = "MEDIA_ENGINE_WEIGHTS_DIR"
ENV_TIMEOUT = "MEDIA_ENGINE_MULTIMODAL_TIMEOUT_SECONDS"

#: capability -> loader, each taking that model's own weights directory
LOADERS: dict[str, Callable[[str], object]] = {
    "audio.transcribe": runtimes.load_whisper,
    "audio.tts": runtimes.load_qwen_tts,
    "audio.voice_clone": lambda directory: runtimes.load_qwen_tts(directory, clone=True),
    "audio.translate": runtimes.load_translator,
    "image.edit": runtimes.load_image_edit,
    "music.generate": runtimes.load_music,
    "video.text_to_video": runtimes.load_wan,
}


def weights_dir_for(root: Path, capability: str) -> Path:
    """Where this capability's primary model keeps its files."""
    spec = primary_for(capability)
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

    def get(self, capability, default=None):  # type: ignore[override]
        present = super().get(capability)
        if present is not None:
            return present
        name = str(capability)
        loader = LOADERS.get(name)
        if loader is None:
            return default
        # One model at a time. The controller's own gate already refuses a
        # second capability while one is resident, so keeping the previous
        # model would hold gigabytes nothing can reach.
        if self.resident is not None and self.resident != capability:
            super().pop(self.resident, None)
            self.resident = None
        directory = str(weights_dir_for(self.root, name))
        runtime = Deferred(name, lambda: loader(directory))
        self[capability] = runtime
        self.resident = name
        return runtime


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
