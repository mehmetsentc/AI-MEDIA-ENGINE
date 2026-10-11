"""Dependency probe. Importing this module does not import torch or download weights."""
from __future__ import annotations

import importlib.util

DEPENDENCIES = {
    "audio.transcribe": ("faster_whisper",),
    "audio.tts": ("qwen_tts",),
    "audio.voice_clone": ("qwen_tts",),
    "audio.translate": ("transformers",),
    "image.edit": ("diffusers",),
    "music.generate": ("diffusers",),
    "video.text_to_video": ("diffusers",),
}


def dependency_status() -> dict[str, bool]:
    names = sorted({name for group in DEPENDENCIES.values() for name in group})
    return {name: importlib.util.find_spec(name) is not None for name in names}


def missing_for(kind: str) -> tuple[str, ...]:
    present = dependency_status()
    return tuple(name for name in DEPENDENCIES.get(kind, ()) if not present[name])
