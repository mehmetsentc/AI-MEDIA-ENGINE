"""Local weight loaders. A hub id is refused, so this module cannot download a model."""
from __future__ import annotations

import importlib
from pathlib import Path

from media_engine.models.errors import AdapterError
from media_engine.models.registry import get_model

_FILES = {
    "Systran/faster-whisper-large-v3": ("model.bin", "config.json"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": ("model.safetensors", "config.json"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": ("model.safetensors", "config.json"),
    "Qwen/Qwen2.5-1.5B-Instruct": ("model.safetensors", "config.json"),
    "Qwen/Qwen-Image-Edit": ("model_index.json",),
    "ACE-Step/ACE-Step-v1-3.5B": ("config.json",),
    "Wan-AI/Wan2.1-T2V-1.3B": ("config.json",),
}


def weights_ready(model_id: str, weights_dir: str) -> Path:
    spec = get_model(model_id)
    if spec.commercial_use != "allowed" or spec.role != "primary":
        raise AdapterError("LICENSE_BLOCKED")
    root = Path(weights_dir)
    if not root.is_dir():
        raise AdapterError("WEIGHTS_NOT_PRESENT")
    for name in _FILES[model_id]:
        if not (root / name).is_file():
            raise AdapterError("WEIGHTS_NOT_PRESENT")
    marker = root / "REVISION"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() != spec.revision:
        raise AdapterError("REVISION_MISMATCH")
    return root


def load_whisper(weights_dir: str):
    root = weights_ready("Systran/faster-whisper-large-v3", weights_dir)
    module = _import("faster_whisper")
    return module.WhisperModel(str(root), device="cpu", compute_type="int8")


def load_qwen_tts(weights_dir: str, *, clone: bool = False):
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" if clone else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    root = weights_ready(model_id, weights_dir)
    module = _import("qwen_tts")
    return module.Qwen3TTSModel.from_pretrained(str(root), local_files_only=True)


def load_translator(weights_dir: str):
    root = weights_ready("Qwen/Qwen2.5-1.5B-Instruct", weights_dir)
    transformers = _import("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(str(root), local_files_only=True)
    return _ChatModel(model, tokenizer)


def load_image_edit(weights_dir: str):
    root = weights_ready("Qwen/Qwen-Image-Edit", weights_dir)
    diffusers = _import("diffusers")
    return diffusers.QwenImageEditPipeline.from_pretrained(str(root), local_files_only=True)


def load_music(weights_dir: str):
    root = weights_ready("ACE-Step/ACE-Step-v1-3.5B", weights_dir)
    diffusers = _import("diffusers")
    return diffusers.DiffusionPipeline.from_pretrained(str(root), local_files_only=True)


def load_wan(weights_dir: str):
    root = weights_ready("Wan-AI/Wan2.1-T2V-1.3B", weights_dir)
    diffusers = _import("diffusers")
    pipeline = getattr(diffusers, "WanPipeline", None)
    if pipeline is None:
        raise AdapterError("DEPENDENCY_MISSING")
    return pipeline.from_pretrained(str(root), local_files_only=True)


class _ChatModel:
    """Real Qwen2.5 generate call. Tests inject an object with translate() instead."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def translate(self, messages: list[dict]) -> str:
        template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        tokens = self.tokenizer(template, return_tensors="pt")
        generated = self.model.generate(**tokens, max_new_tokens=512)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)


def _import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise AdapterError("DEPENDENCY_MISSING") from exc
