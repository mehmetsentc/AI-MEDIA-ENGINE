"""Local weight loaders. A hub id is refused, so this module cannot download a model."""
from __future__ import annotations

import importlib
import io
import json
import os
from pathlib import Path

from media_engine.models.errors import AdapterError
from media_engine.models.registry import get_model

ENV_DTYPE = "MEDIA_ENGINE_LOCAL_DTYPE"
ENV_DEVICE = "MEDIA_ENGINE_LOCAL_DEVICE"
ENV_AUDIOVAE = "MEDIA_ENGINE_AUDIOVAE_PATH"

_FILES = {
    "Systran/faster-whisper-large-v3": ("model.bin", "config.json"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": ("model.safetensors", "config.json"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": ("model.safetensors", "config.json"),
    "Qwen/Qwen2.5-1.5B-Instruct": ("model.safetensors", "config.json"),
    "Qwen/Qwen-Image-Edit": ("model_index.json",),
    "ACE-Step/ACE-Step-v1-3.5B": ("config.json",),
    # model_index.json is what DiffusionPipeline reads to find its pipeline
    # class. A repository without it cannot be loaded at all, so requiring it
    # here turns a late runtime failure into a readiness check.
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": ("model_index.json",),
    "freyavoice/Freya-TTS": ("model.safetensors", "config.json"),
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
    return module.Qwen3TTSModel.from_pretrained(
        str(root), local_files_only=True, dtype=load_dtype(),
    )


def load_translator(weights_dir: str):
    root = weights_ready("Qwen/Qwen2.5-1.5B-Instruct", weights_dir)
    transformers = _import("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(root), local_files_only=True, dtype=load_dtype(),
    )
    return _ChatModel(model, tokenizer)


def load_image_edit(weights_dir: str):
    root = weights_ready("Qwen/Qwen-Image-Edit", weights_dir)
    diffusers = _import("diffusers")
    return diffusers.QwenImageEditPipeline.from_pretrained(
        str(root), local_files_only=True, dtype=load_dtype(),
    )


def load_music(weights_dir: str):
    """ACE-Step does not load through diffusers, so this refuses early.

    DiffusionPipeline finds its pipeline class in model_index.json, and the
    ACE-Step repository has no such file: its own config.json declares
    ACEStepTransformer2DModel, a class diffusers does not contain, and there is
    no acestep package on PyPI either. Running it means installing ACE-Step
    from its own repository and calling its pipeline. Until that exists,
    refusing here is honest; attempting the load would fail later with a
    diffusers error that says nothing about the real cause.
    """
    weights_ready("ACE-Step/ACE-Step-v1-3.5B", weights_dir)
    raise AdapterError("DEPENDENCY_MISSING")


def load_wan(weights_dir: str):
    root = weights_ready("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", weights_dir)
    diffusers = _import("diffusers")
    pipeline = getattr(diffusers, "WanPipeline", None)
    if pipeline is None:
        raise AdapterError("DEPENDENCY_MISSING")
    return pipeline.from_pretrained(
        str(root), local_files_only=True, dtype=load_dtype(),
    )


def load_device() -> str:
    """Where local models run. CPU unless an operator points elsewhere."""
    raw = os.environ.get(ENV_DEVICE, "").strip()
    return raw or "cpu"


def _audiovae_path(root: Path) -> Path:
    """Freya decodes through the VoxCPM2 AudioVAE.

    Its own helper downloads that file from the hub, which this module must not
    do, so the checkpoint is taken from the staged VoxCPM2 directory beside it.
    """
    raw = os.environ.get(ENV_AUDIOVAE, "").strip()
    if raw:
        return Path(raw)
    return Path(root).parents[1] / "openbmb" / "VoxCPM2" / "audiovae.pth"


class FreyaVoice:
    """Gives Freya the interface invoke_tts expects.

    Freya takes neither a speaker nor a language: it is Turkish only, and it has
    no speaker conditioning at all -- the vendor documents that the noise seed
    *is* the voice and that other seeds give other, uncurated people. The seed
    is therefore left at the pinned default, so the one voice on offer is the
    one the registry names.
    """

    sample_rate = 48_000

    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def generate_custom_voice(self, *, text, speaker=None, language=None, **_kwargs):
        wav = self.pipeline.synthesize(str(text))
        rate = int(getattr(self.pipeline, "sample_rate", self.sample_rate))
        return [wav], rate


def load_freya(weights_dir: str) -> FreyaVoice:
    root = weights_ready("freyavoice/Freya-TTS", weights_dir)
    torch = _import("torch")
    pipeline_module = _import("freyatts.pipeline")
    loader = _import("safetensors.torch")
    audiovae = _import("voxcpm.modules.audiovae")

    with io.open(root / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    model = pipeline_module.FreyaDiT(
        vocab=config["vocab"], d=config["d"], depth=config["depth"],
        heads=config["heads"], ff=config["ff"],
    )
    model.load_state_dict(loader.load_file(str(root / "model.safetensors")), strict=True)
    device = load_device()
    model = model.to(device).eval()

    checkpoint = _audiovae_path(root)
    if not checkpoint.is_file():
        raise AdapterError("WEIGHTS_NOT_PRESENT")
    vae = audiovae.AudioVAEV2(audiovae.AudioVAEConfigV2())
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    vae.load_state_dict(state.get("state_dict", state), strict=False)
    vae = vae.to(device).float().eval()
    for parameter in vae.parameters():
        parameter.requires_grad = False

    vocabulary = Path(pipeline_module.__file__).parent / "char_vocab.json"
    with io.open(vocabulary, encoding="utf-8") as handle:
        char_to_id = json.load(handle)
    return FreyaVoice(pipeline_module.FreyaTTS(model, vae, char_to_id, device=device))


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


def load_dtype():
    """The dtype these checkpoints are stored in, unless an operator says otherwise.

    The published weights are bfloat16. Letting transformers widen them to the
    default float32 doubles resident memory and buys nothing, which is the
    difference between fitting in an 8 GB machine and not.
    """
    torch = _import("torch")
    raw = os.environ.get(ENV_DTYPE, "").strip()
    if not raw:
        return torch.bfloat16
    dtype = getattr(torch, raw, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise AdapterError("DTYPE_UNSUPPORTED")
    return dtype
