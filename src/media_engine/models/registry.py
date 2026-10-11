"""Pinned model catalog. Readiness never starts at PRODUCTION_READY."""
from __future__ import annotations

from dataclasses import dataclass

READINESS = frozenset({
    "DISCOVERED",
    "LICENSE_REVIEWED",
    "INTEGRATED",
    "OFFLINE_TESTED",
    "GPU_VERIFIED",
    "PRODUCTION_READY",
})

ADA_VRAM_MIB = 48 * 1024


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    capability: str
    revision: str
    license_name: str
    license_status: str
    commercial_use: str
    storage_bytes: int | None
    vram_mib: int | None
    vram_note: str
    languages: tuple[str, ...]
    modes: tuple[str, ...]
    runtime: str
    output_format: str
    security_notes: str
    worker: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    defaults: tuple[tuple[str, str], ...]
    cost_profile: str
    readiness: str
    role: str
    evidence: str
    ada_fit: str

    def to_public_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "capability": self.capability,
            "revision": self.revision,
            "license_name": self.license_name,
            "license_status": self.license_status,
            "commercial_use": self.commercial_use,
            "storage_bytes": self.storage_bytes,
            "vram_mib": self.vram_mib,
            "vram_note": self.vram_note,
            "languages": list(self.languages),
            "modes": list(self.modes),
            "runtime": self.runtime,
            "output_format": self.output_format,
            "worker": self.worker,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "defaults": {key: value for key, value in self.defaults},
            "cost_profile": self.cost_profile,
            "readiness": self.readiness,
            "role": self.role,
            "evidence": self.evidence,
            "ada_fit": self.ada_fit,
            "executable": self.executable,
        }

    @property
    def executable(self) -> bool:
        return (
            self.role == "primary"
            and self.commercial_use == "allowed"
            and self.readiness in {"GPU_VERIFIED", "PRODUCTION_READY"}
        )


def _spec(**kwargs: object) -> ModelSpec:
    readiness = str(kwargs["readiness"])
    if readiness not in READINESS:
        raise ValueError("READINESS")
    if readiness == "PRODUCTION_READY":
        raise ValueError("PRODUCTION_READY")
    return ModelSpec(**kwargs)  # type: ignore[arg-type]


_TTS_LANGS = (
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
)
_ACE_LANGS = (
    "en", "zh", "de", "fr", "es", "it", "pt", "pl", "tr", "ru",
    "cs", "nl", "ar", "ja", "hu", "ko", "hi",
)

# Storage and revision values are the Hugging Face model API snapshots read
# for this catalog. VRAM is recorded only when the vendor documented it or
# this engine already ran the model. Unmeasured peaks stay empty.
CATALOG: tuple[ModelSpec, ...] = (
    _spec(
        model_id="Qwen/Qwen-Image",
        capability="image.generate",
        revision="75e0b4be04f60ec59a75f475837eced720f823b6",
        license_name="apache-2.0",
        license_status="reviewed",
        commercial_use="allowed",
        storage_bytes=57_954_375_058,
        vram_mib=ADA_VRAM_MIB,
        vram_note="Ran on an RTX 6000 Ada 48GB worker. Peak memory was not profiled below that card.",
        languages=("en", "zh"),
        modes=("text-to-image",),
        runtime="diffusers QwenImagePipeline",
        output_format="image/png",
        security_notes="Weights are not downloaded on import. The browser does not receive provider credentials.",
        worker="media_engine.engines.image.qwen.QwenImageEngine",
        inputs=("prompt", "width", "height", "seed"),
        outputs=("image/png",),
        defaults=(("width", "1024"), ("height", "1024"), ("steps", "20")),
        cost_profile="qwen-image-1024",
        readiness="GPU_VERIFIED",
        role="primary",
        evidence="Pinned revision matches the working image engine. Real PNGs were stored from that worker.",
        ada_fit="verified",
    ),
    _spec(
        model_id="Qwen/Qwen-Image-Edit",
        capability="image.edit",
        revision="ac7f9318f633fc4b5778c59367c8128225f1e3de",
        license_name="apache-2.0",
        license_status="reviewed",
        commercial_use="allowed",
        storage_bytes=57_710_671_286,
        vram_mib=None,
        vram_note="BF16 parameter count matches Qwen-Image. Peak VRAM was not measured.",
        languages=("en", "zh"),
        modes=("image-edit",),
        runtime="diffusers QwenImageEditPipeline",
        output_format="image/png",
        security_notes="Source image must belong to the signed-in user. Weights are not downloaded here.",
        worker="media_engine.models.invoke",
        inputs=("prompt", "source_image"),
        outputs=("image/png",),
        defaults=(("steps", "unmeasured"),),
        cost_profile="unpriced",
        readiness="OFFLINE_TESTED",
        role="primary",
        evidence="Hub license is apache-2.0 and the revision is pinned. No local or GPU run.",
        ada_fit="unmeasured",
    ),
    _spec(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        capability="audio.tts",
        revision="0c0e3051f131929182e2c023b9537f8b1c68adfe",
        license_name="apache-2.0",
        license_status="reviewed",
        commercial_use="allowed",
        storage_bytes=4_617_449_338,
        vram_mib=None,
        vram_note="Hub stores about 4.3GB. Peak VRAM was not measured.",
        languages=_TTS_LANGS,
        modes=("preset-voice",),
        runtime="qwen-tts",
        output_format="audio/wav",
        security_notes="Preset speakers only. This checkpoint is not the voice-clone model.",
        worker="media_engine.models.invoke",
        inputs=("text", "language", "speaker"),
        outputs=("audio/wav",),
        defaults=(("sample_rate", "24000"),),
        cost_profile="unpriced",
        readiness="OFFLINE_TESTED",
        role="primary",
        evidence="Hub license is apache-2.0. No local or GPU run.",
        ada_fit="unmeasured",
    ),
    _spec(
        model_id="SWivid/F5-TTS",
        capability="audio.tts",
        revision="84e5a410d9cead4de2f847e7c9369a6440bdfaca",
        license_name="cc-by-nc-4.0",
        license_status="blocked",
        commercial_use="blocked",
        storage_bytes=14_834_664_139,
        vram_mib=None,
        vram_note="Not selected. Peak VRAM was not measured.",
        languages=("zh", "en"),
        modes=("reference-voice",),
        runtime="f5-tts",
        output_format="audio/wav",
        security_notes="Non-commercial weights. Reference audio would also require voice-owner consent.",
        worker="",
        inputs=("text", "reference_audio"),
        outputs=("audio/wav",),
        defaults=(),
        cost_profile="unpriced",
        readiness="DISCOVERED",
        role="blocked",
        evidence="Hub license is cc-by-nc-4.0 because of the Emilia training data. The code license does not free the weights.",
        ada_fit="blocked",
    ),
    _spec(
        model_id="Systran/faster-whisper-large-v3",
        capability="audio.transcribe",
        revision="edaa852ec7e145841d8ffdb056a99866b5f0a478",
        license_name="mit",
        license_status="reviewed",
        commercial_use="allowed",
        storage_bytes=3_145_762_253,
        vram_mib=4525,
        vram_note="faster-whisper documents a 4525MB fp16 peak. This engine did not remeasure it.",
        languages=("multilingual",),
        modes=("transcription", "timestamps"),
        runtime="faster-whisper CTranslate2",
        output_format="application/json",
        security_notes="Uploaded audio is private to the owner. The model.bin pickle import was not loaded.",
        worker="media_engine.models.invoke",
        inputs=("audio",),
        outputs=("text", "srt", "vtt"),
        defaults=(("compute_type", "int8"), ("word_timestamps", "true")),
        cost_profile="unpriced",
        readiness="OFFLINE_TESTED",
        role="primary",
        evidence="Hub license is mit and the revision is pinned. No local or GPU run.",
        ada_fit="documented",
    ),
    _spec(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        capability="audio.translate",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        license_name="apache-2.0",
        license_status="reviewed",
        commercial_use="allowed",
        storage_bytes=10_208_517_569,
        vram_mib=None,
        vram_note="About 1.5B BF16 weights. Peak VRAM and translation quality were not measured.",
        languages=("en",),
        modes=("text-translation",),
        runtime="transformers",
        output_format="text/plain",
        security_notes="The model card language tag is en. Multilingual quality is not verified.",
        worker="media_engine.models.invoke",
        inputs=("text", "source_language", "target_language"),
        outputs=("text", "srt"),
        defaults=(),
        cost_profile="unpriced",
        readiness="OFFLINE_TESTED",
        role="primary",
        evidence="Hub license is apache-2.0. This is an instruct model, not a measured translation system.",
        ada_fit="unmeasured",
    ),
    _spec(
        model_id="facebook/nllb-200-distilled-600M",
        capability="audio.translate",
        revision="",
        license_name="cc-by-nc-4.0",
        license_status="blocked",
        commercial_use="blocked",
        storage_bytes=None,
        vram_mib=None,
        vram_note="Not selected.",
        languages=("flores-200",),
        modes=("translation",),
        runtime="fairseq",
        output_format="text/plain",
        security_notes="Non-commercial license covers modified versions as well.",
        worker="",
        inputs=("text", "source_language", "target_language"),
        outputs=("text",),
        defaults=(),
        cost_profile="unpriced",
        readiness="DISCOVERED",
        role="blocked",
        evidence="Model card license is CC-BY-NC. Revision was not pinned because the license blocks commercial use.",
        ada_fit="blocked",
    ),
    _spec(
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        capability="audio.voice_clone",
        revision="fd4b254389122332181a7c3db7f27e918eec64e3",
        license_name="apache-2.0",
        license_status="reviewed",
        commercial_use="allowed",
        storage_bytes=4_539_846_851,
        vram_mib=None,
        vram_note="Hub stores about 4.2GB. Peak VRAM was not measured.",
        languages=_TTS_LANGS,
        modes=("voice-clone",),
        runtime="qwen-tts",
        output_format="audio/wav",
        security_notes="A voice-owner consent record is required. Third-party impersonation is refused.",
        worker="media_engine.models.invoke",
        inputs=("text", "language", "consent_id", "reference_audio"),
        outputs=("audio/wav",),
        defaults=(),
        cost_profile="unpriced",
        readiness="OFFLINE_TESTED",
        role="primary",
        evidence="Hub license is apache-2.0. The vendor documents 3-second cloning. No local or GPU run.",
        ada_fit="unmeasured",
    ),
    _spec(
        model_id="Wan-AI/Wan2.1-T2V-1.3B",
        capability="video.text_to_video",
        revision="37ec512624d61f7aa208f7ea8140a131f93afc9a",
        license_name="apache-2.0",
        license_status="reviewed",
        commercial_use="allowed",
        storage_bytes=46_519_465_450,
        vram_mib=8190,
        vram_note="Model card says the 1.3B text-to-video model requires 8.19GB. This engine did not remeasure it.",
        languages=("en", "zh"),
        modes=("text-to-video-480p",),
        runtime="diffusers",
        output_format="video/mp4",
        security_notes="Repository is about 43GB. It is not downloaded in this build and must not share a disk blindly with Qwen-Image.",
        worker="media_engine.models.invoke",
        inputs=("prompt", "aspect_ratio", "duration_seconds"),
        outputs=("video/mp4",),
        defaults=(("resolution", "480p"), ("duration_seconds", "5"), ("fps", "16")),
        cost_profile="wan-t2v-5s",
        readiness="OFFLINE_TESTED",
        role="primary",
        evidence="Hub license is apache-2.0 and the revision is pinned. No local or GPU run.",
        ada_fit="documented",
    ),
    _spec(
        model_id="Lightricks/LTX-Video",
        capability="video.text_to_video",
        revision="",
        license_name="LTXV-Open-Weights-License-0.X",
        license_status="blocked",
        commercial_use="blocked",
        storage_bytes=None,
        vram_mib=None,
        vram_note="Not selected.",
        languages=("en",),
        modes=("text-to-video", "image-to-video"),
        runtime="diffusers",
        output_format="video/mp4",
        security_notes="Commercial entities at or above $10M annual revenue need a separate paid license. Use restrictions also apply.",
        worker="",
        inputs=("prompt",),
        outputs=("video/mp4",),
        defaults=(),
        cost_profile="unpriced",
        readiness="DISCOVERED",
        role="blocked",
        evidence="License text dated 2025-04-15 blocks an unconditional commercial grant. Revision was not pinned.",
        ada_fit="blocked",
    ),
    _spec(
        model_id="Wan-AI/Wan2.1-I2V-14B-480P",
        capability="video.image_to_video",
        revision="",
        license_name="unverified",
        license_status="blocked",
        commercial_use="uncertain",
        storage_bytes=None,
        vram_mib=None,
        vram_note="14B image-to-video peak VRAM, storage, and license file were not verified.",
        languages=("en", "zh"),
        modes=("image-to-video",),
        runtime="unverified",
        output_format="video/mp4",
        security_notes="Do not assume the 1.3B text-to-video license or VRAM applies to this checkpoint.",
        worker="",
        inputs=("image", "prompt"),
        outputs=("video/mp4",),
        defaults=(),
        cost_profile="unpriced",
        readiness="DISCOVERED",
        role="blocked",
        evidence="Named by the Wan2.1 model card. This pass did not pin the repository.",
        ada_fit="blocked",
    ),
    _spec(
        model_id="ACE-Step/ACE-Step-v1-3.5B",
        capability="music.generate",
        revision="82cd0d7b6322bd28cd4e830fe675ddb6180ce36c",
        license_name="apache-2.0",
        license_status="reviewed",
        commercial_use="allowed",
        storage_bytes=8_276_582_969,
        vram_mib=8192,
        vram_note="Project README says the 2025-05-10 update reduced max VRAM to 8GB with offload. This engine did not remeasure it.",
        languages=_ACE_LANGS,
        modes=("text-to-music",),
        runtime="ace-step",
        output_format="audio/wav",
        security_notes="Training-data provenance is separate from the Apache weight license and is not re-audited here.",
        worker="media_engine.models.invoke",
        inputs=("prompt", "duration_seconds", "style"),
        outputs=("audio/wav",),
        defaults=(),
        cost_profile="unpriced",
        readiness="OFFLINE_TESTED",
        role="primary",
        evidence="Hub license is apache-2.0 and the revision is pinned. No local or GPU run.",
        ada_fit="documented",
    ),
    _spec(
        model_id="facebook/musicgen-medium",
        capability="music.generate",
        revision="d3bd7b00761b78ad7a8a05145ee31e7832e9916c",
        license_name="cc-by-nc-4.0",
        license_status="blocked",
        commercial_use="blocked",
        storage_bytes=36_073_996_801,
        vram_mib=None,
        vram_note="Not selected.",
        languages=("en",),
        modes=("text-to-music",),
        runtime="transformers",
        output_format="audio/wav",
        security_notes="Non-commercial weights. The repository is about 34GB and is not downloaded.",
        worker="",
        inputs=("prompt",),
        outputs=("audio/wav",),
        defaults=(),
        cost_profile="unpriced",
        readiness="DISCOVERED",
        role="blocked",
        evidence="Hub license is cc-by-nc-4.0.",
        ada_fit="blocked",
    ),
)

_BY_ID = {spec.model_id: spec for spec in CATALOG}


def get_model(model_id: str) -> ModelSpec:
    try:
        return _BY_ID[model_id]
    except KeyError as exc:
        raise KeyError(model_id) from exc


def models_for(capability: str) -> tuple[ModelSpec, ...]:
    return tuple(spec for spec in CATALOG if spec.capability == capability)


def primary_for(capability: str) -> ModelSpec | None:
    chosen = [spec for spec in models_for(capability) if spec.role == "primary"]
    if not chosen:
        return None
    return chosen[0]
