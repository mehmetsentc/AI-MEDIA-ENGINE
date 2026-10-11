"""Multimodal submissions. Unverified kinds never become jobs or credit holds."""
from __future__ import annotations

from pathlib import Path

from decimal import Decimal

from media_engine.models.consent import ConsentError, ConsentStore
from media_engine.models.cost import estimate
from media_engine.models.registry import models_for, primary_for
from media_engine.models.uploads import UploadError, UploadStore

KINDS = (
    "image.generate",
    "image.edit",
    "audio.tts",
    "audio.transcribe",
    "audio.translate",
    "audio.voice_clone",
    "video.text_to_video",
    "video.image_to_video",
    "music.generate",
)
LEGACY_JOB = {"image.generate": "IMAGE_GENERATE"}
_ASPECTS = {"16:9", "9:16", "1:1"}
_DUBBING = ("audio.transcribe", "audio.translate", "audio.tts", "audio.voice_clone")

_FIELDS = {
    "image.generate": ("prompt", "resolution", "steps", "seed"),
    "image.edit": ("prompt", "source image"),
    "audio.tts": ("text", "language", "voice"),
    "audio.transcribe": ("audio upload", "timestamps", "subtitle export"),
    "audio.translate": ("source language", "target language", "text"),
    "audio.voice_clone": ("text", "language", "voice-owner authorization"),
    "video.text_to_video": ("prompt", "aspect ratio", "duration"),
    "video.image_to_video": ("image", "prompt", "aspect ratio", "duration"),
    "music.generate": ("prompt", "duration", "style"),
}


class SchemaError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def catalog() -> dict:
    capabilities = []
    for kind in KINDS:
        spec = primary_for(kind)
        executable = bool(spec and spec.executable)
        reason = ""
        if not executable:
            reason = "Unavailable until this model is GPU-verified."
        quote = None
        if spec is not None:
            quote = estimate(spec.cost_profile, hourly_usd=Decimal("0.66"))
        capabilities.append({
            "kind": kind,
            "label": kind,
            "fields": list(_FIELDS[kind]),
            "executable": executable,
            "unavailable_reason": reason,
            "legacy_job_type": LEGACY_JOB.get(kind, ""),
            "model": None if spec is None else spec.to_public_dict(),
            "estimate": quote,
            "blocked_alternatives": [
                item.model_id for item in models_for(kind) if item.role == "blocked"
            ],
        })
    return {
        "capabilities": capabilities,
        "pipelines": [{
            "id": "dubbing",
            "stages": list(_DUBBING),
            "executable": False,
            "unavailable_reason": "Dubbing stays off until transcription, translation, and speech are GPU-verified.",
        }],
    }


def submit_capability(controller, user_id: str, kind: str, payload: dict, book) -> tuple[int, dict]:
    if kind not in KINDS:
        return 400, {"error": "UNKNOWN_CAPABILITY"}
    try:
        _validate(kind, payload)
    except SchemaError as exc:
        return 400, {"error": exc.code}
    if kind == "audio.voice_clone":
        try:
            _require_consent(controller, user_id, str(payload.get("consent_id") or ""))
        except ConsentError as exc:
            return 403, {"error": exc.code}
    spec = primary_for(kind)
    if kind == "video.image_to_video" or spec is None or spec.role != "primary":
        return 409, {"error": "CAPABILITY_UNAVAILABLE", "kind": kind}
    if spec.commercial_use != "allowed":
        return 409, {"error": "LICENSE_BLOCKED", "kind": kind}
    provider_name = getattr(getattr(controller, "provider", None), "name", "")
    if provider_name == "vast" and not spec.executable:
        return 409, {"error": "CAPABILITY_UNAVAILABLE", "kind": kind}
    if kind == "image.generate":
        from media_engine.studio.account import submit_for_studio
        return submit_for_studio(controller, user_id, payload, book)
    from media_engine.models.runner import submit_local
    return submit_local(controller, user_id, kind, payload, book)


def record_consent(controller, user_id: str, payload: dict) -> tuple[int, dict]:
    try:
        return 201, _consent_store(controller).record(user_id, payload)
    except ConsentError as exc:
        return 403, {"error": exc.code}


def accept_upload(controller, user_id: str, payload: dict) -> tuple[int, dict]:
    try:
        header = bytes.fromhex(str(payload.get("header_hex") or ""))
    except ValueError:
        return 400, {"error": "UPLOAD_TYPE"}
    try:
        record = _upload_store(controller).accept(
            user_id,
            filename=str(payload.get("filename") or ""),
            header=header,
            declared_bytes=payload.get("declared_bytes"),
            kind=str(payload.get("kind") or ""),
            duration_seconds=payload.get("duration_seconds"),
        )
    except UploadError as exc:
        return 400, {"error": exc.code}
    return 201, record


def delete_upload(controller, user_id: str, upload_id: str) -> tuple[int, dict]:
    if not _upload_store(controller).delete(user_id, upload_id):
        return 404, {"error": "UPLOAD_NOT_FOUND"}
    return 200, {"deleted": upload_id}


def _validate(kind: str, payload: dict) -> None:
    if kind == "image.generate":
        _text(payload.get("prompt"), "PROMPT_REQUIRED")
        return
    if kind == "image.edit":
        _text(payload.get("prompt"), "PROMPT_REQUIRED")
        _text(payload.get("source_upload_id"), "UPLOAD_REQUIRED")
        return
    if kind == "audio.tts":
        _text(payload.get("text"), "TEXT_REQUIRED")
        _language(payload.get("language"), "audio.tts")
        _text(payload.get("voice"), "VOICE_REQUIRED")
        return
    if kind == "audio.transcribe":
        _text(payload.get("upload_id"), "UPLOAD_REQUIRED")
        return
    if kind == "audio.translate":
        _text(payload.get("text"), "TEXT_REQUIRED")
        _text(payload.get("source_language"), "LANGUAGE_REQUIRED")
        _text(payload.get("target_language"), "LANGUAGE_REQUIRED")
        return
    if kind == "audio.voice_clone":
        _text(payload.get("text"), "TEXT_REQUIRED")
        _language(payload.get("language"), "audio.voice_clone")
        _text(payload.get("consent_id"), "CONSENT_REQUIRED")
        return
    if kind in {"video.text_to_video", "video.image_to_video"}:
        _text(payload.get("prompt"), "PROMPT_REQUIRED")
        if payload.get("aspect_ratio") not in _ASPECTS:
            raise SchemaError("ASPECT_UNSUPPORTED")
        _duration(payload.get("duration_seconds"), 5 if kind == "video.text_to_video" else 10)
        if kind == "video.image_to_video":
            _text(payload.get("upload_id"), "UPLOAD_REQUIRED")
        return
    if kind == "music.generate":
        _text(payload.get("prompt"), "PROMPT_REQUIRED")
        _text(payload.get("style"), "STYLE_REQUIRED")
        _duration(payload.get("duration_seconds"), 60)
        return
    raise SchemaError("UNKNOWN_CAPABILITY")


def _text(value: object, code: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise SchemaError(code)


def _language(value: object, kind: str) -> None:
    spec = primary_for(kind)
    if not isinstance(value, str) or spec is None or value not in spec.languages:
        raise SchemaError("LANGUAGE_UNSUPPORTED")


def _duration(value: object, limit: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > limit:
        raise SchemaError("DURATION_LIMIT")


def _require_consent(controller, user_id: str, consent_id: str) -> dict:
    found = _consent_store(controller).get(user_id, consent_id)
    if found is None:
        raise ConsentError("CONSENT_REQUIRED")
    return found


def _consent_store(controller) -> ConsentStore:
    store = getattr(controller, "consent_store", None)
    if store is None:
        root = Path(controller.jobs.db_path).parent
        store = ConsentStore(str(root / "voice_consent.sqlite3"))
        controller.consent_store = store
    return store


def _upload_store(controller) -> UploadStore:
    store = getattr(controller, "upload_store", None)
    if store is None:
        root = Path(controller.jobs.db_path).parent / "uploads"
        store = UploadStore(root)
        controller.upload_store = store
    return store
