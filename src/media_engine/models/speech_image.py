"""Speech image reference. A tag without a digest is not a built image."""
from __future__ import annotations

import json
import re
from pathlib import Path

from media_engine.providers.base import WorkerStageError

QWEN_REPOSITORY = "ghcr.io/mehmetsentc/ai-media-engine-worker"
SPEECH_REPOSITORY = "ghcr.io/mehmetsentc/ai-media-engine-speech-worker"
_DIGEST = r"^sha256:[0-9a-f]{64}$"
_TAG = r"^sha-[0-9a-f]{40}$"


def speech_image_ref_path() -> Path:
    return Path(__file__).resolve().parents[3] / "deploy" / "worker" / "speech" / "image_ref.json"


def qwen_image_ref_path() -> Path:
    return Path(__file__).resolve().parents[3] / "deploy" / "worker" / "image_ref.json"


def load_image_ref(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerStageError("IMAGE_UNVERIFIED") from exc
    if not isinstance(payload, dict):
        raise WorkerStageError("IMAGE_UNVERIFIED")
    return payload


def assess_speech_image_ref(document: dict | None) -> dict:
    """Accept only the speech repository pinned by digest. CUDA is not claimed here."""
    if not isinstance(document, dict):
        raise WorkerStageError("SPEECH_IMAGE_NOT_BUILT")
    if document.get("cuda_gpu_support") is True:
        raise WorkerStageError("IMAGE_UNVERIFIED")
    repository = str(document.get("repository") or "")
    tag = str(document.get("tag") or "")
    digest = str(document.get("digest") or "")
    pinned = str(document.get("pinned_reference") or "")
    if repository == QWEN_REPOSITORY or repository.endswith("/ai-media-engine-worker"):
        raise WorkerStageError("IMAGE_SUBSTITUTION")
    if not repository or not tag or not digest or not pinned:
        raise WorkerStageError("IMAGE_UNVERIFIED")
    if repository != SPEECH_REPOSITORY:
        raise WorkerStageError("IMAGE_SUBSTITUTION")
    if re.fullmatch(_DIGEST, digest) is None:
        raise WorkerStageError("IMAGE_DIGEST")
    if re.fullmatch(_TAG, tag) is None or tag == "latest":
        raise WorkerStageError("IMAGE_MUTABLE")
    if "@sha256:" not in pinned or pinned != f"{repository}@{digest}":
        raise WorkerStageError("IMAGE_MUTABLE")
    previous = document.get("previous_digest", "")
    if previous not in ("", None) and re.fullmatch(_DIGEST, str(previous)) is None:
        raise WorkerStageError("IMAGE_DIGEST")
    qwen = load_image_ref(qwen_image_ref_path())
    if isinstance(qwen, dict) and digest == qwen.get("digest"):
        raise WorkerStageError("IMAGE_SUBSTITUTION")
    return document
