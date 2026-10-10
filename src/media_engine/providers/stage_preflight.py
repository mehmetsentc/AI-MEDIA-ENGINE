"""Checks that can fail before a GPU is rented.

They do not download the model. A manifest read, when used, is one small object.
"""
from __future__ import annotations

import json
from typing import Mapping

from media_engine.engines.image.model_cache import MANIFEST_NAME
from media_engine.engines.image.qwen import MODEL_ID
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.phase2d import r2_worker_env
from media_engine.storage.r2_media import MODEL_BUCKET

_STAGE_SNIPPETS = (
    "python3 /workspace/qwen_remote.py cache-sync",
    "/workspace/phase2d-r2.env",
    "/workspace/cache-sync.log",
)


def local_stage_problems(env: Mapping[str, str], command: str) -> list[str]:
    """Configuration problems visible without contacting R2."""
    problems: list[str] = []
    if r2_worker_env(env) is None:
        problems.append("R2_ENV_INCOMPLETE")
    endpoint = (env.get("R2_ENDPOINT") or "").strip()
    if endpoint and not endpoint.startswith("https://"):
        problems.append("R2_ENDPOINT")
    bucket = (env.get("R2_BUCKET") or "").strip()
    if bucket and bucket != MODEL_BUCKET:
        problems.append("R2_BUCKET")
    if env.get("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", "").strip() == "1":
        problems.append("RELIABLE_TRANSFER_NOT_SHIPPED")
    for snippet in _STAGE_SNIPPETS:
        if snippet not in command:
            problems.append("STAGE_COMMAND")
            break
    return problems


def assess_manifest_body(body: bytes) -> str:
    """Return an empty string when the body is the pinned complete manifest."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError):
        return "MANIFEST_INVALID"
    if not isinstance(payload, dict):
        return "MANIFEST_INVALID"
    if payload.get("state") != "COMPLETE":
        return "MANIFEST_INCOMPLETE"
    if payload.get("repo_id") != MODEL_ID:
        return "MANIFEST_MODEL"
    if payload.get("revision") != REVISION:
        return "MANIFEST_REVISION"
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return "MANIFEST_FILES"
    return ""


def manifest_object_key() -> str:
    return MANIFEST_NAME
