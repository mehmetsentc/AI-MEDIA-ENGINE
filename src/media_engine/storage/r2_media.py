"""Private media bucket. Model objects stay in the model bucket."""
from __future__ import annotations

import os
from pathlib import Path

from media_engine.engines.image.r2_cache import R2Client, R2Config
from media_engine.platform.repository import PlatformBlocked

MODEL_BUCKET = "ai-media-engine-models"
MEDIA_BUCKET = "daidoi-media"


class MediaStorageError(RuntimeError):
    pass


class UnavailableMedia:
    """Live mode has no configured media bucket. Uploads fail closed."""

    bucket = MEDIA_BUCKET

    def put(self, key: str, data: bytes) -> None:
        raise PlatformBlocked("MEDIA_STORAGE_NOT_CONFIGURED")

    def get(self, key: str) -> bytes:
        raise PlatformBlocked("MEDIA_STORAGE_NOT_CONFIGURED")

    def head(self, key: str) -> int:
        raise PlatformBlocked("MEDIA_STORAGE_NOT_CONFIGURED")


class R2MediaStore:
    def __init__(self, config: R2Config) -> None:
        self.bucket = config.bucket
        self._client = R2Client(config)

    def put(self, key: str, data: bytes) -> None:
        self._client.put(key, data)

    def get(self, key: str) -> bytes:
        loaded = self._client.get(key)
        if loaded is None:
            raise PlatformBlocked("ARTIFACT_VERIFY_FAILED")
        body, _meta = loaded
        return body

    def head(self, key: str) -> int:
        status, headers, _body = self._client._request("HEAD", key, b"")
        if status != 200:
            raise PlatformBlocked("ARTIFACT_VERIFY_FAILED")
        length = headers.get("Content-Length") or headers.get("content-length")
        if length is None:
            raise PlatformBlocked("ARTIFACT_VERIFY_FAILED")
        return int(length)


def env_file() -> Path:
    return Path(__file__).resolve().parents[3] / ".env.r2"


def merged_config(source: dict | None = None) -> dict[str, str]:
    """File values, then the process environment, then an explicit mapping."""
    merged: dict[str, str] = {}
    path = env_file()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            merged[name.strip()] = value.strip().strip("'\"")
    for name, value in os.environ.items():
        if name.startswith("R2_") or name.startswith("MEDIA_"):
            merged[name] = value
    if source:
        merged.update(source)
    return merged


def media_config(source: dict | None = None) -> R2Config:
    merged = merged_config(source)
    model_bucket = (merged.get("R2_BUCKET") or MODEL_BUCKET).strip()
    bucket = (merged.get("MEDIA_R2_BUCKET") or merged.get("MEDIA_ENGINE_MEDIA_BUCKET") or MEDIA_BUCKET).strip()
    if not bucket or bucket == model_bucket:
        raise MediaStorageError("MEDIA_BUCKET_COLLIDES_WITH_MODEL_BUCKET")
    endpoint = (merged.get("MEDIA_R2_ENDPOINT") or merged.get("R2_ENDPOINT") or "").strip().rstrip("/")
    access = merged.get("MEDIA_R2_ACCESS_KEY_ID") or merged.get("R2_ACCESS_KEY_ID") or ""
    secret = merged.get("MEDIA_R2_SECRET_ACCESS_KEY") or merged.get("R2_SECRET_ACCESS_KEY") or ""
    if not endpoint or not access or not secret:
        raise MediaStorageError("MEDIA_STORAGE_NOT_CONFIGURED")
    return R2Config(endpoint, bucket, access, secret)


def open_media_store(source: dict | None = None) -> R2MediaStore:
    return R2MediaStore(media_config(source))
