"""Revision-addressed model cache. A hit does not download the model again."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT, REPO_ID
from media_engine.providers.live7_contract import REVISION

PINNED_REVISION = REVISION
EXPECTED_FILES = QWEN_FILE_COUNT
EXPECTED_BYTES = QWEN_CACHE_MIN_BYTES
MODEL_BUCKET = "ai-media-engine-models"


class CacheError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RevisionCache:
    """Publish a revision atomically. Temporary instance disk is not this cache."""

    def __init__(self, root: str | Path, *, revision: str = PINNED_REVISION,
                 expected_files: int = EXPECTED_FILES, expected_bytes: int = EXPECTED_BYTES,
                 enforce_pin: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.revision = revision
        self.expected_files = int(expected_files)
        self.expected_bytes = int(expected_bytes)
        self.enforce_pin = enforce_pin

    def health(self, revision: str = PINNED_REVISION) -> dict:
        published = self._published(revision)
        manifest = self._read_manifest(published)
        problem = self._problem(published, manifest, revision)
        return {
            "revision": revision,
            "repo_id": REPO_ID,
            "bucket": MODEL_BUCKET,
            "state": "HIT" if problem == "" else ("EMPTY" if manifest is None and not published.is_dir() else "INVALID"),
            "problem": problem,
            "path": str(published),
            "expected_files": EXPECTED_FILES,
            "expected_bytes": EXPECTED_BYTES,
            "files": 0 if not isinstance(manifest, dict) else len(manifest.get("files") or []),
            "bytes": 0 if not isinstance(manifest, dict) else int(manifest.get("expected_total_bytes") or 0),
        }

    def lookup(self, revision: str) -> str:
        return "HIT" if self.health(revision)["state"] == "HIT" else "MISS"

    def stage(self, revision: str, source, files: list[dict]) -> str:
        """Copy each missing file once, then publish. A valid publication is a hit."""
        lock = self._lockfile()
        try:
            return self._stage_locked(revision, source, files)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def _stage_locked(self, revision: str, source, files: list[dict]) -> str:
        if self.lookup(revision) == "HIT":
            return "HIT"
        self._check_plan(revision, files)
        if self._published(revision).exists():
            raise CacheError("CACHE_INVALID")
        staging = self._staging(revision)
        staging.mkdir(parents=True, exist_ok=True)
        for item in files:
            relative = self._relative(item["path"])
            target = staging / relative
            expected = int(item["bytes"])
            if target.is_file() and target.stat().st_size == expected:
                continue
            body = source.read(relative)
            if len(body) != expected:
                raise CacheError("PARTIAL_OBJECT")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            temporary.write_bytes(body)
            os.replace(temporary, target)
        manifest = {
            "state": "COMPLETE",
            "repo_id": REPO_ID,
            "revision": revision,
            "expected_file_count": len(files),
            "expected_total_bytes": sum(int(item["bytes"]) for item in files),
            "files": files,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        published = self._published(revision)
        if published.exists():
            raise CacheError("PUBLISH_CONFLICT")
        os.replace(staging, published)
        if self.lookup(revision) != "HIT":
            raise CacheError("PUBLISH_INVALID")
        return "STAGED"

    def invalidate(self, revision: str) -> None:
        published = self._published(revision)
        if published.exists():
            retired = self.root / "revisions" / (revision + ".invalid")
            if retired.exists():
                raise CacheError("INVALID_EXISTS")
            os.replace(published, retired)

    def _check_plan(self, revision: str, files: list[dict]) -> None:
        if self.enforce_pin and revision != self.revision:
            raise CacheError("REVISION_MISMATCH")
        if len(files) != self.expected_files:
            raise CacheError("FILE_COUNT")
        total = sum(int(item["bytes"]) for item in files)
        if total != self.expected_bytes:
            raise CacheError("BYTE_COUNT")

    def _lockfile(self):
        path = self.root / ".cache.lock"
        handle = path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _problem(self, published: Path, manifest: dict | None, revision: str) -> str:
        if manifest is None:
            return "MISSING"
        if manifest.get("state") != "COMPLETE" or manifest.get("revision") != revision:
            return "REVISION_MISMATCH"
        if manifest.get("repo_id") != REPO_ID:
            return "MODEL_MISMATCH"
        files = manifest.get("files")
        if not isinstance(files, list):
            return "MANIFEST_INVALID"
        total = 0
        for item in files:
            if not isinstance(item, dict):
                return "MANIFEST_INVALID"
            try:
                relative = self._relative(str(item.get("path") or ""))
                expected = int(item["bytes"])
            except (CacheError, KeyError, TypeError, ValueError):
                return "MANIFEST_INVALID"
            target = published / relative
            if not target.is_file() or target.stat().st_size != expected:
                return "PARTIAL"
            total += expected
        if total != int(manifest.get("expected_total_bytes") or -1):
            return "BYTE_COUNT"
        return ""

    def _read_manifest(self, published: Path) -> dict | None:
        path = published / "manifest.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"state": "INVALID"}
        return payload if isinstance(payload, dict) else {"state": "INVALID"}

    def _published(self, revision: str) -> Path:
        return self.root / "revisions" / revision / "published"

    def _staging(self, revision: str) -> Path:
        return self.root / "revisions" / revision / "staging"

    def _relative(self, value: str) -> str:
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise CacheError("PATH_UNSAFE")
        return value
