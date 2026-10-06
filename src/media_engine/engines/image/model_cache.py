"""Persistent Qwen cache contract. This module does not create Vast volumes."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ID = "Qwen/Qwen-Image"
MOUNT_PATH = "/models"
HF_CACHE_DIRNAME = "hf-cache"
MANIFEST_NAME = "phase2d-cache.json"
# Smallest of 100/120/150 GB that covers a finished 53.74 GiB snapshot
# plus parallel incomplete shards and the hf-xet chunk cache.
VOLUME_SIZE_GB = 120
# Sum of the public Qwen/Qwen-Image tree on 2026-10-06. Not a checksum.
QWEN_CACHE_MIN_BYTES = 57_704_594_653
QWEN_FILE_COUNT = 31
EMPTY = "EMPTY"
FILLING = "FILLING"
COMPLETE = "COMPLETE"
INVALID = "INVALID"
GENERATE = "PHASE2D_GENERATE"
FILL = "MODEL_CACHE_FILL"


class ModelCacheNotReady(Exception):
    def __init__(self, state: str) -> None:
        super().__init__("MODEL_CACHE_NOT_READY")
        self.code = "MODEL_CACHE_NOT_READY"
        self.state = state


def hf_cache(mount: Path) -> Path:
    return mount / HF_CACHE_DIRNAME


def manifest_path(mount: Path) -> Path:
    return mount / MANIFEST_NAME


def repo_dirname(repo_id: str = REPO_ID) -> str:
    return "models--" + repo_id.replace("/", "--")


def snapshot_dir(mount: Path, revision: str, repo_id: str = REPO_ID) -> Path:
    return hf_cache(mount) / repo_dirname(repo_id) / "snapshots" / revision


def write_manifest(mount: Path, payload: dict) -> None:
    mount.mkdir(parents=True, exist_ok=True)
    manifest_path(mount).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_manifest(mount: Path) -> dict | None:
    path = manifest_path(mount)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"state": INVALID}
    if not isinstance(payload, dict):
        return {"state": INVALID}
    return payload


def mark_filling(mount: Path) -> None:
    write_manifest(mount, {"state": FILLING, "repo_id": REPO_ID, "validation_status": FILLING})


def mark_complete(mount: Path, *, revision: str, files: list[dict], completed_at: str) -> None:
    total = sum(int(item["bytes"]) for item in files)
    write_manifest(mount, {
        "state": COMPLETE,
        "repo_id": REPO_ID,
        "revision": revision,
        "expected_file_count": len(files),
        "expected_total_bytes": total,
        "completed_at": completed_at,
        "validation_status": COMPLETE,
        "files": files,
    })


def assess_cache(mount: Path, *, min_bytes: int = QWEN_CACHE_MIN_BYTES) -> str:
    """Cheap check: manifest, snapshot path, symlink targets, and byte total."""
    manifest = read_manifest(mount)
    if manifest is None:
        return EMPTY
    state = str(manifest.get("state") or "")
    if state in {EMPTY, FILLING, INVALID}:
        return state
    if state != COMPLETE:
        return INVALID
    if manifest.get("repo_id") != REPO_ID:
        return INVALID
    revision = str(manifest.get("revision") or "")
    files = manifest.get("files")
    if not revision or not isinstance(files, list) or not files:
        return INVALID
    snapshot = snapshot_dir(mount, revision)
    if not snapshot.is_dir():
        return INVALID
    total = 0
    for entry in files:
        if not isinstance(entry, dict):
            return INVALID
        relative = str(entry.get("path") or "")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            return INVALID
        target = snapshot / relative
        if not target.exists():
            return INVALID
        size = target.stat().st_size
        expected = entry.get("bytes")
        if isinstance(expected, int) and not isinstance(expected, bool) and size != expected:
            return INVALID
        total += size
    expected_count = manifest.get("expected_file_count")
    if expected_count is not None and expected_count != len(files):
        return INVALID
    if total < min_bytes:
        return INVALID
    return COMPLETE


def prepare_generation(mount: Path, download, *, min_bytes: int = QWEN_CACHE_MIN_BYTES) -> str:
    """Load from the persistent cache. Never starts a remote snapshot download."""
    state = assess_cache(mount, min_bytes=min_bytes)
    if state != COMPLETE:
        raise ModelCacheNotReady(state)
    return download(REPO_ID, cache_dir=str(hf_cache(mount)), local_files_only=True)


def fill_model_cache(mount: Path, download, *, revision: str, files: list[dict], completed_at: str) -> str:
    """Separate from generation. The caller supplies the downloader."""
    mark_filling(mount)
    local = download(REPO_ID, cache_dir=str(hf_cache(mount)), local_files_only=False)
    mark_complete(mount, revision=revision, files=files, completed_at=completed_at)
    return local
