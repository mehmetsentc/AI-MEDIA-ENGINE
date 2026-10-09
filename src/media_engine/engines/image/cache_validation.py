"""Offline Qwen cache check. It never downloads, writes, or marks a partial cache complete."""
from __future__ import annotations

import json
from pathlib import Path

from media_engine.engines.image.model_cache import (
    MANIFEST_NAME,
    QWEN_CACHE_MIN_BYTES,
    QWEN_FILE_COUNT,
    REPO_ID,
    hf_cache,
    snapshot_dir,
)

REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
CACHE_COMPLETE = "CACHE_COMPLETE"
CACHE_INCOMPLETE = "CACHE_INCOMPLETE"
CACHE_REVISION_MISMATCH = "CACHE_REVISION_MISMATCH"
CACHE_MANIFEST_INVALID = "CACHE_MANIFEST_INVALID"
CACHE_PATH_MISSING = "CACHE_PATH_MISSING"
CACHE_UNVERIFIED = "CACHE_UNVERIFIED"


def validate_cache(mount: Path, expected: dict | None = None) -> dict:
    """Compare a local mount with an already loaded manifest.

    `expected` is the verified manifest document. This function does not fetch it.
    """
    mount = Path(mount)
    report = {
        "status": CACHE_PATH_MISSING,
        "revision": "",
        "expected_revision": REVISION,
        "files_expected": 0,
        "files_present": 0,
        "missing": [],
        "truncated": [],
        "hash_mismatches": [],
        "hashes_checked": 0,
        "bytes_present": 0,
        "complete": False,
        "mutated": False,
    }
    if not mount.exists() or not mount.is_dir():
        return report
    document = expected if expected is not None else _read_local(mount)
    if document is None:
        report["status"] = CACHE_UNVERIFIED
        return report
    if not isinstance(document, dict):
        report["status"] = CACHE_MANIFEST_INVALID
        return report
    revision = str(document.get("revision") or "")
    report["revision"] = revision
    if revision and revision != REVISION:
        report["status"] = CACHE_REVISION_MISMATCH
        return report
    files = document.get("files")
    if (
        document.get("repo_id") not in (None, REPO_ID)
        or document.get("state") not in (None, "COMPLETE")
        or not revision
        or not isinstance(files, list)
        or not files
    ):
        report["status"] = CACHE_MANIFEST_INVALID
        return report
    report["files_expected"] = len(files)
    for item in files:
        if not isinstance(item, dict):
            report["status"] = CACHE_MANIFEST_INVALID
            return report
        relative = str(item.get("path") or "")
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts:
            report["status"] = CACHE_MANIFEST_INVALID
            return report
        expected_size = item.get("bytes")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            report["status"] = CACHE_MANIFEST_INVALID
            return report
    declared_count = document.get("expected_file_count")
    if declared_count is not None and declared_count != len(files):
        report["status"] = CACHE_MANIFEST_INVALID
        return report
    snapshot = snapshot_dir(mount, revision)
    if not snapshot.is_dir():
        report["status"] = CACHE_INCOMPLETE
        report["missing"] = [str(item.get("path") or "") for item in files if isinstance(item, dict)]
        return report
    present = 0
    total = 0
    missing: list[str] = []
    truncated: list[str] = []
    hash_mismatches: list[str] = []
    hashes = 0
    for item in files:
        if not isinstance(item, dict):
            report["status"] = CACHE_MANIFEST_INVALID
            return report
        relative = str(item.get("path") or "")
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts:
            report["status"] = CACHE_MANIFEST_INVALID
            return report
        target = snapshot / relative
        expected_size = item.get("bytes")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            report["status"] = CACHE_MANIFEST_INVALID
            return report
        if not target.exists():
            missing.append(relative)
            continue
        size = target.stat().st_size
        total += size
        if size != expected_size:
            truncated.append(relative)
            continue
        present += 1
        digest = item.get("sha256") or item.get("blob_sha256")
        if isinstance(digest, str) and digest:
            hashes += 1
            link = ""
            if target.is_symlink():
                link = target.readlink().as_posix()
            if digest not in link:
                hash_mismatches.append(relative)
    report["files_present"] = present
    report["bytes_present"] = total
    report["missing"] = missing
    report["truncated"] = truncated
    report["hash_mismatches"] = hash_mismatches
    report["hashes_checked"] = hashes
    if missing or truncated or hash_mismatches or present != len(files):
        report["status"] = CACHE_INCOMPLETE
        return report
    declared_count = document.get("expected_file_count")
    declared_bytes = document.get("expected_total_bytes")
    if declared_count is not None and declared_count != len(files):
        report["status"] = CACHE_MANIFEST_INVALID
        return report
    if declared_bytes is not None and declared_bytes != total:
        report["status"] = CACHE_INCOMPLETE
        return report
    report["status"] = CACHE_COMPLETE
    report["complete"] = True
    return report


def qwen_contract(manifest: dict) -> bool:
    """True when a manifest document matches the known Qwen revision and size."""
    files = manifest.get("files")
    return (
        isinstance(manifest, dict)
        and manifest.get("state") == "COMPLETE"
        and manifest.get("revision") == REVISION
        and manifest.get("expected_file_count") == QWEN_FILE_COUNT
        and manifest.get("expected_total_bytes") == QWEN_CACHE_MIN_BYTES
        and isinstance(files, list)
        and len(files) == QWEN_FILE_COUNT
    )


def _read_local(mount: Path) -> dict | None:
    path = mount / MANIFEST_NAME
    if not path.is_file():
        cache = hf_cache(mount)
        if not cache.exists():
            return None
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "INVALID"}
    return payload if isinstance(payload, dict) else {"state": "INVALID"}
