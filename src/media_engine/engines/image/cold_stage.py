"""Bounded Qwen cold stage. Tests use a fake source. This module does not open a network connection."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT, REPO_ID
from media_engine.providers.live7_contract import REVISION

MARKER = ".qwen-stage-complete"
CHUNK = 1024 * 1024


class StageStopped(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def accept_manifest(document: dict, *, enforce_qwen: bool = True) -> None:
    """Refuse a substituted model or a manifest that is not the expected inventory."""
    if not isinstance(document, dict):
        raise StageStopped("CACHE_MANIFEST_INVALID")
    if document.get("repo_id") != REPO_ID or document.get("revision") != REVISION:
        raise StageStopped("MODEL_SUBSTITUTION")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise StageStopped("CACHE_MANIFEST_INVALID")
    total = 0
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise StageStopped("CACHE_MANIFEST_INVALID")
        relative = str(item.get("path") or "")
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts or relative in seen:
            raise StageStopped("CACHE_MANIFEST_INVALID")
        size = item.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise StageStopped("CACHE_MANIFEST_INVALID")
        digest = item.get("sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
            raise StageStopped("CACHE_MANIFEST_INVALID")
        seen.add(relative)
        total += size
    if enforce_qwen and (len(files) != QWEN_FILE_COUNT or total != QWEN_CACHE_MIN_BYTES):
        raise StageStopped("CACHE_INCOMPLETE")


def run_cold_stage(
    manifest: dict,
    dest: Path,
    source,
    clock,
    *,
    transfer_deadline: float,
    spending_deadline: float,
    tail_seconds: float,
    stall_seconds: float = 30.0,
    max_attempts: int = 2,
    enforce_qwen: bool = False,
) -> dict:
    """Copy one manifest. Retries stay inside the same deadlines and never loop without a bound."""
    accept_manifest(manifest, enforce_qwen=enforce_qwen)
    if max_attempts < 1 or max_attempts > 3:
        raise StageStopped("TRANSFER_DEADLINE")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = manifest["files"]
    if _marker_matches(dest, files):
        return _report(files, dest, reads=0, resumed=False, hashes=0, already=True)
    resumed = False
    hashes = 0
    try:
        for item in files:
            relative = str(item["path"])
            final = dest / relative
            part = dest / (relative + ".partial")
            if _final_matches(final, item):
                continue
            _guard(clock, transfer_deadline, spending_deadline, tail_seconds)
            wrote = _one_file(
                item, final, part, source, clock,
                transfer_deadline=transfer_deadline,
                spending_deadline=spending_deadline,
                tail_seconds=tail_seconds,
                stall_seconds=stall_seconds,
                max_attempts=max_attempts,
            )
            resumed = resumed or wrote["resumed"]
            hashes += wrote["hashed"]
        _guard(clock, transfer_deadline, spending_deadline, tail_seconds)
        _write_marker(dest, files)
    except StageStopped:
        _cleanup_partials(dest)
        raise
    return _report(files, dest, reads=getattr(source, "reads", 0), resumed=resumed, hashes=hashes, already=False)


def _one_file(item, final: Path, part: Path, source, clock, **limits) -> dict:
    relative = str(item["path"])
    expected = int(item["bytes"])
    resumed = part.is_file() and part.stat().st_size > 0
    attempts = 0
    while attempts < limits["max_attempts"]:
        attempts += 1
        _guard(clock, limits["transfer_deadline"], limits["spending_deadline"], limits["tail_seconds"])
        part.parent.mkdir(parents=True, exist_ok=True)
        offset = part.stat().st_size if part.is_file() else 0
        if offset > expected:
            part.unlink(missing_ok=True)
            offset = 0
        last = float(clock())
        empty = 0
        try:
            with part.open("ab") as handle:
                while offset < expected:
                    _guard(clock, limits["transfer_deadline"], limits["spending_deadline"], limits["tail_seconds"])
                    chunk = source.read(relative, offset, min(CHUNK, expected - offset))
                    if not isinstance(chunk, (bytes, bytearray)) or not chunk:
                        empty += 1
                        if empty >= 2 or float(clock()) - last >= limits["stall_seconds"]:
                            raise StageStopped("CACHE_STALL")
                        continue
                    empty = 0
                    handle.write(chunk)
                    handle.flush()
                    offset += len(chunk)
                    last = float(clock())
        except KeyError as exc:
            raise StageStopped("CACHE_INCOMPLETE") from exc
        if offset != expected or not _digest_matches(part, item):
            part.unlink(missing_ok=True)
            if attempts >= limits["max_attempts"]:
                raise StageStopped("CACHE_CORRUPT")
            continue
        os.replace(part, final)
        hashed = 1 if item.get("sha256") else 0
        return {"resumed": resumed, "hashed": hashed}
    raise StageStopped("CACHE_CORRUPT")


def _guard(clock, transfer_deadline: float, spending_deadline: float, tail_seconds: float) -> None:
    now = float(clock())
    if now >= float(transfer_deadline):
        raise StageStopped("TRANSFER_DEADLINE")
    if now >= float(spending_deadline) - float(tail_seconds):
        raise StageStopped("BUDGET_STOP")


def _digest_matches(path: Path, item: dict) -> bool:
    if path.stat().st_size != int(item["bytes"]):
        return False
    digest = item.get("sha256")
    if not digest:
        return True
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest() == digest


def _final_matches(path: Path, item: dict) -> bool:
    return path.is_file() and _digest_matches(path, item)


def _marker_matches(dest: Path, files: list) -> bool:
    marker = dest / MARKER
    if not marker.is_file():
        return False
    return all(_final_matches(dest / str(item["path"]), item) for item in files)


def _write_marker(dest: Path, files: list) -> None:
    body = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "files": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
    }
    temporary = dest / (MARKER + ".partial")
    temporary.write_text(json_dumps(body), encoding="utf-8")
    os.replace(temporary, dest / MARKER)


def json_dumps(body: dict) -> str:
    import json
    return json.dumps(body, sort_keys=True)


def _cleanup_partials(dest: Path) -> None:
    if not dest.exists():
        return
    for path in dest.rglob("*.partial"):
        path.unlink(missing_ok=True)


def _report(files, dest: Path, *, reads: int, resumed: bool, hashes: int, already: bool) -> dict:
    present = 0
    total = 0
    for item in files:
        path = dest / str(item["path"])
        if _final_matches(path, item):
            present += 1
            total += path.stat().st_size
    return {
        "status": "COMPLETE",
        "files": present,
        "bytes": total,
        "hashes_checked": hashes,
        "resumed": resumed,
        "already_complete": already,
        "source_reads": reads,
        "marker": (dest / MARKER).is_file(),
    }
