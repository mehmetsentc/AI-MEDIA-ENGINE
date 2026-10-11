"""Resumable speech-model staging. This module does not download weights.

An R2 transfer runs only when ``allow_weight_transfer`` is true. The default
refuses before any object body is requested. The Qwen image prefix and volume
54653022 are not writable from here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from media_engine.infra.volume_audit import VOLUME_ID
from media_engine.models.lifecycle import SCRATCH_BYTES
from media_engine.models.speech_inventory import TARGETS

CHUNK = 1024 * 1024
SPEECH_PREFIX = "speech"
IMAGE_MANIFEST = "phase2d-cache.json"
IMAGE_PREFIX = "hf-cache/"


class SpeechPrepError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def targets() -> tuple[dict, ...]:
    return TARGETS


def target(model_id: str) -> dict:
    for item in TARGETS:
        if item["model_id"] == model_id:
            return item
    raise SpeechPrepError("MODEL_UNKNOWN")


def file_bytes(model_id: str) -> int:
    return sum(int(item["bytes"]) for item in target(model_id)["files"])


def staged_disk_bytes() -> int:
    """Both trees on disk, plus scratch. Only one model is resident in memory."""
    return sum(file_bytes(item["model_id"]) for item in TARGETS) + SCRATCH_BYTES


def object_key(model_id: str, revision: str, relative: str) -> str:
    path = _relative(relative)
    if revision != target(model_id)["revision"]:
        raise SpeechPrepError("REVISION_MISMATCH")
    key = f"{SPEECH_PREFIX}/{model_id}/{revision}/{path.as_posix()}"
    if key == IMAGE_MANIFEST or key.startswith(IMAGE_PREFIX) or "phase2d-cache.json" in key:
        raise SpeechPrepError("PATH_INVALID")
    return key


def verify_tree(dest: Path, model_id: str) -> dict:
    """Compare a local directory with the pinned inventory. No network."""
    spec = target(model_id)
    return verify_files(dest, revision=spec["revision"], files=spec["files"])


def verify_files(dest: Path, *, revision: str, files) -> dict:
    dest = Path(dest)
    missing: list[str] = []
    truncated: list[str] = []
    hash_mismatches: list[str] = []
    present = 0
    total = 0
    expected = 0
    for item in files:
        expected += int(item["bytes"])
        path = dest / _relative(item["path"])
        if not path.is_file():
            missing.append(item["path"])
            continue
        size = path.stat().st_size
        total += size
        if size != int(item["bytes"]):
            truncated.append(item["path"])
            continue
        if _sha256(path) != item["sha256"]:
            hash_mismatches.append(item["path"])
            continue
        present += 1
    found = ""
    marker = dest / "REVISION"
    if marker.is_file():
        found = marker.read_text(encoding="utf-8").strip()
    complete = present == len(tuple(files)) and not missing and not truncated and not hash_mismatches
    if found and found != revision:
        status = "REVISION_MISMATCH"
        complete = False
    elif complete:
        status = "COMPLETE"
    elif present or truncated or hash_mismatches:
        status = "INCOMPLETE"
    else:
        status = "ABSENT"
    return {
        "status": status,
        "revision": found or revision,
        "expected_revision": revision,
        "files_expected": len(tuple(files)),
        "files_present": present,
        "missing": missing,
        "truncated": truncated,
        "hash_mismatches": hash_mismatches,
        "bytes_present": total,
        "bytes_expected": expected,
        "complete": complete and status == "COMPLETE",
    }


def remote_gap(model_id: str, entries: list[tuple[str, int]] | None) -> dict:
    """Compare a key listing with the inventory. Entries are relative paths and sizes."""
    spec = target(model_id)
    if entries is None:
        return {
            "status": "NOT_OBSERVED",
            "missing": [item["path"] for item in spec["files"]],
            "download_bytes": file_bytes(model_id),
        }
    sizes = {path: size for path, size in entries}
    missing = []
    download = 0
    for item in spec["files"]:
        size = sizes.get(item["path"])
        if size != int(item["bytes"]):
            missing.append(item["path"])
            download += int(item["bytes"])
    return {
        "status": "COMPLETE" if not missing else "INCOMPLETE" if entries else "ABSENT",
        "missing": missing,
        "download_bytes": download,
    }


class ResidentGate:
    """One loaded model. A busy model is not unloaded or replaced."""

    def __init__(self) -> None:
        self.model_id: str | None = None
        self.busy = False

    def plan(self, model_id: str) -> dict:
        target(model_id)
        if self.busy and self.model_id != model_id:
            return {"action": "REFUSE", "error": "MODEL_BUSY", "resident": self.model_id}
        if self.model_id == model_id:
            return {"action": "REUSE", "model_id": model_id}
        if self.model_id:
            return {"action": "UNLOAD_THEN_LOAD", "unload": self.model_id, "load": model_id}
        return {"action": "LOAD", "model_id": model_id}

    def unload(self) -> str | None:
        if self.busy:
            raise SpeechPrepError("MODEL_BUSY")
        previous = self.model_id
        self.model_id = None
        return previous

    def begin(self, model_id: str) -> str:
        decision = self.plan(model_id)
        if decision["action"] == "REFUSE":
            raise SpeechPrepError("MODEL_BUSY")
        if decision["action"] == "UNLOAD_THEN_LOAD":
            self.unload()
        if decision["action"] == "REUSE":
            return "REUSE"
        self.model_id = model_id
        self.busy = True
        return "LOAD"

    def finish(self) -> None:
        self.busy = False


def stage_model(model_id: str, dest: Path, source, **kwargs) -> dict:
    """Copy the pinned tree for one speech model."""
    return stage_tree(target(model_id), dest, source, **kwargs)


def stage_tree(
    spec: dict,
    dest: Path,
    source,
    *,
    free_disk_bytes: int,
    clock,
    volume_id: int | None = None,
    transfer_deadline: float | None = None,
    spending_deadline: float | None = None,
    tail_seconds: float = 0,
    max_attempts: int = 2,
    chunk_bytes: int = CHUNK,
) -> dict:
    """Copy one file list. Partials remain after a deadline so the next call resumes."""
    _refuse_volume(volume_id)
    dest = Path(dest)
    if max_attempts < 1 or max_attempts > 3:
        raise SpeechPrepError("TRANSFER_DEADLINE")
    started = float(clock())
    _guard(clock, transfer_deadline, spending_deadline, tail_seconds)
    existing = verify_files(dest, revision=spec["revision"], files=spec["files"])
    if existing["complete"]:
        return _report(spec, dest, started, clock, resumed=False, hashes=0, already=True)
    _require_disk(dest, spec["files"], free_disk_bytes)
    dest.mkdir(parents=True, exist_ok=True)
    resumed = False
    hashes = 0
    for item in spec["files"]:
        relative = item["path"]
        final = dest / _relative(relative)
        partial = Path(str(final) + ".partial")
        if _file_matches(final, item):
            continue
        _guard(clock, transfer_deadline, spending_deadline, tail_seconds)
        wrote = _one_file(
            item, final, partial, source, clock,
            transfer_deadline=transfer_deadline,
            spending_deadline=spending_deadline,
            tail_seconds=tail_seconds,
            max_attempts=max_attempts,
            chunk_bytes=chunk_bytes,
        )
        resumed = resumed or wrote["resumed"]
        hashes += 1
    _guard(clock, transfer_deadline, spending_deadline, tail_seconds)
    _write_marker(dest, spec)
    checked = verify_files(dest, revision=spec["revision"], files=spec["files"])
    if not checked["complete"]:
        raise SpeechPrepError("CACHE_INCOMPLETE")
    return _report(spec, dest, started, clock, resumed=resumed, hashes=hashes, already=False)


def load_model(gate: ResidentGate, model_id: str, loader, clock) -> dict:
    """Time one injected loader. A second call for the same model does not load again."""
    action = gate.begin(model_id)
    if action == "REUSE":
        return {"action": "REUSE", "model_id": model_id, "load_seconds": 0.0}
    started = float(clock())
    try:
        loader()
    except Exception:
        gate.busy = False
        gate.model_id = None
        raise
    seconds = float(clock()) - started
    gate.finish()
    return {"action": "LOAD", "model_id": model_id, "load_seconds": seconds}


def stage_from_r2(client, model_id: str, dest: Path, **kwargs) -> dict:
    """Real R2 range reads. Refused until a caller sets allow_weight_transfer."""
    if kwargs.pop("allow_weight_transfer", False) is not True:
        raise SpeechPrepError("WEIGHTS_NOT_APPROVED")
    spec = target(model_id)
    source = _R2RangeSource(client, spec["model_id"], spec["revision"])
    return stage_model(model_id, dest, source, **kwargs)


class _R2RangeSource:
    def __init__(self, client, model_id: str, revision: str) -> None:
        self.client = client
        self.model_id = model_id
        self.revision = revision

    def read(self, relative: str, offset: int, length: int) -> bytes:
        key = object_key(self.model_id, self.revision, relative)
        end = offset + length - 1
        status, _headers, body = self.client._request(
            "GET", key, b"", {"range": f"bytes={offset}-{end}"},
        )
        if offset and status != 206:
            raise SpeechPrepError("RANGE_REQUIRED")
        if status not in (200, 206):
            raise SpeechPrepError("CACHE_INCOMPLETE")
        if not isinstance(body, (bytes, bytearray)) or not body:
            raise SpeechPrepError("CACHE_INCOMPLETE")
        return bytes(body[:length])


def _one_file(item, final: Path, partial: Path, source, clock, *, chunk_bytes: int = CHUNK, **limits) -> dict:
    expected = int(item["bytes"])
    resumed = partial.is_file() and 0 < partial.stat().st_size <= expected
    attempts = 0
    while attempts < limits["max_attempts"]:
        attempts += 1
        _guard(clock, limits["transfer_deadline"], limits["spending_deadline"], limits["tail_seconds"])
        partial.parent.mkdir(parents=True, exist_ok=True)
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > expected:
            partial.unlink(missing_ok=True)
            offset = 0
            resumed = False
        try:
            with partial.open("ab") as handle:
                while offset < expected:
                    _guard(
                        clock, limits["transfer_deadline"], limits["spending_deadline"], limits["tail_seconds"],
                    )
                    chunk = source.read(item["path"], offset, min(chunk_bytes, expected - offset))
                    if not isinstance(chunk, (bytes, bytearray)) or not chunk:
                        raise SpeechPrepError("CACHE_INCOMPLETE")
                    if offset + len(chunk) > expected:
                        raise SpeechPrepError("CACHE_CORRUPT")
                    handle.write(chunk)
                    handle.flush()
                    offset += len(chunk)
        except SpeechPrepError as exc:
            if exc.code in {"BUDGET_STOP", "TRANSFER_DEADLINE"}:
                raise
            partial.unlink(missing_ok=True)
            if attempts >= limits["max_attempts"]:
                raise
            continue
        if offset != expected or _sha256(partial) != item["sha256"]:
            partial.unlink(missing_ok=True)
            if attempts >= limits["max_attempts"]:
                raise SpeechPrepError("CACHE_CORRUPT")
            continue
        os.replace(partial, final)
        return {"resumed": resumed}
    raise SpeechPrepError("CACHE_CORRUPT")


def _require_disk(dest: Path, files: tuple, free_disk_bytes: int) -> None:
    present = 0
    total = 0
    for item in files:
        total += int(item["bytes"])
        path = dest / _relative(item["path"])
        if path.is_file() and path.stat().st_size == int(item["bytes"]):
            present += int(item["bytes"])
    remaining = total - present
    if int(free_disk_bytes) < remaining + SCRATCH_BYTES:
        raise SpeechPrepError("DISK_CAPACITY")


def _guard(clock, transfer_deadline, spending_deadline, tail_seconds: float) -> None:
    now = float(clock())
    if transfer_deadline is not None and now >= float(transfer_deadline):
        raise SpeechPrepError("TRANSFER_DEADLINE")
    if spending_deadline is not None and now >= float(spending_deadline) - float(tail_seconds):
        raise SpeechPrepError("BUDGET_STOP")


def _write_marker(dest: Path, spec: dict) -> None:
    body = {
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "files": [
            {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
            for item in spec["files"]
        ],
        "bytes": sum(int(item["bytes"]) for item in spec["files"]),
    }
    temporary = dest / "manifest.json.partial"
    temporary.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    os.replace(temporary, dest / "manifest.json")
    revision = dest / "REVISION.partial"
    revision.write_text(spec["revision"] + "\n", encoding="utf-8")
    os.replace(revision, dest / "REVISION")


def _report(spec, dest: Path, started: float, clock, *, resumed: bool, hashes: int, already: bool) -> dict:
    checked = verify_files(dest, revision=spec["revision"], files=spec["files"])
    return {
        "status": "COMPLETE",
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "bytes": checked["bytes_present"],
        "hashes_checked": hashes if not already else checked["files_present"],
        "resumed": resumed,
        "already_complete": already,
        "stage_seconds": float(clock()) - started,
    }


def _file_matches(path: Path, item: dict) -> bool:
    return path.is_file() and path.stat().st_size == int(item["bytes"]) and _sha256(path) == item["sha256"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise SpeechPrepError("PATH_INVALID")
    return path


def _refuse_volume(volume_id: int | None) -> None:
    if volume_id is None:
        return
    if int(volume_id) == VOLUME_ID:
        raise SpeechPrepError("VOLUME_FORBIDDEN")
