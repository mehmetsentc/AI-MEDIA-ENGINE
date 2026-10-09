"""Resumable transfer helpers. Disabled unless MEDIA_ENGINE_R2_RELIABLE_TRANSFER=1.

The default cache copy is unchanged. These helpers are covered by simulated
streams. They do not download the Qwen weights.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from media_engine.engines.image.r2_cache import R2StorageError

RETRYABLE = frozenset({0, 408, 429, 500, 502, 503, 504})
AUTH_FAILURES = frozenset({401, 403})


def reliable_transfer_enabled() -> bool:
    return os.environ.get("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", "").strip() == "1"


class ProgressBook:
    """In-flight bytes replace the previous sample for that object. Retries do not add."""

    def __init__(self) -> None:
        self.received: dict[str, int] = {}
        self.validated: dict[str, int] = {}
        self.completed: set[str] = set()

    def note_received(self, key: str, nbytes: int) -> None:
        if key in self.completed:
            return
        self.received[key] = max(0, int(nbytes))

    def note_completed(self, key: str, nbytes: int) -> None:
        self.validated[key] = max(0, int(nbytes))
        self.completed.add(key)
        self.received.pop(key, None)

    def snapshot(self, total: int) -> dict:
        network = sum(self.validated.values()) + sum(self.received.values())
        validated = sum(self.validated.values())
        return {
            "network_bytes": network,
            "validated_bytes": validated,
            "completed_objects": len(self.completed),
            "remaining_bytes": max(0, int(total) - validated),
            "bytes_done": network,
            "bytes_total": int(total),
        }


def resume_plan(partial_bytes: int, object_size: int, *, accepts_range: bool) -> dict:
    """Decide whether a partial file may continue. A full 200 response must not be appended."""
    if partial_bytes < 0 or object_size < 0:
        return {"mode": "replace", "start": 0}
    if partial_bytes == object_size and object_size > 0:
        return {"mode": "already_complete", "start": partial_bytes}
    if partial_bytes <= 0 or not accepts_range:
        return {"mode": "replace", "start": 0}
    if partial_bytes > object_size:
        return {"mode": "replace", "start": 0}
    return {"mode": "resume", "start": partial_bytes, "header": f"bytes={partial_bytes}-"}


def accept_range_body(status: int, content_range: str, *, start: int, total: int) -> str:
    """Return append, replace, or reject. A 200 body is a full object and must replace."""
    if status == 200:
        return "replace"
    if status != 206:
        return "reject"
    text = content_range.strip().lower()
    prefix = f"bytes {start}-"
    if not text.startswith(prefix):
        return "reject"
    try:
        range_part, size_part = text.split("/", 1)
        end = int(range_part.split("-", 1)[1])
        declared = int(size_part)
    except (ValueError, IndexError):
        return "reject"
    if declared != total or end < start:
        return "reject"
    return "append"


def retry_delay(status: int, attempt: int, *, max_attempts: int = 4) -> float | None:
    """Seconds to wait, or None when the failure must stop. Auth failures are not retried."""
    if status in AUTH_FAILURES or attempt >= max_attempts:
        return None
    if status not in RETRYABLE:
        return None
    return min(8.0, 0.5 * (2 ** attempt))


def promote_complete(partial: Path, destination: Path, expected_size: int) -> None:
    """Publish a finished object only when its size matches. The partial stays otherwise."""
    if not partial.is_file() or partial.stat().st_size != expected_size:
        raise R2StorageError(0)
    os.replace(partial, destination)


def download_with_resume(
    client,
    key: str,
    destination: Path,
    *,
    object_size: int,
    accepts_range: bool = True,
    should_stop=None,
    on_received=None,
    max_attempts: int = 4,
    sleep=time.sleep,
) -> dict[str, str]:
    """Stream one object. Partial data is appended only after a matching 206 response."""
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        if should_stop is not None and should_stop():
            raise TimeoutError("CACHE_SYNC_DEADLINE")
        held = partial.stat().st_size if partial.is_file() else 0
        plan = resume_plan(held, object_size, accepts_range=accepts_range)
        if plan["mode"] == "already_complete":
            promote_complete(partial, destination, object_size)
            return {}
        if plan["mode"] == "replace" and partial.exists():
            partial.unlink()
            held = 0
        incoming = destination.with_name(destination.name + ".incoming")
        extra = {"range": plan["header"]} if plan["mode"] == "resume" else None
        received = held

        def on_chunk(n: int) -> None:
            nonlocal received
            received += n
            if on_received is not None:
                on_received(received)

        try:
            status, headers, _body = client._request(
                "GET", key, b"", extra, stream_to=incoming, should_stop=should_stop,
                stream_mode="wb", on_chunk=on_chunk,
            )
        except R2StorageError as exc:
            incoming.unlink(missing_ok=True)
            delay = retry_delay(exc.status, attempt, max_attempts=max_attempts)
            if delay is None:
                raise
            attempt += 1
            sleep(delay)
            continue
        content_range = ""
        for name, value in headers.items():
            if name.lower() == "content-range":
                content_range = value
        if plan["mode"] == "resume":
            decision = accept_range_body(status, content_range, start=held, total=object_size)
        elif status == 200:
            decision = "replace"
        else:
            decision = "reject"
        if decision == "reject":
            incoming.unlink(missing_ok=True)
            delay = retry_delay(status, attempt, max_attempts=max_attempts)
            if delay is None:
                raise R2StorageError(status or 0)
            attempt += 1
            sleep(delay)
            continue
        if decision == "replace":
            partial.unlink(missing_ok=True)
            os.replace(incoming, partial)
        else:
            with partial.open("ab") as target, incoming.open("rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            incoming.unlink(missing_ok=True)
        if partial.is_file() and partial.stat().st_size == object_size:
            promote_complete(partial, destination, object_size)
            if on_received is not None:
                on_received(object_size)
            return {}
        delay = retry_delay(0, attempt, max_attempts=max_attempts)
        if delay is None:
            raise R2StorageError(0)
        attempt += 1
        sleep(delay)
