"""Heartbeat for the live GPU session. The supervising thread writes it.

A detached timer is intentionally absent. If the controller process or the
session call stops, this file stops changing and the watchdog's existing
300-second stale limit deletes the GPU. A fresh file does not move the
spending deadline; the watchdog still owns that decision.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HEARTBEAT_INTERVAL_SECONDS = 20.0


class HeartbeatOwnershipError(Exception):
    """The file belongs to a different execution or instance."""


class HeartbeatReadError(Exception):
    """The file could not be read. Leave it unchanged so the watchdog still applies."""


def refresh_heartbeat(
    path: Path,
    *,
    execution_id: str,
    instance_id: str,
    now: float,
    previous_at: float | None,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> float | None:
    """Replace the heartbeat, or return None when this interval already has one.

    The replacement is atomic. The payload names the execution and instance
    that are allowed to refresh it. Another pair is refused.
    """
    if previous_at is not None and float(now) - float(previous_at) < float(interval):
        return None
    path = Path(path)
    if path.is_file():
        _require_owner(path, execution_id, instance_id)
    payload = json.dumps(
        {
            "at": float(now),
            "execution_id": str(execution_id),
            "instance_id": str(instance_id),
            "pid": os.getpid(),
        },
        sort_keys=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return float(now)


def _require_owner(path: Path, execution_id: str, instance_id: str) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HeartbeatReadError(type(exc).__name__) from exc
    try:
        current = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HeartbeatReadError("JSON") from exc
    if not isinstance(current, dict):
        raise HeartbeatOwnershipError("HEARTBEAT_OWNERSHIP")
    if str(current.get("execution_id") or "") != str(execution_id):
        raise HeartbeatOwnershipError("HEARTBEAT_OWNERSHIP")
    if str(current.get("instance_id") or "") != str(instance_id):
        raise HeartbeatOwnershipError("HEARTBEAT_OWNERSHIP")
