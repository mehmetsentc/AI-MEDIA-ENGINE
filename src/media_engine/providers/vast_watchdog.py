"""Detached Vast termination. This module never creates an instance or a volume."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from media_engine.providers.vast_lifecycle import (
    InstanceListError,
    collect_instance_pages,
    list_failure_code,
)

LABEL_PREFIX = "media-engine:"
HEARTBEAT_STALE_SECONDS = 300.0
_INSTANCE_ID = re.compile(r"^[0-9]+$")
# Destroy stays on the documented per-instance route. Only the collection route moved to v1.
DELETE_URL = "https://console.vast.ai/api/v0/instances/%s/"


def matching_instances(rows, machine_id: int) -> list[str]:
    found: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            machine = int(row.get("machine_id") or 0)
        except (TypeError, ValueError):
            continue
        if machine != int(machine_id):
            continue
        if not str(row.get("label") or "").startswith(LABEL_PREFIX):
            continue
        identity = str(row.get("id") or "")
        if identity:
            found.append(identity)
    return found


def instances_to_stop(
    rows,
    *,
    machine_id: int,
    now: float,
    deadline: float | None,
    heartbeat_age: float | None,
    stale_after: float = HEARTBEAT_STALE_SECONDS,
) -> list[str]:
    """Instance ids to delete. Two owned instances on the machine are both stopped."""
    owned = matching_instances(rows, machine_id)
    if len(owned) > 1:
        return owned
    if len(owned) != 1:
        return []
    if deadline is not None and now >= deadline:
        return owned
    if heartbeat_age is not None and heartbeat_age > stale_after:
        return owned
    return []


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def arm_watchdog(spec_path: Path, *, popen=None) -> int:
    """Start one detached watchdog. A live lock is left in place."""
    spec_path = Path(spec_path)
    lock = spec_path.with_name(spec_path.name + ".lock")
    if lock.is_file():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = 0
        if _alive(pid):
            return pid
    launcher = popen or subprocess.Popen
    proc = launcher(
        [sys.executable, "-m", "media_engine.providers.vast_watchdog", str(spec_path)],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(proc.pid), encoding="utf-8")
    if hasattr(proc, "wait"):
        threading.Thread(target=_reap_when_done, args=(proc,), daemon=True).start()
    return int(proc.pid)


def _reap_when_done(proc) -> None:
    try:
        proc.wait()
    except Exception:
        return


def _request(key: str, method: str, url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=b"" if method == "DELETE" else None,
        method=method,
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def list_instances(key: str) -> list[dict]:
    """Authenticated v1 inventory. HTTP failures, timeouts, and bad pages raise."""

    def request(url: str) -> tuple[int, bytes]:
        try:
            status, raw = _request(key, "GET", url)
        except TimeoutError as exc:
            raise InstanceListError("LIST_TIMEOUT", 0) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            code = "LIST_TIMEOUT" if isinstance(reason, TimeoutError) else "LIST_NETWORK"
            raise InstanceListError(code, 0) from exc
        if status != 200:
            raise InstanceListError(list_failure_code(status), status)
        return status, raw

    return collect_instance_pages(request)


def _fetch_instances(key: str) -> list:
    return list_instances(key)


def _delete_instance(key: str, instance_id: str) -> int:
    if _INSTANCE_ID.fullmatch(str(instance_id)) is None:
        return 400
    status, _raw = _request(key, "DELETE", DELETE_URL % instance_id)
    return status


def persisted_instance_id(spec: dict) -> str:
    """The one id the parent recorded. Anything else is ignored."""
    path = spec.get("instance_file")
    if not path:
        return ""
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if _INSTANCE_ID.fullmatch(value) is None:
        return ""
    return value


def _cleanup_ids(known: list[str], persisted: str, foreign: set[str]) -> list[str]:
    chosen: list[str] = []
    for instance_id in known:
        if instance_id and instance_id not in chosen:
            chosen.append(instance_id)
    if persisted and persisted not in foreign and persisted not in chosen:
        chosen.append(persisted)
    return chosen


def _finish(stopped: list[str], deadline, failures: list[dict], confirmed: list[str], error: str = "") -> dict:
    result = {
        "stopped": stopped,
        "deadline": deadline,
        "termination_failures": failures,
        "confirmed_absent": confirmed,
    }
    if error:
        result["error"] = error
    return result


def _absent(rows: list[dict], instance_id: str) -> bool:
    return all(str(row.get("id") or "") != str(instance_id) for row in rows if isinstance(row, dict))


def run_until_stopped(spec: dict, *, fetch, delete, now, sleep) -> dict:
    """One control loop used by the detached process and by tests.

    A failed inventory is not an empty fleet. Cleanup then deletes only ids
    already matched on a good list, plus the single id the parent persisted.
    """
    machine_id = int(spec["machine_id"])
    safe_seconds = float(spec["safe_seconds"])
    heartbeat_path = Path(spec["heartbeat"])
    stop_path = Path(spec["stop"])
    stale_after = float(spec.get("stale_after") or HEARTBEAT_STALE_SECONDS)
    seen: float | None = None
    deadline: float | None = None
    armed: float | None = None
    known: list[str] = []
    foreign: set[str] = set()
    locked = spec.get("spending_deadline_wall")
    while True:
        current = float(now())
        if armed is None:
            armed = current
        persisted = persisted_instance_id(spec)
        list_error = ""
        try:
            rows = fetch()
        except InstanceListError as exc:
            rows = None
            list_error = exc.code
        if locked is not None:
            deadline = float(locked)
            window_end = deadline
        else:
            absolute = spec.get("deadline_epoch")
            window_end = float(absolute) if absolute is not None else armed + safe_seconds
        if rows is not None:
            owned = matching_instances(rows, machine_id)
            for instance_id in owned:
                if instance_id not in known:
                    known.append(instance_id)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                identity = str(row.get("id") or "")
                label = str(row.get("label") or "")
                if identity and not label.startswith(LABEL_PREFIX):
                    foreign.add(identity)
            if owned and seen is None:
                seen = current
            if not locked:
                if owned and seen is not None and deadline is None:
                    deadline = float(absolute) if absolute is not None else min(seen + safe_seconds, armed + safe_seconds)
                if absolute is None:
                    window_end = deadline if deadline is not None else armed + safe_seconds
            if not owned and current >= window_end:
                return _finish([], deadline, [], [])
            heartbeat_age = None
            if owned and seen is not None and current - seen > stale_after:
                if heartbeat_path.is_file():
                    heartbeat_age = current - heartbeat_path.stat().st_mtime
                else:
                    heartbeat_age = current - seen
            if stop_path.is_file():
                targets = owned
            else:
                targets = instances_to_stop(
                    rows,
                    machine_id=machine_id,
                    now=current,
                    deadline=deadline,
                    heartbeat_age=heartbeat_age,
                    stale_after=stale_after,
                )
            if targets:
                return _delete_targets(targets, fetch, delete, deadline, "")
            if stop_path.is_file():
                return _finish([], deadline, [], [])
        else:
            heartbeat_age = current - armed
            if heartbeat_path.is_file():
                heartbeat_age = current - heartbeat_path.stat().st_mtime
            cleanup = stop_path.is_file() or current >= window_end or heartbeat_age > stale_after
            if cleanup:
                targets = _cleanup_ids(known, persisted, foreign)
                if not targets:
                    return _finish([], deadline, [], [], list_error or "LIST_UNAVAILABLE")
                return _delete_targets(targets, fetch, delete, deadline, list_error)
        sleep(float(spec.get("poll_seconds") or 15))


def _delete_targets(targets: list[str], fetch, delete, deadline, list_error: str) -> dict:
    """Ask for deletion, then believe it only when a later good list omits the id."""
    failures = []
    confirmed: list[str] = []
    for instance_id in targets:
        status_code = delete(instance_id)
        if status_code not in (None, 200, 202, 204, 404):
            failures.append({"id": instance_id, "status": status_code})
            continue
        try:
            rows = fetch()
        except InstanceListError as exc:
            failures.append({"id": instance_id, "status": "UNCONFIRMED", "error": exc.code})
            continue
        if isinstance(rows, list) and _absent(rows, instance_id):
            confirmed.append(instance_id)
        else:
            failures.append({"id": instance_id, "status": "STILL_PRESENT"})
    unresolved = set(targets) != set(confirmed)
    return _finish(targets, deadline, failures, confirmed, list_error if unresolved else "")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2
    spec_path = Path(args[0])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result_path = Path(spec["result"])
    if spec.get("mode") == "local":
        outcome = run_until_stopped(
            spec,
            fetch=lambda: _read_rows(Path(spec["instances_path"])),
            delete=lambda instance_id: _local_delete(spec, instance_id),
            now=time.time,
            sleep=time.sleep,
        )
        result_path.write_text(json.dumps(outcome), encoding="utf-8")
        return 0
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        result_path.write_text(json.dumps({"error": "KEY_MISSING"}), encoding="utf-8")
        return 2

    def fetch():
        return _fetch_instances(key)

    def delete(instance_id: str) -> int:
        return _delete_instance(key, instance_id)

    outcome = run_until_stopped(spec, fetch=fetch, delete=delete, now=time.time, sleep=time.sleep)
    result_path.write_text(json.dumps(outcome), encoding="utf-8")
    if outcome.get("error"):
        return 1
    return 0


def _read_rows(path: Path) -> list:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _local_delete(spec: dict, instance_id: str) -> int:
    """Record a local delete. This path never contacts a provider."""
    status_code = 500 if spec.get("fail_delete") else 200
    log = Path(spec["delete_log"])
    rows = _read_rows(log)
    rows.append({"id": instance_id, "status": status_code})
    log.write_text(json.dumps(rows), encoding="utf-8")
    inventory = spec.get("instances_path")
    if status_code == 200 and inventory:
        path = Path(inventory)
        if path.is_file():
            kept = [
                row for row in _read_rows(path)
                if not isinstance(row, dict) or str(row.get("id")) != str(instance_id)
            ]
            path.write_text(json.dumps(kept), encoding="utf-8")
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
