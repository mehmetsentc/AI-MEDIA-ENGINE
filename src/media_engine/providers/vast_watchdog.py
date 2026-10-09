"""Detached Vast termination. This module never creates an instance or a volume."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

LABEL_PREFIX = "media-engine:"
HEARTBEAT_STALE_SECONDS = 300.0
INSTANCES_URL = "https://console.vast.ai/api/v0/instances/?owner=me"


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


def _fetch_instances(key: str) -> list:
    status, raw = _request(key, "GET", INSTANCES_URL)
    if status != 200:
        return []
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return []
    rows = payload.get("instances") if isinstance(payload, dict) else None
    return rows if isinstance(rows, list) else []


def _delete_instance(key: str, instance_id: str) -> int:
    status, _raw = _request(key, "DELETE", "https://console.vast.ai/api/v0/instances/%s/" % instance_id)
    return status


def run_until_stopped(spec: dict, *, fetch, delete, now, sleep) -> dict:
    """One control loop used by the detached process and by tests."""
    machine_id = int(spec["machine_id"])
    safe_seconds = float(spec["safe_seconds"])
    heartbeat_path = Path(spec["heartbeat"])
    stop_path = Path(spec["stop"])
    stale_after = float(spec.get("stale_after") or HEARTBEAT_STALE_SECONDS)
    seen: float | None = None
    deadline: float | None = None
    armed: float | None = None
    while True:
        current = float(now())
        if armed is None:
            armed = current
        rows = fetch()
        owned = matching_instances(rows, machine_id)
        absolute = spec.get("deadline_epoch")
        if owned and seen is None:
            seen = current
            deadline = float(absolute) if absolute is not None else min(seen + safe_seconds, armed + safe_seconds)
        window_end = float(absolute) if absolute is not None else armed + safe_seconds
        if not owned and current >= window_end:
            return {"stopped": [], "deadline": deadline, "termination_failures": []}
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
            failures = []
            for instance_id in targets:
                status_code = delete(instance_id)
                if status_code not in (None, 200, 202, 204, 404):
                    failures.append({"id": instance_id, "status": status_code})
            return {
                "stopped": targets,
                "deadline": deadline,
                "termination_failures": failures,
            }
        if stop_path.is_file():
            return {"stopped": [], "deadline": deadline, "termination_failures": []}
        sleep(float(spec.get("poll_seconds") or 15))


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
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
