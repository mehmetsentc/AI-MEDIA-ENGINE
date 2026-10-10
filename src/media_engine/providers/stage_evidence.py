"""Copy staging logs off a GPU before it is deleted.

The capture is bounded. A hung SSH session records a timeout and returns.
Deletion stays the caller's next step.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

CAPTURE_TIMEOUT_SECONDS = 8
_LOG_LIMIT = 16_000

REMOTE_STAGE_EVIDENCE = r"""
echo '---EXIT---'
if [ -f /workspace/cache-sync.exit ]; then cat /workspace/cache-sync.exit; else echo MISSING; fi
echo '---PROGRESS---'
cat /workspace/r2-sync-progress.json 2>/dev/null || true
echo '---REPORT---'
cat /workspace/phase2d_report.json 2>/dev/null || true
echo '---LOG---'
tail -c 16000 /workspace/cache-sync.log 2>/dev/null || true
"""

_MARKERS = ("---EXIT---", "---PROGRESS---", "---REPORT---", "---LOG---")
_SIGNED = (
    re.compile(r"(?i)(X-Amz-Signature=)[^&\s\"']+"),
    re.compile(r"(?i)(X-Amz-Credential=)[^&\s\"']+"),
    re.compile(r"(?i)(X-Amz-Security-Token=)[^&\s\"']+"),
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def redact_stage_text(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Remove credentials. Keep the tail of a long worker log."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    for pattern in _SIGNED:
        text = pattern.sub("[redacted]", text)
    if len(text) > _LOG_LIMIT:
        text = text[-_LOG_LIMIT:]
    return text


def parse_stage_bundle(text: str) -> dict[str, str]:
    """Split the remote evidence command into its labeled sections."""
    positions = [(text.find(marker), marker) for marker in _MARKERS]
    found = [(pos, marker) for pos, marker in positions if pos >= 0]
    found.sort()
    sections = {marker: "" for marker in _MARKERS}
    for index, (pos, marker) in enumerate(found):
        start = pos + len(marker)
        end = found[index + 1][0] if index + 1 < len(found) else len(text)
        sections[marker] = text[start:end].strip()
    return {
        "exit_code": sections["---EXIT---"].splitlines()[-1].strip() if sections["---EXIT---"] else "",
        "progress": sections["---PROGRESS---"],
        "report": sections["---REPORT---"],
        "log": sections["---LOG---"],
    }


def _progress_bytes(progress: str) -> Optional[int]:
    for line in reversed(progress.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        done = payload.get("bytes_done") if isinstance(payload, dict) else None
        if isinstance(done, int) and not isinstance(done, bool):
            return done
    return None


def preserve_stage_evidence(
    *,
    destination: Path,
    fetch: Callable[[], tuple[str, str]],
    secrets: tuple[str, ...] = (),
    exit_code: str = "",
    bytes_done: Optional[int] = None,
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    last_progress_at: Optional[float] = None,
    timeout_seconds: float = CAPTURE_TIMEOUT_SECONDS,
) -> dict:
    """Write a local record. Fetch failures stay in the record and do not raise."""
    record: dict = {
        "captured_at": time.time(),
        "stage_started_at": started_at,
        "stage_ended_at": ended_at if ended_at is not None else time.time(),
        "exit_code": exit_code,
        "bytes_transferred": bytes_done,
        "last_progress_at": last_progress_at,
        "cache_sync_log": "",
        "report": "",
        "stderr": "",
        "capture_error": "",
    }
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fetch()
        except Exception as exc:
            box["error"] = type(exc).__name__

    worker = threading.Thread(target=run, name="stage-evidence", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        record["capture_error"] = "CAPTURE_TIMEOUT"
    elif "error" in box:
        record["capture_error"] = str(box["error"])
    else:
        stdout, stderr = box.get("value") or ("", "")
        parsed = parse_stage_bundle(str(stdout))
        if parsed["exit_code"]:
            record["exit_code"] = parsed["exit_code"]
        remote_bytes = _progress_bytes(parsed["progress"])
        if remote_bytes is not None:
            record["bytes_transferred"] = remote_bytes
        record["cache_sync_log"] = redact_stage_text(parsed["log"], secrets)
        record["report"] = redact_stage_text(parsed["report"], secrets)
        record["stderr"] = redact_stage_text(str(stderr), secrets)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


REMOTE_RUNTIME_EVIDENCE = r"""
echo '---PYTHON---'
cat /workspace/runtime-python.json 2>/dev/null || true
echo '---IMPORT_EXIT---'
cat /workspace/import-probe.exit 2>/dev/null || true
echo '---IMPORT_ERR---'
tail -c 4000 /workspace/import-probe.err 2>/dev/null || true
echo '---REPORT---'
cat /workspace/phase2d_report.json 2>/dev/null || true
echo '---STAMP---'
date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true
"""

_RUNTIME_MARKERS = (
    "---PYTHON---",
    "---IMPORT_EXIT---",
    "---IMPORT_ERR---",
    "---REPORT---",
    "---STAMP---",
)


def parse_runtime_bundle(text: str) -> dict[str, str]:
    """Split the model-load evidence command into its labeled sections."""
    positions = [(text.find(marker), marker) for marker in _RUNTIME_MARKERS]
    found = [(pos, marker) for pos, marker in positions if pos >= 0]
    found.sort()
    sections = {marker: "" for marker in _RUNTIME_MARKERS}
    for index, (pos, marker) in enumerate(found):
        start = pos + len(marker)
        end = found[index + 1][0] if index + 1 < len(found) else len(text)
        sections[marker] = text[start:end].strip()
    return {
        "python": sections["---PYTHON---"],
        "import_exit": sections["---IMPORT_EXIT---"].splitlines()[-1].strip() if sections["---IMPORT_EXIT---"] else "",
        "import_err": sections["---IMPORT_ERR---"],
        "report": sections["---REPORT---"],
        "stamp": sections["---STAMP---"].splitlines()[-1].strip() if sections["---STAMP---"] else "",
    }


def preserve_runtime_evidence(
    *,
    destination: Path,
    fetch: Callable[[], tuple[str, str]],
    secrets: tuple[str, ...] = (),
    python_executable: str = "",
    python_version: str = "",
    dependencies: Optional[list] = None,
    import_exit_code: str = "",
    stderr: str = "",
    model_load_exception: str = "",
    timeout_seconds: float = CAPTURE_TIMEOUT_SECONDS,
) -> dict:
    """Write model-load evidence locally. A hung fetch records a timeout and returns."""
    record: dict = {
        "captured_at": time.time(),
        "failure_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_executable": python_executable,
        "python_version": python_version,
        "dependencies": _redact_tree(dependencies or [], secrets),
        "import_exit_code": import_exit_code,
        "stderr": redact_stage_text(stderr or "", secrets),
        "model_load_exception": model_load_exception,
        "report": "",
        "capture_error": "",
    }
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fetch()
        except Exception as exc:
            box["error"] = type(exc).__name__

    worker = threading.Thread(target=run, name="runtime-evidence", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        record["capture_error"] = "CAPTURE_TIMEOUT"
    elif "error" in box:
        record["capture_error"] = str(box["error"])
    else:
        stdout, remote_err = box.get("value") or ("", "")
        parsed = parse_runtime_bundle(str(stdout))
        _merge_runtime_evidence(record, parsed, redact_stage_text(str(remote_err), secrets), secrets)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _redact_tree(value: object, secrets: tuple[str, ...]) -> object:
    """Redact strings without rewriting the JSON around them."""
    if isinstance(value, str):
        return redact_stage_text(value, secrets)
    if isinstance(value, list):
        return [_redact_tree(item, secrets) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_tree(item, secrets) for key, item in value.items()}
    return value


def _merge_runtime_evidence(record: dict, parsed: dict[str, str], remote_err: str, secrets: tuple[str, ...]) -> None:
    if parsed["python"]:
        try:
            document = json.loads(parsed["python"])
        except json.JSONDecodeError:
            document = {}
        if isinstance(document, dict):
            if document.get("selected") and not record["python_executable"]:
                record["python_executable"] = str(document["selected"])
            if document.get("version"):
                record["python_version"] = str(document["version"])
            candidates = document.get("candidates")
            if isinstance(candidates, list):
                record["dependencies"] = _redact_tree(candidates, secrets)
            if document.get("failure") and record["model_load_exception"] in {"", "NO_INTERPRETER"}:
                record["model_load_exception"] = str(document["failure"])
    if parsed["import_exit"]:
        record["import_exit_code"] = parsed["import_exit"]
    if parsed["import_err"]:
        record["stderr"] = redact_stage_text(parsed["import_err"], secrets)
    elif remote_err:
        record["stderr"] = remote_err
    if parsed["report"]:
        record["report"] = redact_stage_text(parsed["report"], secrets)
        try:
            body = json.loads(parsed["report"])
        except json.JSONDecodeError:
            body = {}
        if isinstance(body, dict):
            exception = body.get("exception") or body.get("detail") or body.get("error") or ""
            if exception:
                record["model_load_exception"] = str(exception)[:180]
    if parsed["stamp"]:
        record["failure_timestamp"] = parsed["stamp"]


def release_after_evidence(
    capture: Callable[[], dict],
    terminate: Callable[[], None],
    *,
    timeout_seconds: float = CAPTURE_TIMEOUT_SECONDS,
) -> dict:
    """Collect evidence, then terminate. Capture problems do not skip termination."""
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = capture()
        except Exception as exc:
            box["error"] = type(exc).__name__

    worker = threading.Thread(target=run, name="stage-evidence-release", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        record = {"capture_error": "CAPTURE_TIMEOUT"}
    elif "error" in box:
        record = {"capture_error": str(box["error"])}
    else:
        value = box.get("value")
        record = dict(value) if isinstance(value, dict) else {"capture_error": ""}
    terminate()
    return record
