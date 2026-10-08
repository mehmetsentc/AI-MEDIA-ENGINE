"""In-container readiness server. It does not import or load Qwen.

Vast SSH mode replaces the image command, so onstart starts this process.
/health reports that the generation files are present. Model weights stay unloaded.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_LOCK = threading.Lock()
_STATUS: dict[str, object] = {
    "state": "container_starting",
    "runtime": False,
    "model_loaded": False,
}
_GENERATION_FILES = (
    "/workspace/qwen_remote.py",
    "/workspace/model_cache.py",
    "/workspace/r2_cache.py",
)
_SECRET = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|bearer)(\s*[=:]\s*)(\S+)"
)


def listen_address() -> tuple[str, int]:
    """The process must accept Vast's published port, not only loopback."""
    return "0.0.0.0", int(os.environ.get("WORKER_PORT", "8080"))


def current_status() -> dict[str, object]:
    with _LOCK:
        return dict(_STATUS)


def publish(status: dict[str, object]) -> None:
    with _LOCK:
        _STATUS.clear()
        _STATUS.update(status)
    _log("state=%s runtime=%s" % (status.get("state"), status.get("runtime")))


def failure_status(exc: BaseException) -> dict[str, object]:
    """A secret-free record of a startup exception."""
    detail = _SECRET.sub(r"\1\2[redacted]", " ".join(str(exc).split()))
    return {
        "state": "startup_failed",
        "runtime": False,
        "model_loaded": False,
        "error": type(exc).__name__,
        "detail": detail[:180],
    }


def assess_runtime() -> dict[str, object]:
    """Confirm the generation files are present. This does not import or load Qwen."""
    marker = Path(os.environ.get("WORKER_RUNTIME_MARKER", "/opt/ai-media-engine/runtime-ready"))
    configured = os.environ.get("WORKER_GENERATION_FILES", "")
    files = tuple(item for item in configured.split(",") if item) if configured else _GENERATION_FILES
    missing = []
    if not marker.is_file():
        missing.append(marker.name)
    for name in files:
        if not Path(name).is_file():
            missing.append(Path(name).name)
    if missing:
        return {
            "state": "startup_failed",
            "runtime": False,
            "model_loaded": False,
            "error": "RuntimeIncomplete",
            "detail": "missing " + ",".join(missing),
        }
    return {"state": "worker_ready", "runtime": True, "model_loaded": False}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/health":
            self.send_error(404)
            return
        body = json.dumps(current_status()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def serve(address: tuple[str, int] | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(address or listen_address(), _Handler)


def main() -> None:
    address = listen_address()
    _write_pid()
    try:
        server = serve(address)
    except Exception as exc:
        publish(failure_status(exc))
        raise
    _log("listen=%s:%s event=start" % address)
    try:
        publish(assess_runtime())
    except Exception as exc:
        publish(failure_status(exc))
    try:
        server.serve_forever()
    except Exception as exc:
        publish(failure_status(exc))
        raise


def _write_pid() -> None:
    path = Path(os.environ.get("WORKER_PID_FILE", "/workspace/worker-ready.pid"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    except OSError:
        return


def _log(message: str) -> None:
    line = "%s pid=%s %s\n" % (_utc(), os.getpid(), message)
    path = Path(os.environ.get("WORKER_LOG", "/workspace/worker-ready.log"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        return


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    main()
