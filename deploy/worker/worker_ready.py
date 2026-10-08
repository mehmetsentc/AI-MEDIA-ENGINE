"""In-container readiness server. It does not load Qwen weights.

Vast SSH mode replaces the image command, so this process is started by
onstart. /health answers before the runtime imports finish.
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
_STATUS: dict[str, object] = {"state": "container_starting", "runtime": False}
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
        "error": type(exc).__name__,
        "detail": detail[:180],
    }


def assess_runtime() -> dict[str, object]:
    """Import the preinstalled runtime. This never downloads model weights."""
    marker = Path("/opt/ai-media-engine/runtime-ready")
    try:
        import torch  # noqa: F401
        import diffusers
        import transformers  # noqa: F401
        import accelerate  # noqa: F401
        import safetensors  # noqa: F401
        from diffusers import QwenImagePipeline

        pipeline = QwenImagePipeline.__name__
        ready = marker.is_file() and pipeline == "QwenImagePipeline" and bool(diffusers.__name__)
    except Exception as exc:
        return failure_status(exc)
    if not ready:
        return {"state": "container_starting", "runtime": False}
    return {"state": "worker_ready", "runtime": True}


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
    _log("listen=%s:%s event=start" % address)
    threading.Thread(target=_assess, name="worker-assess", daemon=True).start()
    try:
        serve(address).serve_forever()
    except Exception as exc:
        publish(failure_status(exc))
        raise


def _assess() -> None:
    try:
        publish(assess_runtime())
    except Exception as exc:
        publish(failure_status(exc))


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
