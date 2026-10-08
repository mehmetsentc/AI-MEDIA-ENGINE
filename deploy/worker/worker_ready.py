"""In-container readiness server. It does not load Qwen weights.

Vast can reach GET /health without SSH. The process reports
container_starting until the preinstalled runtime imports succeed.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def assess_runtime() -> dict[str, object]:
    """Return a secret-free readiness record."""
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
    except Exception:
        ready = False
    if not ready:
        return {"state": "container_starting", "runtime": False}
    return {"state": "worker_ready", "runtime": True}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/health":
            self.send_error(404)
            return
        body = json.dumps(assess_runtime()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def listen_address() -> tuple[str, int]:
    """The process must accept Vast's published port, not only loopback."""
    return "0.0.0.0", int(os.environ.get("WORKER_PORT", "8080"))


def serve(address: tuple[str, int] | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(address or listen_address(), _Handler)


def main() -> None:
    serve().serve_forever()


if __name__ == "__main__":
    main()
