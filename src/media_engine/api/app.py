"""Local HTTP API. Image jobs return immediately and finish on the runner."""
from __future__ import annotations

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from media_engine.auth.clients import UnknownClient
from media_engine.jobs.model import IllegalTransition, JobError, JobNotFound, UnsupportedJobType
from media_engine.orchestrator.controller import MediaController

_JOB = re.compile(r"^/v1/jobs/([A-Za-z0-9_-]+)$")
_CANCEL = re.compile(r"^/v1/jobs/([A-Za-z0-9_-]+)/cancel$")
_ARTIFACT = re.compile(r"^/v1/artifacts/([A-Za-z0-9_-]+)$")
_MAX_BODY = 65536


def authorize(path: str, headers: Optional[dict], api_key: str) -> Optional[tuple[int, dict]]:
    """Health stays open. Every other route fails closed without the bearer."""
    if path.split("?", 1)[0] == "/health":
        return None
    presented = ""
    if headers:
        presented = str(headers.get("Authorization") or headers.get("authorization") or "")
    if not api_key or presented != "Bearer " + api_key:
        return 401, {"error": "UNAUTHORIZED"}
    return None


def dispatch(controller: MediaController, method: str, path: str,
             body: Optional[bytes]) -> tuple[int, dict]:
    path = path.split("?", 1)[0]
    try:
        if method == "GET" and path == "/health":
            return 200, controller.health()
        if method == "POST" and path == "/v1/images/generations":
            payload = _object(body)
            if "negative_prompt" in payload:
                raise JobError("NEGATIVE_PROMPT_UNSUPPORTED")
            prompt = payload.get("prompt")
            if not isinstance(prompt, str):
                raise JobError("PROMPT_REQUIRED")
            width = payload.get("width", 1024)
            height = payload.get("height", 1024)
            seed = payload.get("seed")
            if isinstance(width, bool) or not isinstance(width, int):
                raise JobError("RESOLUTION_UNSUPPORTED")
            if isinstance(height, bool) or not isinstance(height, int):
                raise JobError("RESOLUTION_UNSUPPORTED")
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
                raise JobError("SEED_INVALID")
            job = controller.submit_image(prompt, width=width, height=height, seed=seed)
            return 202, {"job_id": job.id, "status": "queued"}
        if method == "POST" and path == "/v1/jobs":
            payload = _object(body)
            prompt, task = _request_input(payload)
            job = controller.submit_job(
                str(payload.get("client_id", "")), str(payload.get("type", "")), prompt, task=task,
            )
            return 201, job.to_public_dict()
        cancel = _CANCEL.fullmatch(path)
        if method == "POST" and cancel:
            job = controller.cancel_job(cancel.group(1))
            return 200, job.to_public_dict()
        found = _JOB.fullmatch(path)
        if method == "GET" and found:
            job = controller.get_job(found.group(1))
            if job.public_status is not None:
                return 200, job.to_image_dict()
            return 200, job.to_public_dict()
    except JobNotFound as exc:
        return 404, {"error": exc.code}
    except IllegalTransition as exc:
        return 409, {"error": exc.code, "detail": str(exc)}
    except (UnknownClient, UnsupportedJobType, JobError, ValueError) as exc:
        code = getattr(exc, "code", "BAD_REQUEST")
        if code == "JOB_ERROR" and str(exc).isidentifier():
            code = str(exc)
        return 400, {"error": code}
    return 404, {"error": "NOT_FOUND"}


def _object(body: Optional[bytes]) -> dict:
    if not body:
        raise JobError("BODY_REQUIRED")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise JobError("INVALID_JSON") from exc
    if not isinstance(payload, dict):
        raise JobError("BODY_REQUIRED")
    return payload


def _request_input(payload: dict) -> tuple[str, Optional[str]]:
    raw_input = payload.get("input")
    if not isinstance(raw_input, dict):
        raise JobError("PROMPT_REQUIRED")
    prompt = raw_input.get("prompt")
    if not isinstance(prompt, str):
        raise JobError("PROMPT_REQUIRED")
    task = raw_input.get("task")
    if task is not None and not isinstance(task, str):
        raise JobError("TASK_REQUIRED")
    return prompt, task


class _LoopbackServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = False

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler]) -> None:
        self.ready = threading.Event()
        super().__init__(address, handler)

    def service_actions(self) -> None:
        self.ready.set()


class LocalAPIServer:
    """One process: boot the controller, serve loopback HTTP, then join both."""

    def __init__(self, controller: MediaController, host: str = "127.0.0.1", port: int = 0,
                 api_key: Optional[str] = None) -> None:
        if host != "127.0.0.1":
            raise ValueError("local API binds to 127.0.0.1 only")
        self.controller = controller
        self.api_key = os.environ.get("AI_MEDIA_ENGINE_API_KEY", "") if api_key is None else api_key
        self._httpd = _LoopbackServer((host, port), _handler(controller, self.api_key))
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        if not self.controller.runner_alive:
            self.controller.start()
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="media-engine-http", daemon=False,
        )
        self._thread.start()
        if not self._httpd.ready.wait(timeout=2):
            raise RuntimeError("http server did not start")

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        thread = self._thread
        if thread is not None:
            thread.join()
            self._thread = None
        self.controller.stop()


def _handler(controller: MediaController, api_key: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            self._respond()

        def do_POST(self) -> None:
            self._respond()

        def _respond(self) -> None:
            path = self.path.split("?", 1)[0]
            denied = authorize(path, {key: value for key, value in self.headers.items()}, api_key)
            if denied is not None:
                self._json(*denied)
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length < 0 or length > _MAX_BODY:
                self._json(400, {"error": "BODY_REQUIRED"})
                return
            body = self.rfile.read(length) if length else None
            artifact = _ARTIFACT.fullmatch(path)
            if self.command == "GET" and artifact:
                try:
                    data = controller.artifact_bytes(artifact.group(1))
                except JobNotFound:
                    self._json(404, {"error": "JOB_NOT_FOUND"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            status, payload = dispatch(controller, self.command, self.path, body)
            self._json(status, payload)

        def _json(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler
