"""Signed-in studio account. The browser never receives engine credentials."""
from __future__ import annotations

import json
import re
import threading
from typing import Optional

from media_engine.auth.clients import UnknownClient
from media_engine.jobs.model import IMAGE_GENERATE, Job, JobError, JobNotFound
from media_engine.orchestrator.controller import MediaController

IMAGE_PRICE = 2
_KEY = re.compile(r"^[A-Za-z0-9:_-]{8,80}$")
_PENDING = {
    "queued", "planning", "waiting_capacity",
}
_RUNNING = {
    "provisioning", "booting", "runtime_preparing", "model_loading", "generating", "saving",
}


class SessionBook:
    """Binds the loopback cookie to a client id. Idempotency keys stay on the server."""

    def __init__(self) -> None:
        self._users: dict[str, str] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def sign_in(self, token: str, user_id: str) -> None:
        self._users[token] = user_id

    def sign_out(self, token: str) -> None:
        self._users.pop(token, None)

    def user_for(self, token: str) -> Optional[str]:
        return self._users.get(token)

    def claim_key(self, user_id: str, key: str) -> Optional[str]:
        with self._lock:
            current = self._keys.get((user_id, key))
            if current is None:
                self._keys[(user_id, key)] = ""
                return None
            return current

    def finish_key(self, user_id: str, key: str, job_id: str) -> None:
        with self._lock:
            self._keys[(user_id, key)] = job_id

    def release_key(self, user_id: str, key: str) -> None:
        with self._lock:
            if self._keys.get((user_id, key)) == "":
                self._keys.pop((user_id, key), None)


def browser_route(controller: MediaController, method: str, path: str, body: Optional[bytes],
                  user_id: Optional[str], book: SessionBook, cookie: str) -> Optional[tuple[int, dict]]:
    """Studio-cookie routes. None lets the existing engine route handle the request."""
    if method == "POST" and path == "/v1/session":
        try:
            user = accept_client(controller, _object(body).get("client_id"))
        except UnknownClient:
            return 400, {"error": "UNKNOWN_CLIENT"}
        except ValueError as exc:
            return 400, {"error": str(exc) if str(exc).isidentifier() else "BAD_REQUEST"}
        book.sign_in(cookie, user)
        return 200, account_view(controller, user)
    if method == "DELETE" and path == "/v1/session":
        book.sign_out(cookie)
        return 200, {"signed_in": False}
    guarded = (
        path in {"/v1/account", "/v1/library", "/v1/jobs", "/v1/images/generations", "/v1/capabilities", "/v1/voices/consent", "/v1/uploads"}
        or path.startswith("/v1/jobs/")
        or path.startswith("/v1/capabilities/")
        or path.startswith("/v1/uploads/")
    )
    if not guarded:
        return None
    if not user_id:
        return 401, {"error": "SIGN_IN_REQUIRED"}
    if method == "GET" and path == "/v1/account":
        return 200, account_view(controller, user_id)
    if method == "GET" and path == "/v1/capabilities":
        from media_engine.models.schema import catalog
        return 200, catalog()
    if method == "POST" and path.startswith("/v1/capabilities/"):
        from media_engine.models.schema import submit_capability
        try:
            payload = _object(body)
        except ValueError as exc:
            return 400, {"error": str(exc) if str(exc).isidentifier() else "BAD_REQUEST"}
        return submit_capability(controller, user_id, path.split("/", 3)[-1], payload, book)
    if method == "POST" and path == "/v1/voices/consent":
        from media_engine.models.schema import record_consent
        try:
            payload = _object(body)
        except ValueError as exc:
            return 400, {"error": str(exc) if str(exc).isidentifier() else "BAD_REQUEST"}
        return record_consent(controller, user_id, payload)
    if method == "POST" and path == "/v1/uploads":
        from media_engine.models.schema import accept_upload
        try:
            payload = _object(body)
        except ValueError as exc:
            return 400, {"error": str(exc) if str(exc).isidentifier() else "BAD_REQUEST"}
        return accept_upload(controller, user_id, payload)
    upload = re.fullmatch(r"/v1/uploads/([A-Za-z0-9_]+)", path)
    if method == "DELETE" and upload:
        from media_engine.models.schema import delete_upload
        return delete_upload(controller, user_id, upload.group(1))
    if method == "GET" and path == "/v1/library":
        return 200, {"artifacts": library_for(controller, user_id)}
    if method == "GET" and path == "/v1/jobs":
        return 200, {"jobs": history_for(controller, user_id)}
    if method == "POST" and path == "/v1/images/generations":
        try:
            return submit_for_studio(controller, user_id, _object(body), book)
        except UnknownClient:
            return 400, {"error": "UNKNOWN_CLIENT"}
        except JobError as exc:
            code = str(exc)
            return 400, {"error": code if code.isidentifier() else exc.code}
        except ValueError as exc:
            code = str(exc)
            return 400, {"error": code if code.isidentifier() else "BAD_REQUEST"}
    found = re.fullmatch(r"/v1/jobs/([A-Za-z0-9_-]+)", path)
    if method == "GET" and found:
        try:
            job = controller.get_job(found.group(1))
        except JobNotFound:
            return 404, {"error": "JOB_NOT_FOUND"}
        if job.client_id != user_id:
            return 404, {"error": "JOB_NOT_FOUND"}
        payload = job.to_image_dict() if job.public_status is not None else job.to_public_dict()
        payload["credit_status"] = credit_status(controller, job.id)
        payload["group"] = job_group(str(payload.get("status") or ""))
        payload["prompt"] = job.prompt
        return 200, payload
    return None


def _object(body: Optional[bytes]) -> dict:
    if not body:
        raise ValueError("BODY_REQUIRED")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("INVALID_JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("BODY_REQUIRED")
    return payload


def accept_client(controller: MediaController, client_id: object) -> str:
    if not isinstance(client_id, str):
        raise UnknownClient("")
    return controller.clients.require(client_id)


def account_view(controller: MediaController, user_id: str) -> dict:
    platform = getattr(controller, "platform", None)
    balance = None if platform is None else int(platform.balance(user_id))
    price = IMAGE_PRICE if platform is None else int(platform.image_price())
    ready = gpu_authorization_ready(controller)
    reason = ""
    if balance is None:
        reason = "NO_CREDIT_ACCOUNT"
    elif balance < price:
        reason = "INSUFFICIENT_CREDITS"
    elif not ready:
        reason = "AUTHORIZATION_REQUIRED"
    return {
        "user_id": user_id,
        "balance": balance,
        "image_price": price,
        "can_generate": reason == "",
        "block_reason": reason,
        "resolution": "1024x1024",
        "steps": 20,
    }


def gpu_authorization_ready(controller: MediaController) -> bool:
    """A browser submission is not a paid GPU authorization."""
    if getattr(controller, "studio_readonly", False):
        return False
    provider = controller.provider
    if getattr(provider, "name", "fake") != "vast":
        return True
    return getattr(provider, "execution_authorization", None) is not None


def submit_for_studio(controller: MediaController, user_id: str, payload: dict,
                      book: SessionBook) -> tuple[int, dict]:
    if not user_id:
        return 401, {"error": "SIGN_IN_REQUIRED"}
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        return 400, {"error": "PROMPT_REQUIRED"}
    width = payload.get("width", 1024)
    height = payload.get("height", 1024)
    if width != 1024 or height != 1024 or isinstance(width, bool) or isinstance(height, bool):
        return 400, {"error": "RESOLUTION_UNSUPPORTED"}
    seed = payload.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        return 400, {"error": "SEED_INVALID"}
    key = payload.get("idempotency_key")
    if not isinstance(key, str) or _KEY.fullmatch(key) is None:
        return 400, {"error": "IDEMPOTENCY_REQUIRED"}
    if not gpu_authorization_ready(controller):
        return 409, {"error": "AUTHORIZATION_REQUIRED"}
    platform = getattr(controller, "platform", None)
    price = IMAGE_PRICE if platform is None else int(platform.image_price())
    if platform is not None and int(platform.balance(user_id)) < price:
        return 402, {
            "error": "INSUFFICIENT_CREDITS",
            "required": price,
            "balance": int(platform.balance(user_id)),
        }
    claimed = book.claim_key(user_id, key)
    if claimed == "":
        return 409, {"error": "DUPLICATE_SUBMISSION"}
    if claimed:
        return 202, {"job_id": claimed, "status": _status(controller, claimed), "idempotent": True}
    try:
        job = controller.submit_image(
            prompt, width=width, height=height, seed=seed, idempotency_key=key, client_id=user_id,
        )
    except Exception:
        book.release_key(user_id, key)
        raise
    book.finish_key(user_id, key, job.id)
    return 202, {"job_id": job.id, "status": job.public_status or "queued"}


def library_for(controller: MediaController, user_id: str) -> list[dict]:
    items: dict[str, dict] = {}
    platform = getattr(controller, "platform", None)
    if platform is not None:
        for row in platform.list_media(user_id):
            if str(row.get("user_id") or "") != user_id:
                continue
            items[str(row["id"])] = _platform_item(row)
    for artifact in controller.artifacts.list_recent():
        if artifact_owner(controller, artifact.artifact_id) != user_id:
            continue
        items[artifact.artifact_id] = {
            "artifact_id": artifact.artifact_id,
            "job_id": artifact.job_id,
            "prompt": artifact.prompt,
            "seed": artifact.seed,
            "width": artifact.width,
            "height": artifact.height,
            "sha256": artifact.sha256,
            "byte_count": artifact.byte_count,
            "created_at": artifact.created_at,
            "model": artifact.model,
        }
    return sorted(items.values(), key=lambda item: str(item.get("created_at") or ""), reverse=True)


def history_for(controller: MediaController, user_id: str) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for job in controller.jobs.list():
        if job.client_id != user_id or job.job_type != IMAGE_GENERATE:
            continue
        rows.append(history_item(controller, job))
        seen.add(job.id)
    platform = getattr(controller, "platform", None)
    extra = getattr(platform, "list_image_jobs", None) if platform is not None else None
    if extra is not None:
        for item in extra(user_id):
            if item["job_id"] in seen or item.get("user_id") != user_id:
                continue
            rows.append(item)
            seen.add(item["job_id"])
    rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return rows


def history_item(controller: MediaController, job: Job) -> dict:
    status = job.public_status or job.status.lower()
    return {
        "job_id": job.id,
        "user_id": job.client_id,
        "status": status,
        "group": job_group(status),
        "prompt": job.prompt,
        "seed": job.seed,
        "width": job.width,
        "height": job.height,
        "artifact_id": job.artifact_id,
        "error_code": job.error_code,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "credit_status": credit_status(controller, job.id),
    }


def job_group(status: str) -> str:
    if status in _PENDING:
        return "pending"
    if status in _RUNNING:
        return "running"
    if status == "completed":
        return "completed"
    if status == "failed":
        return "failed"
    return "pending"


def credit_status(controller: MediaController, job_id: str) -> str:
    platform = getattr(controller, "platform", None)
    if platform is None:
        return "none"
    kinds = [str(item["entry_type"]) for item in platform.job_ledger(job_id)]
    if "refund" in kinds:
        return "refunded"
    if "consume" in kinds:
        return "settled"
    if "reserve" in kinds:
        return "reserved"
    return "none"


def artifact_owner(controller: MediaController, artifact_id: str) -> Optional[str]:
    record = controller.artifacts.get(artifact_id)
    if record is not None:
        return _job_owner(controller, record.job_id)
    platform = getattr(controller, "platform", None)
    repo = getattr(platform, "repo", None) if platform is not None else None
    row = None if repo is None else repo.media_row(artifact_id)
    if not row:
        return None
    owner = str(row.get("user_id") or "")
    return owner or None


def _job_owner(controller: MediaController, job_id: str) -> Optional[str]:
    try:
        return controller.jobs.get(job_id).client_id
    except JobNotFound:
        return None


def _status(controller: MediaController, job_id: str) -> str:
    try:
        job = controller.get_job(job_id)
    except JobNotFound:
        return "queued"
    return job.public_status or "queued"


def _platform_item(row: dict) -> dict:
    settings = _settings(row.get("settings"))
    return {
        "artifact_id": str(row["id"]),
        "job_id": str(row.get("job_id") or ""),
        "prompt": str(settings.get("prompt") or ""),
        "seed": settings.get("seed"),
        "width": row.get("width"),
        "height": row.get("height"),
        "sha256": row.get("sha256"),
        "byte_count": row.get("byte_count"),
        "created_at": str(row.get("created_at") or ""),
        "model": row.get("model"),
    }


def _settings(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
