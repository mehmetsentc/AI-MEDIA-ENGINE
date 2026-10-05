"""Minimum HTTP API: create, read, and cancel a job."""
from __future__ import annotations

import json
import re
from typing import Optional

from media_engine.auth.clients import UnknownClient
from media_engine.jobs.model import IllegalTransition, JobError, JobNotFound, UnsupportedJobType
from media_engine.orchestrator.controller import MediaController

_JOB = re.compile(r"^/v1/jobs/([A-Za-z0-9_-]+)$")
_CANCEL = re.compile(r"^/v1/jobs/([A-Za-z0-9_-]+)/cancel$")


def dispatch(controller: MediaController, method: str, path: str,
             body: Optional[bytes]) -> tuple[int, dict]:
    path = path.split("?", 1)[0]
    try:
        if method == "POST" and path == "/v1/jobs":
            payload = _object(body)
            prompt = _prompt(payload)
            job = controller.submit_job(str(payload.get("client_id", "")), str(payload.get("type", "")), prompt)
            return 201, job.to_public_dict()
        cancel = _CANCEL.fullmatch(path)
        if method == "POST" and cancel:
            job = controller.cancel_job(cancel.group(1))
            return 200, job.to_public_dict()
        found = _JOB.fullmatch(path)
        if method == "GET" and found:
            job = controller.get_job(found.group(1))
            return 200, job.to_public_dict()
    except JobNotFound as exc:
        return 404, {"error": exc.code}
    except IllegalTransition as exc:
        return 409, {"error": exc.code, "detail": str(exc)}
    except (UnknownClient, UnsupportedJobType, JobError, ValueError) as exc:
        code = getattr(exc, "code", "BAD_REQUEST")
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


def _prompt(payload: dict) -> str:
    raw_input = payload.get("input")
    if not isinstance(raw_input, dict):
        raise JobError("PROMPT_REQUIRED")
    prompt = raw_input.get("prompt")
    if not isinstance(prompt, str):
        raise JobError("PROMPT_REQUIRED")
    return prompt
