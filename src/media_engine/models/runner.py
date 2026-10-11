"""Queue a local multimodal job. This path never calls provider.create."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from pathlib import Path

from media_engine.jobs.model import MULTIMODAL, PUBLIC_PROGRESS, Job, JobStatus, new_job_id
from media_engine.models.cost import CREDIT_RULES
from media_engine.models.errors import AdapterError
from media_engine.models.invoke import (
    Invocation,
    invoke_image_edit,
    invoke_music,
    invoke_translation,
    invoke_transcribe,
    invoke_tts,
    invoke_video,
    invoke_voice_clone,
)
from media_engine.models.registry import primary_for
from media_engine.models.schema import _upload_store
from media_engine.storage.artifacts import Artifact

_KEY = re.compile(r"^[A-Za-z0-9:_-]{8,80}$")
_MEDIA = {
    "audio.transcribe": ("audio", "json", "speech", "application/json"),
    "audio.tts": ("audio", "wav", "speech", "audio/wav"),
    "audio.voice_clone": ("voice", "wav", "voice", "audio/wav"),
    "audio.translate": ("audio", "json", "speech", "application/json"),
    "image.edit": ("images", "png", "image", "image/png"),
    "music.generate": ("music", "wav", "music", "audio/wav"),
    "video.text_to_video": ("video", "mp4", "video", "video/mp4"),
}


def submit_local(controller, user_id: str, kind: str, payload: dict, book) -> tuple[int, dict]:
    if bound_runtime(controller, kind) is None:
        return 409, {"error": "CAPABILITY_UNAVAILABLE", "kind": kind}
    key = payload.get("idempotency_key")
    if not isinstance(key, str) or _KEY.fullmatch(key) is None:
        return 400, {"error": "IDEMPOTENCY_REQUIRED"}
    claimed = book.claim_key(user_id, key)
    if claimed == "":
        return 409, {"error": "DUPLICATE_SUBMISSION"}
    if claimed:
        return 202, {"job_id": claimed, "status": "queued", "idempotent": True}
    try:
        _precheck(controller, user_id, kind, payload)
        price = _price(controller, kind, user_id)
    except AdapterError as exc:
        book.release_key(user_id, key)
        status = 402 if exc.code == "INSUFFICIENT_CREDITS" else 409 if exc.code == "PRICING_UNAVAILABLE" else 400
        body = {"error": exc.code}
        if exc.code == "INSUFFICIENT_CREDITS":
            body["required"] = CREDIT_RULES[kind][2]
            body["balance"] = int(controller.platform.balance(user_id))
        return status, body
    prompt = str(payload.get("prompt") or payload.get("text") or kind)
    now = controller.clock.iso()
    job = Job(
        id=new_job_id(),
        client_id=user_id,
        job_type=MULTIMODAL,
        status=JobStatus.QUEUED,
        created_at=now,
        updated_at=now,
        attempt_count=0,
        result_uri=None,
        error_code=None,
        prompt=prompt[:4000],
        task=json.dumps({"kind": kind, "payload": payload}, separators=(",", ":")),
        public_status="queued",
        progress=PUBLIC_PROGRESS["queued"],
    )
    controller.jobs.insert(job)
    book.finish_key(user_id, key, job.id)
    controller.queue.enqueue(job.id)
    controller._settled.clear()
    controller._notify()
    return 202, {"job_id": job.id, "status": "queued", "credits": price}


def execute_multimodal(controller, job: Job) -> None:
    body = json.loads(job.task or "{}")
    kind = str(body.get("kind") or "")
    payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
    spec = primary_for(kind)
    gate = _gate(controller)
    if spec is None or gate["resident"] not in (None, spec.model_id):
        controller._fail(job, "MODEL_BUSY" if spec is not None else "CAPABILITY_UNAVAILABLE")
        return
    gate["resident"] = spec.model_id
    try:
        controller._transition(job, JobStatus.WAITING_FOR_WORKER)
        controller._transition(job, JobStatus.RUNNING)
        job.attempt_count += 1
        controller.jobs.save(job)
        controller._set_public(job, "generating")
        try:
            service, model, _credits = CREDIT_RULES[kind]
            controller.platform.authorize_paid(
                job.client_id, job.id, service=service, model=model, provider="local",
            )
        except Exception as exc:
            controller._fail(job, getattr(exc, "code", "BILLING_UNAVAILABLE"))
            return
        runtime = bound_runtime(controller, kind)
        timeout = float(getattr(controller, "multimodal_timeout", 30.0))
        cancel = getattr(controller, "multimodal_cancel", None)
        if cancel is None:
            cancel = threading.Event()
            controller.multimodal_cancel = cancel
        ctx = Invocation(time.monotonic() + timeout, cancel)
        try:
            result = controller._unlocked(lambda: _invoke(controller, job, kind, payload, runtime, ctx))
        except AdapterError as exc:
            controller._fail(job, exc.code)
            return
        except Exception:
            controller._fail(job, "GENERATION_FAILED")
            return
        controller._set_public(job, "saving")
        try:
            artifact_id, uri, digest, count = _persist(controller, job, kind, result)
        except Exception:
            controller._fail(job, "ARTIFACT_TRANSFER_FAILED")
            return
        try:
            controller._settle_job(job, success=True)
        except Exception:
            controller._fail(job, "SETTLEMENT_FAILED")
            return
        job.result_uri = uri
        job.artifact_id = artifact_id
        job.sha256 = digest
        job.byte_count = count
        controller._transition(job, JobStatus.SUCCEEDED)
        controller._set_public(job, "completed")
        controller._record_usage(job, JobStatus.SUCCEEDED)
    finally:
        if gate.get("resident") == getattr(spec, "model_id", None):
            gate["resident"] = None


def bound_runtime(controller, kind: str):
    table = getattr(controller, "multimodal_runtimes", None) or {}
    return table.get(kind)


def _gate(controller) -> dict:
    gate = getattr(controller, "model_gate", None)
    if gate is None:
        gate = {"resident": None}
        controller.model_gate = gate
    return gate


def _price(controller, kind: str, user_id: str) -> int:
    platform = getattr(controller, "platform", None)
    if platform is None or kind not in CREDIT_RULES:
        raise AdapterError("PRICING_UNAVAILABLE")
    service, model, credits = CREDIT_RULES[kind]
    rule = platform.repo.price_at(service, model, platform.now_iso())
    if rule is None or int(rule["credit_price"]) != credits:
        raise AdapterError("PRICING_UNAVAILABLE")
    if int(platform.balance(user_id)) < credits:
        raise AdapterError("INSUFFICIENT_CREDITS")
    return credits


def _precheck(controller, user_id: str, kind: str, payload: dict) -> None:
    if kind == "audio.transcribe":
        _owned_bytes(controller, user_id, payload.get("upload_id"), "audio")
    elif kind == "audio.voice_clone":
        _owned_bytes(controller, user_id, payload.get("reference_upload_id"), "audio")
    elif kind == "image.edit":
        _owned_bytes(controller, user_id, payload.get("source_upload_id"), "image")
    elif kind == "audio.tts":
        from media_engine.models.invoke import SPEAKERS
        if payload.get("voice") not in SPEAKERS:
            raise AdapterError("SPEAKER_UNSUPPORTED")


def _owned_bytes(controller, user_id: str, upload_id: object, kind: str) -> bytes:
    if not isinstance(upload_id, str) or not upload_id:
        raise AdapterError("UPLOAD_REQUIRED")
    store = _upload_store(controller)
    row = store.owned(user_id, upload_id)
    if row is None or row.get("kind") != kind:
        raise AdapterError("UPLOAD_REQUIRED")
    data = store.read(user_id, upload_id)
    if not data:
        raise AdapterError("UPLOAD_REQUIRED")
    return data


def _invoke(controller, job: Job, kind: str, payload: dict, runtime, ctx: Invocation):
    if runtime is None:
        raise AdapterError("CAPABILITY_UNAVAILABLE")
    if kind == "audio.transcribe":
        wav = _owned_bytes(controller, job.client_id, payload.get("upload_id"), "audio")
        return invoke_transcribe(runtime, wav, ctx)
    if kind == "audio.tts":
        return invoke_tts(runtime, payload, ctx)
    if kind == "audio.voice_clone":
        reference = _owned_bytes(controller, job.client_id, payload.get("reference_upload_id"), "audio")
        return invoke_voice_clone(runtime, payload, reference, ctx)
    if kind == "audio.translate":
        return invoke_translation(runtime, payload, ctx)
    if kind == "image.edit":
        image = _owned_bytes(controller, job.client_id, payload.get("source_upload_id"), "image")
        return invoke_image_edit(runtime, image, str(payload.get("prompt") or ""), ctx)
    if kind == "music.generate":
        return invoke_music(runtime, payload, ctx)
    if kind == "video.text_to_video":
        return invoke_video(runtime, payload, ctx)
    raise AdapterError("CAPABILITY_UNAVAILABLE")


def _persist(controller, job: Job, kind: str, result) -> tuple[str, str, str, int]:
    media_kind, ext, scene, mime = _MEDIA[kind]
    if isinstance(result, (bytes, bytearray)):
        data = bytes(result)
    else:
        data = json.dumps(result, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    artifact_id = "art_" + uuid.uuid4().hex
    platform = getattr(controller, "platform", None)
    if platform is None:
        uri = controller.storage.put(
            client_id=job.client_id, job_id=job.id, name=artifact_id + "." + ext, data=data,
        )
        path = str(Path(controller.storage.root) / job.client_id / job.id / (artifact_id + "." + ext))
    else:
        temporary = Path(controller.storage.root) / "_tmp" / (artifact_id + "." + ext)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(data)
        meta = platform.persist_media(
            user_id=job.client_id, project_id="direct", scene_id=scene, job_id=job.id,
            artifact_id=artifact_id, kind=media_kind, ext=ext, data=data, mime_type=mime,
            settings={"capability": kind, "prompt": job.prompt}, local_path=temporary,
        )
        path = "r2://" + platform.bucket + "/" + meta.object_key
        uri = path
        digest = meta.sha256
    controller.artifacts.insert(Artifact(
        artifact_id=artifact_id, job_id=job.id, engine="multimodal-local", model=primary_for(kind).model_id,
        prompt=job.prompt, seed=None, width=None, height=None, created_at=controller.clock.iso(),
        byte_count=len(data), sha256=digest, path=path,
    ))
    return artifact_id, uri, digest, len(data)
