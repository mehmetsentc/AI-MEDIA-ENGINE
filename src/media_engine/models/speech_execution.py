"""Speech rehearsal on a fake GPU. This module never rents and never downloads weights."""
from __future__ import annotations

from pathlib import Path

from media_engine.infra.volume_audit import VOLUME_ID
from media_engine.models.audio import encode_wav
from media_engine.models.cost import CREDIT_RULES
from media_engine.models.errors import AdapterError
from media_engine.models.invoke import Invocation, invoke_transcribe, invoke_tts
from media_engine.models.speech_stage import ResidentGate, SpeechPrepError, verify_tree
from media_engine.platform.repository import PlatformBlocked
from media_engine.providers.base import OBSERVE_GONE, WorkerStageError
from media_engine.providers.execution_authorization import accept_speech_record
from media_engine.providers.live6_prepare import claim_execution_id
from media_engine.providers.session_heartbeat import HEARTBEAT_INTERVAL_SECONDS
from media_engine.providers.vast_watchdog import HEARTBEAT_STALE_SECONDS, instances_to_stop
from media_engine.safety.execution_budget import (
    LIVE7_CEILING_USD,
    LIVE7_RESERVE_SECONDS,
    open_live7_budget,
)

KNOWN_TEXT = "known courtyard line"
WHISPER = "Systran/faster-whisper-large-v3"
TTS = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


class _Cancel:
    def is_set(self) -> bool:
        return False


def run_speech_rehearsal(
    authorization,
    *,
    now: float,
    code_sha: str,
    recorded_digest: str | None,
    provider,
    platform,
    user_id: str,
    whisper_runtime,
    tts_runtime,
    claim_path: Path,
    machine_id: int,
    offer_id: int,
    instance_id: str,
    clock,
    volume_id: int | None = None,
    pause_after=None,
    weights_root: Path | None = None,
) -> dict:
    """Run both models on one fake instance. Vast providers are refused."""
    if getattr(provider, "name", "") == "vast":
        raise SpeechPrepError("GPU_RENTAL_REFUSED")
    accept_speech_record(
        authorization,
        now=now,
        code_sha=code_sha,
        recorded_digest=recorded_digest,
        volume_id=volume_id,
    )
    if (int(machine_id), int(offer_id)) not in authorization.allowed_pairs:
        raise WorkerStageError("MACHINE_SCOPE")
    claim_execution_id(claim_path, authorization.execution_id)
    weights = verify_tree(
        Path(weights_root) if weights_root is not None else Path("/no/speech/weights"),
        WHISPER,
    )
    budget = open_live7_budget(
        execution_id=authorization.execution_id,
        grant_execution_id=authorization.execution_id,
        grant_ceiling=LIVE7_CEILING_USD,
        hourly=authorization.max_hourly_usd,
        wall_now=clock.wall,
        monotonic_now=clock.mono,
    )
    created = provider.create(user_id)
    if provider.create_calls != 1 or not created.resource_id:
        raise SpeechPrepError("INSTANCE_LIMIT")
    gate = ResidentGate()
    counts = {"stages": 0, "loads": 0, "transcriptions": 0, "tts": 0}
    transcript = _transcribe(platform, user_id, whisper_runtime, gate, counts, budget, clock)
    if pause_after is not None:
        pause_after(budget)
    tts_status = "completed"
    wav = b""
    try:
        wav = _speak(platform, user_id, tts_runtime, gate, counts, budget, clock)
    except WorkerStageError as exc:
        if exc.code != "BUDGET_STOP":
            raise
        tts_status = "BUDGET_STOP"
    deleted = _delete(
        provider, created.resource_id, user_id,
        machine_id=machine_id, instance_id=instance_id,
        now=budget.spending_deadline_wall, deadline=budget.spending_deadline_wall,
    )
    return {
        "rehearsal": True,
        "weights": weights["status"],
        "gpu_verified": False,
        "cuda_gpu_support": False,
        "provider_billing_cap": False,
        "resource_id": created.resource_id,
        "creates": provider.create_calls,
        "stages": counts["stages"],
        "loads": counts["loads"],
        "transcriptions": counts["transcriptions"],
        "tts": counts["tts"],
        "transcript": transcript,
        "wav": wav,
        "tts_status": tts_status,
        "deleted": deleted,
        "resident": gate.model_id,
        "deletion_reserve_seconds": LIVE7_RESERVE_SECONDS,
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "heartbeat_stale_seconds": HEARTBEAT_STALE_SECONDS,
        "spending_deadline_wall": budget.spending_deadline_wall,
        "forbidden_volume_id": VOLUME_ID,
    }


def _transcribe(platform, user_id, runtime, gate, counts, budget, clock) -> str:
    _guard(budget, clock)
    _stage(gate, WHISPER, counts)
    _load(gate, WHISPER, counts)
    job_id = "speech-transcribe"
    service, model, price = CREDIT_RULES["audio.transcribe"]
    platform.authorize_paid(user_id, job_id, service=service, model=model, provider="local")
    try:
        wav = encode_wav([0.2, -0.2, 0.1, -0.1], 16000)
        result = invoke_transcribe(runtime, wav, Invocation(10**12, _Cancel()))
        if result["text"] != KNOWN_TEXT:
            raise AdapterError("TRANSCRIPT_MISMATCH")
        platform.settle(user_id, job_id, price)
    except Exception:
        platform.settle_failure(user_id, job_id, 0)
        raise
    counts["transcriptions"] += 1
    gate.finish()
    gate.unload()
    return result["text"]


def _speak(platform, user_id, runtime, gate, counts, budget, clock) -> bytes:
    job_id = "speech-tts"
    service, model, price = CREDIT_RULES["audio.tts"]
    platform.authorize_paid(user_id, job_id, service=service, model=model, provider="local")
    try:
        _guard(budget, clock)
        if gate.model_id == WHISPER:
            raise SpeechPrepError("MODEL_BUSY")
        _stage(gate, TTS, counts)
        _load(gate, TTS, counts)
        wav = invoke_tts(
            runtime,
            {"text": "A short line", "language": "English", "voice": "Ryan"},
            Invocation(10**12, _Cancel()),
        )
        if not wav.startswith(b"RIFF") or b"WAVE" not in wav[:16]:
            raise AdapterError("AUDIO_FORMAT")
        platform.settle(user_id, job_id, price)
    except Exception:
        platform.settle_failure(user_id, job_id, 0)
        raise
    counts["tts"] += 1
    gate.finish()
    gate.unload()
    return wav


def _stage(gate: ResidentGate, model_id: str, counts: dict) -> None:
    if counts["stages"] >= 2:
        raise SpeechPrepError("INSTANCE_LIMIT")
    decision = gate.plan(model_id)
    if decision["action"] == "REFUSE":
        raise SpeechPrepError("MODEL_BUSY")
    counts["stages"] += 1


def _load(gate: ResidentGate, model_id: str, counts: dict) -> None:
    if counts["loads"] >= 2:
        raise SpeechPrepError("INSTANCE_LIMIT")
    action = gate.begin(model_id)
    if action not in {"LOAD", "REUSE"}:
        raise SpeechPrepError("MODEL_BUSY")
    counts["loads"] += 1


def _guard(budget, clock) -> None:
    if budget.stop(clock.wall, clock.mono):
        raise WorkerStageError("BUDGET_STOP")


def _delete(provider, resource_id: str, owner: str, *, machine_id: int, instance_id: str, now: float, deadline: float) -> bool:
    rows = [{
        "id": str(instance_id),
        "machine_id": int(machine_id),
        "label": "media-engine:speech",
    }]
    found = instances_to_stop(
        rows, machine_id=int(machine_id), now=now, deadline=deadline, heartbeat_age=0,
    )
    if found != [str(instance_id)]:
        raise SpeechPrepError("DELETE_NOT_DUE")
    provider.terminate(resource_id)
    if provider.observe(resource_id, owner) != OBSERVE_GONE:
        raise SpeechPrepError("DELETE_UNVERIFIED")
    return True


def classify_probe(payload: dict) -> dict:
    """Imports do not prove a CUDA device. The GPU flag stays false."""
    return {
        "imports_ok": payload.get("imports_ok") is True,
        "torch": payload.get("torch"),
        "torchaudio": payload.get("torchaudio"),
        "faster_whisper": payload.get("faster_whisper"),
        "ctranslate2": payload.get("ctranslate2"),
        "qwen_tts": payload.get("qwen_tts"),
        "transformers": payload.get("transformers"),
        "accelerate": payload.get("accelerate"),
        "torch_cuda_is_available": payload.get("torch_cuda_is_available"),
        "cuda_gpu_support": False,
        "reason": "imports_are_not_gpu_proof",
    }
