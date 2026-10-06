"""Phase 2D: one owned Vast GPU, one Qwen-Image PNG, then destroy.

Tests pass a fake generator. The manual command is the only live caller.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Mapping, Optional, TextIO

from media_engine.engines.image.qwen import (
    FIRST_HEIGHT,
    FIRST_PROMPT,
    FIRST_SEED,
    FIRST_WIDTH,
    INFERENCE_STEPS,
    MODEL_ID,
    QwenImageEngine,
    sha256_hex,
)
from media_engine.engines.image.qwen_remote import __file__ as _REMOTE_FILE
from media_engine.providers.base import OBSERVE_FOREIGN, OBSERVE_GONE, OBSERVE_PRESENT, OBSERVE_UNKNOWN
from media_engine.providers.offers import GpuOffer, model_key
from media_engine.providers.planner import ResourceRequirements, max_estimated_gpu_cost, plan
from media_engine.providers.vast import VastDiscovery
from media_engine.providers.vast_lifecycle import (
    VastLifecycle,
    approval_token,
    instances_list_url,
    live_transport,
    ownership_label,
)
from media_engine.resources.ledger import ResourceLedger, ResourceState
from media_engine.safety.limits import SafetyLimits
from media_engine.usage import UsageEvent, UsageStore

APPROVED_GPU_MODEL = "RTX 5090"
MIN_VRAM_GB = Decimal("24")
MIN_RELIABILITY = Decimal("0.98")
MAX_HOURLY_USD = Decimal("0.40")
MAX_ESTIMATED_COST_USD = Decimal("0.20")
MAX_LIFETIME_SECONDS = 1800
MAX_CREATE_ATTEMPTS = 1
OWNER = "phase2d"
INSTANCE_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"
DISK_GB = 100
RUNTYPE = "ssh"


class Phase2DStop(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def policy_fields() -> dict[str, str]:
    return {
        "provider": "vast",
        "gpu_model": APPROVED_GPU_MODEL,
        "min_vram_gb": "24",
        "gpu_count": "1",
        "min_reliability": "0.98",
        "max_hourly_price_usd": "0.40",
        "max_gpu_lifetime_seconds": str(MAX_LIFETIME_SECONDS),
        "max_create_attempts": str(MAX_CREATE_ATTEMPTS),
        "max_gpu_workers": "1",
        "max_estimated_gpu_cost_usd": "0.20",
    }


def fingerprint_for(fields: Mapping[str, str]) -> str:
    raw = json.dumps(dict(fields), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def policy_fingerprint() -> str:
    return fingerprint_for(policy_fields())


def format_policy() -> str:
    fields = policy_fields()
    lines = ["PHASE 2D POLICY"]
    for key in (
        "provider", "gpu_model", "min_vram_gb", "gpu_count", "min_reliability",
        "max_hourly_price_usd", "max_gpu_lifetime_seconds", "max_create_attempts",
        "max_gpu_workers", "max_estimated_gpu_cost_usd",
    ):
        lines.append(f"{key}: {fields[key]}")
    lines.append("POLICY FINGERPRINT: " + policy_fingerprint())
    lines.append("STATUS: HUMAN_POLICY_APPROVAL_REQUIRED")
    return "\n".join(lines) + "\n"


def policy_requirements() -> ResourceRequirements:
    return ResourceRequirements(
        min_vram_gb=MIN_VRAM_GB,
        gpu_count=1,
        max_hourly_price_usd=MAX_HOURLY_USD,
        min_reliability=MIN_RELIABILITY,
        compatible_gpu_models=(APPROVED_GPU_MODEL,),
        max_job_seconds=MAX_LIFETIME_SECONDS,
    )


def preconditions(env: Mapping[str, str]) -> str:
    key = env.get("VAST_API_KEY") or ""
    if not key.strip():
        raise Phase2DStop("VAST_API_KEY_MISSING")
    if env.get("LIVE_EXTERNAL_PROVIDERS") != "true":
        raise Phase2DStop("LIVE_EXTERNAL_PROVIDERS_DISABLED")
    if env.get("VAST_PROVISIONING") != "1":
        raise Phase2DStop("PROVISION_DISABLED")
    approval = env.get("VAST_HUMAN_POLICY_APPROVAL") or ""
    if approval.strip() == "":
        raise Phase2DStop("HUMAN_POLICY_APPROVAL_REQUIRED")
    if approval != "approve-policy:" + policy_fingerprint():
        raise Phase2DStop("APPROVAL_MISMATCH")
    if SafetyLimits().max_gpu_workers != 1 or SafetyLimits().max_job_attempts != 1:
        raise Phase2DStop("MAX_GPU_WORKERS")
    return key


def accept_offer(offer: GpuOffer) -> None:
    if offer.provider != "vast":
        raise Phase2DStop("OFFER_MISMATCH")
    if model_key(offer.gpu_model) != model_key(APPROVED_GPU_MODEL):
        raise Phase2DStop("OFFER_MISMATCH")
    if offer.gpu_count != 1 or offer.vram_gb < MIN_VRAM_GB:
        raise Phase2DStop("OFFER_MISMATCH")
    if offer.reliability is None or offer.reliability < MIN_RELIABILITY:
        raise Phase2DStop("OFFER_MISMATCH")
    if offer.availability not in (None, "rentable"):
        raise Phase2DStop("OFFER_GONE")
    cost = max_estimated_gpu_cost(offer.hourly_price_usd, MAX_LIFETIME_SECONDS)
    if offer.hourly_price_usd > MAX_HOURLY_USD:
        raise Phase2DStop("PRICE_ABOVE_CEILING")
    if cost > MAX_ESTIMATED_COST_USD:
        raise Phase2DStop("COST_ABOVE_CEILING")


def select_one(offers: list[GpuOffer]) -> GpuOffer:
    selected = plan(offers, policy_requirements()).selected
    if selected is None:
        raise Phase2DStop("OFFER_GONE")
    accept_offer(selected)
    return selected


def run_phase2d(
    env: Mapping[str, str],
    *,
    db_path: str,
    artifact_dir: str,
    search_transport,
    lifecycle_transport,
    generate: Callable[[str, GpuOffer, float], tuple[bytes, dict]],
    now: Callable[[], float],
    stdout: Optional[TextIO] = None,
) -> int:
    """One discovery, one create, one image, then destroy. No second create."""
    out = sys.stdout if stdout is None else stdout
    try:
        key = preconditions(env)
        if ResourceLedger(db_path).unresolved():
            raise Phase2DStop("UNRESOLVED_RESOURCE")
    except Phase2DStop as exc:
        _say(out, exc.code)
        return 2
    started = now()
    deadline = started + MAX_LIFETIME_SECONDS
    requirements = policy_requirements()
    try:
        offers = VastDiscovery(transport=search_transport).search(
            requirements.to_search(gpu_model=APPROVED_GPU_MODEL),
            api_key=key,
            read_only=True,
            limit=64,
        )
        selected = select_one(offers)
    except Phase2DStop as exc:
        _say(out, exc.code)
        return 3
    _say(out, "OFFER " + selected.offer_id)
    _say(out, "GPU " + selected.gpu_model)
    _say(out, "VRAM " + str(selected.vram_gb))
    _say(out, "HOURLY " + str(selected.hourly_price_usd))
    _say(out, "RELIABILITY " + str(selected.reliability))
    _say(out, "PERFORMANCE " + str(selected.performance))
    _say(out, "COST600 " + str(max_estimated_gpu_cost(selected.hourly_price_usd, 600)))
    _say(out, "COST1800 " + str(max_estimated_gpu_cost(selected.hourly_price_usd, MAX_LIFETIME_SECONDS)))
    lifecycle = VastLifecycle(
        db_path,
        limits=SafetyLimits(
            live_external_providers=True,
            max_hourly_price_usd=MAX_HOURLY_USD,
            max_gpu_workers=1,
            max_job_attempts=1,
            max_worker_lifetime_seconds=float(MAX_LIFETIME_SECONDS),
        ),
        transport=lifecycle_transport,
        now=now,
    )
    try:
        created = lifecycle.create(
            offers=[selected],
            requirements=requirements,
            offer_id=selected.offer_id,
            owner=OWNER,
            api_key=key,
            provision=True,
            human_approval=approval_token(selected.offer_id),
            image=INSTANCE_IMAGE,
            disk_gb=DISK_GB,
            runtype=RUNTYPE,
        )
    except Exception as exc:
        _say(out, str(getattr(exc, "code", "CREATE_BLOCKED")))
        return 4
    if created.outcome == "definite_failure":
        _say(out, "CREATE_FAILED")
        return 4
    if created.outcome != "created" or not created.resource_id:
        _say(out, "AMBIGUOUS_CREATE")
        return 4
    instance_id = created.resource_id
    if instance_id == selected.offer_id:
        _say(out, "CREATE_FAILED")
        return 4
    try:
        kind = lifecycle.observe(instance_id, OWNER, api_key=key, provision=True)
    except Exception:
        _destroy_if_present(lifecycle, instance_id, key)
        _say(out, OBSERVE_UNKNOWN)
        return 5
    if kind != OBSERVE_PRESENT:
        _say(out, "FOREIGN" if kind == OBSERVE_FOREIGN else "OWNERSHIP_UNKNOWN")
        return 5
    png = b""
    report: dict = {}
    generated = False
    try:
        if now() >= deadline or _over_budget(selected, now() - started):
            raise Phase2DStop("COST_ABOVE_CEILING")
        png, report = generate(instance_id, selected, deadline)
        image = QwenImageEngine(lambda **_kwargs: png).render(
            FIRST_PROMPT, width=FIRST_WIDTH, height=FIRST_HEIGHT, seed=FIRST_SEED,
        )
        path = _write_artifact(artifact_dir, instance_id, image, selected, report, started, now())
        generated = True
        _say(out, "ARTIFACT " + str(path))
        _say(out, "SHA256 " + sha256_hex(image))
    except Exception as exc:
        _say(out, getattr(exc, "code", "GENERATION_FAILED"))
    destroyed = _destroy_owned(lifecycle, instance_id, key)
    if not destroyed:
        _say(out, "MANUAL_CLEANUP_REQUIRED")
        _say(out, "INSTANCE " + instance_id)
        return 8
    try:
        after = lifecycle.observe(instance_id, OWNER, api_key=key, provision=True)
    except Exception:
        after = OBSERVE_UNKNOWN
    elapsed = now() - started
    final = "TERMINATED" if after == OBSERVE_GONE else "STILL_PRESENT"
    _mark_terminated(lifecycle, instance_id, final)
    _record_usage(db_path, selected, instance_id, started, now(), final, elapsed, report)
    _say(out, final)
    _say(out, "INSTANCE " + instance_id)
    if not generated:
        return 7
    return 0 if final == "TERMINATED" else 6


def ssh_generate(instance_id: str, offer: GpuOffer, deadline: float) -> tuple[bytes, dict]:
    """Copy the runner to the owned instance, generate once, and return the PNG."""
    key = os.environ.get("VAST_API_KEY") or ""
    row = _wait_ssh(instance_id, key, deadline)
    host = str(row.get("ssh_host") or "")
    port = str(row.get("ssh_port") or "")
    if not host or not port.isdigit():
        raise Phase2DStop("RUNTIME_NOT_READY")
    identity, public = _ephemeral_ssh_key()
    _attach_ssh_key(instance_id, public, key)
    script = Path(_REMOTE_FILE).read_bytes()
    job = json.dumps({
        "prompt": FIRST_PROMPT,
        "seed": FIRST_SEED,
        "width": FIRST_WIDTH,
        "height": FIRST_HEIGHT,
        "steps": INFERENCE_STEPS,
    }).encode("utf-8")
    remaining = max(1, int(deadline - time.time()))
    _ssh(host, port, identity, "mkdir -p /workspace && cat > /workspace/phase2d_gen.py", script, 60)
    _ssh(host, port, identity, "cat > /workspace/phase2d_job.json", job, 30)
    try:
        code, _out, err = _ssh(
            host,
            port,
            identity,
            "python3 -m pip install -q 'diffusers>=0.35' transformers accelerate safetensors pillow "
            "&& python3 /workspace/phase2d_gen.py",
            None,
            remaining,
        )
    except subprocess.TimeoutExpired as exc:
        raise Phase2DStop("MODEL_DOWNLOAD_FAILED") from exc
    local_dir = Path("runtime/artifacts/phase2d")
    local_dir.mkdir(parents=True, exist_ok=True)
    report_path = local_dir / f".{instance_id}.remote.json"
    png_path = local_dir / f".{instance_id}.download.png"
    _scp(host, port, identity, "/workspace/phase2d_report.json", report_path, 60)
    report: dict = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    if code != 0:
        detail = str(report.get("error") or "GENERATION_FAILED")
        if "DOWNLOAD" in detail:
            raise Phase2DStop("MODEL_DOWNLOAD_FAILED")
        if "IMPORT" in detail or "LOAD" in detail:
            raise Phase2DStop("MODEL_LOAD_FAILED")
        raise Phase2DStop("GENERATION_FAILED")
    _scp(host, port, identity, "/workspace/phase2d_output.png", png_path, 60)
    if not png_path.exists() or png_path.stat().st_size <= 0:
        raise Phase2DStop("GENERATION_FAILED")
    _ = err
    return png_path.read_bytes(), report


def _wait_ssh(instance_id: str, api_key: str, deadline: float) -> dict:
    headers = {
        "Authorization": "Bearer " + api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    while time.time() < deadline:
        raw = live_transport("GET", instances_list_url(), b"", headers)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise Phase2DStop("RUNTIME_NOT_READY") from exc
        for row in payload.get("instances") or []:
            if str(row.get("id")) != instance_id:
                continue
            if row.get("ssh_host") and row.get("ssh_port"):
                _reject_over_price(row)
                return row
        time.sleep(5)
    raise Phase2DStop("RUNTIME_NOT_READY")


def _ssh(host: str, port: str, identity: str, command: str, data: Optional[bytes], timeout: int) -> tuple[int, bytes, bytes]:
    proc = subprocess.run(
        [
            "ssh", "-i", identity, "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-p", port, "root@" + host, command,
        ],
        input=data,
        capture_output=True,
        timeout=max(1, timeout),
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _scp(host: str, port: str, identity: str, remote: str, local: Path, timeout: int) -> None:
    subprocess.run(
        [
            "scp", "-i", identity, "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-P", port, "root@" + host + ":" + remote, str(local),
        ],
        capture_output=True,
        timeout=max(1, timeout),
        check=False,
    )


_SSH_ATTACH = re.compile(r"https://console\.vast\.ai/api/v0/instances/([0-9]+)/ssh/")


def ssh_attach_allowed(method: str, url: str) -> bool:
    return method == "POST" and _SSH_ATTACH.fullmatch(url) is not None


def attach_ssh_key(transport, instance_id: str, public_key: str, api_key: str) -> None:
    """Attach one public key to one owned instance. The private key never leaves this machine."""
    if not instance_id.isdigit():
        raise Phase2DStop("OWNERSHIP_UNKNOWN")
    text = public_key.strip()
    if not text.startswith("ssh-ed25519 ") or "PRIVATE" in text or "\n" in text:
        raise Phase2DStop("SSH_KEY_REFUSED")
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/ssh/"
    if not ssh_attach_allowed("POST", url):
        raise Phase2DStop("COMMAND_BLOCKED")
    body = json.dumps({"ssh_key": text}, separators=(",", ":")).encode("utf-8")
    headers = {
        "Authorization": "Bearer " + api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    transport("POST", url, body, headers)


def _attach_ssh_key(instance_id: str, public_key: str, api_key: str) -> None:
    attach_ssh_key(_ssh_attach_transport, instance_id, public_key, api_key)


def _ssh_attach_transport(method: str, url: str, body: bytes, headers: dict) -> bytes:
    if not ssh_attach_allowed(method, url):
        raise Phase2DStop("COMMAND_BLOCKED")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _ephemeral_ssh_key() -> tuple[str, str]:
    root = Path("runtime/phase2d-ssh")
    root.mkdir(parents=True, exist_ok=True)
    private = root / "id_ed25519"
    public = root / "id_ed25519.pub"
    if not private.exists() or not public.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(private), "-N", "", "-q"],
            check=True,
            capture_output=True,
        )
    os.chmod(private, 0o600)
    return str(private), public.read_text(encoding="utf-8").strip()


def _reject_over_price(row: dict) -> None:
    raw = row.get("dph_total")
    if raw is None:
        return
    try:
        hourly = Decimal(str(raw))
    except Exception:
        return
    if hourly > MAX_HOURLY_USD:
        raise Phase2DStop("PRICE_ABOVE_CEILING")
    if max_estimated_gpu_cost(hourly, MAX_LIFETIME_SECONDS) > MAX_ESTIMATED_COST_USD:
        raise Phase2DStop("COST_ABOVE_CEILING")


def _over_budget(offer: GpuOffer, elapsed: float) -> bool:
    if elapsed <= 0:
        return False
    return max_estimated_gpu_cost(offer.hourly_price_usd, int(elapsed) + 1) > MAX_ESTIMATED_COST_USD


def _destroy_owned(lifecycle: VastLifecycle, instance_id: str, key: str) -> bool:
    for _ in range(2):
        try:
            lifecycle.terminate(
                instance_id,
                OWNER,
                api_key=key,
                provision=True,
                human_approval="approve-terminate:" + instance_id,
            )
            return True
        except Exception:
            continue
    return False


def _destroy_if_present(lifecycle: VastLifecycle, instance_id: str, key: str) -> None:
    try:
        kind = lifecycle.observe(instance_id, OWNER, api_key=key, provision=True)
    except Exception:
        return
    if kind == OBSERVE_PRESENT:
        _destroy_owned(lifecycle, instance_id, key)


def _mark_terminated(lifecycle: VastLifecycle, instance_id: str, status: str) -> None:
    record = lifecycle.ledger.get(instance_id)
    if record is None or status not in {"TERMINATED", "STILL_PRESENT"}:
        return
    if status == "TERMINATED":
        record.state = ResourceState.TERMINATED
        record.last_error = status
        record.updated_at = lifecycle._now()
        lifecycle.ledger.upsert(record)


def _write_artifact(
    artifact_dir: str,
    instance_id: str,
    png: bytes,
    offer: GpuOffer,
    report: dict,
    started: float,
    finished: float,
) -> Path:
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"phase2d-{instance_id}.png"
    if path.exists():
        stamp = datetime.fromtimestamp(finished, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = root / f"phase2d-{instance_id}-{stamp}.png"
    path.write_bytes(png)
    meta = {
        "model_id": MODEL_ID,
        "prompt": FIRST_PROMPT,
        "seed": FIRST_SEED,
        "width": FIRST_WIDTH,
        "height": FIRST_HEIGHT,
        "steps": INFERENCE_STEPS,
        "provider": "vast",
        "gpu_model": offer.gpu_model,
        "offer_id": offer.offer_id,
        "instance_id": instance_id,
        "hourly_price_usd": str(offer.hourly_price_usd),
        "owner_label": ownership_label(OWNER),
        "sha256": sha256_hex(png),
        "bytes": len(png),
        "created_at": _iso(started),
        "finished_at": _iso(finished),
        "remote": report,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def _record_usage(
    db_path: str,
    offer: GpuOffer,
    instance_id: str,
    started: float,
    finished: float,
    status: str,
    elapsed: float,
    report: dict,
) -> None:
    estimated = offer.hourly_price_usd * Decimal(str(max(elapsed, 0))) / Decimal(3600)
    UsageStore(db_path).record(UsageEvent(
        job_id="phase2d-" + instance_id,
        client_id=OWNER,
        job_type="PHASE2D_IMAGE",
        engine_id="qwen-image",
        started_at=_iso(started),
        finished_at=_iso(finished),
        duration_seconds=float(elapsed),
        attempt_count=1,
        estimated_cost_usd=str(estimated),
        status=status,
        provider="vast",
        gpu_model=offer.gpu_model,
        hourly_price_usd=str(offer.hourly_price_usd),
        gpu_seconds=float(elapsed),
        actual_cost_usd=None,
    ))
    _ = report


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _say(out: TextIO, line: str) -> None:
    out.write(line + "\n")


def main(environ: Optional[Mapping[str, str]] = None, stdout: Optional[TextIO] = None) -> int:
    env = os.environ if environ is None else environ
    root = Path("runtime")
    root.mkdir(parents=True, exist_ok=True)
    return run_phase2d(
        env,
        db_path=str(root / "phase2d.sqlite3"),
        artifact_dir=str(root / "artifacts" / "phase2d"),
        search_transport=None,
        lifecycle_transport=live_transport,
        generate=ssh_generate,
        now=lambda: datetime.now(timezone.utc).timestamp(),
        stdout=stdout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
