"""One paid image through the public API. Unit tests do not import this module."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import urllib.error
import urllib.request
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

from media_engine.api.app import LocalAPIServer
from media_engine.engines.image.qwen import validate_png
from media_engine.orchestrator.controller import MediaController, WallClock
from media_engine.providers.vast_image import MACHINE_ID, VOLUME_ID, VastImageProvider
from media_engine.safety.limits import SafetyLimits, limits_from_env

ROOT = Path(__file__).resolve().parents[3]
PROMPT = (
    "A cinematic luxury Mediterranean hotel terrace at sunset, elegant modern architecture, "
    "palm trees, warm ambient lighting, sea in the background, premium travel photography, "
    "photorealistic, high detail, natural colors, no logos, no readable text"
)
CAP = Decimal("0.30")


def _api_key() -> str:
    value = os.environ.get("VAST_API_KEY", "").strip()
    if not value:
        raise SystemExit("VAST_API_KEY missing")
    return value


def _request(method: str, url: str, key: str, body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": "Bearer " + key, "Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status, json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        return exc.code, payload if isinstance(payload, dict) else {}


def _instances(key: str) -> list[dict]:
    from media_engine.providers.vast_lifecycle import instances_list_url
    status, payload = _request("GET", instances_list_url(), key)
    if status != 200:
        return []
    rows = payload.get("instances")
    return rows if isinstance(rows, list) else []


def _local(server: LocalAPIServer, method: str, path: str, token: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(server.url + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=30) as response:
            content = response.headers.get("Content-Type", "")
            raw = response.read()
            if content.startswith("image/"):
                return response.status, raw
            return response.status, json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main() -> int:
    if os.environ.get("PHASE2E_ACCEPTANCE") != "1":
        return 2
    vast_key = _api_key()
    token = os.environ.get("AI_MEDIA_ENGINE_API_KEY", "").strip() or hashlib.sha256(os.urandom(32)).hexdigest()
    limits = limits_from_env(SafetyLimits(live_external_providers=True, max_gpu_workers=1, max_job_attempts=1))
    clock = WallClock()
    runtime = ROOT / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    provider = VastImageProvider(api_key=vast_key, limits=limits, now=clock.now, runtime=runtime)
    storage = runtime / "phase2e-store"
    controller = MediaController(str(runtime / "phase2e.sqlite3"), str(storage), clock=clock, limits=limits, provider=provider)
    server = LocalAPIServer(controller, api_key=token)
    report: dict = {"base_url": "", "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()}
    started = time.time()
    existing = _instances(vast_key)
    if existing:
        report["state"] = "ACTIVE_INSTANCE_PRESENT"
        report["instances"] = [row.get("id") for row in existing]
        print(json.dumps({"state": report["state"], "instances": report["instances"]}))
        return 2
    try:
        server.start()
        report["base_url"] = server.url
        status, created = _local(server, "POST", "/v1/images/generations", token, {
            "prompt": PROMPT, "width": 1024, "height": 1024,
        })
        report["post_status"] = status
        report["job_id"] = created.get("job_id") if isinstance(created, dict) else None
        report["post_body_status"] = created.get("status") if isinstance(created, dict) else None
        if status != 202 or not report["job_id"]:
            report["state"] = "API_REJECTED"
            return 3
        deadline = time.time() + 1500
        done = {}
        while time.time() < deadline:
            hourly = provider.hourly_amount or Decimal("0.38")
            spent = (hourly * Decimal(str(max(0.0, time.time() - started))) / Decimal(3600)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
            report["spent"] = str(spent)
            if spent >= CAP - Decimal("0.02"):
                report["state"] = "BUDGET_STOP"
                return 4
            _status, done = _local(server, "GET", "/v1/jobs/" + report["job_id"], token)
            if isinstance(done, dict) and done.get("status") in {"completed", "failed"}:
                break
            time.sleep(5)
        else:
            report["state"] = "WORKER_TIMEOUT"
            return 5
        report["job"] = {key: done.get(key) for key in ("job_id", "status", "error_code", "artifact_id", "width", "height", "seed", "sha256", "progress")}
        if done.get("status") != "completed":
            report["state"] = str(done.get("error_code") or "FAILED")
            return 6
        status, png = _local(server, "GET", "/v1/artifacts/" + done["artifact_id"], token)
        if status != 200 or not isinstance(png, (bytes, bytearray)):
            report["state"] = "ARTIFACT_TRANSFER_FAILED"
            return 7
        width, height = validate_png(bytes(png))
        digest = hashlib.sha256(png).hexdigest()
        if (width, height) != (1024, 1024) or digest != done.get("sha256") or struct.unpack(">II", png[16:24]) != (1024, 1024):
            report["state"] = "ARTIFACT_INVALID"
            return 8
        out = runtime / "artifacts" / "phase2e" / "acceptance.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png)
        report["artifact"] = {"bytes": len(png), "sha256": digest, "width": width, "height": height}
        report["state"] = "IMAGE_API_MVP_COMPLETE"
        return 0
    finally:
        report["instance_id"] = provider._resource_id
        report["gpu"] = provider.gpu_model
        report["hourly"] = str(provider.hourly_amount)
        born = provider._started_at.get(report["instance_id"]) if report["instance_id"] else None
        try:
            controller.release_worker("ACCEPTANCE_SHUTDOWN")
        except Exception as exc:
            report["release_error"] = type(exc).__name__
        server.stop()
        gone = False
        for _ in range(12):
            rows = _instances(vast_key)
            report["instances"] = [row.get("id") for row in rows]
            if not rows:
                gone = True
                break
            for row in rows:
                if str(row.get("label") or "").startswith("media-engine:"):
                    _request("DELETE", f"https://console.vast.ai/api/v0/instances/{row.get('id')}/", vast_key)
            time.sleep(5)
        report["active_paid_instances"] = 0 if gone else len(report.get("instances") or [])
        _status, volumes = _request("GET", "https://console.vast.ai/api/v0/volumes/?owner=me", vast_key)
        found = False
        for row in volumes.get("volumes") or []:
            if int(row.get("id") or 0) == VOLUME_ID and int(row.get("machine_id") or 0) == MACHINE_ID:
                found = True
        report["volume_preserved"] = found
        report["lifetime_seconds"] = round(time.time() - (born or started), 3)
        hourly = provider.hourly_amount or Decimal("0")
        report["calculated_cost"] = str((hourly * Decimal(str(report["lifetime_seconds"])) / Decimal(3600)).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
        (runtime / "phase2e-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report.get(key) for key in (
            "state", "base_url", "job_id", "post_body_status", "job", "artifact",
            "gpu", "hourly", "instance_id", "lifetime_seconds", "calculated_cost",
            "active_paid_instances", "volume_preserved",
        )}))


if __name__ == "__main__":
    raise SystemExit(main())
