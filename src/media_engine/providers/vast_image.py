"""Vast image worker. Prefer the warm cache, otherwise stage it from R2.

One GPU at a time. The warm volume is attached only when the chosen
machine is the machine that already holds it.
"""
from __future__ import annotations

import json
import os
import shlex
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional

from media_engine.engines.image.model_cache import (
    MANIFEST_NAME,
    QWEN_CACHE_MIN_BYTES,
    QWEN_FILE_COUNT,
    REPO_ID,
)
from media_engine.engines.image.qwen import FIRST_HEIGHT, FIRST_WIDTH, INFERENCE_STEPS
from media_engine.providers.base import (
    OBSERVE_FOREIGN,
    OBSERVE_GONE,
    OBSERVE_PRESENT,
    OBSERVE_UNKNOWN,
    CreateResult,
    GPUProvider,
    Quote,
    WorkerStageError,
)
from media_engine.providers.phase2d import INSTANCE_IMAGE
from media_engine.providers.phase_clock import PHASE_LIMITS, PhaseWatch, note_transfer_text
from media_engine.providers.vast_lifecycle import classify_create_response, instances_list_url, ownership_label
from media_engine.safety.limits import SafetyLimits

MACHINE_ID = 47281
VOLUME_ID = 54653022
MOUNT_PATH = "/models"
REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
WARM_DISK_GB = 40
COLD_DISK_GB = 140
MIN_VRAM_MIB = 44 * 1024
OPERATION_BUDGET_USD = Decimal("1.00")


def active_operation_budget() -> Decimal:
    """The approved image test can tighten the planner with STUDIO_GPU_BUDGET_USD."""
    raw = os.environ.get("STUDIO_GPU_BUDGET_USD", "").strip()
    if not raw:
        return OPERATION_BUDGET_USD
    try:
        amount = Decimal(raw)
    except Exception:
        return OPERATION_BUDGET_USD
    if amount <= 0:
        return OPERATION_BUDGET_USD
    return min(amount, OPERATION_BUDGET_USD)


OVERHEAD_SECONDS = 600
GENERATE_ALLOWANCE_SECONDS = 180
# Planning rate is a quarter of the advertised download, between the slow
# and fast transfers already measured on Vast.
LINE_FRACTION = 0.25
BUNDLES = "https://console.vast.ai/api/v0/bundles/"
VOLUMES = "https://console.vast.ai/api/v0/volumes/?owner=me"
VENV = "/workspace/phase2d-venv"
PYTHON = VENV + "/bin/python"

Transport = Callable[[str, str, bytes, dict[str, str]], tuple[int, bytes]]
SSHRun = Callable[[str, str, str, int], tuple[int, str, str]]


def _live_transport(method: str, url: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    if not _url_allowed(method, url):
        raise WorkerStageError("PROVIDER_CREATE_FAILED")
    request = urllib.request.Request(url, data=body or None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return int(response.status), response.read(1_000_000)
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(20_000)
    except Exception:
        return 0, b""


def _url_allowed(method: str, url: str) -> bool:
    if method == "POST" and url == BUNDLES:
        return True
    if method == "GET" and url == VOLUMES:
        return True
    if method == "PUT" and url.startswith("https://console.vast.ai/api/v0/asks/") and url.endswith("/"):
        return True
    if method == "GET" and url.startswith("https://console.vast.ai/api/v1/instances/?"):
        return True
    if method == "POST" and url.startswith("https://console.vast.ai/api/v0/instances/") and url.endswith("/ssh/"):
        return True
    if method == "DELETE" and url.startswith("https://console.vast.ai/api/v0/instances/") and url.count("/") >= 6:
        return True
    return False


class VastImageProvider(GPUProvider):
    name = "vast"

    def __init__(self, *, api_key: str, limits: Optional[SafetyLimits] = None,
                 transport: Optional[Transport] = None, now: Optional[Callable[[], float]] = None,
                 ssh_run: Optional[SSHRun] = None, sleep: Optional[Callable[[float], None]] = None,
                 runtime: Optional[Path] = None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise WorkerStageError("PROVIDER_CREATE_FAILED")
        self._api_key = api_key.strip()
        self.limits = limits or SafetyLimits()
        self._transport = transport or _live_transport
        self._now = now or time.time
        self._ssh_run = ssh_run
        self._sleep = sleep or time.sleep
        self.runtime = runtime or Path(os.environ.get("MEDIA_ENGINE_RUNTIME", "runtime"))
        self.gpu_model = ""
        self.machine_id = ""
        self.hourly_amount = Decimal("0")
        self.offer_id = ""
        self.cache_source = ""
        self.disk_gb = WARM_DISK_GB
        self.fits_operation_budget = False
        self.projected_cost_usd = Decimal("0")
        self.transfer_seconds = None
        self.transfer_mb_per_sec = None
        self.runtime_ready = False
        self.worker_state = ""
        self.cost_before_generation_usd = Decimal("0")
        self.last_instance_id = ""
        self._put_done = False
        self._owner = ""
        self._label = ""
        self._resource_id = ""
        self._host = ""
        self._port = ""
        self._busy: set[str] = set()
        self._gone: set[str] = set()
        self._started_at: dict[str, float] = {}
        self.commands: list[str] = []

    def quote(self, owner: str) -> Quote:
        if not owner:
            raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
        self._owner = owner
        self._label = ownership_label(owner)
        if not self._volume_ready():
            raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
        chosen = self._select_offer()
        if chosen is None:
            raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
        self.offer_id = chosen.offer_id
        self.gpu_model = chosen.gpu_model
        self.machine_id = str(chosen.machine_id)
        self.hourly_amount = chosen.hourly
        self.cache_source = chosen.source
        self.disk_gb = chosen.disk_gb
        self.projected_cost_usd = chosen.projected_cost
        self.cost_before_generation_usd = chosen.cost_before_generation
        self.fits_operation_budget = chosen.projected_cost <= OPERATION_BUDGET_USD
        if chosen.source == "r2_cold_stage" and not _r2_configured():
            raise WorkerStageError("MODEL_CACHE_INVALID")
        return Quote(hourly_price_usd=self.hourly_amount, provider="vast")

    def create(self, owner: str) -> CreateResult:
        if self._put_done:
            return CreateResult(outcome="definite_failure")
        self._put_done = True
        if not self.offer_id:
            self.quote(owner)
        label = ownership_label(owner)
        image = worker_image()
        body = {
            "label": label,
            "image": image,
            "disk": self.disk_gb,
            "runtype": "ssh",
            "cancel_unavail": True,
        }
        if portable_worker(image):
            body["onstart"] = WORKER_ONSTART
            body["env"] = {
                "HF_HOME": "/models/hf-cache",
                "HUGGINGFACE_HUB_CACHE": "/models/hf-cache",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "WORKER_PORT": "8080",
            }
        if self.cache_source == "warm_cache":
            body["volume_info"] = {
                "mount_path": MOUNT_PATH, "create_new": False, "volume_id": VOLUME_ID,
            }
        status, raw = self._request("PUT", f"https://console.vast.ai/api/v0/asks/{self.offer_id}/", body)
        if status <= 0:
            outcome, instance_id = "ambiguous", None
        else:
            outcome, instance_id, _error = classify_create_response(status, raw)
        if outcome == "ambiguous":
            found = self._await_owned(label)
            if len(found) == 1:
                outcome, instance_id = "created", found[0]
        if outcome == "created" and instance_id:
            self._resource_id = instance_id
            self.last_instance_id = instance_id
            self._owner = owner
            self._label = label
            self._started_at[instance_id] = self._now()
            return CreateResult(outcome="created", resource_id=instance_id)
        if outcome == "definite_failure":
            return CreateResult(outcome="definite_failure")
        return CreateResult(outcome="ambiguous")

    def status(self, resource_id: str) -> str:
        if resource_id in self._gone:
            return "TERMINATED"
        if resource_id in self._busy:
            return "BUSY"
        if resource_id == self._resource_id:
            return "READY"
        return "TERMINATED"

    def begin_busy(self, resource_id: str) -> None:
        self._busy.add(resource_id)

    def end_busy(self, resource_id: str) -> None:
        self._busy.discard(resource_id)

    def observe(self, resource_id: str, owner: str) -> str:
        row = self._instance_row(resource_id)
        if row is None:
            return OBSERVE_GONE if resource_id not in self._gone else OBSERVE_GONE
        label = str(row.get("label") or "")
        if label != ownership_label(owner) or not self._same_machine(row):
            return OBSERVE_FOREIGN
        return OBSERVE_PRESENT

    def terminate(self, resource_id: str) -> None:
        row = self._instance_row(resource_id)
        if row is None:
            self._gone.add(resource_id)
            self._put_done = False
            return
        label = str(row.get("label") or "")
        if not label.startswith("media-engine:") or not self._same_machine(row):
            raise WorkerStageError("OWNERSHIP_MISMATCH")
        if str(row.get("id")) != str(resource_id):
            raise WorkerStageError("OWNERSHIP_MISMATCH")
        self._request("DELETE", f"https://console.vast.ai/api/v0/instances/{resource_id}/", None)
        self._gone.add(resource_id)
        self._busy.discard(resource_id)
        if self._resource_id == resource_id:
            self._resource_id = ""
        self._put_done = False

    def list_owned(self) -> list[dict]:
        rows = []
        for row in self._instances():
            if str(row.get("label") or "").startswith("media-engine:"):
                rows.append({
                    "resource_id": str(row.get("id")),
                    "owner": self._owner,
                    "status": str(row.get("actual_status") or ""),
                    "hourly_price_usd": str(row.get("dph_total") or self.hourly_amount),
                })
        return rows

    def boot(self, resource_id: str) -> None:
        if portable_worker(worker_image()):
            self._wait_until_ready(resource_id)
            return
        if self._ssh_run is None:
            self._attach_and_wait(resource_id)
            return
        code, _out, _err = self._ssh_run(self._host or "worker", self._port or "22", "echo ready", 30)
        if code != 0:
            raise WorkerStageError("WORKER_BOOT_FAILED")

    def confirm_cache(self, resource_id: str) -> None:
        if self.cache_source == "r2_cold_stage":
            self._stage_r2()
        code, out, _err = self._remote(_CACHE_CHECK, 120)
        if code != 0 or not _cache_ok(out, require_mount=self.cache_source != "r2_cold_stage"):
            raise WorkerStageError("MODEL_CACHE_INVALID")

    def prepare_runtime(self, resource_id: str) -> None:
        if self.runtime_ready:
            return
        worker = getattr(self, "_worker_ssh", None)
        if worker is not None and self._host and self._port:
            worker.stage(self._host, self._port)
        code, out, _err = self._remote(_RUNTIME_READY, 60, "RUNTIME_TIMEOUT")
        if code == 0 and "RUNTIME_READY" in out:
            self.runtime_ready = True
            return
        limit = int(PHASE_LIMITS["runtime_preparing"][0])
        code, _out, _err = self._remote(_RUNTIME_INSTALL, limit, "RUNTIME_TIMEOUT")
        if code != 0:
            raise WorkerStageError("RUNTIME_PREPARE_FAILED")

    def ensure_model(self, resource_id: str) -> None:
        code, out, _err = self._remote(_IMPORT_CHECK, 120)
        if code != 0 or "QwenImagePipeline" not in out:
            raise WorkerStageError("MODEL_LOAD_FAILED")

    def generate(self, *, prompt: str, width: int, height: int, seed: int) -> bytes:
        python = "python3" if getattr(self, "runtime_ready", False) else PYTHON
        command = _generate_command(prompt, width, height, seed, python=python)
        self.commands.append(command)
        code, out, _err = self._remote(command, 900)
        if code != 0 or "PNG_READY" not in out:
            if "LOAD_FAILED" in out:
                raise WorkerStageError("MODEL_LOAD_FAILED")
            raise WorkerStageError("GENERATION_FAILED")
        return self._fetch_png()

    def _remote(
        self,
        command: str,
        timeout: int,
        timeout_code: str = "WORKER_TIMEOUT",
    ) -> tuple[int, str, str]:
        if self._ssh_run is None:
            self._ensure_ssh()
        if self._ssh_run is None:
            raise WorkerStageError("WORKER_BOOT_FAILED")
        try:
            return self._ssh_run(self._host or "worker", self._port or "22", command, timeout)
        except TimeoutError as exc:
            raise WorkerStageError(timeout_code) from exc

    def _ensure_ssh(self) -> None:
        """Attach the existing SSH session the first time cache or generate needs it."""
        if self._ssh_run is not None:
            return
        if not self._resource_id:
            raise WorkerStageError("WORKER_BOOT_FAILED")
        self._attach_and_wait(self._resource_id)

    def _fetch_png(self) -> bytes:
        code, out, _err = self._remote("base64 /workspace/phase2d_output.png", 120)
        if code != 0 or not out.strip():
            raise WorkerStageError("ARTIFACT_TRANSFER_FAILED")
        import base64
        alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n")
        cleaned = "".join(char for char in out if char in alphabet)
        try:
            data = base64.b64decode(cleaned.encode("ascii"), validate=False)
        except Exception as exc:
            raise WorkerStageError("ARTIFACT_TRANSFER_FAILED") from exc
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise WorkerStageError("ARTIFACT_TRANSFER_FAILED")
        return data

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": "Bearer " + self._api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, url: str, body: Optional[dict]) -> tuple[int, bytes]:
        raw = b"" if body is None else json.dumps(body).encode("utf-8")
        try:
            return self._transport(method, url, raw, self._headers())
        except WorkerStageError:
            raise
        except Exception:
            return 0, b""

    def _volume_ready(self) -> bool:
        status, raw = self._request("GET", VOLUMES, None)
        if status != 200:
            return False
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return False
        volumes = payload.get("volumes") if isinstance(payload, dict) else None
        if not isinstance(volumes, list):
            return False
        for row in volumes:
            if not isinstance(row, dict):
                continue
            if int(row.get("id") or 0) == VOLUME_ID and int(row.get("machine_id") or 0) == MACHINE_ID:
                return True
        return False

    def _same_machine(self, row: dict) -> bool:
        if not self.machine_id:
            return True
        return int(row.get("machine_id") or 0) == int(self.machine_id)

    def _select_offer(self) -> Optional["ImageChoice"]:
        # 1. An already-running worker is reused by the controller.
        # 2. A rentable GPU on the warm-cache machine.
        # 3. Another provider can be added later.
        # 4. R2 cold stage only when MEDIA_ENGINE_ALLOW_COLD_STAGE is set.
        warm = choose_offer(
            self._bundle_offers({
                "limit": 16,
                "type": "on-demand",
                "num_gpus": {"eq": 1},
                "machine_id": {"eq": MACHINE_ID},
                "gpu_ram": {"gte": MIN_VRAM_MIB},
            }),
            volume_machine=MACHINE_ID,
            volume_ready=True,
            max_hourly=self.limits.max_hourly_price_usd,
            operation_budget=active_operation_budget(),
            allow_cold_stage=False,
        )
        if warm is not None or not cold_stage_allowed():
            return warm
        return choose_offer(
            self._bundle_offers({
                "limit": 64,
                "type": "on-demand",
                "rentable": {"eq": True},
                "num_gpus": {"eq": 1},
                "gpu_ram": {"gte": MIN_VRAM_MIB},
                "order": [["dph_total", "asc"]],
            }),
            volume_machine=MACHINE_ID,
            volume_ready=True,
            max_hourly=self.limits.max_hourly_price_usd,
            operation_budget=active_operation_budget(),
            allow_cold_stage=True,
        )

    def _bundle_offers(self, query: dict) -> list:
        status, raw = self._request("POST", BUNDLES, query)
        if status != 200:
            return []
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return []
        offers = payload.get("offers") if isinstance(payload, dict) else None
        return offers if isinstance(offers, list) else []

    def _stage_r2(self) -> None:
        from media_engine.engines.image import model_cache, qwen_remote, r2_cache
        from media_engine.providers.phase2d import r2_worker_env
        from media_engine.providers.vast_ssh import WorkerSSH
        env = dict(os.environ)
        env["MODEL_CACHE_PROVIDER"] = "r2"
        payload = r2_worker_env(env)
        if payload is None or not self._host or not self._port:
            raise WorkerStageError("MODEL_CACHE_INVALID")
        worker = getattr(self, "_worker_ssh", None) or WorkerSSH(self.runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        env_path = self.runtime / "phase2e-r2.env"
        env_path.write_bytes(payload)
        os.chmod(env_path, 0o600)
        worker.run(self._host, self._port, "mkdir -p /workspace /models", 30)
        for module in (model_cache, r2_cache, qwen_remote):
            path = Path(module.__file__)
            worker._copy(self._host, self._port, path, "/workspace/" + path.name)
        worker._copy(self._host, self._port, env_path, "/workspace/phase2d-r2.env")
        started = time.perf_counter()
        self._watch_cache_sync()
        self.transfer_seconds = round(time.perf_counter() - started, 3)

    def _instances(self) -> list[dict]:
        status, raw = self._request("GET", instances_list_url(), None)
        if status != 200:
            return []
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return []
        rows = payload.get("instances") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict)]

    def _instance_row(self, resource_id: str) -> Optional[dict]:
        for row in self._instances():
            if str(row.get("id")) == str(resource_id):
                return row
        return None

    def _await_owned(self, label: str) -> list[str]:
        found = self._owned_ids(label)
        if found or self._sleep is not time.sleep:
            return found
        for _ in range(8):
            self._sleep(3)
            found = self._owned_ids(label)
            if found:
                return found
        return found

    def _owned_ids(self, label: str) -> list[str]:
        return [
            str(row.get("id"))
            for row in self._instances()
            if str(row.get("label") or "") == label and self._same_machine(row)
        ]

    def _attach_and_wait(self, resource_id: str) -> None:
        from media_engine.providers.vast_ssh import WorkerSSH
        worker = getattr(self, "_connect_worker", None) or WorkerSSH(self.runtime)
        watch = PhaseWatch.start("connecting", self._now())
        attached = False
        seen_status = ""
        while True:
            if watch.expired(self._now()):
                raise WorkerStageError("CONNECT_TIMEOUT")
            row = self._instance_row(resource_id)
            if row is None:
                raise WorkerStageError("WORKER_BOOT_FAILED")
            status_name = str(row.get("actual_status") or "")
            if status_name != seen_status:
                seen_status = status_name
                watch.mark(self._now())
            if status_name == "running" and not attached:
                status, raw = self._request(
                    "POST", f"https://console.vast.ai/api/v0/instances/{resource_id}/ssh/",
                    {"ssh_key": worker.public_key()},
                )
                if status != 200:
                    raise WorkerStageError("WORKER_BOOT_FAILED")
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError):
                    payload = {}
                if not isinstance(payload, dict) or payload.get("success") is not True:
                    raise WorkerStageError("WORKER_BOOT_FAILED")
                attached = True
                watch.mark(self._now())
            host = str(row.get("ssh_host") or "")
            port = "" if row.get("ssh_port") is None else str(row.get("ssh_port"))
            if attached and host and port.isdigit() and self._same_machine(row):
                try:
                    code, out, _err = worker.run(host, port, "echo ready", 15)
                except WorkerStageError as exc:
                    if exc.code != "WORKER_TIMEOUT":
                        raise
                    code, out = 1, ""
                if code == 0 and "ready" in out:
                    self._host = host
                    self._port = port
                    self._ssh_run = worker.run
                    self._worker_ssh = worker
                    return
            self._sleep(5)

    def _wait_until_ready(self, resource_id: str) -> None:
        """Wait until the worker process reports ready. Instance running is not enough."""
        watch = PhaseWatch.start("worker_ready", self._now())
        seen = ""
        while True:
            if watch.expired(self._now()):
                raise WorkerStageError("WORKER_READY_TIMEOUT")
            row = self._instance_row(resource_id)
            if row is None:
                raise WorkerStageError("WORKER_BOOT_FAILED")
            status_name = str(row.get("actual_status") or "")
            endpoint = worker_http_endpoint(row)
            health = self._fetch_health(endpoint) if endpoint else None
            state = classify_readiness(status_name, endpoint is not None, health)
            self.worker_state = state
            if state != seen:
                seen = state
                watch.mark(self._now())
            if state == "worker_ready":
                self.runtime_ready = True
                return
            if state == "worker_failed":
                raise WorkerStageError("WORKER_BOOT_FAILED")
            self._sleep(5)

    def _fetch_health(self, endpoint: tuple[str, str]) -> Optional[dict]:
        url = f"http://{endpoint[0]}:{endpoint[1]}/health"
        getter = getattr(self, "_health_get", None)
        try:
            if getter is not None:
                status, body = getter(url)
            else:
                with urllib.request.urlopen(url, timeout=5) as response:
                    status, body = int(response.status), response.read(10_000)
        except Exception:
            return None
        if status != 200:
            return None
        try:
            payload = json.loads(body)
        except (UnicodeError, json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _watch_cache_sync(self) -> None:
        """Start the R2 copy, then keep it only while bytes keep moving."""
        watch = PhaseWatch.start("cache_staging", self._now())
        code, out, _err = self._remote(_R2_SYNC, 60, "CACHE_TIMEOUT")
        if code != 0 or "STARTED" not in out:
            raise WorkerStageError("MODEL_CACHE_INVALID")
        while True:
            _code, progress, _err = self._remote(_CACHE_PROGRESS, 30, "CACHE_TIMEOUT")
            note_transfer_text(watch, progress, self._now())
            if "MB_per_sec" in progress:
                for line in progress.splitlines():
                    if not line.strip().startswith("{"):
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and "MB_per_sec" in record:
                        self.transfer_mb_per_sec = record.get("MB_per_sec")
            _code, exit_out, _err = self._remote(_CACHE_EXIT, 30, "CACHE_TIMEOUT")
            status = ""
            for line in reversed(exit_out.splitlines()):
                if line.strip():
                    status = line.strip()
                    break
            if status.isdigit():
                if status != "0":
                    raise WorkerStageError("MODEL_CACHE_INVALID")
                return
            reason = watch.expired(self._now())
            if reason == "PHASE_STALL":
                self._remote(_CACHE_KILL, 20, "CACHE_STALL")
                raise WorkerStageError("CACHE_STALL")
            if reason == "PHASE_TIMEOUT":
                self._remote(_CACHE_KILL, 20, "CACHE_TIMEOUT")
                raise WorkerStageError("CACHE_TIMEOUT")
            self._sleep(10)


@dataclass(frozen=True)
class ImageChoice:
    offer_id: str
    machine_id: int
    gpu_model: str
    hourly: Decimal
    disk_gb: int
    source: str
    projected_cost: Decimal
    inet_down: float
    reliability: float
    transfer_seconds: float = 0.0
    cost_before_generation: Decimal = Decimal("0")


def supports_bf16(name: str) -> bool:
    """True for GPU families that can run the proven BF16 pipeline."""
    folded = " ".join(str(name).casefold().replace("_", " ").split())
    if "rtx 8000" in folded or folded.startswith("cmp") or " cmp" in folded:
        return False
    compact = folded.replace(" ", "")
    allowed = (
        "a40",
        "a100",
        "a800",
        "a" + "6000",
        "l40",
        "409" + "0",
        "6000ada",
        "5880ada",
        "h100",
        "h200",
        "b200",
        "b300",
        "pro6000",
        "pro5000",
    )
    return any(token in compact for token in allowed)


def cold_stage_allowed() -> bool:
    """Cold copy is off unless an operator sets MEDIA_ENGINE_ALLOW_COLD_STAGE."""
    value = os.environ.get("MEDIA_ENGINE_ALLOW_COLD_STAGE", "").strip().casefold()
    return value in {"1", "true", "yes"}


def _r2_configured() -> bool:
    return all(os.environ.get(name) for name in (
        "R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
    ))


def expected_transfer_seconds(inet_down: float) -> Optional[float]:
    """Seconds to move the Qwen cache at a quarter of the advertised link."""
    if inet_down <= 0:
        return None
    mb_per_sec = (inet_down / 8.0) * LINE_FRACTION
    if mb_per_sec <= 0:
        return None
    return QWEN_CACHE_MIN_BYTES / (mb_per_sec * 1_000_000)


def projected_transfer_seconds(inet_down: float) -> Optional[float]:
    raw = expected_transfer_seconds(inet_down)
    if raw is None:
        return None
    return raw + OVERHEAD_SECONDS


def plan_cold_stage(hourly: Decimal, inet_down: float) -> Optional[dict[str, Decimal | float]]:
    """Hourly rate, transfer time, pre-generation GPU cost, and total."""
    transfer = expected_transfer_seconds(inet_down)
    if transfer is None or hourly <= 0:
        return None
    before_seconds = transfer + OVERHEAD_SECONDS
    total_seconds = before_seconds + GENERATE_ALLOWANCE_SECONDS
    before = (hourly * Decimal(str(before_seconds)) / Decimal(3600)).quantize(Decimal("0.000001"))
    total = (hourly * Decimal(str(total_seconds)) / Decimal(3600)).quantize(Decimal("0.000001"))
    return {
        "hourly": hourly,
        "transfer_seconds": transfer,
        "cost_before_generation": before,
        "projected_cost": total,
    }


WORKER_ONSTART = (
    "if ! python3 -c 'import urllib.request; "
    "urllib.request.urlopen(\"http://127.0.0.1:8080/health\", timeout=2)' "
    ">/dev/null 2>&1; then "
    "python3 /workspace/worker_ready.py >> /workspace/worker-ready.log 2>&1 & "
    "fi"
)


def portable_worker(image: str) -> bool:
    """True when the instance image is the published runtime, not the stock CUDA image."""
    return image.startswith("ghcr.io/") and "@sha256:" in image


def classify_readiness(instance_status: str, has_endpoint: bool, health: Optional[dict]) -> str:
    """Separate a running VM from a worker that has answered /health."""
    if str(instance_status) != "running":
        return "provisioning"
    if isinstance(health, dict) and health.get("state") == "worker_failed":
        return "worker_failed"
    if (
        isinstance(health, dict)
        and health.get("state") == "worker_ready"
        and health.get("runtime") is True
    ):
        return "worker_ready"
    if has_endpoint:
        return "container_starting"
    return "instance_running"


def worker_http_endpoint(row: dict) -> Optional[tuple[str, str]]:
    if str(row.get("actual_status") or "") != "running":
        return None
    ports = row.get("ports")
    host = str(row.get("public_ipaddr") or "")
    if not isinstance(ports, dict) or not host:
        return None
    mapped = ports.get("8080/tcp")
    if not isinstance(mapped, list) or not mapped or not isinstance(mapped[0], dict):
        return None
    port = str(mapped[0].get("HostPort") or "")
    if port.isdigit():
        return host, port
    return None


def pinned_worker_reference() -> str:
    path = Path(__file__).resolve().parents[3] / "deploy" / "worker" / "image_ref.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    ref = str(payload.get("pinned_reference") or "").strip()
    if portable_worker(ref):
        return ref
    return ""


def worker_image() -> str:
    """Published runtime image, or the stock CUDA image until one is set."""
    configured = os.environ.get("MEDIA_ENGINE_WORKER_IMAGE", "").strip()
    if configured:
        return configured
    return pinned_worker_reference() or INSTANCE_IMAGE


def choose_offer(
    offers: list,
    *,
    volume_machine: int,
    volume_ready: bool,
    max_hourly: Decimal,
    operation_budget: Decimal,
    allow_cold_stage: bool = False,
) -> Optional[ImageChoice]:
    """Use the warm-cache machine. A cold copy is only considered when policy allows it."""
    warm: list[ImageChoice] = []
    cold: list[ImageChoice] = []
    for row in offers:
        if not isinstance(row, dict) or row.get("rentable") is not True:
            continue
        if not supports_bf16(str(row.get("gpu_name") or "")):
            continue
        try:
            ram = float(row.get("gpu_ram") or 0)
            machine = int(row.get("machine_id") or 0)
        except (TypeError, ValueError):
            continue
        if ram < MIN_VRAM_MIB or machine <= 0:
            continue
        on_warm = volume_ready and machine == volume_machine
        disk = WARM_DISK_GB if on_warm else COLD_DISK_GB
        inet = 0.0
        transfer = 0.0
        before = Decimal("0")
        if on_warm:
            seconds = float(OVERHEAD_SECONDS)
        else:
            try:
                free = float(row.get("disk_space") or 0)
                inet = float(row.get("inet_down") or 0)
            except (TypeError, ValueError):
                continue
            if free < COLD_DISK_GB:
                continue
        try:
            hourly = _hourly(row, disk)
            reliability = float(row.get("reliability2") or 0)
        except (WorkerStageError, TypeError, ValueError):
            continue
        if on_warm:
            projected = (hourly * Decimal(str(seconds)) / Decimal(3600)).quantize(Decimal("0.000001"))
            before = projected
        else:
            planned = plan_cold_stage(hourly, inet)
            if planned is None:
                continue
            transfer = float(planned["transfer_seconds"])
            before = planned["cost_before_generation"]
            projected = planned["projected_cost"]
            if not isinstance(before, Decimal) or not isinstance(projected, Decimal):
                continue
        if projected > operation_budget:
            continue
        choice = ImageChoice(
            offer_id=str(row.get("id")),
            machine_id=machine,
            gpu_model=str(row.get("gpu_name") or ""),
            hourly=hourly,
            disk_gb=disk,
            source="warm_cache" if on_warm else "r2_cold_stage",
            projected_cost=projected,
            inet_down=inet,
            reliability=reliability,
            transfer_seconds=transfer,
            cost_before_generation=before,
        )
        (warm if on_warm else cold).append(choice)
    affordable = [item for item in warm if item.hourly <= max_hourly]
    if affordable:
        affordable.sort(key=lambda item: (item.hourly, -item.reliability))
        return affordable[0]
    if not allow_cold_stage:
        return None
    pool = [item for item in cold if item.hourly <= max_hourly] or cold
    if not pool:
        return None
    pool.sort(key=lambda item: (item.projected_cost, -item.reliability))
    return pool[0]


def _hourly(row: dict, disk_gb: int) -> Decimal:
    storage = row.get("storage_cost")
    base = row.get("dph_base")
    if storage in (None, "") or base in (None, ""):
        raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
    try:
        monthly = Decimal(str(storage))
        amount = Decimal(str(base)) + (monthly * Decimal(disk_gb) / Decimal(720))
    except Exception as exc:
        raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE") from exc
    if monthly <= 0 or amount <= 0:
        raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
    total = row.get("dph_total")
    if total not in (None, ""):
        try:
            amount = max(amount, Decimal(str(total)))
        except Exception as exc:
            raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE") from exc
    return amount


def _cache_ok(text: str, *, require_mount: bool = True) -> bool:
    line = ""
    for item in text.splitlines():
        if item.startswith("{"):
            line = item
    if not line:
        return False
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return False
    return (
        payload.get("state") == "COMPLETE"
        and payload.get("repo") == REPO_ID
        and payload.get("revision") == REVISION
        and payload.get("files") == QWEN_FILE_COUNT
        and payload.get("bytes") == QWEN_CACHE_MIN_BYTES
        and (payload.get("mount") is True or not require_mount)
    )


_CACHE_CHECK = r"""python3 - << 'PY'
import json, os, sys
sys.path.insert(0, "/workspace")
manifest = json.load(open("/models/%s"))
state = manifest.get("state")
try:
    import model_cache
    state = model_cache.assess_cache(model_cache.Path("/models"))
except Exception:
    pass
print(json.dumps({
    "state": state,
    "repo": manifest.get("repo_id"),
    "revision": manifest.get("revision"),
    "files": manifest.get("expected_file_count"),
    "bytes": manifest.get("expected_total_bytes"),
    "mount": os.path.ismount("/models"),
}))
PY""" % MANIFEST_NAME

_R2_SYNC = r"""
set -eu
mkdir -p /models /workspace
set -a
. /workspace/phase2d-r2.env
set +a
printf '%s\n' '{"cache_mount":"/models"}' > /workspace/phase2d_job.json
rm -f /workspace/cache-sync.exit /workspace/r2-sync-progress.json
nohup sh -c 'PHASE2D_CACHE_SYNC_SECONDS=5000 python3 /workspace/qwen_remote.py cache-sync > /workspace/cache-sync.log 2>&1; echo $? > /workspace/cache-sync.exit' >/dev/null 2>&1 &
echo $! > /workspace/cache-sync.pid
echo STARTED
"""

_CACHE_PROGRESS = "cat /workspace/r2-sync-progress.json 2>/dev/null || true"
_CACHE_EXIT = "if [ -f /workspace/cache-sync.exit ]; then cat /workspace/cache-sync.exit; else echo RUNNING; fi"
_CACHE_KILL = "if [ -f /workspace/cache-sync.pid ]; then kill $(cat /workspace/cache-sync.pid) >/dev/null 2>&1 || true; fi"

_RUNTIME_READY = (
    "python3 -c \""
    "import pathlib; "
    "p=pathlib.Path('/opt/ai-media-engine/runtime-ready'); "
    "import torch,diffusers,transformers,accelerate,safetensors; "
    "print('RUNTIME_READY' if p.is_file() else 'MISSING')\""
)

_RUNTIME_INSTALL = f"""
set -eu
if ! {PYTHON} -c 'import torch,diffusers' >/dev/null 2>&1; then
  if [ ! -x {PYTHON} ]; then
    if ! python3 -m venv --system-site-packages {VENV}; then
      rm -rf {VENV}
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv
      python3 -m venv --system-site-packages {VENV}
    fi
  fi
  if ! {PYTHON} -c 'import torch' >/dev/null 2>&1; then
    if [ -x /opt/conda/bin/python ] && /opt/conda/bin/python -c 'import torch' >/dev/null 2>&1; then
      rm -rf {VENV}
      /opt/conda/bin/python -m venv --system-site-packages {VENV}
    fi
  fi
  {PYTHON} -m pip install --disable-pip-version-check 'diffusers>=0.35' transformers accelerate safetensors pillow
fi
{PYTHON} -c 'import torch,diffusers,transformers,accelerate,safetensors'
"""

_IMPORT_CHECK = (
    f"{PYTHON} -c 'from diffusers import QwenImagePipeline; print(QwenImagePipeline.__name__)'"
)


def _generate_command(prompt: str, width: int, height: int, seed: int, python: str = PYTHON) -> str:
    if width != FIRST_WIDTH or height != FIRST_HEIGHT:
        raise WorkerStageError("GENERATION_FAILED")
    job = json.dumps({
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": INFERENCE_STEPS,
        "cache_mount": MOUNT_PATH,
    }, separators=(",", ":"))
    quoted = shlex.quote(job)
    return (
        "mkdir -p /workspace && "
        f"printf %s {quoted} > /workspace/phase2d_job.json && "
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 "
        "HF_HOME=/models/hf-cache HUGGINGFACE_HUB_CACHE=/models/hf-cache "
        "PHASE2D_MODEL_LOAD_SECONDS=1200 PHASE2D_GENERATION_SECONDS=600 && "
        f"if [ ! -f /workspace/qwen_remote.py ]; then echo MISSING_REMOTE; exit 3; fi && "
        f"{python} -u /workspace/qwen_remote.py generate && "
        "python3 -c 'import struct; d=open(\"/workspace/phase2d_output.png\",\"rb\").read(); "
        "assert d[:8]==b\"\\x89PNG\\r\\n\\x1a\\n\" and len(d)>0; "
        "w,h=struct.unpack(\">II\", d[16:24]); assert (w,h)==(1024,1024); print(\"PNG_READY\", len(d))'"
    )
