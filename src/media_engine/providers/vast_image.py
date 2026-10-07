"""Vast image worker for the persistent Qwen cache on one machine.

This provider creates at most one GPU at a time, attaches the existing
volume, and never starts a model download.
"""
from __future__ import annotations

import json
import os
import shlex
import time
import urllib.error
import urllib.request
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
from media_engine.providers.vast_lifecycle import classify_create_response, instances_list_url, ownership_label
from media_engine.safety.limits import SafetyLimits

MACHINE_ID = 47281
VOLUME_ID = 54653022
MOUNT_PATH = "/models"
REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
DISK_GB = 40
MIN_VRAM_MIB = 44 * 1024
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
        self.machine_id = str(MACHINE_ID)
        self.hourly_amount = Decimal("0")
        self.offer_id = ""
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
        self.offer_id = str(chosen["id"])
        self.gpu_model = str(chosen.get("gpu_name") or "")
        self.hourly_amount = _hourly(chosen)
        return Quote(hourly_price_usd=self.hourly_amount, provider="vast")

    def create(self, owner: str) -> CreateResult:
        if self._put_done:
            return CreateResult(outcome="definite_failure")
        self._put_done = True
        if not self.offer_id:
            self.quote(owner)
        label = ownership_label(owner)
        body = {
            "label": label,
            "image": INSTANCE_IMAGE,
            "disk": DISK_GB,
            "runtype": "ssh",
            "cancel_unavail": True,
            "volume_info": {"mount_path": MOUNT_PATH, "create_new": False, "volume_id": VOLUME_ID},
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
        if label != ownership_label(owner) or int(row.get("machine_id") or 0) != MACHINE_ID:
            return OBSERVE_FOREIGN
        return OBSERVE_PRESENT

    def terminate(self, resource_id: str) -> None:
        row = self._instance_row(resource_id)
        if row is None:
            self._gone.add(resource_id)
            self._put_done = False
            return
        label = str(row.get("label") or "")
        if not label.startswith("media-engine:") or int(row.get("machine_id") or 0) != MACHINE_ID:
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
        if self._ssh_run is None:
            self._attach_and_wait(resource_id)
            return
        code, _out, _err = self._ssh_run(self._host or "worker", self._port or "22", "echo ready", 30)
        if code != 0:
            raise WorkerStageError("WORKER_BOOT_FAILED")

    def confirm_cache(self, resource_id: str) -> None:
        code, out, _err = self._remote(_CACHE_CHECK, 60)
        if code != 0 or not _cache_ok(out):
            raise WorkerStageError("MODEL_CACHE_INVALID")

    def prepare_runtime(self, resource_id: str) -> None:
        worker = getattr(self, "_worker_ssh", None)
        if worker is not None and self._host and self._port:
            worker.stage(self._host, self._port)
        code, _out, _err = self._remote(_RUNTIME_INSTALL, 900)
        if code != 0:
            raise WorkerStageError("RUNTIME_PREPARE_FAILED")

    def ensure_model(self, resource_id: str) -> None:
        code, out, _err = self._remote(_IMPORT_CHECK, 120)
        if code != 0 or "QwenImagePipeline" not in out:
            raise WorkerStageError("MODEL_LOAD_FAILED")

    def generate(self, *, prompt: str, width: int, height: int, seed: int) -> bytes:
        command = _generate_command(prompt, width, height, seed)
        self.commands.append(command)
        code, out, _err = self._remote(command, 900)
        if code != 0 or "PNG_READY" not in out:
            if "LOAD_FAILED" in out:
                raise WorkerStageError("MODEL_LOAD_FAILED")
            raise WorkerStageError("GENERATION_FAILED")
        return self._fetch_png()

    def _remote(self, command: str, timeout: int) -> tuple[int, str, str]:
        if self._ssh_run is None:
            raise WorkerStageError("WORKER_BOOT_FAILED")
        try:
            return self._ssh_run(self._host or "worker", self._port or "22", command, timeout)
        except TimeoutError as exc:
            raise WorkerStageError("WORKER_TIMEOUT") from exc

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

    def _select_offer(self) -> Optional[dict]:
        status, raw = self._request("POST", BUNDLES, {
            "limit": 16,
            "type": "on-demand",
            "rentable": {"eq": True},
            "num_gpus": {"eq": 1},
            "machine_id": {"eq": MACHINE_ID},
        })
        if status != 200:
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return None
        offers = payload.get("offers") if isinstance(payload, dict) else None
        if not isinstance(offers, list):
            return None
        ranked = []
        for row in offers:
            if not isinstance(row, dict) or row.get("rentable") is not True:
                continue
            if int(row.get("machine_id") or MACHINE_ID) != MACHINE_ID:
                continue
            ram = float(row.get("gpu_ram") or 0)
            if ram < MIN_VRAM_MIB:
                continue
            try:
                price = _hourly(row)
            except WorkerStageError:
                continue
            preferred = "RTX A" + "6000"
            prefer = 0 if str(row.get("gpu_name") or "") == preferred else 1
            ranked.append((prefer, price, row))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1]))
        return ranked[0][2]

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
            if str(row.get("label") or "") == label and int(row.get("machine_id") or 0) == MACHINE_ID
        ]

    def _attach_and_wait(self, resource_id: str) -> None:
        from media_engine.providers.vast_ssh import WorkerSSH
        worker = WorkerSSH(self.runtime)
        deadline = self._now() + 900
        attached = False
        while self._now() < deadline:
            row = self._instance_row(resource_id)
            if row is None:
                raise WorkerStageError("WORKER_BOOT_FAILED")
            if str(row.get("actual_status") or "") == "running" and not attached:
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
            host = str(row.get("ssh_host") or "")
            port = "" if row.get("ssh_port") is None else str(row.get("ssh_port"))
            if attached and host and port.isdigit() and int(row.get("machine_id") or 0) == MACHINE_ID:
                code, out, _err = worker.run(host, port, "echo ready", 40)
                if code == 0 and "ready" in out:
                    self._host = host
                    self._port = port
                    self._ssh_run = worker.run
                    self._worker_ssh = worker
                    return
            self._sleep(5)
        raise WorkerStageError("WORKER_TIMEOUT")


def _hourly(row: dict) -> Decimal:
    storage = row.get("storage_cost")
    base = row.get("dph_base")
    if storage in (None, "") or base in (None, ""):
        raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
    try:
        monthly = Decimal(str(storage))
        amount = Decimal(str(base)) + (monthly * Decimal(DISK_GB) / Decimal(720))
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


def _cache_ok(text: str) -> bool:
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
        and payload.get("mount") is True
    )


_CACHE_CHECK = r"""python3 - << 'PY'
import json, os
manifest = json.load(open("/models/%s"))
print(json.dumps({
    "state": manifest.get("state"),
    "repo": manifest.get("repo_id"),
    "revision": manifest.get("revision"),
    "files": manifest.get("expected_file_count"),
    "bytes": manifest.get("expected_total_bytes"),
    "mount": os.path.ismount("/models"),
}))
PY""" % MANIFEST_NAME

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


def _generate_command(prompt: str, width: int, height: int, seed: int) -> str:
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
        f"{PYTHON} -u /workspace/qwen_remote.py generate && "
        "python3 -c 'import struct; d=open(\"/workspace/phase2d_output.png\",\"rb\").read(); "
        "assert d[:8]==b\"\\x89PNG\\r\\n\\x1a\\n\" and len(d)>0; "
        "w,h=struct.unpack(\">II\", d[16:24]); assert (w,h)==(1024,1024); print(\"PNG_READY\", len(d))'"
    )
