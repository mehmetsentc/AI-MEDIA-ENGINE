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
from media_engine.providers.vast_lifecycle import classify_create_response, ownership_label
from media_engine.safety.limits import SafetyLimits

MACHINE_ID = 47281
VOLUME_ID = 54653022
MOUNT_PATH = "/models"
REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
WARM_DISK_GB = 40
COLD_DISK_GB = 140
MIN_VRAM_MIB = 44 * 1024
OPERATION_BUDGET_USD = Decimal("1.00")
# Explicit process ceilings. Any other value leaves the $1 cap in place.
AUTHORIZED_RUN_BUDGET_USD = Decimal("3.00")
VERIFY_RUN_BUDGET_USD = Decimal("0.50")
AUTHORIZED_RUN_BUDGETS = frozenset({AUTHORIZED_RUN_BUDGET_USD, VERIFY_RUN_BUDGET_USD})
# Previously observed warm hourly for machine 47281 with the 40 GB instance disk.
VERIFY_MAX_HOURLY_USD = Decimal("0.375")


def _operation_ceiling() -> Decimal:
    raw = os.environ.get("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", "").strip()
    if not raw:
        return OPERATION_BUDGET_USD
    try:
        granted = Decimal(raw)
    except Exception:
        return OPERATION_BUDGET_USD
    if granted in AUTHORIZED_RUN_BUDGETS:
        return granted
    if granted == Decimal("1.50"):
        profile = os.environ.get("MEDIA_ENGINE_EXECUTION_PROFILE", "").strip()
        if profile not in {"live7", "live8"}:
            raise WorkerStageError("BUDGET_MISMATCH")
        return granted
    if granted == Decimal("5.00"):
        profile = os.environ.get("MEDIA_ENGINE_EXECUTION_PROFILE", "").strip()
        if profile != "live8":
            raise WorkerStageError("BUDGET_MISMATCH")
        return granted
    return OPERATION_BUDGET_USD


def active_operation_budget() -> Decimal:
    """The default cap stays $1. STUDIO_GPU_BUDGET_USD can only tighten it.

    MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD must be exactly 3.00 or 0.50,
    or 1.50 with MEDIA_ENGINE_EXECUTION_PROFILE=live7 or live8,
    or 5.00 with MEDIA_ENGINE_EXECUTION_PROFILE=live8.
    0.50 is the cache-verification ceiling. 0.49, 0.51, and 9.00 stay at $1.
    1.50 is not rewritten as $1 or $3. 5.00 on any other profile is rejected.
    """
    ceiling = _operation_ceiling()
    raw = os.environ.get("STUDIO_GPU_BUDGET_USD", "").strip()
    if not raw:
        return ceiling
    try:
        amount = Decimal(raw)
    except Exception:
        return ceiling
    if amount <= 0:
        return ceiling
    return min(amount, ceiling)


def safe_runtime_seconds(
    hourly: Decimal,
    *,
    network_ceiling: Decimal,
    reserve: Decimal,
    ceiling: Decimal,
) -> Optional[int]:
    """Seconds of GPU time that stay inside the ceiling after network and reserve."""
    if hourly <= 0 or ceiling <= 0:
        return None
    try:
        spendable = ceiling - network_ceiling - reserve
    except Exception:
        return None
    if spendable <= 0:
        return None
    return int((spendable / hourly) * Decimal(3600))


def decide_staging_budget(
    samples: list[tuple[float, int]],
    *,
    now: float,
    deadline: Optional[float],
    reserve_seconds: float,
    total_bytes: int,
) -> dict:
    """Decide whether staging still fits. The first interval is not a rate.

    After warm-up, the sustained rate is the later bytes divided by the
    later time. One slow interval does not set that rate. The 180-second
    stall timer is separate.
    """
    latest_bytes = samples[-1][1] if samples else 0
    elapsed = 0.0 if not samples else max(0.0, now - samples[0][0])
    decision = {
        "action": "continue",
        "reason": "warmup",
        "sampled_mb_per_sec": None,
        "sustained_mb_per_sec": None,
        "remaining_seconds": None,
        "elapsed_seconds": round(elapsed, 3),
        "bytes_done": latest_bytes,
    }
    remaining_bytes = max(0, int(total_bytes) - latest_bytes)
    if deadline is not None and now >= float(deadline):
        decision["action"] = "stop"
        decision["reason"] = "BUDGET_DEADLINE"
        return decision
    if (
        deadline is not None
        and remaining_bytes > 0
        and float(deadline) - now <= max(0.0, reserve_seconds)
    ):
        decision["action"] = "stop"
        decision["reason"] = "BUDGET_RESERVE"
        return decision
    if deadline is None or total_bytes <= 0:
        decision["reason"] = "no_deadline"
        return decision
    intervals: list[tuple[float, float]] = []
    for (earlier, earlier_bytes), (later, later_bytes) in zip(samples, samples[1:]):
        span = later - earlier
        moved = later_bytes - earlier_bytes
        if span <= 0 or moved <= 0:
            continue
        intervals.append((float(moved), float(span)))
    if (
        len(samples) < STAGING_WARMUP_SAMPLES
        or latest_bytes < STAGING_WARMUP_BYTES
        or samples[-1][0] - samples[0][0] < STAGING_WARMUP_SECONDS
        or len(intervals) < 3
    ):
        return decision
    later = intervals[1:]
    sustained = sum(moved for moved, _span in later) / sum(span for _moved, span in later)
    decision["sampled_mb_per_sec"] = round((intervals[-1][0] / intervals[-1][1]) / 1_000_000, 6)
    decision["sustained_mb_per_sec"] = round(sustained / 1_000_000, 6)
    if sustained <= 0:
        decision["reason"] = "no_sustained_progress"
        return decision
    remaining_seconds = remaining_bytes / sustained
    decision["remaining_seconds"] = round(remaining_seconds, 3)
    if remaining_seconds + max(0.0, reserve_seconds) > float(deadline) - now:
        decision["action"] = "stop"
        decision["reason"] = "SUSTAINED_RATE_OVER_BUDGET"
        return decision
    decision["reason"] = "within_budget"
    return decision


def session_fits(
    safe_seconds: Optional[int],
    *,
    transfer_seconds: float,
    model_load_seconds: float,
    inference_seconds: float,
) -> bool:
    if safe_seconds is None:
        return False
    return safe_seconds >= transfer_seconds + model_load_seconds + inference_seconds


OVERHEAD_SECONDS = 600
GENERATE_ALLOWANCE_SECONDS = 180
WARM_CACHE_INSPECT_SECONDS = 120.0


def warm_session_seconds() -> float:
    """Provisioning, cache inspection, runtime, load, inference, and cleanup."""
    return (
        float(PHASE_LIMITS["worker_ready"][0])
        + WARM_CACHE_INSPECT_SECONDS
        + float(PHASE_LIMITS["runtime_preparing"][0])
        + float(PHASE_LIMITS["model_loading"][0])
        + float(PHASE_LIMITS["generating"][0])
        + float(GENERATE_ALLOWANCE_SECONDS)
    )


def warm_verify_window(hourly: Decimal, ceiling: Decimal) -> dict:
    """Warm-cache session window. Network transfer is not part of this budget.

    The $0.50 verification run uses the shared planning allocation. Other
    ceilings keep a 20 percent reserve.
    """
    required = warm_session_seconds()
    if Decimal(ceiling) == VERIFY_RUN_BUDGET_USD:
        from media_engine.safety.execution_budget import live6_planning_window
        return live6_planning_window(hourly, required)
    reserve = (ceiling * Decimal("0.20")).quantize(Decimal("0.01"))
    safe = safe_runtime_seconds(
        hourly,
        network_ceiling=Decimal("0"),
        reserve=reserve,
        ceiling=ceiling,
    )
    work = (hourly * Decimal(str(required)) / Decimal(3600)).quantize(Decimal("0.000001"))
    fits = (
        safe is not None
        and safe >= required
        and work + reserve <= ceiling
    )
    return {
        "safe_seconds": safe,
        "reserve_usd": reserve,
        "required_seconds": required,
        "work_cost": work,
        "fits": fits,
        "ceiling": ceiling,
        "network_usd": Decimal("0"),
    }


def half_dollar_offer_allowed(provider: "VastImageProvider") -> bool:
    """The $0.50 process may rent only the warm verified GPU with the existing volume."""
    if os.environ.get("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", "").strip() != "0.50":
        return False
    if os.environ.get("MEDIA_ENGINE_FORBID_R2_STAGE", "").strip() != "1":
        return False
    if provider.cache_source != "warm_cache":
        return False
    if str(provider.machine_id) != str(MACHINE_ID):
        return False
    compact = str(provider.gpu_model).casefold().replace(" ", "").replace("_", "")
    if "a6000" not in compact or "ada" in compact:
        return False
    if provider.disk_gb != WARM_DISK_GB:
        return False
    if provider.hourly_amount > VERIFY_MAX_HOURLY_USD:
        return False
    return bool(warm_verify_window(provider.hourly_amount, VERIFY_RUN_BUDGET_USD)["fits"])


# The first progress report is not a sustained rate. LIVE-5 stopped on
# 0.37 MB/s after 199,078 bytes. These bounds require later samples.
STAGING_WARMUP_SAMPLES = 4
STAGING_WARMUP_BYTES = 8 * 1024 * 1024
STAGING_WARMUP_SECONDS = 60.0
# Planning rate is a quarter of the advertised download, between the slow
# and fast transfers already measured on Vast.
LINE_FRACTION = 0.25
BUNDLES = "https://console.vast.ai/api/v0/bundles/"
VOLUMES = "https://console.vast.ai/api/v0/volumes/?owner=me"

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
        self.transfer_bytes_done = None
        self.transfer_bytes_total = None
        self.transfer_samples: list[tuple[float, int]] = []
        self.staging_events: list[dict] = []
        self.staging_deadline: Optional[float] = None
        self.staging_reserve_seconds = 0.0
        self.staging_stop_reason = ""
        self.verified_manifest: Optional[dict] = None
        self.cache_validation = ""
        self.runtime_ready = False
        self.python_executable = ""
        self.runtime_python_version = ""
        self.runtime_dependencies: list = []
        self.worker_state = ""
        self.health_transport = ""
        self.health_error = ""
        self.cost_before_generation_usd = Decimal("0")
        self.last_instance_id = ""
        self._put_done = False
        self._create_closed = False
        self.create_request_sent = False
        self.create_classification = ""
        self.billing_exposure = False
        self.offer_observed_at: float | None = None
        self.final_verification_at: float | None = None
        self.create_request_at: float | None = None
        self.provider_response: dict | None = None
        self.selection_attempts: list[dict] = []
        self.execution_grant = None
        self.execution_authorization = None
        self.execution_id = ""
        self.execution_budget = None
        self.authorization_expires_at: float | None = None
        self._monotonic = time.monotonic
        self.live6_stop = ""
        self.launch_trace: list[str] = []
        self.watchdog_pid = 0
        self._watchdog_popen = None
        self.attached_volume_id = VOLUME_ID
        self.single_image_only = False
        self.image_claimed = False
        self.session_limit = 0
        self._session_booted = False
        self._session_results: dict[tuple, bytes] = {}
        self._images_started = 0
        self._inference_attempts = 0
        self._model_stages_used = 0
        self._model_loads_used = 0
        self._creates_used = 0
        self.images_submitted = 0
        self.gpu_ram_mib = None
        self.cuda_max_good = None
        self.compute_cap = None
        self._owner = ""
        self._label = ""
        self._resource_id = ""
        self._host = ""
        self._port = ""
        self._busy: set[str] = set()
        self._gone: set[str] = set()
        self._started_at: dict[str, float] = {}
        self.commands: list[str] = []

    def _live7_bound(self) -> bool:
        grant = self.execution_grant
        return grant is not None and getattr(grant, "profile", "") == "live7"

    def _live8_bound(self) -> bool:
        return self.execution_authorization is not None

    def _live8_budget_agrees(self, expected: Decimal) -> bool:
        """The $5 target is accepted only when the process config names it.

        A missing or different value is a mismatch. It is not rewritten to
        $1 or $1.50. Other LIVE-8 totals keep their existing env behavior.
        """
        from media_engine.safety.execution_budget import LIVE8_JIT_CEILING_USD
        raw = os.environ.get("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", "").strip()
        profile = os.environ.get("MEDIA_ENGINE_EXECUTION_PROFILE", "").strip()
        configured = None
        if raw:
            try:
                configured = Decimal(raw)
            except Exception:
                return False
        authorization = self.execution_authorization
        if expected == LIVE8_JIT_CEILING_USD:
            return (
                authorization is not None
                and authorization.selection_policy == "just_in_time"
                and profile == "live8"
                and configured == LIVE8_JIT_CEILING_USD
            )
        return configured != LIVE8_JIT_CEILING_USD

    def _five_dollar_target_is_authorized(self) -> bool:
        """A $5 process value cannot widen a warm, live7, or background rental."""
        from media_engine.safety.execution_budget import LIVE8_JIT_CEILING_USD
        raw = os.environ.get("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", "").strip()
        if not raw:
            return True
        try:
            configured = Decimal(raw)
        except Exception:
            return True
        if configured != LIVE8_JIT_CEILING_USD:
            return True
        authorization = self.execution_authorization
        profile = os.environ.get("MEDIA_ENGINE_EXECUTION_PROFILE", "").strip()
        return (
            authorization is not None
            and authorization.selection_policy == "just_in_time"
            and authorization.max_total_usd == LIVE8_JIT_CEILING_USD
            and profile == "live8"
        )

    def bind_authorization(self, authorization, *, code_sha: str) -> None:
        """Attach one authorization. The machine is chosen later from its allowed set."""
        from media_engine.providers.execution_authorization import accept_record
        if self.execution_authorization is not None or self.execution_grant is not None:
            raise WorkerStageError("STALE_AUTHORIZATION")
        accept_record(authorization, now=float(self._now()), code_sha=code_sha)
        if os.environ.get("MEDIA_ENGINE_ALLOW_COLD_STAGE", "").strip() != "1":
            raise WorkerStageError("COLD_STAGE_FORBIDDEN")
        if os.environ.get("MEDIA_ENGINE_FORBID_R2_STAGE", "").strip() == "1":
            raise WorkerStageError("R2_FORBIDDEN")
        self.execution_authorization = authorization
        self.execution_id = authorization.execution_id
        self.authorization_expires_at = float(authorization.expires_at)
        self.execution_budget = None

    def bind_execution(self, grant, *, execution_id: str) -> None:
        """Attach one fresh grant. A second bind in this process is refused."""
        from media_engine.safety.spend_grant import SpendGrant
        if self.execution_grant is not None:
            raise WorkerStageError("STALE_AUTHORIZATION")
        if not isinstance(grant, SpendGrant) or grant.execution_id != execution_id:
            raise WorkerStageError("MISSING_AUTHORIZATION")
        if getattr(grant, "profile", "live6") == "live8":
            self._bind_live8(grant, execution_id)
            return
        if getattr(grant, "profile", "live6") == "live7":
            self._bind_live7(grant, execution_id)
            return
        if grant.ceiling_usd != VERIFY_RUN_BUDGET_USD or grant.max_instances != 1:
            raise WorkerStageError("BUDGET_MISMATCH")
        if grant.machine_id != MACHINE_ID:
            raise WorkerStageError("MACHINE_SCOPE")
        if grant.expires_at <= grant.issued_at or float(self._now()) >= grant.expires_at:
            raise WorkerStageError("EXPIRED_AUTHORIZATION")
        self.execution_grant = grant
        self.execution_id = execution_id
        self.authorization_expires_at = float(grant.expires_at)
        self.execution_budget = None

    def _bind_live7(self, grant, execution_id: str) -> None:
        """Pin the alternative GPU. The warm volume stays detached."""
        from media_engine.providers.live7_contract import (
            FORBIDDEN_VOLUME_ID,
            LIVE7_DISK_GB,
            LIVE7_MACHINE_ID,
            accept_live7_grant,
        )
        accept_live7_grant(grant)
        if float(self._now()) >= grant.expires_at:
            raise WorkerStageError("EXPIRED_AUTHORIZATION")
        if os.environ.get("MEDIA_ENGINE_ALLOW_COLD_STAGE", "").strip() != "1":
            raise WorkerStageError("COLD_STAGE_FORBIDDEN")
        if os.environ.get("MEDIA_ENGINE_FORBID_R2_STAGE", "").strip() == "1":
            raise WorkerStageError("R2_FORBIDDEN")
        self.execution_grant = grant
        self.execution_id = execution_id
        self.authorization_expires_at = float(grant.expires_at)
        self.execution_budget = None
        self.machine_id = str(LIVE7_MACHINE_ID)
        self.disk_gb = LIVE7_DISK_GB
        self.cache_source = "r2_cold_stage"
        self.attached_volume_id = 0
        self.single_image_only = True
        if int(FORBIDDEN_VOLUME_ID) == int(getattr(self, "attached_volume_id", 0) or 0):
            raise WorkerStageError("VOLUME_MISMATCH")

    def _bind_live8(self, grant, execution_id: str) -> None:
        """Bind the grant to the offer this authorization just selected."""
        authorization = self.execution_authorization
        if authorization is None or execution_id != authorization.execution_id:
            raise WorkerStageError("STALE_AUTHORIZATION")
        if grant.execution_id != authorization.execution_id or grant.profile != "live8":
            raise WorkerStageError("STALE_AUTHORIZATION")
        if grant.ceiling_usd != authorization.max_total_usd or grant.max_instances != 1:
            raise WorkerStageError("BUDGET_MISMATCH")
        if int(grant.machine_id or 0) not in authorization.allowed_machine_ids:
            raise WorkerStageError("MACHINE_SCOPE")
        if int(grant.offer_id or 0) not in authorization.allowed_offer_ids:
            raise WorkerStageError("OFFER_REJECTED")
        if float(self._now()) >= grant.expires_at or grant.expires_at != authorization.expires_at:
            raise WorkerStageError("EXPIRED_AUTHORIZATION")
        if grant.forbidden_volume_id != VOLUME_ID or grant.cold_stage is not True:
            raise WorkerStageError("VOLUME_MISMATCH")
        if int(self.machine_id or 0) != int(grant.machine_id or 0):
            raise WorkerStageError("MACHINE_SCOPE")
        if int(self.offer_id or 0) != int(grant.offer_id or 0):
            raise WorkerStageError("OFFER_REJECTED")
        self.execution_grant = grant
        self.execution_id = execution_id
        self.authorization_expires_at = float(grant.expires_at)
        self.execution_budget = None
        self.disk_gb = int(grant.disk_gb or 0)
        self.cache_source = "r2_cold_stage"
        self.attached_volume_id = 0
        self.single_image_only = True
        self.arm_live_session()

    def _reject_unverified_run(self) -> str:
        """The $0.50 path rents nothing until this execution's grant matches the offer."""
        if int(self.attached_volume_id) != VOLUME_ID:
            return "VOLUME_MISMATCH"
        if not half_dollar_offer_allowed(self):
            return "OFFER_REJECTED"
        try:
            listed = self._instances()
        except WorkerStageError as exc:
            return exc.code
        owned = []
        for row in listed:
            if not str(row.get("label") or "").startswith("media-engine:"):
                continue
            if str(row.get("machine_id") or "") != str(MACHINE_ID):
                continue
            owned.append(row)
        from media_engine.safety.spend_grant import evaluate_grant
        try:
            machine = int(self.machine_id)
        except (TypeError, ValueError):
            machine = None
        decision = evaluate_grant(
            self.execution_grant,
            now=float(self._now()),
            execution_id=self.execution_id,
            ceiling_usd=VERIFY_RUN_BUDGET_USD,
            instances=len(owned) + 1,
            machine_id=machine,
        )
        if decision["action"] != "continue":
            return str(decision["reason"])
        grant = self.execution_grant
        if grant is None or grant.max_instances != 1:
            return "INSTANCE_LIMIT"
        if grant.ceiling_usd != VERIFY_RUN_BUDGET_USD:
            return "BUDGET_MISMATCH"
        return ""

    def _reject_live7_run(self) -> str:
        """The $1.50 path rents one pinned machine and never attaches the warm volume."""
        from media_engine.providers.live7_contract import (
            FORBIDDEN_VOLUME_ID,
            LIVE7_DISK_GB,
            LIVE7_MACHINE_ID,
            accept_offer,
        )
        from media_engine.safety.execution_budget import LIVE7_CEILING_USD
        from media_engine.safety.spend_grant import evaluate_grant
        if int(getattr(self, "attached_volume_id", 0) or 0) in {FORBIDDEN_VOLUME_ID, VOLUME_ID}:
            return "VOLUME_MISMATCH"
        if self.cache_source != "r2_cold_stage" or int(self.disk_gb) != LIVE7_DISK_GB:
            return "VOLUME_MISMATCH"
        if str(self.machine_id) != str(LIVE7_MACHINE_ID):
            return "MACHINE_SCOPE"
        if self.cuda_max_good is None or self.gpu_ram_mib is None or self.compute_cap is None:
            return "CUDA_MISMATCH"
        reason = accept_offer({
            "id": int(self.offer_id or 0),
            "machine_id": LIVE7_MACHINE_ID,
            "gpu_name": self.gpu_model,
            "gpu_ram": self.gpu_ram_mib,
            "disk_space": LIVE7_DISK_GB,
            "cuda_max_good": self.cuda_max_good,
            "compute_cap": self.compute_cap,
            "num_gpus": 1,
            "rentable": True,
        }, self.hourly_amount)
        if reason:
            return reason
        try:
            listed = self._instances()
        except WorkerStageError as exc:
            return exc.code
        owned = [
            row for row in listed
            if str(row.get("label") or "").startswith("media-engine:")
        ]
        decision = evaluate_grant(
            self.execution_grant,
            now=float(self._now()),
            execution_id=self.execution_id,
            ceiling_usd=LIVE7_CEILING_USD,
            instances=len(owned) + 1,
            machine_id=LIVE7_MACHINE_ID,
        )
        if decision["action"] != "continue":
            return str(decision["reason"])
        return ""

    def spending_stop(self, wall_now: float, monotonic_now: float) -> bool:
        budget = self.execution_budget
        if budget is None:
            return False
        return budget.stop(wall_now, monotonic_now)

    def _arm_guard(self) -> bool:
        """Start the detached watchdog before any instance request."""
        from media_engine.safety.execution_budget import ExecutionBudget, open_budget, retain_deadline
        popen = self._watchdog_popen
        if popen is None:
            if os.environ.get("MEDIA_ENGINE_WATCHDOG_LAUNCH") != "1":
                return False
            import subprocess
            popen = subprocess.Popen
        if self.execution_grant is None:
            return False
        root = Path(self.runtime)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "watchdog.json"
        local = os.environ.get("MEDIA_ENGINE_WATCHDOG_LOCAL", "").strip() == "1"
        try:
            poll = float(os.environ.get("MEDIA_ENGINE_WATCHDOG_POLL_SECONDS", "15"))
        except ValueError:
            poll = 15.0
        if path.is_file():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.live6_stop = "BUDGET_MISMATCH"
                return False
            try:
                retain_deadline(stored, float(self._now()) + 10**9, self.execution_id)
                self.execution_budget = ExecutionBudget.from_mapping(stored)
            except (KeyError, TypeError, ValueError):
                self.live6_stop = "BUDGET_MISMATCH"
                return False
            except WorkerStageError as exc:
                self.live6_stop = exc.code
                return False
        else:
            studio = os.environ.get("STUDIO_GPU_BUDGET_USD", "").strip()
            if self._live8_bound():
                expected_studio = self.execution_authorization.max_total_usd
                if not self._live8_budget_agrees(expected_studio):
                    self.live6_stop = "BUDGET_MISMATCH"
                    return False
            elif self._live7_bound():
                expected_studio = Decimal("1.50")
            else:
                expected_studio = VERIFY_RUN_BUDGET_USD
            if studio:
                try:
                    studio_amount = Decimal(studio)
                except Exception:
                    self.live6_stop = "BUDGET_MISMATCH"
                    return False
                if studio_amount != expected_studio:
                    self.live6_stop = "BUDGET_MISMATCH"
                    return False
            try:
                opener = open_budget
                if self._live8_bound():
                    from media_engine.safety.execution_budget import open_authorized_budget
                    opener = open_authorized_budget
                elif self._live7_bound():
                    from media_engine.safety.execution_budget import open_live7_budget
                    opener = open_live7_budget
                budget = opener(
                    execution_id=self.execution_id,
                    grant_execution_id=self.execution_grant.execution_id,
                    grant_ceiling=self.execution_grant.ceiling_usd,
                    hourly=self.hourly_amount,
                    wall_now=float(self._now()),
                    monotonic_now=float(self._monotonic()),
                    observed_starts=[float(self._now())],
                )
            except WorkerStageError as exc:
                self.live6_stop = exc.code
                return False
            if self._live7_bound() or self._live8_bound():
                from media_engine.safety.execution_budget import LIVE7_TAIL_SECONDS
                self.staging_deadline = budget.spending_deadline_wall
                self.staging_reserve_seconds = float(LIVE7_TAIL_SECONDS)
            pinned_machine = int(self.machine_id) if (self._live7_bound() or self._live8_bound()) else MACHINE_ID
            spec = {
                "mode": "local" if local else "provider",
                "machine_id": pinned_machine,
                "safe_seconds": budget.allowed_seconds,
                "authorization_expires_at": self.authorization_expires_at,
                "heartbeat": str(root / "watchdog.heartbeat"),
                "stop": str(root / "watchdog.stop"),
                "result": str(root / "watchdog.result"),
                "instances_path": str(root / "watchdog-instances.json"),
                "delete_log": str(root / "watchdog-deletes.json"),
                "instance_file": str(root / "watchdog.instance"),
                "poll_seconds": poll,
            }
            spec.update(budget.as_mapping())
            path.write_text(json.dumps(spec), encoding="utf-8")
            self.execution_budget = budget
        from media_engine.providers.vast_watchdog import arm_watchdog
        self.launch_trace.append("watchdog")
        pid = arm_watchdog(path, popen=popen)
        self.watchdog_pid = int(pid)
        return pid > 0

    def quote(self, owner: str) -> Quote:
        if not owner:
            raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
        if not self._five_dollar_target_is_authorized():
            raise WorkerStageError("BUDGET_MISMATCH")
        self._owner = owner
        self._label = ownership_label(owner)
        if self._live8_bound():
            self._refresh_live8(bind_grant=False)
            return Quote(hourly_price_usd=self.hourly_amount, provider="vast")
        if self._live7_bound():
            return self._quote_live7(owner)
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
        self.fits_operation_budget = chosen.projected_cost <= active_operation_budget()
        if chosen.source == "r2_cold_stage" and not _r2_configured():
            raise WorkerStageError("MODEL_CACHE_INVALID")
        return Quote(hourly_price_usd=self.hourly_amount, provider="vast")

    def _quote_live7(self, owner: str) -> Quote:
        """Search only the pinned machine. Another machine in the response is ignored."""
        from media_engine.providers.live7_contract import LIVE7_DISK_GB, LIVE7_MACHINE_ID, accept_offer
        del owner
        rows = self._bundle_offers({
            "limit": 8,
            "type": "on-demand",
            "rentable": {"eq": True},
            "num_gpus": {"eq": 1},
            "machine_id": {"eq": LIVE7_MACHINE_ID},
        })
        accepted = []
        for row in rows:
            if int(row.get("machine_id") or 0) != LIVE7_MACHINE_ID:
                continue
            try:
                hourly = _hourly(row, LIVE7_DISK_GB)
            except WorkerStageError:
                continue
            if accept_offer(row, hourly):
                continue
            accepted.append((row, hourly))
        if not accepted:
            raise WorkerStageError("OFFER_REJECTED")
        accepted.sort(key=lambda item: item[1])
        row, hourly = accepted[0]
        self.offer_id = str(row.get("id"))
        self.gpu_model = str(row.get("gpu_name") or "")
        self.machine_id = str(LIVE7_MACHINE_ID)
        self.hourly_amount = hourly
        self.cache_source = "r2_cold_stage"
        self.disk_gb = LIVE7_DISK_GB
        self.attached_volume_id = 0
        self.gpu_ram_mib = float(row.get("gpu_ram") or 0)
        self.cuda_max_good = float(row.get("cuda_max_good") or 0)
        self.compute_cap = int(row.get("compute_cap") or 0)
        self.projected_cost_usd = hourly
        self.fits_operation_budget = True
        return Quote(hourly_price_usd=hourly, provider="vast")

    def _search_live8(self) -> list:
        """Current rentable offers. A failed search is not an empty market."""
        from media_engine.providers.live7_contract import LIVE7_DISK_GB, LIVE7_MIN_VRAM_MIB
        return self._bundle_rows({
            "limit": 200,
            "type": "on-demand",
            "rentable": {"eq": True},
            "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": LIVE7_MIN_VRAM_MIB},
            "disk_space": {"gte": LIVE7_DISK_GB},
        })

    def _machine_offers(self, machine_id: int) -> list:
        """One machine, including a row that is no longer marked rentable."""
        return self._bundle_rows({
            "limit": 16,
            "type": "on-demand",
            "num_gpus": {"eq": 1},
            "machine_id": {"eq": int(machine_id)},
        })

    def _bundle_rows(self, query: dict) -> list:
        status, raw = self._request("POST", BUNDLES, query)
        if status != 200:
            raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE") from exc
        offers = payload.get("offers") if isinstance(payload, dict) else None
        if not isinstance(offers, list):
            raise WorkerStageError("PROVIDER_CAPACITY_UNAVAILABLE")
        return [row for row in offers if isinstance(row, dict)]

    def _live8_inventory_rows(self) -> list:
        """Fresh inventory. Just-in-time reads only the permitted machines."""
        authorization = self.execution_authorization
        if authorization is None:
            raise WorkerStageError("MISSING_AUTHORIZATION")
        if authorization.selection_policy != "just_in_time":
            return self._search_live8()
        rows: list = []
        seen: set[int] = set()
        for machine_id, _offer_id in authorization.allowed_pairs:
            if machine_id in seen:
                continue
            seen.add(machine_id)
            rows.extend(self._machine_offers(machine_id))
        return rows

    def _refresh_live8(self, *, bind_grant: bool = True) -> None:
        """Re-read inventory and keep only an offer inside this authorization."""
        from media_engine.providers.execution_authorization import make_live8_grant, select_for_authorization
        from media_engine.providers.gpu_discovery import assess_offers
        authorization = self.execution_authorization
        if authorization is None:
            raise WorkerStageError("MISSING_AUTHORIZATION")
        if float(self._now()) >= authorization.expires_at:
            raise WorkerStageError("EXPIRED_AUTHORIZATION")
        candidates = assess_offers(
            self._search_live8(),
            observed_at=float(self._now()),
            max_hourly=authorization.max_hourly_usd,
            ceiling=authorization.max_total_usd,
        )
        chosen, reason = select_for_authorization(authorization, candidates, now=float(self._now()))
        if chosen is None:
            raise WorkerStageError(reason or "OFFER_UNAVAILABLE")
        if self.execution_grant is not None:
            if int(self.execution_grant.machine_id or 0) != chosen.machine_id:
                raise WorkerStageError("OFFER_UNAVAILABLE")
            if int(self.execution_grant.offer_id or 0) != chosen.offer_id:
                raise WorkerStageError("OFFER_UNAVAILABLE")
        self.offer_id = str(chosen.offer_id)
        self.machine_id = str(chosen.machine_id)
        self.gpu_model = chosen.gpu_name
        self.hourly_amount = chosen.hourly_usd
        self.disk_gb = authorization.max_disk_gb
        self.cache_source = "r2_cold_stage"
        self.attached_volume_id = 0
        self.gpu_ram_mib = float(chosen.vram_mib)
        self.cuda_max_good = float(chosen.cuda or 0)
        self.compute_cap = chosen.compute_cap
        self.projected_cost_usd = chosen.session_cost_usd
        self.fits_operation_budget = True
        if bind_grant and self.execution_grant is None:
            grant = make_live8_grant(authorization, chosen)
            self.bind_execution(grant, execution_id=authorization.execution_id)

    def _reject_live8_run(self, listed: list | None = None) -> str:
        """Safety gates for the selected offer. A failed list blocks the rental."""
        authorization = self.execution_authorization
        if authorization is None or self.execution_grant is None:
            return "MISSING_AUTHORIZATION"
        if float(self._now()) >= authorization.expires_at:
            return "EXPIRED_AUTHORIZATION"
        if int(self.machine_id or 0) not in authorization.allowed_machine_ids:
            return "MACHINE_SCOPE"
        if int(self.offer_id or 0) not in authorization.allowed_offer_ids:
            return "OFFER_REJECTED"
        if self.hourly_amount > authorization.max_hourly_usd:
            return "PRICE_CHANGED"
        if int(self.disk_gb) != authorization.max_disk_gb:
            return "DISK_INSUFFICIENT"
        if self.cache_source != "r2_cold_stage" or int(self.attached_volume_id or 0) == VOLUME_ID:
            return "VOLUME_MISMATCH"
        if int(self.execution_grant.machine_id or 0) != int(self.machine_id):
            return "MACHINE_SCOPE"
        if int(self.execution_grant.offer_id or 0) != int(self.offer_id):
            return "OFFER_REJECTED"
        try:
            listed = self._instances() if listed is None else listed
        except WorkerStageError as exc:
            return exc.code
        for row in listed:
            if str(row.get("label") or "").startswith("media-engine:"):
                if int(row.get("machine_id") or 0) == int(self.machine_id):
                    return "INSTANCE_LIMIT"
        return ""

    def _stage_preflight_reason(self) -> str:
        """Refuse to rent when the worker cannot stage the pinned model."""
        from media_engine.providers.stage_preflight import assess_manifest_body, local_stage_problems
        problems = local_stage_problems(os.environ, _R2_SYNC)
        if problems:
            self.stage_problems = problems
            return "STAGE_CONFIG_INVALID"
        probe = getattr(self, "_manifest_ready", None)
        if probe is None:
            probe = self._manifest_probe
        try:
            ready = probe()
        except Exception:
            return "MODEL_CACHE_INVALID"
        if ready is True:
            return ""
        if ready is False or ready is None:
            return "MODEL_CACHE_INVALID"
        self.stage_problems = [str(ready)]
        return "MODEL_CACHE_INVALID"

    def _manifest_probe(self):
        from media_engine.engines.image.r2_cache import R2Client, config_from_env
        from media_engine.providers.stage_preflight import assess_manifest_body, manifest_object_key
        loaded = R2Client(config_from_env(os.environ)).get(manifest_object_key())
        if loaded is None:
            return False
        body, _meta = loaded
        reason = assess_manifest_body(body)
        return reason or True

    def _select_runtime_python(self) -> None:
        from media_engine.providers.runtime_python import (
            RUNTIME_SELECT_COMMAND,
            accepted_executable,
            parse_runtime_selection,
        )
        code, out, err = self._remote(RUNTIME_SELECT_COMMAND, 60, "RUNTIME_TIMEOUT")
        document = parse_runtime_selection(out)
        self.runtime_dependencies = list(document.get("candidates") or [])
        self.runtime_python_version = str(document.get("version") or "")
        selected = accepted_executable(document)
        if code != 0 or not selected:
            self._preserve_runtime_evidence(
                import_exit=str(code),
                stderr=err,
                exception=str(document.get("failure") or "NO_INTERPRETER"),
            )
            raise WorkerStageError("RUNTIME_PREPARE_FAILED")
        self.python_executable = selected
        self.runtime_ready = True

    def _preserve_runtime_evidence(self, *, import_exit: str, stderr: str, exception: str) -> None:
        from media_engine.providers.phase2d import _runtime_secrets
        from media_engine.providers.stage_evidence import CAPTURE_TIMEOUT_SECONDS, preserve_runtime_evidence
        preserve_runtime_evidence(
            destination=Path(self.runtime) / "runtime-evidence.json",
            fetch=self._fetch_runtime_text,
            secrets=_runtime_secrets(self._api_key),
            python_executable=self.python_executable,
            python_version=self.runtime_python_version,
            dependencies=self.runtime_dependencies,
            import_exit_code=import_exit,
            stderr=stderr,
            model_load_exception=exception,
            timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
        )

    def _fetch_runtime_text(self) -> tuple[str, str]:
        from media_engine.providers.stage_evidence import CAPTURE_TIMEOUT_SECONDS, REMOTE_RUNTIME_EVIDENCE
        if self._ssh_run is None or not self._host or not self._port:
            raise WorkerStageError("WORKER_BOOT_FAILED")
        _code, out, err = self._ssh_run(
            self._host, self._port, REMOTE_RUNTIME_EVIDENCE, int(CAPTURE_TIMEOUT_SECONDS),
        )
        return out, err

    def _preserve_stage_evidence(self, exit_code: str) -> None:
        from media_engine.providers.phase2d import _runtime_secrets
        from media_engine.providers.stage_evidence import CAPTURE_TIMEOUT_SECONDS, preserve_stage_evidence
        samples = getattr(self, "transfer_samples", None) or []
        preserve_stage_evidence(
            destination=Path(self.runtime) / "stage-evidence.json",
            fetch=self._fetch_stage_text,
            secrets=_runtime_secrets(self._api_key),
            exit_code=exit_code,
            bytes_done=getattr(self, "transfer_bytes_done", None),
            started_at=getattr(self, "stage_started_wall", None),
            ended_at=time.time(),
            last_progress_at=samples[-1][0] if samples else None,
            timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
        )

    def _fetch_stage_text(self) -> tuple[str, str]:
        from media_engine.providers.stage_evidence import CAPTURE_TIMEOUT_SECONDS, REMOTE_STAGE_EVIDENCE
        if self._ssh_run is None or not self._host or not self._port:
            raise WorkerStageError("WORKER_BOOT_FAILED")
        _code, out, err = self._ssh_run(
            self._host, self._port, REMOTE_STAGE_EVIDENCE, int(CAPTURE_TIMEOUT_SECONDS),
        )
        return out, err

    def _intent_path(self) -> Path:
        return Path(self.runtime) / "create-intent.json"

    def _write_create_intent(self, label: str) -> None:
        authorization = self.execution_authorization
        deadline = None if self.execution_budget is None else self.execution_budget.spending_deadline_wall
        payload = {
            "state": "create_request",
            "authorization_id": authorization.authorization_id,
            "execution_id": self.execution_id,
            "code_sha": authorization.code_sha,
            "machine_id": int(self.machine_id),
            "offer_id": int(self.offer_id),
            "label": label,
            "deadline_wall": deadline,
        }
        self._intent_path().write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _mark_intent(self, state: str, instance_id: str = "") -> None:
        path = self._intent_path()
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        payload["state"] = state
        if instance_id:
            payload["instance_id"] = instance_id
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _reconcile_create_intent(self) -> str:
        """Adopt the labeled instance, or stop. This never sends a second create."""
        path = self._intent_path()
        if not path.is_file():
            return "absent"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.live6_stop = "UNRESOLVED_BILLING"
            return "unresolved"
        authorization = self.execution_authorization
        if not isinstance(payload, dict) or payload.get("authorization_id") != authorization.authorization_id:
            self.live6_stop = "UNRESOLVED_BILLING"
            return "unresolved"
        label = str(payload.get("label") or "")
        machine_id = int(payload.get("machine_id") or 0)
        saved = str(payload.get("instance_id") or "")
        if payload.get("state") == "created" and saved.isdigit():
            return saved
        if payload.get("state") == "rejected":
            return self._reconcile_rejected(label, machine_id)
        try:
            listed = self._instances()
        except WorkerStageError as exc:
            self.live6_stop = exc.code
            return "unresolved"
        found = [
            str(row.get("id"))
            for row in listed
            if str(row.get("label") or "") == label and int(row.get("machine_id") or 0) == machine_id
        ]
        if len(found) == 1 and str(found[0]).isdigit():
            return str(found[0])
        self.live6_stop = "UNRESOLVED_BILLING"
        return "unresolved"

    def _reconcile_rejected(self, label: str, machine_id: int) -> str:
        """A rejected create stays closed. An instance that actually appeared is adopted."""
        try:
            listed = self._instances()
        except WorkerStageError as exc:
            self.live6_stop = exc.code
            return "unresolved"
        found = [
            str(row.get("id"))
            for row in listed
            if str(row.get("label") or "") == label and int(row.get("machine_id") or 0) == machine_id
        ]
        if len(found) == 1 and str(found[0]).isdigit():
            return str(found[0])
        if found:
            self.live6_stop = "UNRESOLVED_BILLING"
            return "unresolved"
        self.create_classification = "provider_create_rejected"
        self.billing_exposure = False
        return "rejected"

    def _stop_before_create(self, reason: str) -> CreateResult:
        """The provider was not asked to create an instance."""
        self.live6_stop = reason
        self.create_request_sent = False
        self.create_classification = "offer_unavailable_before_create"
        self.billing_exposure = False
        self.provider_response = None
        self._write_create_observation()
        return CreateResult(outcome="definite_failure")

    def _write_create_observation(self) -> None:
        selected_machine = int(self.machine_id or 0)
        selected_offer = int(self.offer_id or 0)
        if self.selection_attempts and self.selection_attempts[-1].get("result") != "verified":
            selected_machine = int(self.selection_attempts[-1]["machine_id"])
            selected_offer = int(self.selection_attempts[-1]["offer_id"])
        payload = {
            "offer_observed_at": self.offer_observed_at,
            "selected_machine_id": selected_machine,
            "selected_offer_id": selected_offer,
            "final_verification_at": self.final_verification_at,
            "create_request_at": self.create_request_at,
            "create_request_sent": bool(self.create_request_sent),
            "provider_response": self.provider_response,
            "classification": self.create_classification,
            "billing_exposure": bool(self.billing_exposure),
            "stop": self.live6_stop,
            "attempts": list(self.selection_attempts),
        }
        path = Path(self.runtime) / "create-observation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _apply_live8_candidate(self, chosen) -> None:
        authorization = self.execution_authorization
        self.offer_id = str(chosen.offer_id)
        self.machine_id = str(chosen.machine_id)
        self.gpu_model = chosen.gpu_name
        self.hourly_amount = chosen.hourly_usd
        self.disk_gb = authorization.max_disk_gb
        self.cache_source = "r2_cold_stage"
        self.attached_volume_id = 0
        self.gpu_ram_mib = float(chosen.vram_mib)
        self.cuda_max_good = float(chosen.cuda or 0)
        self.compute_cap = chosen.compute_cap
        self.projected_cost_usd = chosen.session_cost_usd
        self.fits_operation_budget = True

    def _final_verify(self, candidate):
        """Re-read one offer immediately before any create request."""
        from media_engine.providers.gpu_discovery import assess_offer
        authorization = self.execution_authorization
        self.final_verification_at = float(self._now())
        try:
            rows = self._machine_offers(candidate.machine_id)
        except WorkerStageError as exc:
            return None, exc.code
        match = [
            row for row in rows
            if int(row.get("id") or 0) == candidate.offer_id
            and int(row.get("machine_id") or 0) == candidate.machine_id
        ]
        if len(match) != 1:
            return None, "OFFER_UNAVAILABLE"
        assessed = assess_offer(
            match[0],
            observed_at=float(self.final_verification_at),
            max_hourly=authorization.max_hourly_usd,
            ceiling=authorization.max_total_usd,
        )
        if assessed.rejection:
            return None, assessed.rejection
        if authorization.gpu_families and assessed.gpu_family not in authorization.gpu_families:
            return None, "GPU_MISMATCH"
        if assessed.hourly_usd > authorization.max_hourly_usd:
            return None, "PRICE_CHANGED"
        if authorization.selection_policy == "just_in_time":
            pair = (assessed.machine_id, assessed.offer_id)
            if pair not in authorization.allowed_pairs:
                return None, "OFFER_REJECTED"
        elif (
            assessed.offer_id not in authorization.allowed_offer_ids
            or assessed.machine_id not in authorization.allowed_machine_ids
        ):
            return None, "OFFER_REJECTED"
        return assessed, ""

    def _label_absent(self, label: str) -> bool:
        try:
            listed = self._instances()
        except WorkerStageError:
            return False
        return not any(str(row.get("label") or "") == label for row in listed)

    def _record_create_response(self, status: int, outcome: str, label: str) -> None:
        self.provider_response = {"status": int(status), "outcome": outcome}
        if outcome == "created":
            self.create_classification = "instance_created"
            self.billing_exposure = True
        elif outcome == "definite_failure":
            self.create_classification = "provider_create_rejected"
            absent = self._label_absent(label)
            self.billing_exposure = not absent
            if absent:
                self._mark_intent("rejected")
            else:
                self._mark_intent("unresolved")
                if not self.live6_stop:
                    self.live6_stop = "UNRESOLVED_BILLING"
        else:
            self.create_classification = "ambiguous_create"
            self.billing_exposure = True
        self._write_create_observation()

    def _create_live8(self, owner: str) -> CreateResult:
        """Select inside the authorization, verify that offer, then send one create."""
        from media_engine.providers.execution_authorization import (
            candidates_for_authorization,
            make_live8_grant,
        )
        from media_engine.providers.gpu_discovery import assess_offers
        prior = self._reconcile_create_intent()
        if prior == "unresolved":
            self.create_classification = "ambiguous_create"
            self.billing_exposure = True
            self._write_create_observation()
            return CreateResult(outcome="ambiguous")
        if prior == "rejected":
            self._write_create_observation()
            return CreateResult(outcome="definite_failure")
        if prior.isdigit():
            self.create_classification = "instance_created"
            self.billing_exposure = True
            self._resource_id = prior
            self.last_instance_id = prior
            self._remember_instance(prior)
            self._write_create_observation()
            return CreateResult(outcome="created", resource_id=prior)
        try:
            listed = self._instances()
        except WorkerStageError as exc:
            return self._stop_before_create(exc.code)
        try:
            rows = self._live8_inventory_rows()
        except WorkerStageError as exc:
            return self._stop_before_create(exc.code)
        self.offer_observed_at = float(self._now())
        authorization = self.execution_authorization
        assessed = assess_offers(
            rows,
            observed_at=float(self.offer_observed_at),
            max_hourly=authorization.max_hourly_usd,
            ceiling=authorization.max_total_usd,
        )
        ranked, reason = candidates_for_authorization(
            authorization, assessed, now=float(self._now()),
        )
        if not ranked:
            return self._stop_before_create(reason or "OFFER_UNAVAILABLE")
        reason = self._stage_preflight_reason()
        if reason:
            return self._stop_before_create(reason)
        verified = None
        for candidate in ranked:
            checked, rejection = self._final_verify(candidate)
            self.selection_attempts.append({
                "machine_id": candidate.machine_id,
                "offer_id": candidate.offer_id,
                "verified_at": self.final_verification_at,
                "result": rejection or "verified",
            })
            if rejection:
                if authorization.selection_policy != "just_in_time":
                    return self._stop_before_create(rejection)
                continue
            verified = checked
            break
        if verified is None:
            last = self.selection_attempts[-1]["result"] if self.selection_attempts else "OFFER_UNAVAILABLE"
            return self._stop_before_create(str(last))
        occupied = any(
            str(row.get("label") or "").startswith("media-engine:")
            and int(row.get("machine_id") or 0) == verified.machine_id
            for row in listed
        )
        if occupied:
            return self._stop_before_create("INSTANCE_LIMIT")
        if not self._live8_budget_agrees(authorization.max_total_usd):
            return self._stop_before_create("BUDGET_MISMATCH")
        self._apply_live8_candidate(verified)
        try:
            grant = make_live8_grant(authorization, verified)
            self.bind_execution(grant, execution_id=authorization.execution_id)
        except WorkerStageError as exc:
            return self._stop_before_create(exc.code)
        reason = self._reject_live8_run(listed)
        if reason:
            return self._stop_before_create(reason)
        if not self._arm_guard():
            if not self.live6_stop:
                self.live6_stop = "WATCHDOG_NOT_ARMED"
            self.create_request_sent = False
            self.create_classification = "offer_unavailable_before_create"
            self.billing_exposure = False
            self._write_create_observation()
            return CreateResult(outcome="definite_failure")
        return self._dispatch_create(owner)

    def create(self, owner: str) -> CreateResult:
        authorization = self.execution_authorization
        if authorization is not None and self._creates_used >= int(authorization.max_creates):
            self._create_closed = True
            self.create_request_sent = False
            self.billing_exposure = False
            return CreateResult(outcome="definite_failure")
        if self._put_done or self._create_closed:
            return CreateResult(outcome="definite_failure")
        if not self._five_dollar_target_is_authorized():
            self.live6_stop = "BUDGET_MISMATCH"
            self.create_request_sent = False
            self.billing_exposure = False
            if self._live8_bound():
                self._creates_used += 1
                self._create_closed = True
            else:
                self._put_done = True
            return CreateResult(outcome="definite_failure")
        if self._live8_bound():
            self._creates_used += 1
            self._create_closed = True
            return self._create_live8(owner)
        self._put_done = True
        if not self.offer_id:
            self.quote(owner)
        if self._live7_bound():
            reason = self._reject_live7_run()
            if reason:
                self.live6_stop = reason
                return CreateResult(outcome="definite_failure")
            if not self._arm_guard():
                if not self.live6_stop:
                    self.live6_stop = "WATCHDOG_NOT_ARMED"
                return CreateResult(outcome="definite_failure")
        elif not self._live8_bound() and (
            self.execution_grant is not None
            or os.environ.get("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", "").strip() == "0.50"
        ):
            reason = self._reject_unverified_run()
            if reason:
                self.live6_stop = reason
                return CreateResult(outcome="definite_failure")
            if not self._arm_guard():
                if not self.live6_stop:
                    self.live6_stop = "WATCHDOG_NOT_ARMED"
                return CreateResult(outcome="definite_failure")
        return self._dispatch_create(owner)

    def _dispatch_create(self, owner: str) -> CreateResult:
        label = ownership_label(self.execution_id if self._live8_bound() else owner)
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
            # runtype ssh replaces the image command and publishes no ports.
            # Vast maps this docker -p key to a random public HostPort.
            body["env"] = {
                "HF_HOME": "/models/hf-cache",
                "HUGGINGFACE_HUB_CACHE": "/models/hf-cache",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "WORKER_PORT": "8080",
                "-p 8080:8080": "1",
            }
        if self._live7_bound() or self._live8_bound():
            if "volume_info" in body or int(getattr(self, "attached_volume_id", 0) or 0) == VOLUME_ID:
                if self._live8_bound():
                    return self._stop_before_create("VOLUME_MISMATCH")
                self.live6_stop = "VOLUME_MISMATCH"
                return CreateResult(outcome="definite_failure")
        elif self.cache_source == "warm_cache":
            body["volume_info"] = {
                "mount_path": MOUNT_PATH, "create_new": False, "volume_id": int(self.attached_volume_id),
            }
        if self._live8_bound():
            self._write_create_intent(label)
            self._put_done = True
            self.create_request_sent = True
            self.create_request_at = float(self._now())
        self.launch_trace.append("put")
        status, raw = self._request("PUT", f"https://console.vast.ai/api/v0/asks/{self.offer_id}/", body)
        if status <= 0:
            outcome, instance_id = "ambiguous", None
        else:
            outcome, instance_id, _error = classify_create_response(status, raw)
        if outcome == "ambiguous":
            found = self._await_owned(label)
            if len(found) == 1:
                outcome, instance_id = "created", found[0]
            elif self._live8_bound():
                self._mark_intent("unresolved")
                if not self.live6_stop:
                    self.live6_stop = "UNRESOLVED_BILLING"
        if self._live8_bound():
            self._record_create_response(status, outcome, label)
        if outcome == "created" and instance_id:
            self._resource_id = instance_id
            self.last_instance_id = instance_id
            self._owner = owner
            self._label = label
            self._started_at[instance_id] = self._now()
            self._remember_instance(instance_id)
            if self._live8_bound():
                self._mark_intent("created", instance_id)
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
        try:
            row = self._instance_row(resource_id)
        except WorkerStageError:
            return OBSERVE_UNKNOWN
        if row is None:
            return OBSERVE_GONE
        label = str(row.get("label") or "")
        if label != ownership_label(owner) or not self._same_machine(row):
            return OBSERVE_FOREIGN
        return OBSERVE_PRESENT

    def terminate(self, resource_id: str) -> None:
        """Delete one exact id. Gone is recorded only after a later list omits it."""
        if not str(resource_id).isdigit():
            raise WorkerStageError("OWNERSHIP_MISMATCH")
        try:
            row = self._instance_row(resource_id)
        except WorkerStageError:
            row = "unlisted"
        if row is None:
            self._mark_gone(resource_id)
            return
        if row != "unlisted":
            label = str(row.get("label") or "")
            if not label.startswith("media-engine:") or not self._same_machine(row):
                raise WorkerStageError("OWNERSHIP_MISMATCH")
            if str(row.get("id")) != str(resource_id):
                raise WorkerStageError("OWNERSHIP_MISMATCH")
        self._request("DELETE", f"https://console.vast.ai/api/v0/instances/{resource_id}/", None)
        self._confirm_absent(resource_id)

    def _mark_gone(self, resource_id: str) -> None:
        self._gone.add(resource_id)
        self._busy.discard(resource_id)
        if self._resource_id == resource_id:
            self._resource_id = ""
        self._put_done = False

    def _confirm_absent(self, resource_id: str) -> None:
        try:
            rows = self._instances()
        except WorkerStageError as exc:
            raise WorkerStageError("DELETE_UNCONFIRMED") from exc
        if any(str(row.get("id")) == str(resource_id) for row in rows):
            raise WorkerStageError("DELETE_UNCONFIRMED")
        self._mark_gone(resource_id)

    def _remember_instance(self, instance_id: str) -> None:
        if not str(instance_id).isdigit():
            return
        path = Path(self.runtime) / "watchdog.instance"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(instance_id), encoding="utf-8")

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
            if self._worker_cache_already_complete():
                self.cache_validation = "COMPLETE"
                self.cache_source = "worker_cache_hit"
                return
            if os.environ.get("MEDIA_ENGINE_FORBID_R2_STAGE", "").strip() == "1":
                self.cache_validation = "R2_FORBIDDEN"
                raise WorkerStageError("MODEL_CACHE_INVALID")
            self._claim_model_stage()
            self._stage_r2()
        else:
            worker = getattr(self, "_worker_ssh", None)
            if worker is not None and self._host and self._port:
                worker.stage(self._host, self._port)
        code, out, _err = self._remote(_CACHE_CHECK, 180)
        if code != 0 or not _cache_ok(out, require_mount=self.cache_source != "r2_cold_stage"):
            self.cache_validation = self.cache_validation or "CACHE_INCOMPLETE"
            raise WorkerStageError("MODEL_CACHE_INVALID")
        expected = getattr(self, "verified_manifest", None)
        if isinstance(expected, dict):
            matched, reason = inventory_matches(out, expected)
            if not matched:
                self.cache_validation = reason
                raise WorkerStageError("MODEL_CACHE_INVALID")
            code, resolved, _err = self._remote(_CACHE_RESOLVE, 180, "MODEL_LOAD_FAILED")
            if code != 0 or not _resolved_locally(resolved):
                self.cache_validation = "RESOLVE_FAILED"
                raise WorkerStageError("MODEL_CACHE_INVALID")
        self.cache_validation = "COMPLETE"

    def prepare_runtime(self, resource_id: str) -> None:
        # Health can set runtime_ready before any interpreter is known.
        # That flag must not skip selection or imply the virtualenv exists.
        if self.python_executable:
            return
        worker = getattr(self, "_worker_ssh", None)
        if worker is not None and self._host and self._port:
            worker.stage(self._host, self._port)
        self._select_runtime_python()

    def ensure_model(self, resource_id: str) -> None:
        if not self.python_executable:
            self._select_runtime_python()
        from media_engine.providers.runtime_python import import_probe_command
        command = import_probe_command(self.python_executable)
        code, out, err = self._remote(command, 120)
        if code != 0 or "QwenImagePipeline" not in out:
            self._preserve_runtime_evidence(import_exit=str(code), stderr=err, exception="IMPORT_FAILED")
            raise WorkerStageError("MODEL_LOAD_FAILED")

    def _claim_model_stage(self) -> None:
        authorization = self.execution_authorization
        if authorization is None:
            return
        if self._model_stages_used >= int(authorization.max_model_stages):
            raise WorkerStageError("INSTANCE_LIMIT")
        self._model_stages_used += 1

    def _claim_model_load(self) -> None:
        authorization = self.execution_authorization
        if authorization is None:
            return
        if self._model_loads_used >= int(authorization.max_model_loads):
            raise WorkerStageError("INSTANCE_LIMIT")
        self._model_loads_used += 1

    def arm_live_session(self) -> None:
        """Apply the bound record. This does not create an instance or raise its limits."""
        authorization = self.execution_authorization
        self.attached_volume_id = 0
        if authorization is None:
            self.session_limit = 0
            self.single_image_only = True
            return
        images = int(authorization.max_image_count)
        inferences = int(authorization.max_inference_attempts)
        if images not in (1, 3) or inferences != images:
            raise WorkerStageError("INSTANCE_LIMIT")
        if (
            int(authorization.max_instances) != 1
            or int(authorization.max_creates) != 1
            or int(authorization.max_gpu_instances) != 1
            or int(authorization.max_model_stages) != 1
            or int(authorization.max_model_loads) != 1
        ):
            raise WorkerStageError("INSTANCE_LIMIT")
        self.session_limit = images
        self.single_image_only = images == 1

    def generate(self, *, prompt: str, width: int, height: int, seed: int) -> bytes:
        authorization = self.execution_authorization
        if authorization is not None:
            if self.authorization_expires_at is not None and float(self._now()) >= float(self.authorization_expires_at):
                raise WorkerStageError("EXPIRED_AUTHORIZATION")
            if int(self.session_limit) > int(authorization.max_image_count):
                raise WorkerStageError("INSTANCE_LIMIT")
            key = (prompt, width, height, seed)
            cached = self._session_results.get(key)
            if cached is not None:
                return cached
            if self._images_started >= int(authorization.max_image_count):
                raise WorkerStageError("SESSION_IMAGE_LIMIT")
            if self._inference_attempts >= int(authorization.max_inference_attempts):
                raise WorkerStageError("SESSION_IMAGE_LIMIT")
            if int(authorization.max_image_count) > 1:
                if int(self.session_limit) != int(authorization.max_image_count):
                    raise WorkerStageError("INSTANCE_LIMIT")
                return self._session_generate(prompt=prompt, width=width, height=height, seed=seed)
        elif int(getattr(self, "session_limit", 0) or 0) > 1:
            raise WorkerStageError("INSTANCE_LIMIT")
        python = self.python_executable
        if not python:
            raise WorkerStageError("RUNTIME_PREPARE_FAILED")
        if self.execution_authorization is not None:
            self._claim_model_load()
        command = _generate_command(prompt, width, height, seed, python=python)
        self.commands.append(command)
        code, out, err = self._remote(
            command,
            int(PHASE_LIMITS["model_loading"][0] + PHASE_LIMITS["generating"][0]),
        )
        if code != 0 or "PNG_READY" not in out:
            if "LOAD_FAILED" in out or "IMPORT_FAILED" in out:
                self._preserve_runtime_evidence(
                    import_exit=str(code),
                    stderr=err,
                    exception="LOAD_FAILED" if "LOAD_FAILED" in out else "IMPORT_FAILED",
                )
                raise WorkerStageError("MODEL_LOAD_FAILED")
            raise WorkerStageError("GENERATION_FAILED")
        data = self._fetch_png()
        if self.execution_authorization is not None:
            self._images_started += 1
            self._inference_attempts += 1
            self._session_results[(prompt, width, height, seed)] = data
        return data

    def _session_generate(self, *, prompt: str, width: int, height: int, seed: int) -> bytes:
        """Submit one image to the resident worker. A repeated job does not infer again."""
        if self.authorization_expires_at is not None and float(self._now()) >= float(self.authorization_expires_at):
            raise WorkerStageError("EXPIRED_AUTHORIZATION")
        python = self.python_executable
        if not python:
            raise WorkerStageError("RUNTIME_PREPARE_FAILED")
        if width != FIRST_WIDTH or height != FIRST_HEIGHT:
            raise WorkerStageError("GENERATION_FAILED")
        key = (prompt, width, height, seed)
        cached = self._session_results.get(key)
        if cached is not None:
            return cached
        authorization = self.execution_authorization
        limit = int(self.session_limit)
        if authorization is not None:
            if limit > int(authorization.max_image_count):
                raise WorkerStageError("INSTANCE_LIMIT")
            limit = int(authorization.max_image_count)
            if self._inference_attempts >= int(authorization.max_inference_attempts):
                raise WorkerStageError("SESSION_IMAGE_LIMIT")
        if self._images_started >= limit or len(self._session_results) >= limit:
            raise WorkerStageError("SESSION_IMAGE_LIMIT")
        if not self._session_booted:
            self._claim_model_load()
            code, out, _err = self._remote(_session_boot_command(python), 1200)
            if code != 0 or "SESSION_READY" not in out:
                raise WorkerStageError("MODEL_LOAD_FAILED")
            self._session_booted = True
        image_id = "s" + str(len(self._session_results) + 1)
        code, out, _err = self._remote(
            _session_job_command(image_id, prompt, width, height, seed),
            int(PHASE_LIMITS["generating"][0]),
        )
        if code != 0 or "SESSION_IMAGE" not in out:
            raise WorkerStageError("GENERATION_FAILED")
        data = self._fetch_session_png(image_id)
        self._images_started += 1
        self._inference_attempts += 1
        self._session_results[key] = data
        self.commands.append("session:" + image_id)
        return data

    def _fetch_session_png(self, image_id: str) -> bytes:
        if not image_id.startswith("s") or not image_id[1:].isdigit():
            raise WorkerStageError("ARTIFACT_TRANSFER_FAILED")
        code, out, _err = self._remote("base64 /workspace/session-out/" + image_id + ".png", 120)
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
        # 3. An explicitly selected other machine cold-starts from R2.
        # 4. A broad cold search only when MEDIA_ENGINE_ALLOW_COLD_STAGE is set.
        verification = os.environ.get("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", "").strip() == "0.50"
        selected = explicit_machine_id()
        if verification and selected not in (None, MACHINE_ID):
            return None
        if selected is not None and selected != MACHINE_ID:
            return choose_offer(
                self._bundle_offers({
                    "limit": 16,
                    "type": "on-demand",
                    "rentable": {"eq": True},
                    "num_gpus": {"eq": 1},
                    "machine_id": {"eq": selected},
                    "gpu_ram": {"gte": MIN_VRAM_MIB},
                }),
                volume_machine=MACHINE_ID,
                volume_ready=False,
                max_hourly=self.limits.max_hourly_price_usd,
                operation_budget=active_operation_budget(),
                allow_cold_stage=True,
            )
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
        if verification or warm is not None or not cold_stage_allowed():
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

    def _worker_cache_already_complete(self) -> bool:
        """Ask the GPU. A cache directory on the control machine is not a hit."""
        try:
            code, out, _err = self._remote(_CACHE_CHECK, 180)
        except WorkerStageError:
            return False
        return code == 0 and _cache_ok(out, require_mount=False)

    def _stage_r2(self) -> None:
        if os.environ.get("MEDIA_ENGINE_FORBID_R2_STAGE", "").strip() == "1":
            self.cache_validation = "R2_FORBIDDEN"
            raise WorkerStageError("MODEL_CACHE_INVALID")
        from media_engine.engines.image import model_cache, qwen_remote, r2_cache
        from media_engine.providers.phase2d import r2_worker_env
        from media_engine.providers.vast_ssh import WorkerSSH
        env = dict(os.environ)
        env["MODEL_CACHE_PROVIDER"] = "r2"
        payload = r2_worker_env(env)
        if payload is None or not self._host or not self._port:
            self._preserve_stage_evidence("")
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
        self.stage_started_wall = time.time()
        self._watch_cache_sync()
        self.transfer_seconds = round(time.perf_counter() - started, 3)

    def _instances(self) -> list[dict]:
        from media_engine.providers.vast_lifecycle import InstanceListError, collect_instance_pages

        def request(url: str) -> tuple[int, bytes]:
            return self._request("GET", url, None)

        try:
            return collect_instance_pages(request)
        except InstanceListError as exc:
            raise WorkerStageError(exc.code) from exc

    def _instance_row(self, resource_id: str) -> Optional[dict]:
        for row in self._instances():
            if str(row.get("id")) == str(resource_id):
                return row
        return None

    def _await_owned(self, label: str) -> list[str]:
        try:
            found = self._owned_ids(label)
        except WorkerStageError as exc:
            self.live6_stop = exc.code
            return []
        if found or self._sleep is not time.sleep:
            return found
        for _ in range(8):
            self._sleep(3)
            try:
                found = self._owned_ids(label)
            except WorkerStageError as exc:
                self.live6_stop = exc.code
                return []
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
        running_without_port: Optional[float] = None
        self.health_error = ""
        while True:
            row = self._instance_row(resource_id)
            if row is None:
                raise WorkerStageError("WORKER_BOOT_FAILED")
            status_name = str(row.get("actual_status") or "")
            # Image pull has no worker yet. The 180s stall applies only after
            # the instance is running. The startup maximum is unchanged.
            if status_name != "running":
                watch.mark(self._now())
            if watch.expired(self._now()):
                if status_name != "running" and not self.health_error:
                    raise WorkerStageError("INSTANCE_NOT_RUNNING")
                raise WorkerStageError(_readiness_timeout_code(self.health_error))
            endpoint = worker_http_endpoint(row)
            ssh_endpoint = worker_ssh_endpoint(row)
            if status_name == "running" and endpoint is None and ssh_endpoint is None:
                if running_without_port is None:
                    running_without_port = self._now()
                elif self._now() - running_without_port >= PORT_MAP_GRACE_SECONDS:
                    self.worker_state = "instance_running"
                    raise WorkerStageError("PORT_UNPUBLISHED")
            else:
                running_without_port = None
            if ssh_endpoint is not None:
                kind, health = self._ssh_local_health(row)
                if kind == "http":
                    self.worker_state = "http_failed"
                    raise WorkerStageError("HTTP_HEALTH_FAILED")
                if kind in {"ssh", "connect", "not_ready"}:
                    self.health_error = kind
            elif endpoint is not None:
                health = self._fetch_health(endpoint)
                self.health_transport = "public"
            else:
                health = None
            state = classify_readiness(
                status_name,
                endpoint is not None or isinstance(health, dict),
                health,
            )
            self.worker_state = state
            if state != seen:
                seen = state
                watch.mark(self._now())
            if isinstance(health, dict) and health.get("state") == "container_starting":
                watch.mark(self._now())
            if state == "worker_ready":
                self.runtime_ready = True
                return
            if state == "startup_failed":
                raise WorkerStageError("STARTUP_FAILED")
            if state == "worker_failed":
                raise WorkerStageError("WORKER_BOOT_FAILED")
            self._sleep(5)

    def _ssh_local_health(self, row: dict) -> tuple[str, Optional[dict]]:
        """GET loopback /health through the instance SSH proxy. No public HTTP."""
        endpoint = worker_ssh_endpoint(row)
        if endpoint is None:
            return "missing", None
        host, port = endpoint
        if self._ssh_run is None:
            try:
                self._attach_and_wait(str(row.get("id") or ""))
            except WorkerStageError as exc:
                if exc.code in {"CONNECT_TIMEOUT", "WORKER_BOOT_FAILED"}:
                    raise
                self.health_error = "ssh"
                return "ssh", None
        if self._ssh_run is None:
            self.health_error = "ssh"
            return "ssh", None
        try:
            code, out, _err = self._ssh_run(host, port, LOCAL_HEALTH_COMMAND, 15)
        except WorkerStageError:
            self.health_error = "ssh"
            return "ssh", None
        self.health_transport = "ssh"
        kind, health = classify_ssh_probe(code, out)
        self.health_error = kind if kind != "ready" else ""
        return kind, health

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
        if self._live7_bound() and self.execution_budget is not None:
            remaining = float(self.execution_budget.spending_deadline_wall) - float(self._now())
            if remaining <= float(self.staging_reserve_seconds or 0):
                self._preserve_stage_evidence("")
                raise WorkerStageError("CACHE_TIMEOUT")
            watch.max_seconds = remaining
        code, out, _err = self._remote(_R2_SYNC, 60, "CACHE_TIMEOUT")
        if code != 0 or "STARTED" not in out:
            self._preserve_stage_evidence("")
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
                        done = record.get("bytes_done")
                        total = record.get("bytes_total")
                        if isinstance(done, int) and not isinstance(done, bool) and done >= 0:
                            self.transfer_bytes_done = done
                            self.transfer_samples.append((self._now(), done))
                        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                            self.transfer_bytes_total = total
            _code, exit_out, _err = self._remote(_CACHE_EXIT, 30, "CACHE_TIMEOUT")
            status = ""
            for line in reversed(exit_out.splitlines()):
                if line.strip():
                    status = line.strip()
                    break
            if status.isdigit():
                if status != "0":
                    self._preserve_stage_evidence(status)
                    raise WorkerStageError("MODEL_CACHE_INVALID")
                return
            decision = decide_staging_budget(
                self.transfer_samples,
                now=self._now(),
                deadline=getattr(self, "staging_deadline", None),
                reserve_seconds=float(getattr(self, "staging_reserve_seconds", 0) or 0),
                total_bytes=int(self.transfer_bytes_total or 0),
            )
            self.staging_events.append(decision)
            if decision["action"] == "stop":
                self.staging_stop_reason = decision["reason"]
                self._remote(_CACHE_KILL, 20, "CACHE_TIMEOUT")
                self._preserve_stage_evidence("")
                raise WorkerStageError("CACHE_TIMEOUT")
            reason = watch.expired(self._now())
            if reason == "PHASE_STALL":
                self._remote(_CACHE_KILL, 20, "CACHE_STALL")
                self._preserve_stage_evidence("")
                raise WorkerStageError("CACHE_STALL")
            if reason == "PHASE_TIMEOUT":
                self._remote(_CACHE_KILL, 20, "CACHE_TIMEOUT")
                self._preserve_stage_evidence("")
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


def explicit_machine_id() -> Optional[int]:
    """An operator-selected machine. Empty keeps the warm-cache machine preferred."""
    raw = os.environ.get("MEDIA_ENGINE_MACHINE_ID", "").strip()
    if not raw.isdigit():
        return None
    machine = int(raw)
    return machine if machine > 0 else None


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


# A running instance should already have its published ports. This is long
# enough for Vast to fill the ports map and far shorter than the readiness stall.
PORT_MAP_GRACE_SECONDS = 45.0
_LOOPBACK_HEALTH_PROBE = """
import json, urllib.request, urllib.error
url = "http://127.0.0.1:8080/health"
try:
    response = urllib.request.urlopen(url, timeout=2)
    body = response.read(10000).decode()
    print(json.dumps({"probe": "http", "status": int(response.status), "body": body}))
except urllib.error.HTTPError as exc:
    print(json.dumps({"probe": "http", "status": int(exc.code), "body": ""}))
except Exception as exc:
    print(json.dumps({"probe": "connect", "error": type(exc).__name__}))
""".strip()
LOCAL_HEALTH_COMMAND = "python3 -c '" + _LOOPBACK_HEALTH_PROBE + "'"

# SSH mode does not run the image command. This script detaches the server
# into its own session and records a secret-free launch line.
WORKER_ONSTART = "bash /workspace/start_worker.sh"


def portable_worker(image: str) -> bool:
    """True when the instance image is the published runtime, not the stock CUDA image."""
    return image.startswith("ghcr.io/") and "@sha256:" in image


def classify_readiness(instance_status: str, has_endpoint: bool, health: Optional[dict]) -> str:
    """Separate a running VM from a worker that has answered /health."""
    if str(instance_status) != "running":
        return "provisioning"
    if isinstance(health, dict) and health.get("state") == "startup_failed":
        return "startup_failed"
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


def worker_ssh_endpoint(row: dict) -> Optional[tuple[str, str]]:
    if str(row.get("actual_status") or "") != "running":
        return None
    host = str(row.get("ssh_host") or "")
    port = "" if row.get("ssh_port") is None else str(row.get("ssh_port"))
    if host and port.isdigit():
        return host, port
    return None


def parse_health_output(text: str) -> Optional[dict]:
    payload = last_json_object(text)
    if isinstance(payload, dict) and "state" in payload:
        return payload
    return None


def last_json_object(text: str) -> Optional[dict]:
    for line in reversed(str(text).splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def classify_ssh_probe(code: int, text: str) -> tuple[str, Optional[dict]]:
    """Separate a dead SSH session, a refused loopback connection, and an HTTP answer."""
    if code != 0:
        return "ssh", None
    probe = last_json_object(text)
    if not isinstance(probe, dict) or "probe" not in probe:
        return "ssh", None
    if probe.get("probe") == "connect":
        return "connect", None
    if probe.get("probe") != "http":
        return "ssh", None
    if probe.get("status") != 200:
        return "http", None
    body = probe.get("body")
    if not isinstance(body, str):
        return "http", None
    try:
        health = json.loads(body)
    except json.JSONDecodeError:
        return "http", None
    if not isinstance(health, dict) or "state" not in health:
        return "http", None
    if health.get("state") == "worker_ready" and health.get("runtime") is True:
        return "ready", health
    return "not_ready", health


def _readiness_timeout_code(kind: str) -> str:
    if kind == "ssh":
        return "SSH_HEALTH_UNREACHABLE"
    if kind == "connect":
        return "HEALTH_CONNECT_FAILED"
    if kind == "not_ready":
        return "WORKER_NOT_READY"
    return "WORKER_READY_TIMEOUT"


def classify_connect_failure(exc: BaseException) -> str:
    """Separate a dropped handshake from a completed refusal. Neither is an HTTP success."""
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, TimeoutError) or isinstance(exc, TimeoutError):
        return "timeout"
    text = str(reason).lower()
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if isinstance(reason, ConnectionRefusedError) or "refused" in text:
        return "refused"
    return "unreachable"


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
        if not on_warm:
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
            window = warm_verify_window(hourly, operation_budget)
            if not window["fits"]:
                continue
            projected = window["work_cost"]
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


def inventory_matches(text: str, expected: dict) -> tuple[bool, str]:
    """Compare a read-only worker inventory with the offline cache validator."""
    payload = last_json_object(text)
    if not isinstance(payload, dict):
        return False, "CACHE_INCOMPLETE"
    from media_engine.engines.image.cache_validation import assess_inventory
    report = assess_inventory(payload, expected)
    return report["reason"] == "COMPLETE", report["reason"]


def _resolved_locally(text: str) -> bool:
    payload = last_json_object(text)
    if not isinstance(payload, dict):
        return False
    return payload.get("resolved") is True and payload.get("wrote") is False


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
mount = os.environ.get("MEDIA_ENGINE_MODEL_CACHE") or "/models"
manifest_path = mount.rstrip("/") + "/%s"
try:
    manifest = json.load(open(manifest_path))
except Exception:
    print(json.dumps({"state": "MISSING", "mount": os.path.ismount(mount), "entries": []}))
    raise SystemExit(0)
state = manifest.get("state")
entries = []
try:
    import model_cache
    state = model_cache.assess_cache(model_cache.Path(mount))
    revision = str(manifest.get("revision") or "")
    snapshot = model_cache.snapshot_dir(model_cache.Path(mount), revision)
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        relative = str(entry.get("path") or "")
        target = snapshot / relative
        item = {"path": relative, "exists": target.exists()}
        if target.exists():
            item["size"] = target.stat().st_size
            if target.is_symlink():
                item["link"] = os.readlink(target)
        entries.append(item)
except Exception:
    pass
print(json.dumps({
    "state": state,
    "repo": manifest.get("repo_id"),
    "revision": manifest.get("revision"),
    "files": manifest.get("expected_file_count"),
    "bytes": manifest.get("expected_total_bytes"),
    "mount": os.path.ismount(mount),
    "entries": entries,
}))
PY""" % MANIFEST_NAME

_CACHE_RESOLVE = r"""python3 - << 'PY'
import json, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = "/models/hf-cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/models/hf-cache"
sys.path.insert(0, "/workspace")
from pathlib import Path
before = Path("/models/phase2d-cache.json").stat().st_mtime_ns
from huggingface_hub import snapshot_download
import model_cache
local = model_cache.prepare_generation(Path("/models"), snapshot_download)
after = Path("/models/phase2d-cache.json").stat().st_mtime_ns
print(json.dumps({"resolved": True, "local": local, "wrote": before != after}))
PY"""

# The copy runs on the image python. It does not import torch.
# prepare_runtime selects the interpreter used for import, load, and inference.
_R2_SYNC = r"""
set -eu
mkdir -p /models /workspace
set -a
. /workspace/phase2d-r2.env
set +a
printf '%s\n' '{"cache_mount":"/models"}' > /workspace/phase2d_job.json
rm -f /workspace/cache-sync.exit /workspace/r2-sync-progress.json
nohup sh -c 'PHASE2D_CACHE_SYNC_SECONDS=__CACHE_SYNC_SECONDS__ python3 /workspace/qwen_remote.py cache-sync > /workspace/cache-sync.log 2>&1; echo $? > /workspace/cache-sync.exit' >/dev/null 2>&1 &
echo $! > /workspace/cache-sync.pid
echo STARTED
"""
_R2_SYNC = _R2_SYNC.replace(
    "__CACHE_SYNC_SECONDS__", str(int(PHASE_LIMITS["cache_staging"][0])),
)

_CACHE_PROGRESS = "cat /workspace/r2-sync-progress.json 2>/dev/null || true"
_CACHE_EXIT = "if [ -f /workspace/cache-sync.exit ]; then cat /workspace/cache-sync.exit; else echo RUNNING; fi"
_CACHE_KILL = "if [ -f /workspace/cache-sync.pid ]; then kill $(cat /workspace/cache-sync.pid) >/dev/null 2>&1 || true; fi"

def _session_boot_command(python: str) -> str:
    quoted = shlex.quote(python)
    return (
        "mkdir -p /workspace/session-inbox /workspace/session-out && "
        f"if [ ! -f /workspace/qwen_remote.py ]; then echo MISSING_REMOTE; exit 3; fi && "
        "if [ ! -f /workspace/session.ready ]; then "
        f"nohup {quoted} -u /workspace/qwen_remote.py serve >> /workspace/session.log 2>&1 & "
        "echo $! > /workspace/session.pid; fi && "
        "ready=0; i=0; while [ \"$i\" -lt 600 ]; do "
        "if [ -f /workspace/session.ready ]; then ready=1; break; fi; "
        "i=$((i+1)); sleep 2; done; "
        "if [ \"$ready\" -ne 1 ]; then echo SESSION_NOT_READY; exit 4; fi; "
        "echo SESSION_READY"
    )


def _session_job_command(image_id: str, prompt: str, width: int, height: int, seed: int) -> str:
    if not image_id.startswith("s") or not image_id[1:].isdigit():
        raise WorkerStageError("GENERATION_FAILED")
    job = json.dumps({
        "id": image_id,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": INFERENCE_STEPS,
        "cache_mount": MOUNT_PATH,
    }, separators=(",", ":"))
    quoted = shlex.quote(job)
    return (
        f"printf %s {quoted} > /workspace/session-inbox/{image_id}.json && "
        "ready=0; i=0; while [ \"$i\" -lt 300 ]; do "
        f"if [ -f /workspace/session-out/{image_id}.png ]; then ready=1; break; fi; "
        "i=$((i+1)); sleep 2; done; "
        "if [ \"$ready\" -ne 1 ]; then echo SESSION_IMAGE_MISSING; exit 5; fi; "
        "echo SESSION_IMAGE"
    )


def _generate_command(prompt: str, width: int, height: int, seed: int, python: str) -> str:
    if not python or python != python.strip() or any(char in python for char in "\n\r\x00"):
        raise WorkerStageError("RUNTIME_PREPARE_FAILED")
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
        f"{shlex.quote(python)} -u /workspace/qwen_remote.py generate && "
        f"{shlex.quote(python)} -c 'import struct; d=open(\"/workspace/phase2d_output.png\",\"rb\").read(); "
        "assert d[:8]==b\"\\x89PNG\\r\\n\\x1a\\n\" and len(d)>0; "
        "w,h=struct.unpack(\">II\", d[16:24]); assert (w,h)==(1024,1024); print(\"PNG_READY\", len(d))'"
    )
