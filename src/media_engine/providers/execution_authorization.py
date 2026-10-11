"""Immutable single-use permission to rent one GPU. A search result is not permission."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal

from media_engine.engines.image.qwen import MODEL_ID
from media_engine.providers.base import WorkerStageError
from media_engine.providers.gpu_discovery import GpuCandidate, rank_candidates
from media_engine.providers.live7_contract import (
    FORBIDDEN_VOLUME_ID,
    LIVE7_CUDA_MIN,
    LIVE7_DISK_GB,
    REVISION,
)
from media_engine.safety.execution_budget import (
    LIVE7_CEILING_USD,
    LIVE7_MAX_HOURLY_USD,
    LIVE8_JIT_CEILING_USD,
    authorized_billable_seconds,
    live7_transfer_room,
)
from media_engine.safety.spend_grant import SpendGrant

_ID = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_POLICIES = frozenset({"exact_offers", "approved_pool", "just_in_time"})
_TOKEN = re.compile(r"^[a-z0-9]{1,32}$")
_FIELDS = (
    "authorization_id",
    "execution_id",
    "code_sha",
    "allowed_machine_ids",
    "allowed_offer_ids",
    "selection_policy",
    "max_hourly_usd",
    "max_total_usd",
    "max_instances",
    "max_inference_attempts",
    "max_image_count",
    "max_disk_gb",
    "model",
    "revision",
    "issued_at",
    "expires_at",
)
_OPTIONAL = (
    "allowed_pairs",
    "selection_constraints",
    "max_creates",
    "max_gpu_instances",
    "max_model_stages",
    "max_model_loads",
)


def _optional_one(kwargs: dict, name: str) -> int:
    """Missing means one. Any other value is over the authorization."""
    if name not in kwargs:
        return 1
    try:
        value = int(kwargs[name])
    except (TypeError, ValueError):
        raise WorkerStageError("INSTANCE_LIMIT") from None
    if value != 1:
        raise WorkerStageError("INSTANCE_LIMIT")
    return value


@dataclass(frozen=True)
class ExecutionAuthorization:
    authorization_id: str
    execution_id: str
    code_sha: str
    allowed_machine_ids: tuple[int, ...]
    allowed_offer_ids: tuple[int, ...]
    selection_policy: str
    max_hourly_usd: Decimal
    max_total_usd: Decimal
    max_instances: int
    max_inference_attempts: int
    max_image_count: int
    max_disk_gb: int
    model: str
    revision: str
    issued_at: float
    expires_at: float
    allowed_pairs: tuple[tuple[int, int], ...] = ()
    gpu_families: tuple[str, ...] = ()
    max_creates: int = 1
    max_gpu_instances: int = 1
    max_model_stages: int = 1
    max_model_loads: int = 1

    def to_dict(self) -> dict:
        return {
            "authorization_id": self.authorization_id,
            "execution_id": self.execution_id,
            "code_sha": self.code_sha,
            "allowed_machine_ids": list(self.allowed_machine_ids),
            "allowed_offer_ids": list(self.allowed_offer_ids),
            "allowed_pairs": [[machine, offer] for machine, offer in self.allowed_pairs],
            "selection_policy": self.selection_policy,
            "selection_constraints": {"gpu_families": list(self.gpu_families)},
            "max_hourly_usd": str(self.max_hourly_usd),
            "max_total_usd": str(self.max_total_usd),
            "max_instances": self.max_instances,
            "max_creates": self.max_creates,
            "max_gpu_instances": self.max_gpu_instances,
            "max_inference_attempts": self.max_inference_attempts,
            "max_image_count": self.max_image_count,
            "max_model_stages": self.max_model_stages,
            "max_model_loads": self.max_model_loads,
            "max_disk_gb": self.max_disk_gb,
            "model": self.model,
            "revision": self.revision,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


def issue_authorization(**kwargs: object) -> ExecutionAuthorization:
    """Validate and freeze one authorization. Later edits are not a new issuance."""
    unknown = set(kwargs) - set(_FIELDS) - set(_OPTIONAL)
    if unknown:
        raise WorkerStageError("MISSING_AUTHORIZATION")
    if any(name not in kwargs for name in _FIELDS):
        raise WorkerStageError("MISSING_AUTHORIZATION")
    policy = str(kwargs["selection_policy"])
    if policy not in _POLICIES:
        raise WorkerStageError("SELECTION_POLICY")
    pairs = _pairs(kwargs.get("allowed_pairs"))
    families = _gpu_families(kwargs.get("selection_constraints"))
    if policy != "just_in_time" and (pairs or families):
        raise WorkerStageError("SELECTION_POLICY")
    authorization_id = str(kwargs["authorization_id"])
    execution_id = str(kwargs["execution_id"])
    code_sha = str(kwargs["code_sha"])
    if _ID.fullmatch(authorization_id) is None or _ID.fullmatch(execution_id) is None:
        raise WorkerStageError("MISSING_AUTHORIZATION")
    if _SHA.fullmatch(code_sha) is None:
        raise WorkerStageError("STALE_AUTHORIZATION")
    machines = _positive_ids(kwargs["allowed_machine_ids"])
    offers = _positive_ids(kwargs["allowed_offer_ids"])
    if not machines or not offers:
        raise WorkerStageError("MACHINE_SCOPE")
    if policy == "just_in_time":
        if not pairs:
            raise WorkerStageError("MACHINE_SCOPE")
        pair_machines = {machine for machine, _offer in pairs}
        pair_offers = {offer for _machine, offer in pairs}
        if pair_machines != set(machines) or pair_offers != set(offers):
            raise WorkerStageError("MACHINE_SCOPE")
    try:
        hourly = Decimal(str(kwargs["max_hourly_usd"]))
        total = Decimal(str(kwargs["max_total_usd"]))
        issued = float(kwargs["issued_at"])  # type: ignore[arg-type]
        expires = float(kwargs["expires_at"])  # type: ignore[arg-type]
        instances = int(kwargs["max_instances"])  # type: ignore[arg-type]
        inferences = int(kwargs["max_inference_attempts"])  # type: ignore[arg-type]
        images = int(kwargs["max_image_count"])  # type: ignore[arg-type]
        disk = int(kwargs["max_disk_gb"])  # type: ignore[arg-type]
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise WorkerStageError("BUDGET_MISMATCH") from exc
    if hourly <= 0 or hourly > LIVE7_MAX_HOURLY_USD:
        raise WorkerStageError("PRICE_CHANGED")
    if total == LIVE8_JIT_CEILING_USD:
        if policy != "just_in_time":
            raise WorkerStageError("BUDGET_MISMATCH")
    elif total <= 0 or total > LIVE7_CEILING_USD:
        raise WorkerStageError("BUDGET_MISMATCH")
    if instances != 1:
        raise WorkerStageError("INSTANCE_LIMIT")
    if images not in (1, 2, 3) or inferences != images:
        raise WorkerStageError("INSTANCE_LIMIT")
    creates = _optional_one(kwargs, "max_creates")
    gpu_instances = _optional_one(kwargs, "max_gpu_instances")
    stages = _optional_one(kwargs, "max_model_stages")
    loads = _optional_one(kwargs, "max_model_loads")
    if disk != LIVE7_DISK_GB:
        raise WorkerStageError("DISK_INSUFFICIENT")
    if str(kwargs["model"]) != MODEL_ID or str(kwargs["revision"]) != REVISION:
        raise WorkerStageError("MODEL_SUBSTITUTION")
    if expires <= issued:
        raise WorkerStageError("EXPIRED_AUTHORIZATION")
    return ExecutionAuthorization(
        authorization_id=authorization_id,
        execution_id=execution_id,
        code_sha=code_sha,
        allowed_machine_ids=machines,
        allowed_offer_ids=offers,
        selection_policy=policy,
        max_hourly_usd=hourly,
        max_total_usd=total,
        max_instances=instances,
        max_inference_attempts=inferences,
        max_image_count=images,
        max_disk_gb=disk,
        model=MODEL_ID,
        revision=REVISION,
        issued_at=issued,
        expires_at=expires,
        allowed_pairs=pairs,
        gpu_families=families,
        max_creates=creates,
        max_gpu_instances=gpu_instances,
        max_model_stages=stages,
        max_model_loads=loads,
    )


def authorization_from_json(text: str) -> ExecutionAuthorization:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkerStageError("MISSING_AUTHORIZATION") from exc
    if not isinstance(payload, dict):
        raise WorkerStageError("MISSING_AUTHORIZATION")
    return issue_authorization(**payload)


def accept_record(authorization: ExecutionAuthorization, *, now: float, code_sha: str) -> None:
    """Refuse a record that does not match the code that is about to run."""
    if not isinstance(authorization, ExecutionAuthorization):
        raise WorkerStageError("MISSING_AUTHORIZATION")
    if code_sha != authorization.code_sha:
        raise WorkerStageError("STALE_AUTHORIZATION")
    if now >= authorization.expires_at or authorization.expires_at <= authorization.issued_at:
        raise WorkerStageError("EXPIRED_AUTHORIZATION")
    if authorization.max_instances != 1 or not _total_allowed(authorization):
        raise WorkerStageError("BUDGET_MISMATCH")
    if authorization.max_image_count not in (1, 2, 3):
        raise WorkerStageError("INSTANCE_LIMIT")
    if authorization.max_inference_attempts != authorization.max_image_count:
        raise WorkerStageError("INSTANCE_LIMIT")
    if (
        authorization.max_creates != 1
        or authorization.max_gpu_instances != 1
        or authorization.max_model_stages != 1
        or authorization.max_model_loads != 1
    ):
        raise WorkerStageError("INSTANCE_LIMIT")


def select_for_authorization(
    authorization: ExecutionAuthorization,
    candidates: list[GpuCandidate],
    *,
    now: float,
) -> tuple[GpuCandidate | None, str]:
    """Pick one covered offer. An offer outside the record is never a fallback."""
    ranked, reason = candidates_for_authorization(authorization, candidates, now=now)
    if not ranked:
        return None, reason
    return ranked[0], ""


def candidates_for_authorization(
    authorization: ExecutionAuthorization,
    candidates: list[GpuCandidate],
    *,
    now: float,
) -> tuple[list[GpuCandidate], str]:
    """Rank covered offers. Just-in-time keeps every pair; older policies keep the first."""
    if now >= authorization.expires_at:
        return [], "EXPIRED_AUTHORIZATION"
    allowed_offers = set(authorization.allowed_offer_ids)
    allowed_machines = set(authorization.allowed_machine_ids)
    allowed_pairs = set(authorization.allowed_pairs)
    eligible: list[GpuCandidate] = []
    seen: list[str] = []
    for candidate in candidates:
        if authorization.selection_policy == "just_in_time":
            pair = (candidate.machine_id, candidate.offer_id)
            if pair not in allowed_pairs:
                if candidate.machine_id in allowed_machines or candidate.offer_id in allowed_offers:
                    seen.append("OFFER_REJECTED")
                continue
        else:
            offered = candidate.offer_id in allowed_offers
            hosted = candidate.machine_id in allowed_machines
            if not offered and not hosted:
                continue
            if not offered:
                seen.append("OFFER_REJECTED")
                continue
            if not hosted:
                seen.append("MACHINE_SCOPE")
                continue
        if candidate.rejection:
            seen.append(candidate.rejection)
            continue
        if authorization.gpu_families and candidate.gpu_family not in authorization.gpu_families:
            seen.append("GPU_MISMATCH")
            continue
        if candidate.hourly_usd > authorization.max_hourly_usd:
            seen.append("PRICE_CHANGED")
            continue
        if candidate.session_cost_usd > authorization.max_total_usd:
            seen.append("SESSION_TOO_SHORT")
            continue
        eligible.append(candidate)
    if not eligible:
        if "PRICE_CHANGED" in seen:
            return [], "PRICE_CHANGED"
        if "NOT_RENTABLE" in seen:
            return [], "NOT_RENTABLE"
        if seen:
            return [], seen[0]
        return [], "OFFER_UNAVAILABLE"
    ranked = rank_candidates(eligible)
    if authorization.selection_policy == "just_in_time":
        return ranked, ""
    return [ranked[0]], ""


def make_live8_grant(authorization: ExecutionAuthorization, candidate: GpuCandidate) -> SpendGrant:
    """Bind the grant to the offer selected for this authorization, not to a source constant."""
    if authorization.selection_policy == "just_in_time":
        pair = (candidate.machine_id, candidate.offer_id)
        if pair not in authorization.allowed_pairs:
            raise WorkerStageError("OFFER_REJECTED")
    if candidate.offer_id not in authorization.allowed_offer_ids:
        raise WorkerStageError("OFFER_REJECTED")
    if candidate.machine_id not in authorization.allowed_machine_ids:
        raise WorkerStageError("MACHINE_SCOPE")
    if authorization.gpu_families and candidate.gpu_family not in authorization.gpu_families:
        raise WorkerStageError("GPU_MISMATCH")
    seconds = authorized_billable_seconds(candidate.hourly_usd, authorization.max_total_usd)
    if seconds is None:
        raise WorkerStageError("BUDGET_MISMATCH")
    return SpendGrant(
        execution_id=authorization.execution_id,
        ceiling_usd=authorization.max_total_usd,
        max_instances=1,
        issued_at=authorization.issued_at,
        expires_at=authorization.expires_at,
        machine_id=candidate.machine_id,
        offer_id=candidate.offer_id,
        profile="live8",
        gpu_family=candidate.gpu_family,
        min_vram_mib=candidate.vram_mib,
        max_hourly_usd=authorization.max_hourly_usd,
        disk_gb=LIVE7_DISK_GB,
        cold_stage=True,
        max_transfer_seconds=live7_transfer_room(seconds),
        max_execution_seconds=seconds,
        cuda_min=str(LIVE7_CUDA_MIN),
        forbidden_volume_id=FORBIDDEN_VOLUME_ID,
    )


def _total_allowed(authorization: ExecutionAuthorization) -> bool:
    total = authorization.max_total_usd
    if total == LIVE8_JIT_CEILING_USD:
        return authorization.selection_policy == "just_in_time"
    return total > 0 and total <= LIVE7_CEILING_USD


def _pairs(value: object) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise WorkerStageError("MACHINE_SCOPE")
    found = []
    for item in value:
        if isinstance(item, (str, bytes)) or not isinstance(item, (list, tuple)) or len(item) != 2:
            raise WorkerStageError("MACHINE_SCOPE")
        try:
            machine = int(item[0])
            offer = int(item[1])
        except (TypeError, ValueError) as exc:
            raise WorkerStageError("MACHINE_SCOPE") from exc
        if machine <= 0 or offer <= 0:
            raise WorkerStageError("MACHINE_SCOPE")
        found.append((machine, offer))
    return tuple(dict.fromkeys(found))


def _gpu_families(value: object) -> tuple[str, ...]:
    """Narrow a just-in-time set. Price and GPU class alone never add a machine."""
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise WorkerStageError("SELECTION_POLICY")
    unknown = set(value) - {"gpu_families"}
    if unknown:
        raise WorkerStageError("SELECTION_POLICY")
    raw = value.get("gpu_families", [])
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise WorkerStageError("SELECTION_POLICY")
    found = []
    for item in raw:
        token = str(item).strip().casefold()
        if _TOKEN.fullmatch(token) is None:
            raise WorkerStageError("SELECTION_POLICY")
        found.append(token)
    return tuple(dict.fromkeys(found))


def _positive_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise WorkerStageError("MACHINE_SCOPE")
    found = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise WorkerStageError("MACHINE_SCOPE") from exc
        if number <= 0:
            raise WorkerStageError("MACHINE_SCOPE")
        found.append(number)
    return tuple(dict.fromkeys(found))
