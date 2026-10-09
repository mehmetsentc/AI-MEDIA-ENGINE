"""Rank rentable Qwen-Image GPUs. Discovery does not authorize a rental."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from media_engine.providers.base import WorkerStageError
from media_engine.providers.live7_contract import (
    LIVE7_CUDA_MIN,
    LIVE7_DISK_GB,
    LIVE7_MIN_VRAM_MIB,
    gpu_family_token,
)
from media_engine.providers.vast_image import _hourly, expected_transfer_seconds, supports_bf16
from media_engine.safety.execution_budget import (
    LIVE7_CEILING_USD,
    LIVE7_MAX_HOURLY_USD,
    LIVE7_PROVISION_SECONDS,
    LIVE7_RESERVE_SECONDS,
    LIVE7_TAIL_SECONDS,
    LIVE7_TRANSFER_FLOOR_SECONDS,
    authorized_billable_seconds,
)

SHORTLIST_LIMIT = 5


@dataclass(frozen=True)
class GpuCandidate:
    machine_id: int
    offer_id: int
    gpu_name: str
    gpu_family: str
    vram_mib: int
    cuda: str
    compute_cap: int
    disk_gb: float
    hourly_usd: Decimal
    inet_down: float
    reliability: float
    verified: bool
    rentable: bool
    observed_at: float
    transfer_seconds: float
    session_cost_usd: Decimal
    inference_room_seconds: int
    geolocation: str
    rejection: str = ""

    @property
    def eligible(self) -> bool:
        return self.rejection == ""


def assess_offers(
    rows: list,
    *,
    observed_at: float,
    max_hourly: Decimal = LIVE7_MAX_HOURLY_USD,
    ceiling: Decimal = LIVE7_CEILING_USD,
) -> list[GpuCandidate]:
    """Score every row. Unsuitable offers stay in the list with a rejection code."""
    assessed = []
    for row in rows:
        if isinstance(row, dict):
            assessed.append(assess_offer(
                row, observed_at=observed_at, max_hourly=max_hourly, ceiling=ceiling,
            ))
    return assessed


def assess_offer(
    row: dict,
    *,
    observed_at: float,
    max_hourly: Decimal = LIVE7_MAX_HOURLY_USD,
    ceiling: Decimal = LIVE7_CEILING_USD,
) -> GpuCandidate:
    machine_id, offer_id = _ids(row)
    name = str(row.get("gpu_name") or "")
    vram = _int(row.get("gpu_ram"))
    compute = _int(row.get("compute_cap"))
    disk = _float(row.get("disk_space"))
    inet = _float(row.get("inet_down"))
    reliability = _float(row.get("reliability2"))
    cuda = _decimal(row.get("cuda_max_good"))
    rentable = row.get("rentable") is True
    verified = str(row.get("verification") or "") == "verified"
    base = dict(
        machine_id=machine_id,
        offer_id=offer_id,
        gpu_name=name,
        gpu_family=gpu_family_token(name),
        vram_mib=vram,
        cuda="" if cuda is None else str(cuda),
        compute_cap=compute,
        disk_gb=disk,
        hourly_usd=Decimal("0"),
        inet_down=inet,
        reliability=reliability,
        verified=verified,
        rentable=rentable,
        observed_at=float(observed_at),
        transfer_seconds=0.0,
        session_cost_usd=Decimal("0"),
        inference_room_seconds=0,
        geolocation=str(row.get("geolocation") or ""),
    )
    def finish(**updates: object) -> GpuCandidate:
        values = dict(base)
        values.update(updates)
        return GpuCandidate(**values)

    reason = _reject_row(row, name=name, vram=vram, compute=compute, disk=disk, cuda=cuda, rentable=rentable)
    if reason:
        return finish(rejection=reason)
    try:
        hourly = _hourly(row, LIVE7_DISK_GB)
    except WorkerStageError:
        return finish(rejection="PRICE_CHANGED")
    if hourly <= 0 or hourly > Decimal(max_hourly) or hourly > LIVE7_MAX_HOURLY_USD:
        return finish(hourly_usd=hourly, rejection="PRICE_CHANGED")
    transfer = expected_transfer_seconds(inet)
    # A missing or non-positive advertisement is not treated as a fast link.
    if transfer is None or transfer <= 0 or transfer > LIVE7_TRANSFER_FLOOR_SECONDS:
        return finish(hourly_usd=hourly, rejection="TRANSFER_RISK")
    if reliability <= 0:
        return finish(hourly_usd=hourly, transfer_seconds=float(transfer), rejection="RELIABILITY_UNKNOWN")
    allowed = authorized_billable_seconds(hourly, ceiling)
    if allowed is None:
        return finish(hourly_usd=hourly, transfer_seconds=float(transfer), rejection="SESSION_TOO_SHORT")
    room = int(allowed) - LIVE7_PROVISION_SECONDS - int(transfer)
    if room < LIVE7_TAIL_SECONDS:
        return finish(
            hourly_usd=hourly, transfer_seconds=float(transfer),
            inference_room_seconds=max(room, 0), rejection="SESSION_TOO_SHORT",
        )
    session = _session_cost(hourly, transfer)
    if session > Decimal(ceiling):
        return finish(
            hourly_usd=hourly, transfer_seconds=float(transfer),
            session_cost_usd=session, inference_room_seconds=room, rejection="SESSION_TOO_SHORT",
        )
    return finish(
        hourly_usd=hourly,
        transfer_seconds=float(transfer),
        session_cost_usd=session,
        inference_room_seconds=room,
        rejection="",
    )


def rank_candidates(candidates: list[GpuCandidate]) -> list[GpuCandidate]:
    """Eligible hosts only, best completion chance first, then cheaper sessions."""
    eligible = [item for item in candidates if item.eligible]
    eligible.sort(key=_rank_key)
    return eligible


def shortlist(candidates: list[GpuCandidate], *, limit: int = SHORTLIST_LIMIT) -> list[GpuCandidate]:
    return rank_candidates(candidates)[:limit]


def rejected(candidates: list[GpuCandidate]) -> list[GpuCandidate]:
    return [item for item in candidates if not item.eligible]


def _rank_key(candidate: GpuCandidate) -> tuple:
    return (
        0 if candidate.verified else 1,
        candidate.transfer_seconds,
        -candidate.reliability,
        candidate.session_cost_usd,
        -candidate.inference_room_seconds,
        candidate.offer_id,
    )


def _session_cost(hourly: Decimal, transfer_seconds: float) -> Decimal:
    seconds = (
        Decimal(LIVE7_PROVISION_SECONDS)
        + Decimal(str(transfer_seconds))
        + Decimal(LIVE7_TAIL_SECONDS)
        + Decimal(LIVE7_RESERVE_SECONDS)
    )
    return (hourly * seconds / Decimal(3600)).quantize(Decimal("0.000001"))


def _reject_row(
    row: dict,
    *,
    name: str,
    vram: int,
    compute: int,
    disk: float,
    cuda: Decimal | None,
    rentable: bool,
) -> str:
    if not rentable:
        return "NOT_RENTABLE"
    if _int(row.get("num_gpus")) != 1:
        return "GPU_MISMATCH"
    if _advertised_memory_unusable(name):
        return "VRAM_UNUSABLE"
    if not supports_bf16(name):
        return "GPU_MISMATCH"
    if vram < LIVE7_MIN_VRAM_MIB:
        return "VRAM_MISMATCH"
    if cuda is None or cuda < LIVE7_CUDA_MIN:
        return "CUDA_MISMATCH"
    if compute < 800:
        return "COMPUTE_MISMATCH"
    if disk < LIVE7_DISK_GB:
        return "DISK_INSUFFICIENT"
    return ""


def _advertised_memory_unusable(name: str) -> bool:
    """A 24 GB consumer card is not usable just because the listing says 48 GB."""
    compact = str(name).casefold().replace(" ", "").replace("_", "").replace("-", "")
    return any(token in compact for token in ("309" + "0", "409" + "0", "509" + "0"))


def _ids(row: dict) -> tuple[int, int]:
    return _int(row.get("machine_id")), _int(row.get("id"))


def _int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _decimal(value: object) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
