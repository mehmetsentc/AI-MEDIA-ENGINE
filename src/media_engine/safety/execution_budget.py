"""One LIVE-6 spending deadline shared by the planner, studio, provider, and watchdog.

Authorization expiry is not this deadline. The clock starts at the create
request and is not moved forward when later stages are observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from media_engine.providers.base import WorkerStageError

LIVE6_CEILING_USD = Decimal("0.50")
LIVE6_RESERVE_USD = Decimal("0.10")
LIVE6_PLANNING_USD = Decimal("0.40")
LIVE6_MAX_HOURLY_USD = Decimal("0.375")
BILLING_START_EVENT = "create_request"


def billable_seconds(hourly: Decimal) -> Optional[int]:
    """Whole seconds that stay inside the $0.40 planning allocation.

    The division is floored. A price above $0.375 or a second that would
    cross $0.40 returns no duration, which the caller treats as refusal.
    """
    amount = Decimal(hourly)
    if amount <= 0 or amount > LIVE6_MAX_HOURLY_USD:
        return None
    if LIVE6_PLANNING_USD + LIVE6_RESERVE_USD != LIVE6_CEILING_USD:
        return None
    seconds = int((LIVE6_PLANNING_USD / amount) * Decimal(3600))
    if seconds < 1:
        return None
    cost = (amount * Decimal(seconds)) / Decimal(3600)
    if cost > LIVE6_PLANNING_USD:
        return None
    return seconds


def live6_planning_window(hourly: Decimal, required_seconds: float) -> dict:
    """Planner view of the same billable duration and the same $0.10 reserve."""
    seconds = billable_seconds(hourly)
    work = (Decimal(hourly) * Decimal(str(required_seconds)) / Decimal(3600)).quantize(Decimal("0.000001"))
    fits = (
        seconds is not None
        and seconds >= required_seconds
        and work + LIVE6_RESERVE_USD <= LIVE6_CEILING_USD
    )
    return {
        "safe_seconds": seconds,
        "reserve_usd": LIVE6_RESERVE_USD,
        "required_seconds": required_seconds,
        "work_cost": work,
        "fits": fits,
        "ceiling": LIVE6_CEILING_USD,
        "network_usd": Decimal("0"),
        "planning_usd": LIVE6_PLANNING_USD,
    }


def conservative_billing_start(observed: list) -> float:
    """Earliest known start. A missing later stage is not a reason to wait."""
    values = []
    for item in observed:
        if item is None:
            continue
        values.append(float(item))
    if not values:
        raise WorkerStageError("BUDGET_MISMATCH")
    return min(values)


@dataclass(frozen=True)
class ExecutionBudget:
    execution_id: str
    ceiling_usd: Decimal
    reserve_usd: Decimal
    planning_usd: Decimal
    hourly_usd: Decimal
    allowed_seconds: int
    billing_started_wall: float
    billing_started_monotonic: float
    spending_deadline_wall: float
    spending_deadline_monotonic: float
    billing_start_event: str = BILLING_START_EVENT

    def as_mapping(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "ceiling_usd": str(self.ceiling_usd),
            "reserve_usd": str(self.reserve_usd),
            "planning_usd": str(self.planning_usd),
            "hourly_usd": str(self.hourly_usd),
            "allowed_seconds": self.allowed_seconds,
            "billing_started_wall": self.billing_started_wall,
            "billing_started_monotonic": self.billing_started_monotonic,
            "spending_deadline_wall": self.spending_deadline_wall,
            "spending_deadline_monotonic": self.spending_deadline_monotonic,
            "billing_start_event": self.billing_start_event,
            "deadline_epoch": self.spending_deadline_wall,
        }

    @staticmethod
    def from_mapping(payload: dict) -> "ExecutionBudget":
        return ExecutionBudget(
            execution_id=str(payload["execution_id"]),
            ceiling_usd=Decimal(str(payload["ceiling_usd"])),
            reserve_usd=Decimal(str(payload["reserve_usd"])),
            planning_usd=Decimal(str(payload["planning_usd"])),
            hourly_usd=Decimal(str(payload["hourly_usd"])),
            allowed_seconds=int(payload["allowed_seconds"]),
            billing_started_wall=float(payload["billing_started_wall"]),
            billing_started_monotonic=float(payload["billing_started_monotonic"]),
            spending_deadline_wall=float(payload["spending_deadline_wall"]),
            spending_deadline_monotonic=float(payload["spending_deadline_monotonic"]),
            billing_start_event=str(payload.get("billing_start_event") or BILLING_START_EVENT),
        )

    def stop(self, wall_now: float, monotonic_now: float) -> bool:
        """True once either clock reaches the original deadline."""
        if float(monotonic_now) - self.billing_started_monotonic >= self.allowed_seconds:
            return True
        return float(wall_now) >= self.spending_deadline_wall

    def planning_cost_usd(self) -> Decimal:
        return (self.hourly_usd * Decimal(self.allowed_seconds)) / Decimal(3600)


def open_budget(
    *,
    execution_id: str,
    grant_execution_id: str,
    grant_ceiling: Decimal,
    hourly: Decimal,
    wall_now: float,
    monotonic_now: float,
    observed_starts: Optional[list] = None,
) -> ExecutionBudget:
    """Freeze a deadline. The grant expiry is not an input."""
    if not execution_id or execution_id != grant_execution_id:
        raise WorkerStageError("STALE_AUTHORIZATION")
    if Decimal(grant_ceiling) != LIVE6_CEILING_USD:
        raise WorkerStageError("BUDGET_MISMATCH")
    seconds = billable_seconds(hourly)
    if seconds is None:
        raise WorkerStageError("BUDGET_MISMATCH")
    candidates = [wall_now]
    if observed_starts:
        candidates.extend(observed_starts)
    wall = conservative_billing_start(candidates)
    # The monotonic sample is paired with wall_now. An earlier wall start
    # shifts the monotonic origin by the same amount so both deadlines agree.
    shift = float(wall_now) - wall
    mono = float(monotonic_now) - shift
    return ExecutionBudget(
        execution_id=execution_id,
        ceiling_usd=LIVE6_CEILING_USD,
        reserve_usd=LIVE6_RESERVE_USD,
        planning_usd=LIVE6_PLANNING_USD,
        hourly_usd=Decimal(hourly),
        allowed_seconds=seconds,
        billing_started_wall=wall,
        billing_started_monotonic=mono,
        spending_deadline_wall=wall + seconds,
        spending_deadline_monotonic=mono + seconds,
    )


def retain_deadline(stored: dict, proposed_wall: float, proposed_execution_id: str) -> float:
    """A restart keeps the stored deadline. A later proposal cannot extend it."""
    if str(stored.get("execution_id") or "") != proposed_execution_id:
        raise WorkerStageError("STALE_AUTHORIZATION")
    original = float(stored["spending_deadline_wall"])
    if float(proposed_wall) > original:
        return original
    return original
