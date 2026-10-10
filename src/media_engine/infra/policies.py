"""Session budgets are configuration. None of them authorizes a rental."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_UP

PAID_ACTIONS_DISABLED = "PAID_ACTIONS_DISABLED"
HOURLY_CEILING_USD = Decimal("0.95")
DELETION_RESERVE_SECONDS = 840
IDLE_SHUTDOWN_SECONDS = 300


@dataclass(frozen=True)
class SessionPolicy:
    """A future lease length. `allows_paid_create` stays false in this build."""

    name: str
    hours: int
    hourly_ceiling_usd: Decimal = HOURLY_CEILING_USD
    idle_shutdown_seconds: int = IDLE_SHUTDOWN_SECONDS
    deletion_reserve_seconds: int = DELETION_RESERVE_SECONDS
    allows_paid_create: bool = False

    def session_seconds(self) -> int:
        return int(self.hours) * 3600

    def usable_seconds(self) -> int:
        return max(0, self.session_seconds() - self.deletion_reserve_seconds)

    def session_ceiling_usd(self, hourly: Decimal) -> Decimal:
        rate = min(Decimal(hourly), self.hourly_ceiling_usd)
        raw = rate * Decimal(self.session_seconds()) / Decimal(3600)
        return raw.quantize(Decimal("0.000001"), rounding=ROUND_UP)

    def accepts_hourly(self, hourly: Decimal) -> bool:
        amount = Decimal(hourly)
        return amount > 0 and amount <= self.hourly_ceiling_usd


def catalog() -> dict[str, SessionPolicy]:
    return {
        name: SessionPolicy(name=name, hours=hours)
        for name, hours in (("1h", 1), ("8h", 8), ("10h", 10), ("24h", 24))
    }


def budget_layers(*, hourly: Decimal = Decimal("0"), hours: int = 0, credits: int = 0) -> dict:
    """Control-plane polls are read-only. Session math is an estimate, not a lease."""
    policy = SessionPolicy(name="estimate", hours=hours) if hours else None
    session = Decimal("0") if policy is None or hourly <= 0 else policy.session_ceiling_usd(hourly)
    return {
        "control_plane_usd": "0",
        "session_usd": str(session),
        "user_credits": int(credits),
        "paid_create": False,
    }
