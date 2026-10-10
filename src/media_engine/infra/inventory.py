"""Read-only offer ranking and machine-scoped verification. No create calls."""
from __future__ import annotations

from decimal import Decimal

from media_engine.providers.gpu_discovery import assess_offers, rank_candidates
from media_engine.safety.execution_budget import LIVE7_CEILING_USD, LIVE7_MAX_HOURLY_USD


class PollSchedule:
    """Caps discovery calls. Failures back off. A quiet process does not poll."""

    def __init__(self, *, interval_seconds: float = 60.0, maximum_seconds: float = 300.0,
                 calls_per_minute: int = 6) -> None:
        self.interval_seconds = float(interval_seconds)
        self.maximum_seconds = float(maximum_seconds)
        self.calls_per_minute = int(calls_per_minute)
        self.backoff_seconds = self.interval_seconds
        self._calls: list[float] = []

    def allow(self, now: float) -> bool:
        self._calls = [item for item in self._calls if now - item < 60.0]
        return len(self._calls) < self.calls_per_minute

    def record(self, now: float, *, ok: bool) -> None:
        self._calls.append(now)
        if ok:
            self.backoff_seconds = self.interval_seconds
            return
        self.backoff_seconds = min(self.maximum_seconds, max(self.interval_seconds, self.backoff_seconds * 2))


def rank_market(rows: list, *, observed_at: float, ceiling: Decimal = LIVE7_CEILING_USD,
                max_hourly: Decimal = LIVE7_MAX_HOURLY_USD) -> list:
    assessed = assess_offers(rows, observed_at=observed_at, ceiling=ceiling, max_hourly=max_hourly)
    return rank_candidates(assessed)


def verify_exact_pair(rows: list, machine_id: int, offer_id: int, *, observed_at: float,
                      ceiling: Decimal = LIVE7_CEILING_USD,
                      max_hourly: Decimal = LIVE7_MAX_HOURLY_USD):
    """The offer must still be on that machine. A search hit is not enough."""
    matched = [
        row for row in rows
        if isinstance(row, dict)
        and int(row.get("machine_id") or 0) == int(machine_id)
        and int(row.get("id") or 0) == int(offer_id)
    ]
    if not matched:
        return None, "OFFER_UNAVAILABLE"
    assessed = assess_offers(matched, observed_at=observed_at, ceiling=ceiling, max_hourly=max_hourly)
    candidate = assessed[0]
    if candidate.machine_id != int(machine_id) or candidate.offer_id != int(offer_id):
        return candidate, "OFFER_REJECTED"
    if not candidate.eligible:
        return candidate, candidate.rejection or "OFFER_UNAVAILABLE"
    return candidate, ""
