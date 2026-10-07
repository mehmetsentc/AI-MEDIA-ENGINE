"""Phase clocks for a rented image worker.

A long cache copy that is still moving bytes stays alive. A phase that
stops reporting progress does not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


# (maximum seconds, seconds without progress)
PHASE_LIMITS: dict[str, tuple[float, float]] = {
    "provisioning": (180.0, 120.0),
    "connecting": (420.0, 240.0),
    "cache_staging": (2400.0, 180.0),
    "runtime_preparing": (900.0, 300.0),
    "model_loading": (600.0, 300.0),
    "generating": (600.0, 240.0),
    "saving": (180.0, 120.0),
}


@dataclass
class PhaseWatch:
    phase: str
    started: float
    last_progress: float
    bytes_transferred: int
    max_seconds: float
    stall_seconds: float

    @classmethod
    def start(cls, phase: str, now: float) -> "PhaseWatch":
        maximum, stall = PHASE_LIMITS[phase]
        return cls(phase, now, now, 0, maximum, stall)

    def mark(self, now: float, transferred: int | None = None) -> None:
        """Record a new signal. The same byte count is not progress."""
        if transferred is not None:
            if transferred <= self.bytes_transferred:
                return
            self.bytes_transferred = transferred
        self.last_progress = now

    def expired(self, now: float) -> str | None:
        if now - self.started > self.max_seconds:
            return "PHASE_TIMEOUT"
        if now - self.last_progress > self.stall_seconds:
            return "PHASE_STALL"
        return None

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started)


def note_transfer_text(watch: PhaseWatch, text: str, now: float) -> bool:
    """Apply a remote progress record. Returns True when bytes increased."""
    moved = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        raw = payload.get("bytes_done", payload.get("bytes"))
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        before = watch.bytes_transferred
        watch.mark(now, int(raw))
        moved = moved or watch.bytes_transferred > before
    return moved
