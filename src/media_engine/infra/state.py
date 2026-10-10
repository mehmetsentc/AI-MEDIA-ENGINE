"""Durable GPU lifecycle. An ambiguous create is never permission to create again."""
from __future__ import annotations

STOPPED = "STOPPED"
DISCOVERING = "DISCOVERING"
PROVISIONING = "PROVISIONING"
STAGING = "STAGING"
LOADING = "LOADING"
READY = "READY"
BUSY = "BUSY"
IDLE = "IDLE"
STOPPING = "STOPPING"
FAILED = "FAILED"
RECONCILING = "RECONCILING"
AMBIGUOUS = "AMBIGUOUS"

STATES = frozenset({
    STOPPED, DISCOVERING, PROVISIONING, STAGING, LOADING, READY, BUSY, IDLE,
    STOPPING, FAILED, RECONCILING, AMBIGUOUS,
})

_TRANSITIONS = {
    STOPPED: {DISCOVERING, RECONCILING},
    DISCOVERING: {PROVISIONING, STOPPED, FAILED},
    PROVISIONING: {STAGING, AMBIGUOUS, FAILED, STOPPING},
    AMBIGUOUS: {RECONCILING},
    RECONCILING: {STAGING, READY, IDLE, STOPPED, FAILED, AMBIGUOUS},
    STAGING: {LOADING, FAILED, STOPPING},
    LOADING: {READY, FAILED, STOPPING},
    READY: {BUSY, IDLE, STOPPING, FAILED},
    BUSY: {READY, IDLE, FAILED, STOPPING},
    IDLE: {BUSY, READY, STOPPING, FAILED},
    STOPPING: {STOPPED, FAILED},
    FAILED: {RECONCILING, STOPPING, STOPPED},
}


class IllegalGpuTransition(Exception):
    def __init__(self, current: str, proposed: str) -> None:
        super().__init__(f"{current}->{proposed}")
        self.code = "ILLEGAL_GPU_TRANSITION"
        self.current = current
        self.proposed = proposed


def transition(current: str, proposed: str) -> str:
    if proposed not in STATES or proposed not in _TRANSITIONS.get(current, ()):
        raise IllegalGpuTransition(current, proposed)
    return proposed
