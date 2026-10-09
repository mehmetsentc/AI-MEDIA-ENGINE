"""Instance stop rules that do not contact a provider."""
from __future__ import annotations


def lock_action(pid: int, alive) -> str:
    """A dead watchdog lock may be replaced. A live lock stays."""
    if pid <= 0 or not alive(pid):
        return "replace"
    return "keep"


def instances_to_stop(
    rows,
    *,
    machine_id: int,
    now: float,
    deadline: float | None,
    heartbeat_age: float | None,
    stale_after: float = 300.0,
) -> list[str]:
    """Two owned instances are both stopped. One stops at the deadline or a stale heartbeat."""
    owned: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            machine = int(row.get("machine_id") or 0)
        except (TypeError, ValueError):
            continue
        if machine != int(machine_id):
            continue
        if not str(row.get("label") or "").startswith("media-engine:"):
            continue
        identity = str(row.get("id") or "")
        if identity:
            owned.append(identity)
    if len(owned) > 1:
        return owned
    if len(owned) != 1:
        return []
    if deadline is not None and now >= deadline:
        return owned
    if heartbeat_age is not None and heartbeat_age > stale_after:
        return owned
    return []


def termination_report(*, delete_status: int | None, still_listed: bool) -> dict:
    """Estimated cost is not provider billing. An unknown delete stays unknown."""
    if delete_status is None:
        state = "UNKNOWN"
    elif still_listed:
        state = "STILL_LISTED"
    elif delete_status in {200, 204, 404}:
        state = "ABSENT"
    else:
        state = "UNKNOWN"
    return {"provider_state": state, "confirmed_charge_usd": None, "delete_status": delete_status}
