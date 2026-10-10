"""GPU manager. Discovery is read-only. Paid create stays behind a closed gate."""
from __future__ import annotations

from decimal import Decimal

from media_engine.infra.inventory import PollSchedule, rank_market, verify_exact_pair
from media_engine.infra.policies import PAID_ACTIONS_DISABLED, SessionPolicy, catalog
from media_engine.infra.state import (
    AMBIGUOUS,
    BUSY,
    DISCOVERING,
    FAILED,
    IDLE,
    LOADING,
    PROVISIONING,
    READY,
    RECONCILING,
    STAGING,
    STOPPED,
    STOPPING,
    transition,
)
from media_engine.infra.store import ManagerStore
from media_engine.providers.vast_watchdog import instances_to_stop


class GpuManager:
    def __init__(self, store: ManagerStore, *, now, policies: dict[str, SessionPolicy] | None = None,
                 poll: PollSchedule | None = None) -> None:
        self.store = store
        self.now = now
        self.policies = policies or catalog()
        self.poll = poll or PollSchedule()
        self.alerts: list[dict] = []
        self.candidates: list = []
        self.provider_creates = 0
        saved = store.load()
        self.state = STOPPED if saved is None else saved["state"]
        self.instance_id = "" if saved is None else saved["instance_id"]
        self.machine_id = 0 if saved is None else saved["machine_id"]
        self.offer_id = 0 if saved is None else saved["offer_id"]
        self.create_attempted = False if saved is None else saved["create_attempted"]
        self.detail = {} if saved is None else saved["detail"]
        self.hourly_usd = Decimal(str(self.detail.get("hourly_usd") or "0"))

    def pull(self, source, *, ceiling: Decimal) -> list:
        """Read-only discovery plus instance reconciliation. This never creates."""
        current = float(self.now())
        if current < float(getattr(self, "next_pull_at", 0) or 0):
            self.alerts.append({"code": "POLL_BACKOFF", "at": current})
            return self.candidates
        if not self.poll.allow(current):
            self.alerts.append({"code": "POLL_LIMIT", "at": current})
            return self.candidates
        try:
            rows = source.search()
        except Exception as exc:
            self.poll.record(current, ok=False)
            self.next_pull_at = current + self.poll.backoff_seconds
            self.alerts.append({"code": "DISCOVERY_FAILED", "error": type(exc).__name__})
            return self.candidates
        ranked = self.discover(rows, ceiling=ceiling)
        self.next_pull_at = current + self.poll.interval_seconds
        try:
            listed = source.instances()
        except Exception as exc:
            self.alerts.append({"code": "RECONCILE_FAILED", "error": type(exc).__name__})
            return ranked
        self.reconcile(listed)
        return ranked

    def discover(self, rows: list, *, ceiling: Decimal) -> list:
        current = float(self.now())
        if not self.poll.allow(current):
            self.alerts.append({"code": "POLL_LIMIT", "at": current})
            return self.candidates
        try:
            if self.state == STOPPED:
                self._move(DISCOVERING)
            ranked = rank_market(rows, observed_at=current, ceiling=ceiling)
            self.candidates = ranked
            self.poll.record(current, ok=True)
            if self.state == DISCOVERING:
                self._move(STOPPED)
        except Exception as exc:
            self.poll.record(current, ok=False)
            self.alerts.append({"code": "DISCOVERY_FAILED", "error": type(exc).__name__})
            if self.state == DISCOVERING:
                self._move(FAILED)
        return self.candidates

    def verify_pair(self, rows: list, machine_id: int, offer_id: int, *, ceiling: Decimal):
        return verify_exact_pair(
            rows, machine_id, offer_id, observed_at=float(self.now()), ceiling=ceiling,
        )

    def evaluate_capacity(self, policy_name: str, authorization: dict | None) -> dict:
        """Decide whether a future create would be legal. This method never creates one."""
        if self.store.authorizes_rental():
            return {"allowed": False, "reason": PAID_ACTIONS_DISABLED, "creates": 0}
        if self.state == AMBIGUOUS or self.create_attempted:
            return {"allowed": False, "reason": "AMBIGUOUS_CREATE", "creates": 0}
        policy = self.policies.get(policy_name)
        if policy is None or not policy.allows_paid_create:
            return {"allowed": False, "reason": PAID_ACTIONS_DISABLED, "creates": 0}
        if not authorization or not authorization.get("execution_id"):
            return {"allowed": False, "reason": "MISSING_AUTHORIZATION", "creates": 0}
        if float(self.now()) >= float(authorization.get("expires_at") or 0):
            return {"allowed": False, "reason": "EXPIRED_AUTHORIZATION", "creates": 0}
        return {"allowed": False, "reason": "CREATE_TRANSPORT_DISABLED", "creates": 0}

    def note_ambiguous_create(self) -> None:
        """Record one uncertain create. A later call cannot send another."""
        if self.create_attempted:
            raise RuntimeError("AMBIGUOUS_CREATE")
        if self.state == STOPPED:
            self._move(DISCOVERING)
            self._move(PROVISIONING)
        elif self.state == DISCOVERING:
            self._move(PROVISIONING)
        self.create_attempted = True
        self.provider_creates += 1
        self._move(AMBIGUOUS)
        self.detail["create_outcome"] = "ambiguous"

    def reconcile(self, instances: list[dict]) -> str:
        owned = [
            row for row in instances
            if str(row.get("label") or "").startswith("media-engine:")
        ]
        if self.state == AMBIGUOUS:
            self._move(RECONCILING)
            if len(owned) == 1:
                self._adopt(owned[0], READY)
                return self.state
            if len(owned) == 0:
                self._move(AMBIGUOUS)
                self.alerts.append({"code": "UNRESOLVED_BILLING"})
                return self.state
            self._move(FAILED)
            self.alerts.append({"code": "INSTANCE_LIMIT"})
            return self.state
        if self.state in {STOPPED, FAILED}:
            self._move(RECONCILING)
        if len(owned) == 1:
            self._adopt(owned[0], READY)
            return self.state
        if len(owned) > 1:
            self.alerts.append({"code": "UNEXPECTED_INSTANCES", "count": len(owned)})
            if self.state == RECONCILING:
                self._move(FAILED)
            return self.state
        if self.state == RECONCILING:
            self._move(STOPPED)
        return self.state

    def request_shutdown(self, instances: list[dict], *, heartbeat_age: float | None, deadline: float | None) -> list[str]:
        if self.machine_id and self.state not in {STOPPING, STOPPED}:
            self._move(STOPPING)
        targets = instances_to_stop(
            instances,
            machine_id=int(self.machine_id or 0),
            now=float(self.now()),
            deadline=deadline,
            heartbeat_age=heartbeat_age,
        )
        if targets:
            self.alerts.append({"code": "SHUTDOWN_REQUESTED", "instances": targets})
        if self.state == STOPPING:
            self.instance_id = ""
            self._move(STOPPED)
        return targets

    def advance_ready_worker(self) -> None:
        """Local lifecycle only: staging and load do not contact a provider."""
        if self.state == READY:
            return
        if self.state == RECONCILING:
            self._move(STAGING)
        if self.state == STAGING:
            self._move(LOADING)
        if self.state == LOADING:
            self._move(READY)

    def mark_idle(self) -> None:
        if self.state == READY:
            self._move(IDLE)

    def mark_busy(self) -> None:
        if self.state == IDLE:
            self._move(READY)
        if self.state == READY:
            self._move(BUSY)

    def note_spend(self, policy_name: str, spent_usd: Decimal) -> dict:
        """Stop a tracked session when its configured ceiling is reached. Never creates."""
        policy = self.policies.get(policy_name)
        if policy is None:
            return {"allowed": False, "reason": PAID_ACTIONS_DISABLED, "creates": 0}
        ceiling = policy.session_ceiling_usd(policy.hourly_ceiling_usd)
        self.detail["spent_usd"] = str(spent_usd)
        self.detail["session_ceiling_usd"] = str(ceiling)
        if spent_usd >= ceiling and self.state in {READY, IDLE, BUSY}:
            self.alerts.append({"code": "BUDGET_EXHAUSTED", "ceiling_usd": str(ceiling)})
            self._move(STOPPING)
            self.instance_id = ""
            self._move(STOPPED)
            return {"allowed": False, "reason": "BUDGET_EXHAUSTED", "creates": 0}
        return {"allowed": False, "reason": PAID_ACTIONS_DISABLED, "creates": 0}

    def snapshot(self) -> dict:
        hourly = self.hourly_usd
        return {
            "state": self.state,
            "instance_id": self.instance_id,
            "machine_id": self.machine_id,
            "offer_id": self.offer_id,
            "hourly_usd": str(hourly),
            "estimated_daily_usd": str(hourly * Decimal(24)),
            "estimated_monthly_usd": str(hourly * Decimal(24 * 30)),
            "create_attempted": self.create_attempted,
            "provider_creates": self.provider_creates,
            "paid_actions_enabled": False,
            "alerts": list(self.alerts),
            "candidates": len(self.candidates),
            "inventory": [
                {
                    "offer_id": item.offer_id,
                    "machine_id": item.machine_id,
                    "gpu_name": item.gpu_name,
                    "vram_mib": item.vram_mib,
                    "hourly_usd": str(item.hourly_usd),
                    "reliability": item.reliability,
                    "eligible": item.eligible,
                }
                for item in self.candidates[:4]
            ],
            "poll_backoff_seconds": self.poll.backoff_seconds,
            "recovery": self.state,
        }

    def _adopt(self, row: dict, proposed: str) -> None:
        self.instance_id = str(row.get("id") or "")
        self.machine_id = int(row.get("machine_id") or 0)
        self.offer_id = int(row.get("offer_id") or self.offer_id or 0)
        if row.get("dph_total") not in (None, ""):
            self.hourly_usd = Decimal(str(row.get("dph_total")))
            self.detail["hourly_usd"] = str(self.hourly_usd)
        if self.state == RECONCILING:
            self._move(proposed)

    def _move(self, proposed: str) -> None:
        self.state = transition(self.state, proposed)
        self.store.save({
            "state": self.state,
            "instance_id": self.instance_id,
            "machine_id": self.machine_id,
            "offer_id": self.offer_id,
            "create_attempted": self.create_attempted,
            "detail": self.detail,
            "updated_at": str(self.now()),
        })
