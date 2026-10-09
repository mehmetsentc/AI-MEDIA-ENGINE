"""One-execution spend grants and provider-independent stop rules."""
from __future__ import annotations

import unittest
from decimal import Decimal

from media_engine.safety.instance_guard import instances_to_stop, lock_action, termination_report
from media_engine.safety.spend_grant import SpendGrant, evaluate_grant


def _grant(**overrides) -> SpendGrant:
    fields = dict(
        execution_id="run-1",
        ceiling_usd=Decimal("0.50"),
        max_instances=1,
        issued_at=100.0,
        expires_at=200.0,
        machine_id=47281,
    )
    fields.update(overrides)
    return SpendGrant(**fields)


class SpendGrantTests(unittest.TestCase):
    def test_missing_expired_stale_and_mismatch(self) -> None:
        self.assertEqual(evaluate_grant(None, now=120, execution_id="run-1", ceiling_usd=Decimal("0.50"), instances=0)["reason"], "MISSING_AUTHORIZATION")
        self.assertEqual(
            evaluate_grant(_grant(), now=200, execution_id="run-1", ceiling_usd=Decimal("0.50"), instances=0, machine_id=47281)["reason"],
            "EXPIRED_AUTHORIZATION",
        )
        self.assertEqual(
            evaluate_grant(_grant(), now=120, execution_id="run-2", ceiling_usd=Decimal("0.50"), instances=0, machine_id=47281)["reason"],
            "STALE_AUTHORIZATION",
        )
        self.assertEqual(
            evaluate_grant(_grant(), now=120, execution_id="run-1", ceiling_usd=Decimal("1.00"), instances=0, machine_id=47281)["reason"],
            "BUDGET_MISMATCH",
        )

    def test_instance_and_machine_limits(self) -> None:
        self.assertEqual(
            evaluate_grant(_grant(), now=120, execution_id="run-1", ceiling_usd=Decimal("0.50"), instances=2, machine_id=47281)["reason"],
            "INSTANCE_LIMIT",
        )
        self.assertEqual(
            evaluate_grant(_grant(), now=120, execution_id="run-1", ceiling_usd=Decimal("0.50"), instances=1, machine_id=56199)["reason"],
            "MACHINE_SCOPE",
        )
        allowed = evaluate_grant(
            _grant(), now=120, execution_id="run-1", ceiling_usd=Decimal("0.50"), instances=1, machine_id=47281,
        )
        self.assertEqual(allowed["action"], "continue")
        self.assertEqual(allowed["ceiling_usd"], "0.50")


class InstanceGuardTests(unittest.TestCase):
    def test_dead_lock_is_replaced_and_a_live_lock_is_kept(self) -> None:
        self.assertEqual(lock_action(0, lambda _pid: False), "replace")
        self.assertEqual(lock_action(44, lambda pid: pid == 44), "keep")

    def test_two_owned_instances_and_deadline_or_stale_heartbeat(self) -> None:
        rows = [
            {"id": "a", "machine_id": 47281, "label": "media-engine:one"},
            {"id": "b", "machine_id": 47281, "label": "media-engine:two"},
            {"id": "c", "machine_id": 56199, "label": "media-engine:other"},
        ]
        self.assertEqual(instances_to_stop(rows, machine_id=47281, now=10, deadline=100, heartbeat_age=1), ["a", "b"])
        one = [{"id": "a", "machine_id": 47281, "label": "media-engine:one"}]
        self.assertEqual(instances_to_stop(one, machine_id=47281, now=10, deadline=100, heartbeat_age=1), [])
        self.assertEqual(instances_to_stop(one, machine_id=47281, now=100, deadline=100, heartbeat_age=1), ["a"])
        self.assertEqual(instances_to_stop(one, machine_id=47281, now=10, deadline=100, heartbeat_age=301), ["a"])

    def test_unknown_billing_is_not_a_confirmed_charge(self) -> None:
        timeout = termination_report(delete_status=None, still_listed=True)
        failed = termination_report(delete_status=500, still_listed=True)
        gone = termination_report(delete_status=200, still_listed=False)
        self.assertEqual(timeout["provider_state"], "UNKNOWN")
        self.assertEqual(failed["provider_state"], "STILL_LISTED")
        self.assertEqual(gone["provider_state"], "ABSENT")
        self.assertIsNone(timeout["confirmed_charge_usd"])
        self.assertIsNone(gone["confirmed_charge_usd"])
