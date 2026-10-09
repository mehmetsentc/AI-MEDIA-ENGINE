"""Read-only machine classification. No provider call creates an instance."""
from __future__ import annotations

import json
import unittest
from decimal import Decimal

from media_engine.providers.machine_check import (
    MACHINE_ID,
    VOLUME_ID,
    ReadOnlyInventory,
    classify_machine,
    main,
)


def _offer(**overrides) -> dict:
    row = {
        "id": 30565040,
        "machine_id": MACHINE_ID,
        "gpu_name": "NVIDIA RTX A6000",
        "gpu_ram": 49140,
        "rentable": True,
        "rented": False,
        "dph_base": "0.36",
        "dph_total": "0.362962962962963",
        "storage_cost": "0.2666666666666666",
    }
    row.update(overrides)
    return row


def _volume(**overrides) -> dict:
    row = {"id": VOLUME_ID, "machine_id": MACHINE_ID, "instances": []}
    row.update(overrides)
    return row


class MachineCheckTests(unittest.TestCase):
    def test_rentable_offer_with_ready_volume_is_suitable(self) -> None:
        report = classify_machine(offers=[_offer()], volumes=[_volume()], instances=[])
        self.assertEqual(report["code"], "OFFER_RENTABLE")
        self.assertEqual(report["volume_state"], "VOLUME_READY")
        self.assertTrue(report["suitable"])
        self.assertFalse(report["creates_instance"])
        self.assertLessEqual(Decimal(report["hourly_usd"]), Decimal("0.375"))

    def test_existing_offer_that_is_not_rentable(self) -> None:
        report = classify_machine(
            offers=[_offer(rentable=False)], volumes=[_volume()], instances=[],
        )
        self.assertEqual(report["offer_state"], "OFFER_EXISTS_NOT_RENTABLE")
        self.assertFalse(report["suitable"])

    def test_already_rented_offer(self) -> None:
        report = classify_machine(
            offers=[_offer(rented=True, rentable=False)], volumes=[_volume()], instances=[],
        )
        self.assertEqual(report["code"], "OFFER_RENTED")
        self.assertFalse(report["suitable"])

    def test_active_instance_blocks_a_rentable_offer(self) -> None:
        report = classify_machine(
            offers=[_offer()],
            volumes=[_volume()],
            instances=[{"id": 99, "machine_id": MACHINE_ID}],
        )
        self.assertEqual(report["code"], "OFFER_RENTED")
        self.assertEqual(report["active_instance_ids"], ["99"])
        self.assertFalse(report["suitable"])

    def test_missing_machine(self) -> None:
        report = classify_machine(offers=[], volumes=[_volume()], instances=[])
        self.assertEqual(report["code"], "MACHINE_UNAVAILABLE")

    def test_gpu_mismatch(self) -> None:
        report = classify_machine(
            offers=[_offer(gpu_name="NVIDIA L40")], volumes=[_volume()], instances=[],
        )
        self.assertEqual(report["code"], "GPU_MISMATCH")

    def test_price_above_ceiling(self) -> None:
        report = classify_machine(
            offers=[_offer(dph_base="1.20", dph_total="1.25")],
            volumes=[_volume()],
            instances=[],
        )
        self.assertEqual(report["code"], "PRICE_CHANGED")

    def test_volume_missing_and_incompatible(self) -> None:
        missing = classify_machine(offers=[_offer()], volumes=[], instances=[])
        self.assertEqual(missing["code"], "VOLUME_MISSING")
        other = classify_machine(
            offers=[_offer()],
            volumes=[_volume(machine_id=56199)],
            instances=[],
        )
        self.assertEqual(other["code"], "VOLUME_INCOMPATIBLE")
        attached = classify_machine(
            offers=[_offer()],
            volumes=[_volume(instances=[7])],
            instances=[],
        )
        self.assertEqual(attached["volume_state"], "VOLUME_INCOMPATIBLE")

    def test_missing_and_rejected_credentials_fail_closed(self) -> None:
        missing = classify_machine(offers=[_offer()], volumes=[_volume()], instances=[], credential="missing")
        self.assertEqual(missing["code"], "CREDENTIAL_MISSING")
        self.assertFalse(missing["suitable"])
        rejected = classify_machine(offers=[_offer()], volumes=[_volume()], instances=[], credential="rejected")
        self.assertEqual(rejected["code"], "CREDENTIAL_REJECTED")

    def test_inventory_refuses_mutation_and_hides_the_key(self) -> None:
        seen = []

        def transport(method, url, body, headers):
            seen.append((method, url))
            self.assertNotIn("fixture-key", url)
            self.assertTrue(headers["Authorization"].startswith("Bearer "))
            if "volumes" in url:
                return 200, json.dumps({"volumes": [_volume()]}).encode()
            if "instances" in url:
                return 200, json.dumps({"instances": []}).encode()
            return 200, json.dumps({"offers": [_offer()]}).encode()

        inventory = ReadOnlyInventory(transport)
        report = inventory.check("fixture-key")
        self.assertEqual(report["code"], "OFFER_RENTABLE")
        self.assertEqual({method for method, _url in seen}, {"GET", "POST"})
        with self.assertRaises(RuntimeError):
            inventory._send("PUT", "https://console.vast.ai/api/v0/asks/1/", {}, {})
        self.assertEqual(inventory.mutations, ["PUT"])
        encoded = json.dumps(report)
        self.assertNotIn("fixture-key", encoded)

    def test_rejected_http_status_does_not_look_suitable(self) -> None:
        def transport(method, url, body, headers):
            return 401, b""

        report = ReadOnlyInventory(transport).check("fixture-key")
        self.assertEqual(report["code"], "CREDENTIAL_REJECTED")
        self.assertFalse(report["creates_instance"])

    def test_cli_requires_the_read_only_flag(self) -> None:
        self.assertEqual(main([]), 2)
