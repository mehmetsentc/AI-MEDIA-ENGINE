"""LIVE-6 authorization path with mocks. This file never rents a GPU or writes Neon."""
from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.cache_validation import assess_inventory
from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT
from media_engine.platform.service import MemoryMedia, Platform
from media_engine.providers.base import WorkerStageError
from media_engine.providers.vast_image import (
    MACHINE_ID,
    REPO_ID,
    REVISION,
    VOLUME_ID,
    VastImageProvider,
    _CACHE_RESOLVE,
)
from media_engine.providers.vast_watchdog import run_until_stopped
from media_engine.safety.spend_grant import SpendGrant
from tests.test_phase2e import PNG

SIMULATED_E2E_PASS = "SIMULATED_E2E_PASS"
REAL_IMAGE_E2E_PASS = "REAL_IMAGE_E2E_PASS"


def _manifest(revision: str = REVISION) -> dict:
    size = QWEN_CACHE_MIN_BYTES // QWEN_FILE_COUNT
    last = QWEN_CACHE_MIN_BYTES - size * (QWEN_FILE_COUNT - 1)
    files = []
    for index in range(QWEN_FILE_COUNT):
        name = "model_index.json" if index == 0 else f"weights/shard-{index}.safetensors"
        files.append({
            "path": name,
            "bytes": last if index == QWEN_FILE_COUNT - 1 else size,
            "sha256": f"{index:064x}",
        })
    return {
        "state": "COMPLETE",
        "repo_id": REPO_ID,
        "revision": revision,
        "expected_file_count": QWEN_FILE_COUNT,
        "expected_total_bytes": QWEN_CACHE_MIN_BYTES,
        "files": files,
    }


def _inventory(manifest: dict, *, revision: str | None = None) -> dict:
    entries = []
    for item in manifest["files"]:
        entries.append({
            "path": item["path"],
            "exists": True,
            "size": item["bytes"],
            "link": "../../blobs/" + item["sha256"],
        })
    return {
        "state": "COMPLETE",
        "repo": REPO_ID,
        "revision": revision or manifest["revision"],
        "files": QWEN_FILE_COUNT,
        "bytes": QWEN_CACHE_MIN_BYTES,
        "mount": True,
        "entries": entries,
    }


class _Proc:
    pid = 77


class Live6IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        names = (
            "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD",
            "MEDIA_ENGINE_FORBID_R2_STAGE",
            "MEDIA_ENGINE_ALLOW_COLD_STAGE",
            "MEDIA_ENGINE_MACHINE_ID",
            "MEDIA_ENGINE_WATCHDOG_LAUNCH",
            "MEDIA_ENGINE_WORKER_IMAGE",
        )
        self.previous = {name: os.environ.get(name) for name in names}
        self.addCleanup(self._restore)
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "0.50"
        os.environ["MEDIA_ENGINE_FORBID_R2_STAGE"] = "1"
        os.environ.pop("MEDIA_ENGINE_ALLOW_COLD_STAGE", None)
        os.environ.pop("MEDIA_ENGINE_MACHINE_ID", None)
        os.environ.pop("MEDIA_ENGINE_WATCHDOG_LAUNCH", None)
        self.calls: list[str] = []

    def _restore(self) -> None:
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_missing_expired_stale_and_wrong_targets_do_not_rent(self) -> None:
        missing = self._provider()
        self.assertEqual(missing.create("daidoi").outcome, "definite_failure")
        self.assertEqual(missing.live6_stop, "MISSING_AUTHORIZATION")
        self.assertNotIn("put", missing.launch_trace)

        expired = self._provider()
        with self.assertRaises(WorkerStageError) as already:
            self._bind(expired, expires_at=500)
        self.assertEqual(already.exception.code, "EXPIRED_AUTHORIZATION")
        self.assertNotIn("PUT", self.calls)
        later = self._provider()
        self._bind(later, expires_at=1_500.0)
        later._now = lambda: 1_500.0
        self.assertEqual(later.create("daidoi").outcome, "definite_failure")
        self.assertEqual(later.live6_stop, "EXPIRED_AUTHORIZATION")

        stale = self._provider()
        self._bind(stale)
        stale.execution_id = "other-run"
        self.assertEqual(stale.create("daidoi").outcome, "definite_failure")
        self.assertEqual(stale.live6_stop, "STALE_AUTHORIZATION")

        wrong_machine = self._provider()
        self._bind(wrong_machine)
        wrong_machine.machine_id = "56199"
        self.assertEqual(wrong_machine.create("daidoi").outcome, "definite_failure")
        self.assertEqual(wrong_machine.live6_stop, "OFFER_REJECTED")

        wrong_volume = self._provider()
        self._bind(wrong_volume)
        wrong_volume.attached_volume_id = 1
        self.assertEqual(wrong_volume.create("daidoi").outcome, "definite_failure")
        self.assertEqual(wrong_volume.live6_stop, "VOLUME_MISMATCH")

        costly = self._provider()
        self._bind(costly)
        costly.hourly_amount = Decimal("0.376")
        self.assertEqual(costly.create("daidoi").outcome, "definite_failure")
        self.assertEqual(costly.live6_stop, "OFFER_REJECTED")
        self.assertNotIn("PUT", self.calls)

    def test_one_authorized_create_attaches_the_volume_after_the_watchdog(self) -> None:
        provider = self._provider()
        self._bind(provider)
        provider._watchdog_popen = lambda *_args, **_kwargs: _Proc()
        created = provider.create("daidoi")
        self.assertEqual(created.outcome, "created")
        self.assertEqual(provider.launch_trace, ["watchdog", "put"])
        body = json.loads(self.put_body)
        self.assertEqual(body["volume_info"]["volume_id"], VOLUME_ID)
        self.assertFalse(body["volume_info"]["create_new"])
        self.assertEqual(created.resource_id, "9")
        again = provider.create("daidoi")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertEqual(self.calls.count("PUT"), 1)

    def test_listed_instance_blocks_a_second_rental(self) -> None:
        provider = self._provider(instances=[{
            "id": 8, "machine_id": MACHINE_ID, "label": "media-engine:busy",
        }])
        self._bind(provider)
        provider._watchdog_popen = lambda *_args, **_kwargs: _Proc()
        self.assertEqual(provider.create("daidoi").outcome, "definite_failure")
        self.assertEqual(provider.live6_stop, "INSTANCE_LIMIT")
        self.assertNotIn("PUT", self.calls)

    def test_verification_budget_does_not_select_a_fallback_machine(self) -> None:
        os.environ["MEDIA_ENGINE_MACHINE_ID"] = "56199"
        os.environ["MEDIA_ENGINE_ALLOW_COLD_STAGE"] = "true"
        provider = self._provider()
        with self.assertRaises(WorkerStageError) as caught:
            provider.quote("daidoi")
        self.assertEqual(caught.exception.code, "PROVIDER_CAPACITY_UNAVAILABLE")
        self.assertNotIn("PUT", self.calls)

    def test_cache_failure_stops_before_download_and_destroy_is_requested(self) -> None:
        provider = self._ssh_provider(json.dumps({
            "state": "FILLING", "mount": True, "revision": REVISION, "entries": [],
        }))
        with self.assertRaises(WorkerStageError) as incomplete:
            provider.confirm_cache("9")
        self.assertEqual(incomplete.exception.code, "MODEL_CACHE_INVALID")
        self.assertEqual(provider.cache_validation, "CACHE_INCOMPLETE")
        provider.terminate("9")
        self.assertIn("DELETE", self.calls)

        mismatch = self._ssh_provider(json.dumps(_inventory(_manifest())))
        mismatch.verified_manifest = _manifest(revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        with self.assertRaises(WorkerStageError):
            mismatch.confirm_cache("9")
        self.assertEqual(mismatch.cache_validation, "REVISION_MISMATCH")
        report = assess_inventory(
            _inventory(_manifest()),
            _manifest(revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        )
        self.assertEqual(report["status"], "CACHE_REVISION_MISMATCH")
        self.assertFalse(report["complete"])

        cold = self._provider()
        cold.cache_source = "r2_cold_stage"
        with self.assertRaises(WorkerStageError) as forbidden:
            cold.confirm_cache("9")
        self.assertEqual(forbidden.exception.code, "MODEL_CACHE_INVALID")
        self.assertEqual(cold.cache_validation, "R2_FORBIDDEN")
        self.assertIn('HF_HUB_OFFLINE"] = "1"', _CACHE_RESOLVE)

    def test_watchdog_deletes_at_the_deadline(self) -> None:
        deleted: list[str] = []
        rows = [{"id": "9", "machine_id": MACHINE_ID, "label": "media-engine:one"}]
        clock = iter((0.0, 10.0, 30.0))
        outcome = run_until_stopped(
            {
                "machine_id": MACHINE_ID,
                "safe_seconds": 30,
                "heartbeat": str(Path(self.tmp.name) / "beat"),
                "stop": str(Path(self.tmp.name) / "stop"),
                "stale_after": 300,
                "poll_seconds": 1,
            },
            fetch=lambda: rows,
            delete=deleted.append,
            now=clock.__next__,
            sleep=lambda _seconds: None,
        )
        self.assertEqual(deleted, ["9"])
        self.assertEqual(outcome["stopped"], ["9"])

    def test_simulated_path_settles_once_and_refunds_a_failure(self) -> None:
        provider = self._provider()
        self._bind(provider)
        provider._watchdog_popen = lambda *_args, **_kwargs: _Proc()
        os.environ["MEDIA_ENGINE_WORKER_IMAGE"] = "example.invalid/worker:local"
        created = provider.create("nahaber")
        self.assertEqual(created.outcome, "created")
        self.assertEqual(provider.launch_trace[0], "watchdog")
        provider._host = "203.0.113.10"
        provider._port = "22"
        provider._ssh_run = self._ssh
        provider.boot(created.resource_id)
        provider.verified_manifest = _manifest()
        provider.confirm_cache(created.resource_id)
        self.assertEqual(provider.cache_validation, "COMPLETE")
        provider.ensure_model(created.resource_id)
        image = provider.generate(prompt="fixture", width=1024, height=1024, seed=20260804)
        self.assertEqual(image, PNG)
        platform = self._platform("ok")
        platform.authorize_paid("nahaber", "job-sim")
        platform.media.put("users/nahaber/sim.png", image)
        platform.settle("nahaber", "job-sim", 2)
        platform.settle("nahaber", "job-sim", 2)
        self.assertEqual(platform.balance("nahaber"), 1)
        failed = self._platform("fail")
        failed.authorize_paid("nahaber", "job-fail")
        failed.settle_failure("nahaber", "job-fail", legitimate_cost=0)
        failed.settle_failure("nahaber", "job-fail", legitimate_cost=0)
        self.assertEqual(failed.balance("nahaber"), 3)
        provider.terminate(created.resource_id)
        self.assertIn("DELETE", self.calls)
        self.assertEqual(SIMULATED_E2E_PASS, "SIMULATED_E2E_PASS")
        self.assertNotIn(REAL_IMAGE_E2E_PASS, provider.cache_validation)

    def _provider(self, instances: list | None = None) -> VastImageProvider:
        listed = list(instances or [])
        created = {"done": False}
        removed: set[str] = set()
        self.put_body = b""

        def transport(method, url, body, headers):
            self.calls.append(method)
            self.assertNotIn("fixture-key", url)
            if method == "PUT":
                created["done"] = True
                self.put_body = body
                return 200, b'{"success": true, "new_contract": 9}'
            if method == "DELETE":
                removed.add(url.rstrip("/").rsplit("/", 1)[-1])
                return 200, b'{"success": true}'
            if "volumes" in url:
                return 200, json.dumps({"volumes": [{"id": VOLUME_ID, "machine_id": MACHINE_ID}]}).encode()
            if "instances" in url:
                rows = [row for row in listed if str(row.get("id")) not in removed]
                if created["done"] and "9" not in removed:
                    rows.append({
                        "id": 9,
                        "machine_id": MACHINE_ID,
                        "label": "media-engine:nahaber",
                    })
                return 200, json.dumps({"instances": rows, "next_token": None}).encode()
            return 200, b'{"offers": []}'

        provider = VastImageProvider(
            api_key="fixture-key",
            transport=transport,
            now=lambda: 1_000.0,
            sleep=lambda _seconds: None,
            runtime=Path(self.tmp.name),
        )
        provider.offer_id = "30565038"
        provider.cache_source = "warm_cache"
        provider.machine_id = str(MACHINE_ID)
        provider.gpu_model = "RTX A6000"
        provider.hourly_amount = Decimal("0.374")
        provider.disk_gb = 40
        return provider

    def _bind(self, provider: VastImageProvider, *, expires_at: float = 2_000.0) -> None:
        provider.bind_execution(SpendGrant(
            execution_id="live6-run",
            ceiling_usd=Decimal("0.50"),
            max_instances=1,
            issued_at=1_000.0,
            expires_at=expires_at,
            machine_id=MACHINE_ID,
        ), execution_id="live6-run")

    def _ssh_provider(self, payload: str) -> VastImageProvider:
        provider = self._provider(instances=[{
            "id": "9", "machine_id": MACHINE_ID, "label": "media-engine:one",
        }])
        provider._host = "203.0.113.10"
        provider._port = "22"
        provider._resource_id = "9"

        def ssh(_host, _port, command, _timeout):
            return 0, payload + "\n", ""

        provider._ssh_run = ssh
        return provider

    def _ssh(self, _host, _port, command, _timeout):
        if "echo ready" in command:
            return 0, "ready\n", ""
        if "phase2d-cache.json" in command and "snapshot_download" not in command:
            return 0, json.dumps(_inventory(_manifest())) + "\n", ""
        if "snapshot_download" in command:
            return 0, json.dumps({"resolved": True, "wrote": False}) + "\n", ""
        if "QwenImagePipeline" in command:
            return 0, "QwenImagePipeline\n", ""
        if "qwen_remote.py generate" in command:
            return 0, "PNG_READY\n", ""
        if command.startswith("base64 "):
            return 0, base64.b64encode(PNG).decode("ascii") + "\n", ""
        return 0, "", ""

    def _platform(self, name: str) -> Platform:
        platform = Platform(str(Path(self.tmp.name) / (name + ".sqlite3")), media=MemoryMedia())
        platform.add_price(
            version="live6-" + name, service="image", model="qwen-image", quality="sim",
            unit="image", credit_price=2, internal_cost_units=1, effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 3, "topup_purchase", idempotency_key="live6-" + name)
        return platform
