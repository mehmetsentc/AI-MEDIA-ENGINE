"""Three-image authorization is the record. The runtime cannot raise it."""
from __future__ import annotations

import base64
import json
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT, REPO_ID
from media_engine.engines.image.qwen import MODEL_ID
from media_engine.jobs.model import JobError
from media_engine.orchestrator.controller import MediaController
from media_engine.platform.repository import PlatformBlocked
from media_engine.platform.service import Platform
from media_engine.providers.base import CreateResult, WorkerStageError
from media_engine.providers.execution_authorization import (
    accept_record,
    authorization_from_json,
    issue_authorization,
    select_for_authorization,
)
from media_engine.providers.gpu_discovery import assess_offers
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.vast_image import REVISION as CACHE_REVISION
from media_engine.providers.vast_image import VastImageProvider

SHA = "f15f654b0fb5b295b2fcbf785d275b1c772b7ed5"
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


def _fields(**overrides):
    fields = dict(
        authorization_id="live9-auth-limit",
        execution_id="live9-run-limit",
        code_sha=SHA,
        allowed_machine_ids=(501,),
        allowed_offer_ids=(1001,),
        allowed_pairs=((501, 1001),),
        selection_policy="just_in_time",
        max_hourly_usd=Decimal("0.95"),
        max_total_usd=Decimal("5.00"),
        max_instances=1,
        max_creates=1,
        max_gpu_instances=1,
        max_inference_attempts=3,
        max_image_count=3,
        max_model_stages=1,
        max_model_loads=1,
        max_disk_gb=140,
        model=MODEL_ID,
        revision=REVISION,
        issued_at=1_000,
        expires_at=20_000,
    )
    fields.update(overrides)
    return fields


def _three(**overrides):
    return issue_authorization(**_fields(**overrides))


def _one(**overrides):
    fields = _fields(
        authorization_id="live8-auth-one",
        execution_id="live8-run-one",
        selection_policy="exact_offers",
        max_total_usd=Decimal("1.50"),
        max_inference_attempts=1,
        max_image_count=1,
    )
    fields.pop("allowed_pairs")
    fields.update(overrides)
    return issue_authorization(**fields)


def _offer(offer_id: int, machine_id: int) -> dict:
    return {
        "id": offer_id,
        "machine_id": machine_id,
        "gpu_name": "RTX 6000 Ada",
        "gpu_ram": 49140,
        "num_gpus": 1,
        "rentable": True,
        "disk_space": 400,
        "cuda_max_good": 13.0,
        "compute_cap": 890,
        "dph_base": "0.70",
        "storage_cost": "0.10",
        "dph_total": "0.72",
        "inet_down": 8000,
        "reliability2": 0.99,
        "verification": "verified",
        "geolocation": "California, US",
    }


def _provider() -> VastImageProvider:
    def transport(*args):
        del args
        raise AssertionError("transport")

    return VastImageProvider(
        api_key="fixture-key",
        transport=transport,
        now=lambda: 5_000.0,
        runtime=Path(tempfile.mkdtemp()),
    )


class AuthorizationLimitTests(unittest.TestCase):
    def test_three_image_record_is_accepted_and_bound(self) -> None:
        authorization = _three()
        self.assertEqual(authorization.max_creates, 1)
        self.assertEqual(authorization.max_gpu_instances, 1)
        self.assertEqual(authorization.max_instances, 1)
        self.assertEqual(authorization.max_image_count, 3)
        self.assertEqual(authorization.max_inference_attempts, 3)
        self.assertEqual(authorization.max_model_stages, 1)
        self.assertEqual(authorization.max_model_loads, 1)
        accept_record(authorization, now=5_000, code_sha=SHA)
        restored = authorization_from_json(json.dumps(authorization.to_dict()))
        self.assertEqual(restored, authorization)
        provider = _provider()
        provider.execution_authorization = authorization
        provider.arm_live_session()
        self.assertEqual(provider.session_limit, 3)
        self.assertFalse(provider.single_image_only)
        self.assertEqual(provider.attached_volume_id, 0)

    def test_one_image_record_stays_one_image(self) -> None:
        authorization = _one()
        self.assertEqual(authorization.max_image_count, 1)
        self.assertEqual(authorization.max_inference_attempts, 1)
        provider = _provider()
        provider.execution_authorization = authorization
        provider.arm_live_session()
        self.assertEqual(provider.session_limit, 1)
        self.assertTrue(provider.single_image_only)
        provider.session_limit = 3
        provider.single_image_only = False
        provider.python_executable = "/usr/bin/python3"
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="one", width=1024, height=1024, seed=1)
        self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")
        self.assertEqual(provider.commands, [])
        provider.arm_live_session()
        self.assertEqual(provider.session_limit, 1)
        self.assertTrue(provider.single_image_only)

    def test_unmatched_and_out_of_range_counts_are_rejected(self) -> None:
        for images, inferences in ((0, 0), (4, 4), (3, 1), (1, 3), (2, 1), (2, 3)):
            with self.assertRaises(WorkerStageError) as raised:
                _three(max_image_count=images, max_inference_attempts=inferences)
            self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")
        for name in ("max_creates", "max_gpu_instances", "max_model_stages", "max_model_loads", "max_instances"):
            with self.assertRaises(WorkerStageError) as raised:
                _three(**{name: 2})
            self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")

    def test_sha_expiry_pair_and_budget_still_fail(self) -> None:
        authorization = _three()
        with self.assertRaises(WorkerStageError) as raised:
            accept_record(authorization, now=5_000, code_sha="a" * 40)
        self.assertEqual(raised.exception.code, "STALE_AUTHORIZATION")
        with self.assertRaises(WorkerStageError) as raised:
            accept_record(authorization, now=20_000, code_sha=SHA)
        self.assertEqual(raised.exception.code, "EXPIRED_AUTHORIZATION")
        with self.assertRaises(WorkerStageError) as raised:
            _three(max_hourly_usd=Decimal("0.96"))
        self.assertEqual(raised.exception.code, "PRICE_CHANGED")
        with self.assertRaises(WorkerStageError) as raised:
            _three(max_total_usd=Decimal("5.01"))
        self.assertEqual(raised.exception.code, "BUDGET_MISMATCH")
        outside = assess_offers([_offer(1003, 501)], observed_at=5_000)
        chosen, reason = select_for_authorization(authorization, outside, now=5_000)
        self.assertIsNone(chosen)
        self.assertEqual(reason, "OFFER_REJECTED")

    def test_one_image_authorization_cannot_execute_three(self) -> None:
        calls = []

        def ssh(host, port, command, timeout):
            del host, port, timeout
            calls.append(command)
            if "phase2d_job.json" in command:
                return 0, "PNG_READY\n", ""
            if command.startswith("base64 "):
                return 0, base64.b64encode(PNG).decode(), ""
            return 1, command, ""

        provider = _provider()
        provider.execution_authorization = _one()
        provider.authorization_expires_at = 20_000
        provider.arm_live_session()
        provider.python_executable = "/usr/bin/python3"
        provider._ssh_run = ssh
        provider._host = "worker"
        provider._port = "22"
        first = provider.generate(prompt="one", width=1024, height=1024, seed=1)
        self.assertTrue(first.startswith(b"\x89PNG"))
        repeated = provider.generate(prompt="one", width=1024, height=1024, seed=1)
        self.assertEqual(repeated, first)
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="two", width=1024, height=1024, seed=2)
        self.assertEqual(raised.exception.code, "SESSION_IMAGE_LIMIT")
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="three", width=1024, height=1024, seed=3)
        self.assertEqual(raised.exception.code, "SESSION_IMAGE_LIMIT")
        self.assertEqual(sum("qwen_remote.py generate" in item for item in calls), 1)
        self.assertFalse(any("qwen_remote.py serve" in item for item in calls))
        self.assertEqual(provider._model_loads_used, 1)
        self.assertEqual(provider._images_started, 1)

    def test_fourth_image_and_second_load_are_rejected(self) -> None:
        calls = []

        def ssh(host, port, command, timeout):
            del host, port, timeout
            calls.append(command)
            if "qwen_remote.py serve" in command:
                return 0, "SESSION_READY\n", ""
            if command.startswith("base64 "):
                return 0, base64.b64encode(PNG).decode(), ""
            if "session-inbox" in command:
                return 0, "SESSION_IMAGE\n", ""
            return 1, command, ""

        provider = _provider()
        provider.execution_authorization = _three()
        provider.authorization_expires_at = 20_000
        provider.arm_live_session()
        provider.python_executable = "/usr/bin/python3"
        provider._ssh_run = ssh
        provider._host = "worker"
        provider._port = "22"
        for prompt, seed in (("one", 1), ("two", 2), ("three", 3)):
            provider.generate(prompt=prompt, width=1024, height=1024, seed=seed)
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="four", width=1024, height=1024, seed=4)
        self.assertEqual(raised.exception.code, "SESSION_IMAGE_LIMIT")
        self.assertEqual(sum("qwen_remote.py serve" in item for item in calls), 1)
        self.assertEqual(provider._model_loads_used, 1)
        self.assertEqual(provider._inference_attempts, 3)
        with self.assertRaises(WorkerStageError) as raised:
            provider._claim_model_load()
        self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")
        provider.session_limit = 9
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="five", width=1024, height=1024, seed=5)
        self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")
        self.assertEqual(sum("qwen_remote.py serve" in item for item in calls), 1)

    def test_second_stage_and_second_create_are_rejected(self) -> None:
        provider = _provider()
        provider.execution_authorization = _three()
        provider.cache_source = "r2_cold_stage"
        provider._worker_cache_already_complete = lambda: False
        staged = []
        provider._stage_r2 = lambda: staged.append("stage")
        payload = json.dumps({
            "state": "COMPLETE",
            "repo": REPO_ID,
            "revision": CACHE_REVISION,
            "files": QWEN_FILE_COUNT,
            "bytes": QWEN_CACHE_MIN_BYTES,
            "mount": True,
        })
        provider._remote = lambda *args, **kwargs: (0, payload, "")
        provider.confirm_cache("instance")
        with self.assertRaises(WorkerStageError) as raised:
            provider.confirm_cache("instance")
        self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")
        self.assertEqual(staged, ["stage"])
        self.assertEqual(provider._model_stages_used, 1)

        creates = []

        def fake_create(owner: str) -> CreateResult:
            creates.append(owner)
            return CreateResult(outcome="created", resource_id="91")

        provider._create_live8 = fake_create
        first = provider.create("studio")
        second = provider.create("studio")
        provider._create_closed = False
        provider._put_done = False
        third = provider.create("studio")
        self.assertEqual(first.outcome, "created")
        self.assertEqual(second.outcome, "definite_failure")
        self.assertEqual(third.outcome, "definite_failure")
        self.assertEqual(creates, ["studio"])
        self.assertFalse(provider.create_request_sent)

    def test_unbound_arm_does_not_open_three_images(self) -> None:
        provider = _provider()
        provider.arm_live_session()
        self.assertEqual(provider.session_limit, 0)
        self.assertTrue(provider.single_image_only)
        provider.session_limit = 3
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="one", width=1024, height=1024, seed=1)
        self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")

    def test_controller_rejects_a_fourth_submit_and_a_second_one_image_submit(self) -> None:
        root = Path(tempfile.mkdtemp())
        controller = MediaController(str(root / "jobs.sqlite3"), str(root / "storage"))
        controller.provider.execution_authorization = _three()
        controller.provider.single_image_only = False
        controller.provider.session_limit = 3
        controller.provider.images_submitted = 0
        for index in range(3):
            controller.submit_image("frame " + str(index), seed=index)
        with self.assertRaises(JobError) as raised:
            controller.submit_image("frame 4", seed=4)
        self.assertEqual(str(raised.exception), "SESSION_IMAGE_LIMIT")
        self.assertEqual(controller.provider.images_submitted, 3)

        other = MediaController(str(root / "one.sqlite3"), str(root / "one-storage"))
        other.provider.execution_authorization = _one()
        other.provider.single_image_only = True
        other.provider.image_claimed = False
        other.submit_image("only", seed=1)
        with self.assertRaises(JobError) as raised:
            other.submit_image("extra", seed=2)
        self.assertEqual(str(raised.exception), "SINGLE_IMAGE_ONLY")

    def test_three_images_consume_two_credits_each(self) -> None:
        root = Path(tempfile.mkdtemp())
        platform = Platform(
            str(root / "platform.sqlite3"),
            now=lambda: datetime(2026, 10, 11, tzinfo=timezone.utc),
        )
        platform.set_pool_priority(["topup"])
        platform.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=1,
            effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 6, "topup_purchase", idempotency_key="seed-three")
        self.assertEqual(platform.balance("nahaber"), 6)
        for index in range(3):
            job_id = "job-" + str(index)
            platform.authorize_paid("nahaber", job_id)
            self.assertEqual(platform.balance("nahaber"), 6 - (index + 1) * 2)
            platform.persist_media(
                user_id="nahaber", project_id="direct", scene_id="image", job_id=job_id,
                artifact_id="art-" + str(index), kind="images", ext="png", data=PNG,
                mime_type="image/png", settings={"seed": index}, width=1024, height=1024,
            )
            platform.settle("nahaber", job_id, 2)
            platform.settle("nahaber", job_id, 2)
            self.assertEqual(platform.balance("nahaber"), 6 - (index + 1) * 2)
        self.assertEqual(platform.balance("nahaber"), 0)
        self.assertEqual(len(platform.media.objects), 3)
        with self.assertRaises(PlatformBlocked) as raised:
            platform.authorize_paid("nahaber", "job-4")
        self.assertEqual(raised.exception.code, "INSUFFICIENT_CREDIT")


if __name__ == "__main__":
    unittest.main()
