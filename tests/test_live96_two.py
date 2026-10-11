"""Two-image authorization. No Vast rental and no production write."""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.qwen import MODEL_ID
from media_engine.engines.image.qwen_remote import SessionServer, session_image_limit
from media_engine.jobs.model import JobError
from media_engine.orchestrator.controller import MediaController
from media_engine.platform.service import Platform
from media_engine.providers.base import CreateResult, WorkerStageError
from media_engine.providers.execution_authorization import accept_record, issue_authorization
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.vast_image import VastImageProvider

SHA = "e5ad1fefea2cccaf62f20dbd2a0b42a9e5436b17"
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


def _fields(**overrides):
    fields = dict(
        authorization_id="live9-auth-two",
        execution_id="live9-run-two",
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
        max_inference_attempts=2,
        max_image_count=2,
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


def _two(**overrides):
    return issue_authorization(**_fields(**overrides))


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
        "dph_base": "0.40",
        "storage_cost": "0.10",
        "dph_total": "0.42",
        "inet_down": 8000,
        "reliability2": 0.99,
        "verification": "verified",
        "geolocation": "California, US",
    }


class _Popen:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.pid = 5151


class TwoImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            name: os.environ.get(name)
            for name in (
                "MEDIA_ENGINE_ALLOW_COLD_STAGE",
                "MEDIA_ENGINE_FORBID_R2_STAGE",
                "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD",
                "MEDIA_ENGINE_EXECUTION_PROFILE",
                "MEDIA_ENGINE_SESSION_IMAGE_LIMIT",
                "MODEL_CACHE_PROVIDER",
                "R2_ENDPOINT",
                "R2_BUCKET",
                "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY",
            )
        }
        os.environ["MEDIA_ENGINE_ALLOW_COLD_STAGE"] = "1"
        os.environ.pop("MEDIA_ENGINE_FORBID_R2_STAGE", None)
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "5.00"
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live8"
        os.environ["MODEL_CACHE_PROVIDER"] = "r2"
        os.environ["R2_ENDPOINT"] = "https://example.r2.cloudflarestorage.com"
        os.environ["R2_BUCKET"] = "ai-media-engine-models"
        os.environ["R2_ACCESS_KEY_ID"] = "fixture-access"
        os.environ["R2_SECRET_ACCESS_KEY"] = "fixture-secret"

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_two_image_record_is_accepted_and_runs_two_inferences(self) -> None:
        authorization = _two()
        accept_record(authorization, now=5_000, code_sha=SHA)
        self.assertEqual(authorization.max_image_count, 2)
        self.assertEqual(authorization.max_inference_attempts, 2)
        self.assertEqual(authorization.max_creates, 1)
        self.assertEqual(authorization.max_model_stages, 1)
        self.assertEqual(authorization.max_model_loads, 1)
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

        provider = VastImageProvider(
            api_key="fixture-key", transport=lambda *args: (_ for _ in ()).throw(AssertionError("transport")),
            now=lambda: 5_000.0, runtime=Path(tempfile.mkdtemp()),
        )
        provider.execution_authorization = authorization
        provider.authorization_expires_at = 20_000
        provider.execution_id = authorization.execution_id
        provider.arm_live_session()
        self.assertEqual(provider.session_limit, 2)
        self.assertFalse(provider.single_image_only)
        provider.python_executable = "/usr/bin/python3"
        provider._ssh_run = ssh
        provider._host = "worker"
        provider._port = "22"
        first = provider.generate(prompt="one", width=1024, height=1024, seed=1)
        second = provider.generate(prompt="two", width=1024, height=1024, seed=2)
        self.assertTrue(first.startswith(b"\x89PNG"))
        self.assertTrue(second.startswith(b"\x89PNG"))
        self.assertEqual(sum("qwen_remote.py serve" in item for item in calls), 1)
        self.assertTrue(any("MEDIA_ENGINE_SESSION_IMAGE_LIMIT=2" in item for item in calls))
        self.assertFalse(any("MEDIA_ENGINE_SESSION_IMAGE_LIMIT=3" in item for item in calls))
        self.assertEqual(sum("session-inbox/s" in item for item in calls), 2)
        self.assertEqual(provider._model_loads_used, 1)
        self.assertEqual(provider._inference_attempts, 2)
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="three", width=1024, height=1024, seed=3)
        self.assertEqual(raised.exception.code, "SESSION_IMAGE_LIMIT")
        self.assertEqual(sum("qwen_remote.py serve" in item for item in calls), 1)

        provider.session_limit = 3
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="four", width=1024, height=1024, seed=4)
        self.assertEqual(raised.exception.code, "INSTANCE_LIMIT")

        root = Path(tempfile.mkdtemp())
        controller = MediaController(str(root / "jobs.sqlite3"), str(root / "storage"))
        controller.provider.execution_authorization = authorization
        controller.provider.single_image_only = False
        controller.submit_image("one", seed=1)
        controller.submit_image("two", seed=2)
        with self.assertRaises(JobError) as rejected:
            controller.submit_image("three", seed=3)
        self.assertEqual(str(rejected.exception), "SESSION_IMAGE_LIMIT")

    def test_one_image_stays_one_and_three_image_still_opens(self) -> None:
        one = _two(
            authorization_id="live9-auth-one96",
            execution_id="live9-run-one96",
            selection_policy="exact_offers",
            max_total_usd=Decimal("1.50"),
            max_image_count=1,
            max_inference_attempts=1,
            allowed_pairs=(),
        )
        provider = VastImageProvider(
            api_key="fixture-key", transport=lambda *args: (500, b""),
            now=lambda: 5_000.0, runtime=Path(tempfile.mkdtemp()),
        )
        provider.execution_authorization = one
        provider.arm_live_session()
        self.assertEqual(provider.session_limit, 1)
        self.assertTrue(provider.single_image_only)
        three = _two(
            authorization_id="live9-auth-three96",
            execution_id="live9-run-three96",
            max_image_count=3,
            max_inference_attempts=3,
        )
        provider.execution_authorization = three
        provider.execution_grant = None
        provider.arm_live_session()
        self.assertEqual(provider.session_limit, 3)
        self.assertFalse(provider.single_image_only)

    def test_second_create_is_rejected_and_a_live_session_is_not_requoted(self) -> None:
        posts = []
        puts = []
        instances = []
        taken = {"value": False}

        def transport(method, url, body, headers):
            del headers
            if method == "PUT":
                taken["value"] = True
                puts.append(url)
                instances.append({
                    "id": 77,
                    "machine_id": 501,
                    "label": "media-engine:live9-run-two",
                    "actual_status": "running",
                    "ssh_host": "worker.example",
                    "ssh_port": 22,
                })
                return 200, b'{"success": true, "new_contract": 77}'
            if method == "POST":
                posts.append(1)
                if taken["value"]:
                    return 200, json.dumps({"offers": [_offer(1003, 503)]}).encode()
                return 200, json.dumps({"offers": [_offer(1001, 501)]}).encode()
            if method == "GET":
                return 200, json.dumps({
                    "success": True, "instances": instances, "next_token": None,
                }).encode()
            return 500, b""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()),
            now=lambda: 5_000.0,
        )
        provider._watchdog_popen = _Popen
        provider._manifest_ready = lambda: True
        provider.bind_authorization(_two(), code_sha=SHA)
        created = provider.create("nahaber")
        self.assertEqual(created.outcome, "created")
        self.assertEqual(created.resource_id, "77")
        searched = len(posts)
        self.assertEqual(len(puts), 1)
        provider.quote("nahaber")
        provider.quote("nahaber")
        self.assertEqual(len(posts), searched)
        self.assertEqual(provider.offer_id, "1001")
        self.assertEqual(provider.machine_id, "501")
        again = provider.create("nahaber")
        provider._create_closed = False
        provider._put_done = False
        third = provider.create("nahaber")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertEqual(third.outcome, "definite_failure")
        self.assertEqual(len(puts), 1)
        self.assertIsInstance(again, CreateResult)

    def test_worker_limit_and_credit_settlement_stay_exact(self) -> None:
        os.environ["MEDIA_ENGINE_SESSION_IMAGE_LIMIT"] = "2"
        self.assertEqual(session_image_limit(), 2)
        os.environ["MEDIA_ENGINE_SESSION_IMAGE_LIMIT"] = "4"
        with self.assertRaises(RuntimeError):
            session_image_limit()
        os.environ.pop("MEDIA_ENGINE_SESSION_IMAGE_LIMIT", None)
        with self.assertRaises(RuntimeError):
            session_image_limit()

        root = Path(tempfile.mkdtemp()) / "models"
        root.mkdir()
        loads = []

        def ready(path: Path) -> bool:
            return (path / "ready").is_file()

        def sync(path: Path) -> None:
            (path / "ready").write_text("1", encoding="utf-8")

        def load(path: Path) -> dict:
            loads.append(path)
            return {"loaded": True}

        def generate(pipe: dict, job: dict) -> bytes:
            del pipe
            return PNG

        server = SessionServer(root, cache_ready=ready, sync=sync, load=load, generate=generate, limit=2)
        server.submit({"id": "a"})
        server.submit({"id": "b"})
        self.assertEqual(loads, [root])
        with self.assertRaises(RuntimeError):
            server.submit({"id": "c"})

        db = Path(tempfile.mkdtemp()) / "platform.sqlite3"
        platform = Platform(str(db), now=lambda: datetime(2026, 10, 11, tzinfo=timezone.utc))
        platform.set_pool_priority(["topup"])
        platform.repo.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=1, effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 4, "topup_purchase", idempotency_key="live96-seed")
        platform.authorize_paid("nahaber", "job-fail")
        platform.settle_failure("nahaber", "job-fail")
        platform.settle_failure("nahaber", "job-fail")
        self.assertEqual(platform.balance("nahaber"), 4)
        for job_id in ("job-a", "job-b"):
            platform.authorize_paid("nahaber", job_id)
            platform.persist_media(
                user_id="nahaber", project_id="direct", scene_id="image", job_id=job_id,
                artifact_id="art-" + job_id, kind="images", ext="png", data=PNG,
                mime_type="image/png", settings={"seed": 1}, width=1024, height=1024,
            )
            platform.settle("nahaber", job_id, 2)
            platform.settle("nahaber", job_id, 2)
        self.assertEqual(platform.balance("nahaber"), 0)
        with sqlite3.connect(db) as conn:
            counts = dict(conn.execute(
                "SELECT entry_type, COUNT(*) FROM credit_ledger GROUP BY entry_type"
            ).fetchall())
        self.assertEqual(counts.get("consume"), 2)
        self.assertEqual(counts.get("refund"), 1)
        self.assertEqual(counts.get("reserve"), 3)


if __name__ == "__main__":
    unittest.main()
