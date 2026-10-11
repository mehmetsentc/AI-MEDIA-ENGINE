"""An active authorized GPU stays in use after its offer leaves the market."""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import struct
import tempfile
import unittest
import zlib
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from media_engine.engines.image.model_cache import QWEN_CACHE_MIN_BYTES, QWEN_FILE_COUNT, REPO_ID
from media_engine.engines.image.qwen import MODEL_ID
from media_engine.jobs.model import JobError, JobStatus
from media_engine.orchestrator.controller import MediaController
from media_engine.platform.service import Platform
from media_engine.providers.base import WorkerStageError
from media_engine.providers.execution_authorization import issue_authorization
from media_engine.providers.live7_contract import REVISION
from media_engine.providers.runtime_python import pinned_image_selection
from media_engine.providers.vast_image import VastImageProvider
from media_engine.safety.execution_budget import termination_reserve_usd

SHA = "2acecb48afb76d924cd61374df05bcdd32a4b922"


def _png(width: int = 1024, height: int = 1024) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + (b"\x00\x00\x00" * width) for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


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


class _Clock:
    def __init__(self, start: float) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def iso(self) -> str:
        return datetime.fromtimestamp(self.t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Popen:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.pid = 5151

    def poll(self):
        return None

    def wait(self, timeout=None):
        del timeout
        return 0


class ActiveSessionReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            name: os.environ.get(name)
            for name in (
                "MEDIA_ENGINE_ALLOW_COLD_STAGE",
                "MEDIA_ENGINE_FORBID_R2_STAGE",
                "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD",
                "MEDIA_ENGINE_EXECUTION_PROFILE",
                "STUDIO_GPU_BUDGET_USD",
                "MODEL_CACHE_PROVIDER",
                "R2_ENDPOINT",
                "R2_BUCKET",
                "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY",
                "MEDIA_ENGINE_R2_RELIABLE_TRANSFER",
            )
        }
        os.environ["MEDIA_ENGINE_ALLOW_COLD_STAGE"] = "1"
        os.environ.pop("MEDIA_ENGINE_FORBID_R2_STAGE", None)
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "5.00"
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live8"
        os.environ.pop("STUDIO_GPU_BUDGET_USD", None)
        os.environ["MODEL_CACHE_PROVIDER"] = "r2"
        os.environ["R2_ENDPOINT"] = "https://example.r2.cloudflarestorage.com"
        os.environ["R2_BUCKET"] = "ai-media-engine-models"
        os.environ["R2_ACCESS_KEY_ID"] = "fixture-access"
        os.environ["R2_SECRET_ACCESS_KEY"] = "fixture-secret"
        os.environ.pop("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", None)

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_later_jobs_reuse_one_gpu_after_the_offer_leaves_the_market(self) -> None:
        clock = _Clock(5_000.0)
        posts: list[dict] = []
        puts: list[str] = []
        deletes: list[str] = []
        instances: list[dict] = []
        taken = {"value": False}
        commands: list[str] = []
        cache_reads = {"count": 0}
        stages: list[str] = []
        image = _png()
        cache = {
            "state": "COMPLETE",
            "repo": REPO_ID,
            "revision": REVISION,
            "files": QWEN_FILE_COUNT,
            "bytes": QWEN_CACHE_MIN_BYTES,
            "mount": True,
        }

        def transport(method, url, body, headers):
            del headers
            if method == "PUT":
                taken["value"] = True
                puts.append(url)
                instances.append({
                    "id": 77,
                    "machine_id": 501,
                    "label": "media-engine:live9-run-reuse",
                    "actual_status": "running",
                    "ssh_host": "worker.example",
                    "ssh_port": 22,
                    "gpu_name": "RTX 6000 Ada",
                })
                return 200, b'{"success": true, "new_contract": 77}'
            if method == "POST":
                posts.append(json.loads(body.decode("utf-8")))
                if taken["value"]:
                    # A different rentable card must not become a second rental.
                    return 200, json.dumps({"offers": [_offer(1003, 503)]}).encode()
                return 200, json.dumps({"offers": [_offer(1001, 501)]}).encode()
            if method == "GET":
                return 200, json.dumps({
                    "success": True, "instances": [dict(row) for row in instances], "next_token": None,
                }).encode()
            if method == "DELETE":
                deletes.append(url)
                instances.clear()
                return 200, b'{"success": true}'
            return 500, b""

        def ssh(host, port, command, timeout):
            del host, port, timeout
            commands.append(command)
            if "phase2d-cache.json" in command:
                cache_reads["count"] += 1
                if cache_reads["count"] == 1:
                    return 0, json.dumps({"state": "MISSING", "mount": False, "entries": []}) + "\n", ""
                return 0, json.dumps(cache) + "\n", ""
            if "RUNTIME_PYTHON_SELECT" in command:
                return 0, json.dumps(pinned_image_selection()) + "\n", ""
            if "RUNTIME_IMPORT_PROBE" in command:
                return 0, "QwenImagePipeline\n", ""
            if "probe" in command:
                body = json.dumps({"state": "worker_ready", "runtime": True})
                return 0, json.dumps({"probe": "http", "status": 200, "body": body}), ""
            if "qwen_remote.py serve" in command:
                return 0, "SESSION_READY\n", ""
            if "session-inbox" in command:
                return 0, "SESSION_IMAGE\n", ""
            if command.startswith("base64 "):
                return 0, base64.b64encode(image).decode(), ""
            return 1, command, ""

        provider = VastImageProvider(
            api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()), now=clock.now,
        )
        provider._watchdog_popen = _Popen
        provider._manifest_ready = lambda: True
        provider._sleep = lambda *_args, **_kwargs: None
        provider._ssh_run = ssh
        provider._stage_r2 = lambda: stages.append("stage")
        provider.bind_authorization(issue_authorization(
            authorization_id="live9-auth-reuse",
            execution_id="live9-run-reuse",
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
        ), code_sha=SHA)

        root = Path(tempfile.mkdtemp())
        platform_db = root / "platform.sqlite3"
        platform = Platform(str(platform_db), now=lambda: datetime(2026, 10, 11, tzinfo=timezone.utc))
        platform.set_pool_priority(["topup"])
        platform.repo.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=1, effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 6, "topup_purchase", idempotency_key="live95-seed")
        controller = MediaController(str(root / "jobs.sqlite3"), str(root / "storage"), clock=clock)
        controller.provider = provider
        controller.platform = platform
        jobs = [controller.submit_image("frame " + str(index), seed=index) for index in range(3)]
        for job in jobs:
            platform.authorize_paid("nahaber", job.id)

        controller._process_next()
        first = controller.jobs.get(jobs[0].id)
        self.assertEqual(first.status, JobStatus.SUCCEEDED, first.error_code)
        self.assertEqual(puts, ["https://console.vast.ai/api/v0/asks/1001/"])
        self.assertEqual(provider._creates_used, 1)
        self.assertEqual(stages, ["stage"])
        self.assertEqual(provider._model_stages_used, 1)
        self.assertEqual(provider._model_loads_used, 1)
        searched = len(posts)

        clock.advance(300)
        self.assertTrue(controller.queue.pending())
        self.assertIsNone(controller.enforce_lifecycle())
        self.assertEqual(deletes, [])
        self.assertEqual(controller._worker_id, "77")

        controller._process_next()
        controller._process_next()
        for job in jobs:
            stored = controller.jobs.get(job.id)
            self.assertEqual(stored.status, JobStatus.SUCCEEDED, stored.error_code)
            self.assertEqual(stored.attempt_count, 1)
        self.assertEqual(len(posts), searched)
        self.assertEqual(puts, ["https://console.vast.ai/api/v0/asks/1001/"])
        self.assertEqual(stages, ["stage"])
        self.assertEqual(provider._model_stages_used, 1)
        self.assertEqual(provider._model_loads_used, 1)
        self.assertEqual(sum("qwen_remote.py serve" in item for item in commands), 1)
        self.assertEqual(sum("session-inbox/s" in item for item in commands), 3)
        self.assertFalse(any("qwen_remote.py generate" in item for item in commands))
        self.assertEqual(provider.execution_grant.offer_id, 1001)
        self.assertEqual(provider.execution_grant.machine_id, 501)
        again = provider.create("nahaber")
        self.assertEqual(again.outcome, "definite_failure")
        self.assertEqual(len(puts), 1)

        with self.assertRaises(JobError) as raised:
            controller.submit_image("frame 4", seed=4)
        self.assertEqual(str(raised.exception), "SESSION_IMAGE_LIMIT")
        with self.assertRaises(WorkerStageError) as generated:
            provider.generate(prompt="four", width=1024, height=1024, seed=4)
        self.assertEqual(generated.exception.code, "SESSION_IMAGE_LIMIT")
        self.assertEqual(provider._inference_attempts, 3)
        self.assertEqual(sum("qwen_remote.py serve" in item for item in commands), 1)

        self.assertEqual(platform.balance("nahaber"), 0)
        platform.settle("nahaber", jobs[0].id, 2)
        self.assertEqual(platform.balance("nahaber"), 0)
        with sqlite3.connect(platform_db) as conn:
            counts = dict(conn.execute(
                "SELECT entry_type, COUNT(*) FROM credit_ledger GROUP BY entry_type"
            ).fetchall())
        self.assertEqual(counts.get("consume"), 3)
        self.assertEqual(counts.get("reserve"), 3)
        self.assertNotIn("refund", counts)

        spec = json.loads((provider.runtime / "watchdog.json").read_text(encoding="utf-8"))
        self.assertEqual(Decimal(spec["reserve_usd"]), termination_reserve_usd(provider.hourly_amount))
        self.assertEqual(spec["ceiling_usd"], "5.00")

        instances[0]["actual_status"] = "exited"
        with self.assertRaises(WorkerStageError) as unhealthy:
            provider.quote("nahaber")
        self.assertEqual(unhealthy.exception.code, "WORKER_NOT_READY")
        instances[0]["actual_status"] = "running"
        instances[0]["label"] = "media-engine:someone-else"
        with self.assertRaises(WorkerStageError) as foreign:
            provider.quote("nahaber")
        self.assertEqual(foreign.exception.code, "OWNERSHIP_MISMATCH")
        instances.clear()
        with self.assertRaises(WorkerStageError) as closed:
            provider.quote("nahaber")
        self.assertEqual(closed.exception.code, "SESSION_CLOSED")
        self.assertEqual(len(posts), searched)
        self.assertEqual(len(puts), 1)
        self.assertEqual(deletes, [])

        instances.append({
            "id": 77,
            "machine_id": 501,
            "label": "media-engine:live9-run-reuse",
            "actual_status": "running",
            "ssh_host": "worker.example",
            "ssh_port": 22,
        })
        clock.advance(299)
        self.assertIsNone(controller.enforce_lifecycle())
        self.assertEqual(deletes, [])
        clock.advance(1)
        self.assertEqual(controller.enforce_lifecycle(), "GPU_IDLE_SHUTDOWN")
        self.assertEqual(deletes, ["https://console.vast.ai/api/v0/instances/77/"])
        self.assertEqual(instances, [])
        self.assertIsNone(controller._worker_id)


if __name__ == "__main__":
    unittest.main()
