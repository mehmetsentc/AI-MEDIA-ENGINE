"""Three-image session wiring. No Vast create and no production write."""
from __future__ import annotations

import base64
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from media_engine.engines.image.qwen_remote import PINNED_QWEN_REVISION, SessionServer, cache_is_ready
from media_engine.infra.manager import GpuManager
from media_engine.infra.store import ManagerStore
from media_engine.platform.repository import PlatformBlocked
from media_engine.platform.service import Platform
from media_engine.platform.test_grant import grant_test_credits
from media_engine.providers.base import WorkerStageError
from media_engine.providers.vast_image import VastImageProvider

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class SessionTests(unittest.TestCase):
    def test_server_loads_once_and_a_retry_does_not_infer_again(self) -> None:
        root = Path(tempfile.mkdtemp()) / "models"
        root.mkdir()
        loads = []
        generated = []

        def ready(path: Path) -> bool:
            return (path / "ready").is_file()

        def sync(path: Path) -> None:
            (path / "ready").write_text("1", encoding="utf-8")

        def load(path: Path) -> dict:
            loads.append(path)
            return {"loaded": True}

        def generate(pipe: dict, job: dict) -> bytes:
            generated.append(job["id"])
            return PNG + job["id"].encode()

        server = SessionServer(
            root, cache_ready=ready, sync=sync, load=load, generate=generate, limit=3,
        )
        self.assertFalse(cache_is_ready(root, min_bytes=1))
        first = [server.submit({"id": name}) for name in ("a", "b", "c")]
        again = server.submit({"id": "a"})
        self.assertEqual(len(first), 3)
        self.assertEqual(again, first[0])
        self.assertEqual(loads, [root])
        self.assertEqual(generated, ["a", "b", "c"])
        self.assertEqual(server.downloads, 1)
        with self.assertRaises(RuntimeError):
            server.submit({"id": "d"})

    def test_provider_session_uses_one_serve_process(self) -> None:
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
            now=_Clock(100), runtime=Path(tempfile.mkdtemp()),
        )
        provider.arm_live_session()
        provider.python_executable = "/usr/bin/python3"
        provider._ssh_run = ssh
        provider._host = "worker"
        provider._port = "22"
        images = [
            provider.generate(prompt=prompt, width=1024, height=1024, seed=seed)
            for prompt, seed in (("one", 1), ("two", 2), ("three", 3))
        ]
        repeated = provider.generate(prompt="one", width=1024, height=1024, seed=1)
        self.assertEqual(images[0], repeated)
        self.assertEqual(len({image for image in images}), 1)
        self.assertEqual(sum("qwen_remote.py serve" in item for item in calls), 1)
        self.assertEqual(sum("session-inbox/s" in item for item in calls), 3)
        self.assertFalse(any("qwen_remote.py generate" in item for item in calls))
        self.assertEqual(provider.attached_volume_id, 0)
        with self.assertRaises(WorkerStageError) as raised:
            provider.generate(prompt="four", width=1024, height=1024, seed=4)
        self.assertEqual(raised.exception.code, "SESSION_IMAGE_LIMIT")
        provider.authorization_expires_at = 50
        with self.assertRaises(WorkerStageError) as expired:
            provider.generate(prompt="one", width=1024, height=1024, seed=1)
        self.assertEqual(expired.exception.code, "EXPIRED_AUTHORIZATION")

    def test_credits_settle_three_artifacts_and_a_failure_refunds(self) -> None:
        root = Path(tempfile.mkdtemp())
        platform = Platform(str(root / "platform.sqlite3"), now=lambda: datetime(2026, 10, 11, tzinfo=timezone.utc))
        platform.set_pool_priority(["topup"])
        platform.add_price(
            version="daidoi-e2e", service="image", model="qwen-image", quality="standard",
            unit="image", credit_price=2, internal_cost_units=1,
            effective_at="2026-01-01T00:00:00Z",
        )
        platform.grant("nahaber", "topup", 1, "topup_purchase", idempotency_key="existing")
        previous = os.environ.get("MEDIA_ENGINE_ALLOW_TEST_CREDITS")
        admin = os.environ.get("MEDIA_ENGINE_TEST_CREDIT_ADMIN")
        os.environ["MEDIA_ENGINE_ALLOW_TEST_CREDITS"] = "1"
        os.environ["MEDIA_ENGINE_TEST_CREDIT_ADMIN"] = "operator"
        try:
            with self.assertRaises(PlatformBlocked):
                grant_test_credits(
                    platform, admin_id="other", user_id="nahaber", credits=5,
                    idempotency_key="test-grant:live9",
                )
            grant_test_credits(
                platform, admin_id="operator", user_id="nahaber", credits=5,
                idempotency_key="test-grant:live9",
            )
            grant_test_credits(
                platform, admin_id="operator", user_id="nahaber", credits=5,
                idempotency_key="test-grant:live9",
            )
        finally:
            if previous is None:
                os.environ.pop("MEDIA_ENGINE_ALLOW_TEST_CREDITS", None)
            else:
                os.environ["MEDIA_ENGINE_ALLOW_TEST_CREDITS"] = previous
            if admin is None:
                os.environ.pop("MEDIA_ENGINE_TEST_CREDIT_ADMIN", None)
            else:
                os.environ["MEDIA_ENGINE_TEST_CREDIT_ADMIN"] = admin
        self.assertEqual(platform.balance("nahaber"), 6)
        for index in range(2):
            self._finish_image(platform, index)
        platform.authorize_paid("nahaber", "job-fail")
        self.assertEqual(platform.recover_paid("nahaber", "job-fail"), "refunded")
        self.assertEqual(platform.recover_paid("nahaber", "job-fail"), "settled")
        self.assertEqual(platform.balance("nahaber"), 2)
        self._finish_image(platform, 2)
        self.assertEqual(platform.balance("nahaber"), 0)
        self.assertEqual(len(platform.media.objects), 3)

    def _finish_image(self, platform: Platform, index: int) -> None:
        job_id = "job-" + str(index)
        platform.authorize_paid("nahaber", job_id)
        platform.persist_media(
            user_id="nahaber", project_id="direct", scene_id="image", job_id=job_id,
            artifact_id="art-" + str(index), kind="images", ext="png", data=PNG,
            mime_type="image/png", settings={"seed": index}, width=1024, height=1024,
        )
        platform.settle("nahaber", job_id, 2)

    def test_unauthorized_create_is_rejected_and_watchdog_deletes(self) -> None:
        methods = []

        def transport(method, url, body, headers):
            del url, body, headers
            methods.append(method)
            return b""

        previous_budget = os.environ.get("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD")
        previous_profile = os.environ.get("MEDIA_ENGINE_EXECUTION_PROFILE")
        os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = "5.00"
        os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = "live8"
        try:
            provider = VastImageProvider(api_key="fixture-key", transport=transport, runtime=Path(tempfile.mkdtemp()))
            provider.arm_live_session()
            result = provider.create("nahaber")
        finally:
            if previous_budget is None:
                os.environ.pop("MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD", None)
            else:
                os.environ["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] = previous_budget
            if previous_profile is None:
                os.environ.pop("MEDIA_ENGINE_EXECUTION_PROFILE", None)
            else:
                os.environ["MEDIA_ENGINE_EXECUTION_PROFILE"] = previous_profile
        self.assertEqual(result.outcome, "definite_failure")
        self.assertNotIn("PUT", methods)
        self.assertFalse(provider.create_request_sent)
        clock = _Clock(1_000)
        manager = GpuManager(ManagerStore(Path(tempfile.mkdtemp()) / "gpu.sqlite3"), now=clock)
        manager.reconcile([{"id": "77", "machine_id": 10, "label": "media-engine:run", "dph_total": "0.55"}])
        self.assertEqual(manager.request_shutdown(
            [{"id": "77", "machine_id": 10, "label": "media-engine:run"}],
            heartbeat_age=301, deadline=None,
        ), ["77"])
        self.assertEqual(PINNED_QWEN_REVISION, "75e0b4be04f60ec59a75f475837eced720f823b6")


if __name__ == "__main__":
    unittest.main()
