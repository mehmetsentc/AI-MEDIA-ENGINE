"""Simulated transfer streams. These tests do not contact Cloudflare R2."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from media_engine.engines.image.r2_cache import R2StorageError
from media_engine.engines.image.r2_reliability import (
    ProgressBook,
    accept_range_body,
    download_with_resume,
    promote_complete,
    reliable_transfer_enabled,
    resume_plan,
    retry_delay,
)
from media_engine.providers.vast_image import decide_staging_budget


class ProgressTests(unittest.TestCase):
    def test_retries_replace_in_flight_bytes(self) -> None:
        book = ProgressBook()
        book.note_received("shard", 100)
        book.note_received("shard", 40)
        snap = book.snapshot(1000)
        self.assertEqual(snap["network_bytes"], 40)
        self.assertEqual(snap["validated_bytes"], 0)
        self.assertEqual(snap["remaining_bytes"], 1000)
        book.note_completed("shard", 1000)
        book.note_received("shard", 5)
        done = book.snapshot(1000)
        self.assertEqual(done["validated_bytes"], 1000)
        self.assertEqual(done["network_bytes"], 1000)
        self.assertEqual(done["completed_objects"], 1)
        self.assertEqual(done["remaining_bytes"], 0)


class ResumeTests(unittest.TestCase):
    def test_plan_and_range_acceptance(self) -> None:
        self.assertEqual(resume_plan(0, 10, accepts_range=True)["mode"], "replace")
        self.assertEqual(resume_plan(4, 10, accepts_range=False)["mode"], "replace")
        self.assertEqual(resume_plan(4, 10, accepts_range=True)["header"], "bytes=4-")
        self.assertEqual(resume_plan(10, 10, accepts_range=True)["mode"], "already_complete")
        self.assertEqual(accept_range_body(200, "", start=4, total=10), "replace")
        self.assertEqual(accept_range_body(206, "bytes 4-9/10", start=4, total=10), "append")
        self.assertEqual(accept_range_body(206, "bytes 0-9/10", start=4, total=10), "reject")
        self.assertEqual(accept_range_body(206, "bytes 4-9/11", start=4, total=10), "reject")
        self.assertEqual(accept_range_body(403, "", start=0, total=10), "reject")

    def test_auth_is_not_retried_and_delay_is_bounded(self) -> None:
        self.assertIsNone(retry_delay(401, 0))
        self.assertIsNone(retry_delay(403, 1))
        self.assertIsNone(retry_delay(404, 0))
        self.assertEqual(retry_delay(503, 0), 0.5)
        self.assertLessEqual(retry_delay(0, 3), 8.0)
        self.assertIsNone(retry_delay(0, 4))

    def test_promote_requires_the_expected_size(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        partial = root / "obj.partial"
        dest = root / "obj"
        partial.write_bytes(b"no")
        with self.assertRaises(R2StorageError):
            promote_complete(partial, dest, 4)
        self.assertFalse(dest.exists())
        partial.write_bytes(b"full")
        promote_complete(partial, dest, 4)
        self.assertEqual(dest.read_bytes(), b"full")
        self.assertFalse(partial.exists())


class FakeClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def _request(self, method, key, body, extra=None, stream_to=None, stream_mode="wb", on_chunk=None, should_stop=None):
        self.calls.append({"extra": extra, "mode": stream_mode})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status, headers, payload = item
        if stream_to is not None and status in {200, 206}:
            Path(stream_to).write_bytes(payload)
            if on_chunk is not None:
                on_chunk(len(payload))
        return status, headers, payload


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name) / "cache" / "shard.bin"

    def test_resume_appends_only_a_matching_range(self) -> None:
        self.dest.parent.mkdir(parents=True)
        partial = self.dest.with_name(self.dest.name + ".partial")
        partial.write_bytes(b"abcd")
        sleeps = []
        client = FakeClient([
            R2StorageError(0),
            (206, {"Content-Range": "bytes 4-7/8"}, b"efgh"),
        ])
        download_with_resume(
            client, "hf-cache/shard.bin", self.dest, object_size=8, sleep=sleeps.append,
        )
        self.assertEqual(self.dest.read_bytes(), b"abcdefgh")
        self.assertFalse(partial.exists())
        self.assertEqual(sleeps, [0.5])
        self.assertEqual(client.calls[1]["extra"]["range"], "bytes=4-")

    def test_full_response_replaces_a_partial_file(self) -> None:
        self.dest.parent.mkdir(parents=True)
        partial = self.dest.with_name(self.dest.name + ".partial")
        partial.write_bytes(b"OLD")
        client = FakeClient([(200, {}, b"new-body")])
        download_with_resume(
            client, "hf-cache/shard.bin", self.dest, object_size=8, accepts_range=False, sleep=lambda _s: None,
        )
        self.assertEqual(self.dest.read_bytes(), b"new-body")
        self.assertIsNone(client.calls[0]["extra"])

    def test_authentication_failure_stops(self) -> None:
        client = FakeClient([(401, {}, b"")])
        with self.assertRaises(R2StorageError) as caught:
            download_with_resume(client, "k", self.dest, object_size=4, sleep=lambda _s: (_ for _ in ()).throw(AssertionError("slept")))
        self.assertEqual(caught.exception.status, 401)
        self.assertFalse(self.dest.exists())

    def test_flag_defaults_off(self) -> None:
        os.environ.pop("MEDIA_ENGINE_R2_RELIABLE_TRANSFER", None)
        self.assertFalse(reliable_transfer_enabled())
        os.environ["MEDIA_ENGINE_R2_RELIABLE_TRANSFER"] = "1"
        self.addCleanup(os.environ.pop, "MEDIA_ENGINE_R2_RELIABLE_TRANSFER", None)
        self.assertTrue(reliable_transfer_enabled())


class BudgetSampleTests(unittest.TestCase):
    def test_fast_first_sample_does_not_guarantee_success(self) -> None:
        samples = [(0.0, 0), (1.0, 50_000_000), (31.0, 50_100_000), (61.0, 50_200_000)]
        decision = decide_staging_budget(
            samples, now=61.0, deadline=10_000.0, reserve_seconds=30.0, total_bytes=57_704_594_653,
        )
        self.assertEqual(decision["reason"], "SUSTAINED_RATE_OVER_BUDGET")

    def test_slow_first_sample_does_not_stop_a_healthy_window(self) -> None:
        samples = [(0.0, 0), (10.0, 1_000), (40.0, 20_000_000), (70.0, 40_000_000)]
        decision = decide_staging_budget(
            samples, now=70.0, deadline=10_000.0, reserve_seconds=30.0, total_bytes=50_000_000,
        )
        self.assertEqual(decision["reason"], "within_budget")

    def test_one_later_slow_interval_does_not_set_the_rate(self) -> None:
        samples = [(0.0, 0), (10.0, 1_000_000), (20.0, 20_000_000), (80.0, 21_000_000)]
        decision = decide_staging_budget(
            samples, now=80.0, deadline=110.0, reserve_seconds=0.0, total_bytes=21_000_000 + 2_857_140,
        )
        self.assertEqual(decision["action"], "continue")
        self.assertEqual(decision["reason"], "within_budget")
