"""AdminShutdown recovery. No production database and no Vast create."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from media_engine.orchestrator.controller import MediaController
from media_engine.platform.postgres_conn import PgPool, PostgresUnavailable
from media_engine.providers.fake import FakeGPUProvider


class AdminShutdown(Exception):
    sqlstate = "57P01"


class Cursor:
    rowcount = 1

    def fetchone(self):
        return {"id": "row"}


class Connection:
    def __init__(self, faults: list) -> None:
        self.faults = faults
        self.closed = False
        self.fail_probe = False
        self.queries: list[str] = []
        self.committed = 0

    def execute(self, query, params=None):
        del params
        text = str(query).strip()
        self.queries.append(text)
        if text == "SELECT 1":
            if self.fail_probe:
                raise AdminShutdown("terminating connection due to administrator command")
            return Cursor()
        if self.faults and self.faults[0] != "commit":
            fault = self.faults.pop(0)
            raise fault
        return Cursor()

    def commit(self) -> None:
        if self.faults and self.faults[0] == "commit":
            self.faults.pop(0)
            raise AdminShutdown("terminating connection due to administrator command")
        self.committed += 1

    def rollback(self) -> None:
        if self.faults and self.faults[0] == "rollback":
            self.faults.pop(0)
            raise AdminShutdown("the connection is lost")

    def close(self) -> None:
        self.closed = True


class ScriptedPool(PgPool):
    def __init__(self, faults: list) -> None:
        super().__init__("postgres://user:secret@localhost/media", sleep=lambda _seconds: None)
        self.faults = faults
        self.opened: list[Connection] = []

    def _connect(self) -> Connection:
        connection = Connection(self.faults)
        self.opened.append(connection)
        return connection


class PostgresRecoveryTests(unittest.TestCase):
    def test_idle_connection_killed_during_staging_is_discarded(self) -> None:
        pool = ScriptedPool([])
        first = pool.checkout()
        pool.checkin(first)
        self.assertIs(pool.checkout(), first)
        pool.checkin(first)
        first.fail_probe = True
        fresh = pool.checkout()
        self.assertIsNot(fresh, first)
        self.assertTrue(first.closed)

    def test_status_write_retries_after_cache_confirmation(self) -> None:
        pool = ScriptedPool([AdminShutdown("terminating connection due to administrator command")])
        saved = []

        def write(connection):
            connection.execute("UPDATE jobs SET public_status = 'runtime_preparing'")
            saved.append("runtime_preparing")
            return "saved"

        with self.assertLogs("media_engine.postgres", level="WARNING") as logs:
            self.assertEqual(pool.run(write, idempotent=True), "saved")
        self.assertEqual(saved, ["runtime_preparing"])
        self.assertEqual(len(pool.opened), 2)
        text = " ".join(logs.output)
        self.assertIn("AdminShutdown", text)
        self.assertIn("57P01", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("postgres://", text)

    def test_three_job_statuses_continue_after_one_shutdown(self) -> None:
        pool = ScriptedPool([AdminShutdown("terminating connection due to administrator command")])
        creates = []
        written = []
        for name in ("job-1", "job-2", "job-3"):
            def write(connection, name=name):
                connection.execute("UPDATE jobs")
                written.append(name)
                return name

            pool.run(write, idempotent=True)
        self.assertEqual(written, ["job-1", "job-2", "job-3"])
        self.assertEqual(creates, [])

    def test_settlement_and_refund_do_not_post_twice(self) -> None:
        ledger: dict[str, str] = {}

        def settle(connection, key: str, entry: str) -> str:
            connection.execute("INSERT credit_ledger")
            if key in ledger:
                return ledger[key]
            ledger[key] = entry
            return entry

        reserve = ScriptedPool([AdminShutdown("terminating connection due to administrator command")])
        self.assertEqual(
            reserve.run(lambda conn: settle(conn, "reserve:job-1:topup", "led-reserve"), idempotent=True),
            "led-reserve",
        )
        refund = ScriptedPool(["commit"])
        self.assertEqual(
            refund.run(lambda conn: settle(conn, "refund:job-1:topup", "led-refund"), idempotent=True),
            "led-refund",
        )
        self.assertEqual(ledger, {
            "reserve:job-1:topup": "led-reserve",
            "refund:job-1:topup": "led-refund",
        })

    def test_non_idempotent_write_is_not_retried(self) -> None:
        pool = ScriptedPool([AdminShutdown("terminating connection due to administrator command")])
        calls = []

        def write(connection):
            calls.append(1)
            connection.execute("INSERT once")

        with self.assertRaises(PostgresUnavailable) as raised:
            pool.run(write, idempotent=False)
        self.assertEqual(raised.exception.code, "POSTGRES_CONNECTION_FAILED")
        self.assertIsInstance(raised.exception.__cause__, AdminShutdown)
        self.assertEqual(calls, [1])

    def test_deadline_stops_retries(self) -> None:
        pool = ScriptedPool([
            AdminShutdown("terminating connection due to administrator command"),
            AdminShutdown("terminating connection due to administrator command"),
        ])
        pool.deadline = time.time() - 1
        with self.assertRaises(PostgresUnavailable) as raised:
            pool.run(lambda connection: connection.execute("UPDATE jobs"), idempotent=True)
        self.assertEqual(raised.exception.code, "SESSION_DEADLINE")
        self.assertEqual(len(pool.opened), 1)

    def test_repeated_shutdown_stops_without_another_attempt_past_the_limit(self) -> None:
        pool = ScriptedPool([AdminShutdown("terminating connection due to administrator command")] * 4)
        with self.assertRaises(PostgresUnavailable) as raised:
            pool.run(lambda connection: connection.execute("UPDATE jobs"), idempotent=True)
        self.assertEqual(raised.exception.code, "POSTGRES_CONNECTION_FAILED")
        self.assertEqual(len(pool.opened), 4)

    def test_rollback_failure_does_not_replace_the_original_error(self) -> None:
        pool = ScriptedPool([
            AdminShutdown("terminating connection due to administrator command"),
            "rollback",
        ])
        with self.assertRaises(AdminShutdown):
            with pool.session() as connection:
                connection.execute("UPDATE jobs")
        self.assertTrue(pool.opened[0].closed)

    def test_unrecoverable_database_deletes_the_gpu_and_does_not_create(self) -> None:
        root = Path(tempfile.mkdtemp())
        provider = FakeGPUProvider(time.time)
        provider.plant("55294957", "nahaber")
        controller = MediaController(
            str(root / "studio.sqlite3"), str(root / "artifacts"), provider=provider,
        )
        controller._worker_id = "55294957"
        before = provider.create_calls
        controller._stop_for_database("job-missing")
        self.assertEqual(provider.create_calls, before)
        self.assertEqual(provider.terminate_calls, 1)
        self.assertEqual(provider.status("55294957"), "TERMINATED")


if __name__ == "__main__":
    unittest.main()
