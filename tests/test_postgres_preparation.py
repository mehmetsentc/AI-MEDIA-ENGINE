"""PostgreSQL preparation. Migrations are not applied and production does not open SQLite."""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from media_engine.db import SCHEMA
from media_engine.platform.migrate import (
    PLATFORM_TABLES,
    PRESERVED_TABLES,
    STUDIO_TABLES,
    apply_migrations,
    migration_plan,
    migration_tables,
    migration_violations,
    uncovered_sqlite_tables,
)
from media_engine.platform.runtime import (
    MISSING_RUNTIME_ADAPTERS,
    assert_production_postgres_ready,
    verify_project_identity,
)
from media_engine.platform.schema import POSTGRES_SCHEMA, SQLITE_SCHEMA
from media_engine.studio.boot import build_controller

_TABLE = re.compile(r"CREATE TABLE IF NOT EXISTS ([a-z_]+)", re.IGNORECASE)


class _Cursor:
    def __init__(self, value: str) -> None:
        self.value = value
        self.sql = ""

    def execute(self, sql: str) -> None:
        self.sql = sql

    def fetchone(self) -> tuple[str]:
        return (self.value,)


class _Connection:
    def __init__(self, value: str) -> None:
        self.cursor_value = _Cursor(value)

    def cursor(self) -> _Cursor:
        return self.cursor_value


class PostgresPreparationTests(unittest.TestCase):
    def test_migration_plan_is_forward_only_and_blocked(self) -> None:
        plan = migration_plan()
        self.assertEqual([row["version"] for row in plan], ["0001", "0002", "0003"])
        self.assertEqual(len({row["checksum"] for row in plan}), 3)
        created = migration_tables()
        self.assertTrue(set(PRESERVED_TABLES).issubset(created))
        self.assertTrue(set(PLATFORM_TABLES).issubset(created))
        self.assertTrue(set(STUDIO_TABLES).issubset(created))
        self.assertIn("schema_migrations", created)
        platform = {name.casefold() for name in _TABLE.findall(SQLITE_SCHEMA)}
        studio = {name.casefold() for name in _TABLE.findall(SCHEMA)}
        postgres = {name.casefold() for name in _TABLE.findall(POSTGRES_SCHEMA)}
        self.assertEqual(uncovered_sqlite_tables(platform, studio), [])
        self.assertTrue(postgres.issubset(created))
        self.assertEqual(migration_violations("DROP TABLE users"), ["DROP"])
        self.assertEqual(migration_violations("DELETE FROM users"), ["DELETE"])
        self.assertEqual(migration_violations("TRUNCATE credit_ledger"), ["TRUNCATE"])
        with self.assertRaises(RuntimeError) as blocked:
            apply_migrations()
        self.assertEqual(str(blocked.exception), "MIGRATION_APPLY_BLOCKED")
        sql = "\n".join(
            (Path(__file__).parents[1] / "src/media_engine/platform/migrations" / row["path"]).read_text(
                encoding="utf-8"
            )
            for row in plan
        )
        self.assertIn("credit_ledger_block_update", sql)
        self.assertIn("credit_ledger_block_delete", sql)
        self.assertIn("credit_ledger_block_truncate", sql)
        self.assertNotIn("BYTEA", sql.upper())
        self.assertIn("JSONB", sql.upper())
        for name in MISSING_RUNTIME_ADAPTERS:
            self.assertTrue(name)

    def test_production_boot_never_opens_sqlite(self) -> None:
        root = Path(tempfile.mkdtemp())
        environ = {
            "STUDIO_MODE": "production",
            "VAST_API_KEY": "unit-test-vast-secret",
            "DATABASE_URL": "postgresql://daidoi.invalid/daidoi",
        }
        with self.assertRaises(RuntimeError) as missing_adapter:
            build_controller(root, "real", environ)
        self.assertEqual(str(missing_adapter.exception), "POSTGRES_CONNECTION_FAILED")
        self.assertFalse((root / "studio.sqlite3").exists())
        self.assertFalse((root / "platform.sqlite3").exists())
        with self.assertRaises(RuntimeError) as sqlite_url:
            assert_production_postgres_ready({"DATABASE_URL": "sqlite:///tmp/daidoi.sqlite3"})
        self.assertEqual(str(sqlite_url.exception), "POSTGRES_CONFIGURATION_REQUIRED")
        self.assertFalse(Path("/tmp/daidoi.sqlite3").exists())
        with self.assertRaises(RuntimeError) as foreign:
            assert_production_postgres_ready({
                "DATABASE_URL": "postgresql://owner@nahaber.example/media",
            })
        self.assertEqual(str(foreign.exception), "DATABASE_IDENTITY_UNVERIFIED")
        self.assertNotIn("owner", str(foreign.exception))

    def test_project_identity_hides_a_mismatch(self) -> None:
        other = _Connection("some-other-project")
        with self.assertRaises(RuntimeError) as mismatch:
            verify_project_identity(other)
        self.assertEqual(str(mismatch.exception), "DATABASE_IDENTITY_UNVERIFIED")
        self.assertNotIn("some-other-project", str(mismatch.exception))
        self.assertIn("neon.project_id", other.cursor_value.sql)
        self.assertNotIn("*", other.cursor_value.sql)
        matched = _Connection("green-poetry-65686525")
        self.assertEqual(verify_project_identity(matched), "VERIFIED")
