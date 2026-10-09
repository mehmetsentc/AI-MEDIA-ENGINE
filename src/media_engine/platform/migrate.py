"""Forward-only PostgreSQL migrations. Application requires the verified DAIDOI project."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

MIGRATION_DIR = Path(__file__).resolve().parent / "migrations"
EXPECTED_PROJECT_ID = "green-poetry-65686525"

PRESERVED_TABLES = (
    "users",
    "projects",
    "scenes",
    "assets",
    "generation_jobs",
    "artifacts",
    "usage_events",
    "provider_runs",
    "project_exports",
    "credit_accounts",
    "credit_ledger",
    "pricing_rules",
    "continuity_states",
)

PLATFORM_TABLES = (
    "usage_meters",
    "durable_jobs",
    "media_artifacts",
    "media_orphans",
    "kill_switches",
    "plans",
    "entitlements",
    "payment_transactions",
    "webhook_events",
    "subscriptions",
    "cost_facts",
    "benchmarks",
    "platform_config",
    "signed_urls",
    "credit_pools",
    "ledger_entry_types",
)

STUDIO_TABLES = (
    "jobs",
    "job_attempts",
    "external_resources",
    "host_quarantine",
    "studio_assets",
    "job_artifacts",
    "job_usage",
)

SQLITE_NAME_TARGETS = {
    "artifacts": frozenset({"artifacts", "media_artifacts", "job_artifacts"}),
    "usage_events": frozenset({"usage_events", "usage_meters", "job_usage"}),
}

_TABLE = re.compile(r"CREATE TABLE IF NOT EXISTS ([a-z_]+)", re.IGNORECASE)


def migration_plan() -> list[dict[str, str]]:
    """Versioned files in application order. Checksums cover the file bytes."""
    rows: list[dict[str, str]] = []
    for path in sorted(MIGRATION_DIR.glob("*.sql")):
        version, _, name = path.name.partition("_")
        version = version.removesuffix(".sql")
        text = path.read_text(encoding="utf-8")
        violations = migration_violations(text)
        if violations:
            raise RuntimeError("MIGRATION_NOT_FORWARD_ONLY")
        rows.append({
            "version": version,
            "name": name.removesuffix(".sql") or version,
            "checksum": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "path": path.name,
        })
    if [row["version"] for row in rows] != sorted(row["version"] for row in rows):
        raise RuntimeError("MIGRATION_ORDER")
    return rows


def migration_tables() -> set[str]:
    names: set[str] = set()
    for path in MIGRATION_DIR.glob("*.sql"):
        names.update(name.casefold() for name in _TABLE.findall(path.read_text(encoding="utf-8")))
    return names


def uncovered_sqlite_tables(platform_tables: set[str], studio_tables: set[str]) -> list[str]:
    created = migration_tables()
    missing: list[str] = []
    for name in sorted(platform_tables | studio_tables):
        targets = SQLITE_NAME_TARGETS.get(name, frozenset({name}))
        if targets.isdisjoint(created):
            missing.append(name)
    return missing


def apply_migrations(connection: object = None, *, confirm_project: str = "") -> list[str]:
    """Apply missing versions. A call without the verified project id does nothing."""
    if connection is None or confirm_project != EXPECTED_PROJECT_ID:
        raise RuntimeError("MIGRATION_APPLY_BLOCKED")
    connection.rollback()
    _require_identity(connection)
    present = _public_tables(connection)
    recorded = _recorded_versions(connection) if "schema_migrations" in present else {}
    if present and "schema_migrations" not in present:
        raise RuntimeError("SCHEMA_NOT_EMPTY")
    connection.rollback()
    applied: list[str] = []
    for item in migration_plan():
        version = item["version"]
        if version in recorded:
            if recorded[version] != item["checksum"]:
                raise RuntimeError("MIGRATION_CHECKSUM_MISMATCH")
            continue
        text = (MIGRATION_DIR / item["path"]).read_text(encoding="utf-8")
        if migration_violations(text):
            raise RuntimeError("MIGRATION_NOT_FORWARD_ONLY")
        with connection.transaction():
            for statement in _statements(text):
                if statement.strip():
                    connection.execute(statement)
            connection.execute(
                """
                INSERT INTO schema_migrations (version, name, checksum)
                VALUES (%s, %s, %s)
                """,
                (version, item["name"], item["checksum"]),
            )
        applied.append(version)
    return applied


def _require_identity(connection: object) -> None:
    row = connection.execute(
        """
        SELECT current_setting('neon.project_id') AS project_id,
               current_database() AS database_name
        """
    ).fetchone()
    project = "" if row is None else str(row["project_id"])
    database = "" if row is None else str(row["database_name"])
    if project != EXPECTED_PROJECT_ID or database != "neondb" or "nahaber" in project.casefold():
        raise RuntimeError("DATABASE_IDENTITY_UNVERIFIED")


def _public_tables(connection: object) -> set[str]:
    rows = connection.execute(
        """
        SELECT c.relname AS table_name
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r'
        """
    ).fetchall()
    return {str(row["table_name"]) for row in rows}


def _recorded_versions(connection: object) -> dict[str, str]:
    rows = connection.execute(
        "SELECT version, checksum FROM schema_migrations"
    ).fetchall()
    return {str(row["version"]): str(row["checksum"]) for row in rows}


def migration_violations(sql: str) -> list[str]:
    violations: list[str] = []
    for statement in _statements(sql):
        compact = " ".join(statement.split())
        upper = compact.upper()
        if not upper:
            continue
        if re.search(r"\bDROP\b", upper):
            violations.append("DROP")
        if re.search(r"\bDELETE\b", upper) and "BEFORE DELETE" not in upper:
            violations.append("DELETE")
        if re.search(r"\bTRUNCATE\b", upper) and "BEFORE TRUNCATE" not in upper:
            violations.append("TRUNCATE")
        if re.search(r"\bUPDATE\b", upper) and "BEFORE UPDATE" not in upper:
            violations.append("UPDATE")
        if upper.startswith("INSERT") and "ON CONFLICT" not in upper:
            violations.append("INSERT_WITHOUT_CONFLICT")
        if upper.startswith("CREATE TABLE") and "IF NOT EXISTS" not in upper:
            violations.append("CREATE_TABLE_NOT_REPEATABLE")
        if upper.startswith("CREATE INDEX") or upper.startswith("CREATE UNIQUE INDEX"):
            if "IF NOT EXISTS" not in upper:
                violations.append("CREATE_INDEX_NOT_REPEATABLE")
        if upper.startswith("ALTER TABLE") and "ADD COLUMN" in upper and "IF NOT EXISTS" not in upper:
            violations.append("ADD_COLUMN_NOT_REPEATABLE")
        if "BYTEA" in upper or "BLOB" in upper:
            violations.append("MEDIA_BLOB")
    return violations


def _statements(sql: str) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    index = 0
    while index < len(sql):
        if sql.startswith("$$", index):
            end = sql.find("$$", index + 2)
            if end < 0:
                buffer.append(sql[index:])
                break
            buffer.append(sql[index:end + 2])
            index = end + 2
            continue
        if sql[index] == ";":
            parts.append("".join(buffer))
            buffer = []
            index += 1
            continue
        buffer.append(sql[index])
        index += 1
    tail = "".join(buffer).strip()
    if tail:
        parts.append(tail)
    return parts


def apply_isolated_migrations(connection: object) -> list[str]:
    """Apply the reviewed SQL to a local database whose name starts with daidoi_live7_."""
    row = connection.execute("SELECT current_database() AS name, inet_server_addr()::text AS addr").fetchone()
    name = row["name"] if isinstance(row, dict) else row[0]
    addr = row["addr"] if isinstance(row, dict) else row[1]
    if not str(name).startswith("daidoi_live7_"):
        raise RuntimeError("ISOLATED_DATABASE_REQUIRED")
    if addr not in (None, "", "127.0.0.1", "::1"):
        raise RuntimeError("ISOLATED_DATABASE_REQUIRED")
    applied: list[str] = []
    for item in migration_plan():
        path = MIGRATION_DIR / item["path"]
        for statement in _statements(path.read_text(encoding="utf-8")):
            if statement.strip():
                connection.execute(statement)
        connection.execute(
            """
            INSERT INTO schema_migrations (version, name, checksum, applied_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (version) DO NOTHING
            """,
            (item["version"], item["name"], item["checksum"]),
        )
        applied.append(item["version"])
    connection.commit()
    return applied
