"""Production metadata must be PostgreSQL. SQLite stays on the development path."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from media_engine.platform.migrate import EXPECTED_PROJECT_ID

MISSING_RUNTIME_ADAPTERS = ()


def postgres_env_file(environ: dict) -> Path:
    override = str(environ.get("MEDIA_ENGINE_POSTGRES_FILE", "")).strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / ".env.postgres"


def configured_database_url(environ: dict) -> str:
    """Return a DSN from the caller or the local env file. The value is not logged."""
    direct = str(environ.get("DATABASE_URL", "")).strip().strip('"').strip("'")
    if direct:
        return direct
    path = postgres_env_file(environ)
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        name, value = raw.split("=", 1)
        if name.strip() == "DATABASE_URL":
            return value.strip().strip('"').strip("'")
    return ""


def direct_dsn(dsn: str) -> str:
    """Use Neon's direct endpoint for transactional DDL and worker claims."""
    parsed = urlparse(dsn)
    netloc = parsed.netloc.replace("-pooler.", ".", 1)
    if netloc == parsed.netloc:
        return dsn
    return parsed._replace(netloc=netloc).geturl()


def assert_production_postgres_ready(environ: dict) -> str:
    """Return the verified DAIDOI DSN. SQLite and other projects are refused."""
    dsn = configured_database_url(environ)
    if not dsn.startswith("postgres"):
        raise RuntimeError("POSTGRES_CONFIGURATION_REQUIRED")
    if "nahaber" in _public_markers(dsn):
        raise RuntimeError("DATABASE_IDENTITY_UNVERIFIED")
    _verify_live(dsn)
    return direct_dsn(dsn)


def _verify_live(dsn: str) -> None:
    try:
        import psycopg
        from psycopg.rows import dict_row
        connection = psycopg.connect(
            direct_dsn(dsn),
            connect_timeout=8,
            prepare_threshold=None,
            autocommit=True,
            row_factory=dict_row,
            options="-c default_transaction_read_only=on -c application_name=daidoi-media-engine",
        )
    except Exception:
        raise RuntimeError("POSTGRES_CONNECTION_FAILED") from None
    try:
        row = connection.execute(
            """
            SELECT current_setting('neon.project_id') AS project_id,
                   current_database() AS database_name
            """
        ).fetchone()
    except Exception:
        raise RuntimeError("DATABASE_IDENTITY_UNVERIFIED") from None
    finally:
        connection.close()
    project = "" if row is None else str(row["project_id"])
    database = "" if row is None else str(row["database_name"])
    if project != EXPECTED_PROJECT_ID or database != "neondb" or "nahaber" in project.casefold():
        raise RuntimeError("DATABASE_IDENTITY_UNVERIFIED")


def verify_project_identity(connection: object, expected: str = EXPECTED_PROJECT_ID) -> str:
    """Confirm the Neon project. A mismatch does not reveal the other project id."""
    cursor = connection.cursor()
    cursor.execute("SELECT current_setting('neon.project_id')")
    row = cursor.fetchone()
    found = "" if not row else str(row[0])
    if found != expected or "nahaber" in found.casefold():
        raise RuntimeError("DATABASE_IDENTITY_UNVERIFIED")
    return "VERIFIED"


def _public_markers(dsn: str) -> str:
    parsed = urlparse(dsn)
    host = parsed.hostname or ""
    database = (parsed.path or "").lstrip("/")
    user = parsed.username or ""
    return f"{host} {database} {user}".casefold()
