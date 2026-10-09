"""Production process wiring. PostgreSQL is required and SQLite is not opened."""
from __future__ import annotations

from dataclasses import dataclass

from media_engine.platform.postgres_conn import PgPool
from media_engine.platform.postgres_repo import PostgresRepository
from media_engine.platform.postgres_stores import (
    PostgresArtifactStore,
    PostgresAttemptStore,
    PostgresJobStore,
    PostgresResourceLedger,
    PostgresStudioStore,
    PostgresUsageStore,
)
from media_engine.platform.runtime import assert_production_postgres_ready
from media_engine.platform.service import Platform
from media_engine.safety.limits import SafetyLimits


@dataclass
class ProductionStores:
    jobs: PostgresJobStore
    attempts: PostgresAttemptStore
    ledger: PostgresResourceLedger
    usage: PostgresUsageStore
    artifacts: PostgresArtifactStore
    studio: PostgresStudioStore
    platform: Platform
    pool: PgPool


def open_production_stores(environ: dict, *, media: object = None,
                           limits: SafetyLimits | None = None) -> ProductionStores:
    dsn = assert_production_postgres_ready(environ)
    if media is None:
        from media_engine.storage.r2_media import UnavailableMedia, open_media_store
        try:
            media = open_media_store()
        except Exception:
            media = UnavailableMedia()
    pool = PgPool(dsn)
    repository = PostgresRepository(pool)
    platform = Platform(repository=repository, media=media)
    user_id = str(environ.get("DAIDOI_STUDIO_USER", "nahaber")).strip() or "nahaber"
    attempt_limit = 1 if limits is None else limits.max_job_attempts
    return ProductionStores(
        jobs=PostgresJobStore(pool),
        attempts=PostgresAttemptStore(pool, max_attempts=attempt_limit),
        ledger=PostgresResourceLedger(pool),
        usage=PostgresUsageStore(pool),
        artifacts=PostgresArtifactStore(pool),
        studio=PostgresStudioStore(pool, user_id),
        platform=platform,
        pool=pool,
    )
