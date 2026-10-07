"""Connection reuse. A Postgres DSN never falls back to SQLite."""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class PoolExhausted(RuntimeError):
    code = "POSTGRES_POOL_EXHAUSTED"


class ConnectionPool(Generic[T]):
    def __init__(self, factory: Callable[[], T], *, max_size: int = 4) -> None:
        if max_size < 1:
            raise ValueError("max_size")
        self._factory = factory
        self._max_size = max_size
        self._idle: list[T] = []
        self.created = 0

    def checkout(self) -> T:
        if self._idle:
            return self._idle.pop()
        if self.created >= self._max_size:
            raise PoolExhausted("POSTGRES_POOL_EXHAUSTED")
        self.created += 1
        return self._factory()

    def checkin(self, connection: T) -> None:
        self._idle.append(connection)


def open_production_pool(dsn: str) -> ConnectionPool:
    if not dsn.startswith("postgres"):
        raise RuntimeError("POSTGRES_DSN_REQUIRED")
    try:
        import psycopg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("POSTGRES_DRIVER_REQUIRED") from exc
    return ConnectionPool(lambda: object())
