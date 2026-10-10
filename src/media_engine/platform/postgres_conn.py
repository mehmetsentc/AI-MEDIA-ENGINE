"""Short PostgreSQL sessions. Idle connections are not reused after GPU work."""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

from psycopg.rows import dict_row

logger = logging.getLogger("media_engine.postgres")

IDLE_MAX_SECONDS = 20.0
_BACKOFF_SECONDS = (0.05, 0.2, 0.8)
_TRANSIENT_STATES = frozenset({
    "57P01", "57P02", "57P03",
    "08000", "08001", "08003", "08004", "08006", "08007",
})
_TRANSIENT_NAMES = frozenset({
    "AdminShutdown", "CrashShutdown", "CannotConnectNow",
    "ConnectionDoesNotExist", "InterfaceError", "OperationalError",
})


class PostgresUnavailable(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def is_transient_connection(exc: BaseException) -> bool:
    """AdminShutdown and broken sockets. The message is not logged."""
    sqlstate = str(getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None) or "")
    if sqlstate in _TRANSIENT_STATES:
        return True
    return type(exc).__name__ in _TRANSIENT_NAMES


class PgPool:
    def __init__(self, dsn: str, *, max_size: int = 4, sleep: Callable[[float], None] | None = None) -> None:
        self._dsn = dsn
        self._max_size = max_size
        self._sleep = sleep or time.sleep
        self._lock = threading.Lock()
        self._idle: list[tuple[object, float]] = []
        self.created = 0
        self.deadline: float | None = None

    @contextmanager
    def session(self) -> Iterator[object]:
        """One transaction. The body is not replayed."""
        connection = self.checkout()
        try:
            yield connection
            connection.commit()
        except Exception as exc:
            rolled = self._rollback_quietly(connection)
            if is_transient_connection(exc) or not rolled or not self._alive(connection):
                self._retire(connection)
                if is_transient_connection(exc):
                    self._log(exc)
            else:
                self.checkin(connection)
            raise
        else:
            self.checkin(connection)

    def run(self, operation: Callable[[object], object], *, idempotent: bool = False) -> object:
        """Retry a whole transaction only when the caller guarantees one outcome."""
        limit = 4 if idempotent else 1
        last: BaseException | None = None
        for attempt in range(limit):
            if attempt and self._past_deadline():
                raise PostgresUnavailable("SESSION_DEADLINE") from last
            if attempt:
                self._sleep(_BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)])
                if self._past_deadline():
                    raise PostgresUnavailable("SESSION_DEADLINE") from last
            connection = self.checkout()
            try:
                result = operation(connection)
                connection.commit()
            except Exception as exc:
                rolled = self._rollback_quietly(connection)
                broken = is_transient_connection(exc) or not rolled or not self._alive(connection)
                if broken:
                    self._retire(connection)
                else:
                    self.checkin(connection)
                last = exc
                if not broken or not idempotent or attempt + 1 == limit:
                    if is_transient_connection(exc):
                        self._log(exc)
                        raise PostgresUnavailable("POSTGRES_CONNECTION_FAILED") from exc
                    raise
                self._log(exc)
                continue
            self.checkin(connection)
            return result
        raise PostgresUnavailable("POSTGRES_CONNECTION_FAILED") from last

    def checkout(self) -> object:
        while True:
            connection = self._pop_idle()
            if connection is None:
                break
            if self._usable(connection):
                return connection
            self._retire(connection)
        return self._open()

    def checkin(self, connection: object) -> None:
        if not self._alive(connection):
            self._retire(connection)
            return
        with self._lock:
            self._idle.append((connection, time.monotonic()))

    def close(self) -> None:
        with self._lock:
            idle = [connection for connection, _when in self._idle]
            self._idle.clear()
            self.created = 0
        for connection in idle:
            try:
                connection.close()
            except Exception:
                continue

    def _pop_idle(self) -> object | None:
        while True:
            with self._lock:
                if not self._idle:
                    return None
                connection, idle_at = self._idle.pop()
            if time.monotonic() - idle_at > IDLE_MAX_SECONDS:
                self._retire(connection)
                continue
            return connection

    def _usable(self, connection: object) -> bool:
        if not self._alive(connection):
            return False
        try:
            connection.execute("SELECT 1")
            connection.commit()
        except Exception as exc:
            self._log(exc)
            return False
        return True

    def _open(self) -> object:
        with self._lock:
            if self.created >= self._max_size:
                raise RuntimeError("POSTGRES_POOL_EXHAUSTED")
            self.created += 1
        try:
            return self._connect()
        except Exception as exc:
            with self._lock:
                self.created = max(0, self.created - 1)
            if is_transient_connection(exc):
                self._log(exc)
                raise PostgresUnavailable("POSTGRES_CONNECTION_FAILED") from exc
            raise

    def _retire(self, connection: object) -> None:
        try:
            connection.close()
        except Exception:
            pass
        with self._lock:
            self.created = max(0, self.created - 1)

    def _alive(self, connection: object) -> bool:
        try:
            return not bool(connection.closed)
        except Exception:
            return False

    def _rollback_quietly(self, connection: object) -> bool:
        try:
            connection.rollback()
        except Exception as exc:
            self._log(exc)
            return False
        return True

    def _past_deadline(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() >= float(self.deadline)

    def _log(self, exc: BaseException) -> None:
        sqlstate = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
        logger.warning(
            "postgres_connection_failed error_type=%s sqlstate=%s",
            type(exc).__name__,
            sqlstate or "",
        )

    def _connect(self) -> object:
        import psycopg
        return psycopg.connect(
            self._dsn,
            connect_timeout=8,
            prepare_threshold=None,
            autocommit=False,
            row_factory=dict_row,
            options="-c application_name=daidoi-media-engine",
        )
