"""Reusable PostgreSQL connections. Callers never share one connection across threads."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from psycopg.rows import dict_row


class PgPool:
    def __init__(self, dsn: str, *, max_size: int = 4) -> None:
        self._dsn = dsn
        self._max_size = max_size
        self._lock = threading.Lock()
        self._idle: list[object] = []
        self.created = 0

    @contextmanager
    def session(self) -> Iterator[object]:
        connection = self.checkout()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self.checkin(connection)

    def checkout(self) -> object:
        with self._lock:
            if self._idle:
                return self._idle.pop()
            if self.created >= self._max_size:
                raise RuntimeError("POSTGRES_POOL_EXHAUSTED")
            self.created += 1
        try:
            return self._connect()
        except Exception:
            with self._lock:
                self.created -= 1
            raise

    def checkin(self, connection: object) -> None:
        try:
            broken = connection.closed
        except Exception:
            broken = True
        if broken:
            with self._lock:
                self.created -= 1
            return
        with self._lock:
            self._idle.append(connection)

    def close(self) -> None:
        with self._lock:
            idle = list(self._idle)
            self._idle.clear()
            self.created = 0
        for connection in idle:
            connection.close()

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
