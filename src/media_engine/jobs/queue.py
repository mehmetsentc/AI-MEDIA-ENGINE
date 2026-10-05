"""In-process FIFO of job ids. Durable state lives in the job store."""
from __future__ import annotations

import collections
import threading
from typing import Deque, Optional


class JobQueue:
    def __init__(self) -> None:
        self._items: Deque[str] = collections.deque()
        self._lock = threading.Lock()

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            self._items.append(job_id)

    def dequeue(self) -> Optional[str]:
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft()

    def remove(self, job_id: str) -> bool:
        with self._lock:
            try:
                self._items.remove(job_id)
            except ValueError:
                return False
            return True

    def pending(self) -> list[str]:
        with self._lock:
            return list(self._items)
