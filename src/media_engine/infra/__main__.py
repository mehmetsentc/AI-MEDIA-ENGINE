"""Long-running GPU manager. Paid create is disabled and this process never rents.

Run it without Cursor:

    PYTHONPATH=src python -m media_engine.infra

Discovery transport is injected by the operator. The default loop only reconciles
an empty inventory and exits when MEDIA_ENGINE_GPU_ONCE=1.
"""
from __future__ import annotations

import os
import time
from decimal import Decimal
from pathlib import Path

from media_engine.infra.manager import GpuManager
from media_engine.infra.policies import PAID_ACTIONS_DISABLED
from media_engine.infra.store import ManagerStore
from media_engine.infra.vast_read import VastReadOnly, offline_transport


def main() -> int:
    root = Path(os.environ.get("MEDIA_ENGINE_GPU_STATE", "runtime/gpu-manager.sqlite3"))
    store = ManagerStore(root)
    manager = GpuManager(store, now=time.time)
    decision = manager.evaluate_capacity("1h", None)
    if decision["allowed"] or decision["reason"] != PAID_ACTIONS_DISABLED:
        store.close()
        return 2
    interval = float(os.environ.get("MEDIA_ENGINE_GPU_POLL_SECONDS", "60"))
    once = os.environ.get("MEDIA_ENGINE_GPU_ONCE", "1") == "1"
    while True:
        if manager.poll.allow(time.time()):
            manager.pull(VastReadOnly(offline_transport), ceiling=Decimal("5.00"))
        if once:
            break
        time.sleep(max(1.0, interval))
    store.close()
    print({"paid_actions_enabled": False, "state": manager.state})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
