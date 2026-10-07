"""Local studio. Development does not rent a GPU. Real mode uses the image API provider."""
from __future__ import annotations

import os
import threading
import time
from decimal import Decimal
from pathlib import Path

from media_engine.api.app import LocalAPIServer
from media_engine.studio.boot import budget_usd, build_controller, resolve_studio_mode


def main() -> None:
    mode = resolve_studio_mode()
    root = Path(os.environ.get("MEDIA_ENGINE_RUNTIME", "runtime")) / "studio"
    controller = build_controller(root, mode)
    port = int(os.environ.get("STUDIO_PORT", "8765"))
    server = LocalAPIServer(controller, port=port)
    ceiling = budget_usd()
    if mode == "real" and ceiling is not None:
        threading.Thread(target=_supervise, args=(controller, ceiling), daemon=True).start()
    server.start()
    print(server.url, flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()


def _supervise(controller, ceiling: Decimal) -> None:
    """Release the worker when its image job finishes, or before the budget is crossed."""
    watched: set[str] = set()
    while True:
        time.sleep(3)
        try:
            jobs = controller.jobs.list()
        except Exception:
            continue
        for job in jobs:
            status = getattr(job, "public_status", None)
            if status and status not in {"completed", "failed"}:
                watched.add(job.id)
        resource_id = controller._worker_id
        if not resource_id:
            continue
        spent = _spent(controller, resource_id)
        finished = any(
            job.id in watched and getattr(job, "public_status", None) in {"completed", "failed"}
            for job in jobs
        )
        if spent < ceiling and not finished:
            continue
        reason = "BUDGET_STOP" if spent >= ceiling else "ACCEPTANCE_SHUTDOWN"
        try:
            controller.release_worker(reason)
        except Exception:
            continue
        print({"release": reason, "spent": format(spent, "f")}, flush=True)
        watched.clear()
        if spent >= ceiling:
            return


def _spent(controller, resource_id: str) -> Decimal:
    provider = controller.provider
    started = getattr(provider, "_started_at", {}).get(str(resource_id))
    hourly = getattr(provider, "hourly_amount", None) or Decimal("0")
    if started is None:
        return Decimal("0")
    elapsed = max(0.0, time.time() - float(started))
    return Decimal(str(hourly)) * Decimal(str(elapsed)) / Decimal(3600)


if __name__ == "__main__":
    main()
