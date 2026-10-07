"""Local studio. Uses the in-process preview engine and does not rent a GPU."""
from __future__ import annotations

import os
import time
from pathlib import Path

from media_engine.api.app import LocalAPIServer
from media_engine.orchestrator.controller import MediaController, WallClock
from media_engine.providers.fake import FakeGPUProvider
from media_engine.studio.dev_engine import DevImageEngine


def main() -> None:
    root = Path(os.environ.get("MEDIA_ENGINE_RUNTIME", "runtime")) / "studio"
    root.mkdir(parents=True, exist_ok=True)
    clock = WallClock()
    controller = MediaController(
        str(root / "studio.sqlite3"),
        str(root / "artifacts"),
        clock=clock,
        provider=FakeGPUProvider(time.time),
        image_engine=DevImageEngine(),
    )
    port = int(os.environ.get("STUDIO_PORT", "8765"))
    server = LocalAPIServer(controller, port=port)
    server.start()
    print(server.url, flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
