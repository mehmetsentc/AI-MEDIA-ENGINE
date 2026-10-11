"""SSH helpers for one Vast worker. Private keys stay under the runtime directory."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from media_engine.engines.image import model_cache, qwen_remote
from media_engine.providers.base import WorkerStageError


class WorkerSSH:
    def __init__(self, runtime: Path) -> None:
        self.runtime = runtime
        self.directory = runtime / "vast-ssh"
        self.key = self.directory / "id_ed25519"
        self.known = runtime / "vast_known_hosts"

    def public_key(self) -> str:
        self._ensure()
        return self.key.with_suffix(".pub").read_text(encoding="utf-8")

    def run(self, host: str, port: str, command: str, timeout: int, pulse=None) -> tuple[int, str, str]:
        """Run one remote command. pulse is the supervising thread, not a side process."""
        self._ensure()
        proc = subprocess.Popen(
            self._ssh(host, port) + [command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        started = time.monotonic()
        while True:
            if pulse is not None:
                pulse()
            remaining = float(timeout) - (time.monotonic() - started)
            if remaining <= 0:
                proc.kill()
                proc.communicate()
                raise WorkerStageError("WORKER_TIMEOUT")
            try:
                out, err = proc.communicate(timeout=min(20.0, remaining))
            except subprocess.TimeoutExpired:
                continue
            return proc.returncode or 0, out or "", err or ""

    def stage(self, host: str, port: str) -> None:
        self._ensure()
        self.run(host, port, "mkdir -p /workspace", 30)
        for module in (model_cache, qwen_remote):
            path = Path(module.__file__)
            self._copy(host, port, path, "/workspace/" + path.name)

    def _copy(self, host: str, port: str, local: Path, remote: str) -> None:
        subprocess.run(
            [
                "scp", "-i", str(self.key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={self.known}",
                "-P", port, str(local), f"root@{host}:{remote}",
            ],
            text=True, capture_output=True, timeout=120, check=True,
        )

    def _ssh(self, host: str, port: str) -> list[str]:
        return [
            "ssh", "-i", str(self.key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={self.known}",
            "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=15",
            "-p", port, "root@" + host,
        ]

    def _ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.key.exists() and self.key.with_suffix(".pub").exists():
            return
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(self.key), "-C", "ai-media-engine"],
            check=True, capture_output=True, text=True,
        )
        os.chmod(self.key, 0o600)
