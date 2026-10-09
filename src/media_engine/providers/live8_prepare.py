"""Bind one dynamic authorization. This module never creates an instance."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from media_engine.providers.base import WorkerStageError
from media_engine.providers.execution_authorization import (
    ExecutionAuthorization,
    accept_record,
    authorization_from_json,
)
from media_engine.providers.live6_prepare import claim_execution_id
from media_engine.providers.vast_image import VastImageProvider


def head_sha(root: Path | None = None) -> str:
    """The commit that is about to run. An authorization for another commit is refused."""
    repo = Path(__file__).resolve().parents[3] if root is None else root
    try:
        raw = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WorkerStageError("STALE_AUTHORIZATION") from exc
    if len(raw) != 40:
        raise WorkerStageError("STALE_AUTHORIZATION")
    return raw


def bind_from_environ(
    provider: VastImageProvider,
    environ: dict,
    *,
    runtime: Path,
    code_sha: str,
) -> ExecutionAuthorization:
    """Load the authorization file, claim it once, and attach it. Does not rent."""
    path = str(environ.get("MEDIA_ENGINE_AUTHORIZATION_FILE", "")).strip()
    if not path:
        raise WorkerStageError("MISSING_AUTHORIZATION")
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkerStageError("MISSING_AUTHORIZATION") from exc
    authorization = authorization_from_json(text)
    accept_record(authorization, now=float(provider._now()), code_sha=code_sha)
    claim_execution_id(runtime / "live8-authorizations", authorization.authorization_id)
    claim_execution_id(runtime / "live8-executions", authorization.execution_id)
    provider.bind_authorization(authorization, code_sha=code_sha)
    return authorization


def main(argv: list[str] | None = None) -> int:
    """Print the bound authorization and exit. `--execute` is refused here."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--execute" in args:
        print(json.dumps({"creates_instance": False, "error": "EXECUTE_REFUSED"}))
        return 2
    runtime = Path(os.environ.get("MEDIA_ENGINE_RUNTIME", "runtime")) / "live8"
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        print(json.dumps({"creates_instance": False, "error": "KEY_MISSING"}))
        return 2
    provider = VastImageProvider(api_key=key, runtime=runtime)
    try:
        authorization = bind_from_environ(provider, dict(os.environ), runtime=runtime, code_sha=head_sha())
    except WorkerStageError as exc:
        print(json.dumps({"creates_instance": False, "error": exc.code}))
        return 2
    print(json.dumps({
        "creates_instance": False,
        "authorization_id": authorization.authorization_id,
        "offers": list(authorization.allowed_offer_ids),
        "machines": list(authorization.allowed_machine_ids),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
