"""Bind one LIVE-7 grant. This module never creates an instance."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from media_engine.providers.base import WorkerStageError
from media_engine.providers.live6_prepare import claim_execution_id
from media_engine.providers.live7_contract import (
    FORBIDDEN_VOLUME_ID,
    LIVE7_MACHINE_ID,
    accept_live7_grant,
    make_live7_grant,
)
from media_engine.providers.vast_image import VastImageProvider
from media_engine.safety.execution_budget import LIVE7_CEILING_USD

_FIELDS = (
    "MEDIA_ENGINE_EXECUTION_ID",
    "MEDIA_ENGINE_EXECUTION_ISSUED_AT",
    "MEDIA_ENGINE_EXECUTION_EXPIRES_AT",
    "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD",
    "MEDIA_ENGINE_EXECUTION_PROFILE",
)


def grant_from_environ(environ: dict):
    """Build a LIVE-7 grant only from a complete explicit environment."""
    values = {name: str(environ.get(name, "")).strip() for name in _FIELDS}
    if values["MEDIA_ENGINE_EXECUTION_PROFILE"] != "live7":
        raise WorkerStageError("BUDGET_MISMATCH")
    if not all(values.values()):
        raise WorkerStageError("MISSING_AUTHORIZATION")
    if values["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] != "1.50":
        raise WorkerStageError("BUDGET_MISMATCH")
    if str(environ.get("MEDIA_ENGINE_ALLOW_COLD_STAGE", "")).strip() != "1":
        raise WorkerStageError("COLD_STAGE_FORBIDDEN")
    if str(environ.get("MEDIA_ENGINE_FORBID_R2_STAGE", "")).strip() == "1":
        raise WorkerStageError("R2_FORBIDDEN")
    pinned = str(environ.get("MEDIA_ENGINE_MACHINE_ID", "")).strip()
    if pinned and pinned != str(LIVE7_MACHINE_ID):
        raise WorkerStageError("MACHINE_SCOPE")
    try:
        issued = float(values["MEDIA_ENGINE_EXECUTION_ISSUED_AT"])
        expires = float(values["MEDIA_ENGINE_EXECUTION_EXPIRES_AT"])
    except ValueError as exc:
        raise WorkerStageError("EXPIRED_AUTHORIZATION") from exc
    grant = make_live7_grant(
        execution_id=values["MEDIA_ENGINE_EXECUTION_ID"],
        issued_at=issued,
        expires_at=expires,
    )
    accept_live7_grant(grant)
    return grant


def prepare_provider(environ: dict, *, runtime: Path, api_key: str, transport=None, now=None) -> VastImageProvider:
    """Construct the provider and bind the grant. Does not rent."""
    grant = grant_from_environ(environ)
    claim_execution_id(runtime / "live7-executions", grant.execution_id)
    provider = VastImageProvider(api_key=api_key, transport=transport, now=now, runtime=runtime)
    provider.bind_execution(grant, execution_id=grant.execution_id)
    return provider


def main(argv: list[str] | None = None) -> int:
    """Print the bound grant and exit. `--execute` is refused here."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--execute" in args:
        print(json.dumps({"creates_instance": False, "error": "EXECUTE_REFUSED"}))
        return 2
    runtime = Path(os.environ.get("MEDIA_ENGINE_RUNTIME", "runtime")) / "live7"
    key = os.environ.get("VAST_API_KEY", "").strip()
    if not key:
        print(json.dumps({"creates_instance": False, "error": "KEY_MISSING"}))
        return 2
    try:
        provider = prepare_provider(dict(os.environ), runtime=runtime, api_key=key)
    except WorkerStageError as exc:
        print(json.dumps({"creates_instance": False, "error": exc.code}))
        return 2
    grant = provider.execution_grant
    print(json.dumps({
        "creates_instance": False,
        "execution_id": provider.execution_id,
        "authorization_expires_at": provider.authorization_expires_at,
        "spending_deadline_wall": None,
        "ceiling_usd": str(grant.ceiling_usd) if grant is not None else None,
        "machine_id": LIVE7_MACHINE_ID,
        "forbidden_volume_id": FORBIDDEN_VOLUME_ID,
        "volume_attached": False,
        "max_instances": 1,
        "profile": "live7",
        "expected_ceiling": str(LIVE7_CEILING_USD),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
