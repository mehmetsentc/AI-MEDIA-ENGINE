"""Bind one fresh verification grant. This module never creates an instance."""
from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path

from media_engine.providers.base import WorkerStageError
from media_engine.providers.vast_image import MACHINE_ID, VERIFY_RUN_BUDGET_USD, VastImageProvider
from media_engine.safety.spend_grant import SpendGrant

_FIELDS = (
    "MEDIA_ENGINE_EXECUTION_ID",
    "MEDIA_ENGINE_EXECUTION_ISSUED_AT",
    "MEDIA_ENGINE_EXECUTION_EXPIRES_AT",
    "MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD",
)


def grant_from_environ(environ: dict) -> SpendGrant | None:
    """Build a grant only from a complete explicit environment. Nothing is filled in."""
    values = {name: str(environ.get(name, "")).strip() for name in _FIELDS}
    if not any(values.values()):
        return None
    if not all(values.values()):
        raise WorkerStageError("MISSING_AUTHORIZATION")
    if values["MEDIA_ENGINE_AUTHORIZED_RUN_BUDGET_USD"] != "0.50":
        raise WorkerStageError("BUDGET_MISMATCH")
    try:
        issued = float(values["MEDIA_ENGINE_EXECUTION_ISSUED_AT"])
        expires = float(values["MEDIA_ENGINE_EXECUTION_EXPIRES_AT"])
    except ValueError as exc:
        raise WorkerStageError("EXPIRED_AUTHORIZATION") from exc
    if expires <= issued:
        raise WorkerStageError("EXPIRED_AUTHORIZATION")
    return SpendGrant(
        execution_id=values["MEDIA_ENGINE_EXECUTION_ID"],
        ceiling_usd=VERIFY_RUN_BUDGET_USD,
        max_instances=1,
        issued_at=issued,
        expires_at=expires,
        machine_id=MACHINE_ID,
    )


def claim_execution_id(path: Path, execution_id: str) -> None:
    """Remember an execution id. The same id cannot be bound again."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if path.is_file():
        seen = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if execution_id in seen:
        raise WorkerStageError("STALE_AUTHORIZATION")
    seen.add(execution_id)
    path.write_text("\n".join(sorted(seen)) + "\n", encoding="utf-8")


def prepare_provider(environ: dict, *, runtime: Path, api_key: str, transport=None, now=None) -> VastImageProvider:
    """Construct the integrated provider and bind the grant. Does not rent."""
    grant = grant_from_environ(environ)
    if grant is None:
        raise WorkerStageError("MISSING_AUTHORIZATION")
    if Decimal(grant.ceiling_usd) != Decimal("0.50"):
        raise WorkerStageError("BUDGET_MISMATCH")
    claim_execution_id(runtime / "live6-executions", grant.execution_id)
    provider = VastImageProvider(
        api_key=api_key, transport=transport, now=now, runtime=runtime,
    )
    provider.bind_execution(grant, execution_id=grant.execution_id)
    return provider


def main(argv: list[str] | None = None) -> int:
    """Print the bound grant and exit. `--execute` is refused here."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--execute" in args:
        print(json.dumps({"creates_instance": False, "error": "EXECUTE_REFUSED"}))
        return 2
    runtime = Path(os.environ.get("MEDIA_ENGINE_RUNTIME", "runtime")) / "live6"
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
        "provider": provider.name,
        "execution_id": provider.execution_id,
        "authorization_expires_at": provider.authorization_expires_at,
        "spending_deadline_wall": None,
        "ceiling_usd": str(grant.ceiling_usd) if grant is not None else None,
        "machine_id": MACHINE_ID,
        "max_instances": 1,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
