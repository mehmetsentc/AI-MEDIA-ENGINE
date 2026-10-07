"""One explicit test grant. Production databases are refused.

The command does nothing unless MEDIA_ENGINE_ALLOW_TEST_CREDITS=1.
Credit amount and the test price are arguments. Neither is stored in source.

    MEDIA_ENGINE_ALLOW_TEST_CREDITS=1 python -m media_engine.platform.test_grant \\
        --db runtime/studio/platform.sqlite3 \\
        --user nahaber \\
        --credits <units> \\
        --price-credits <units> \\
        --key final-image-1
"""
from __future__ import annotations

import argparse
import os

from media_engine.platform.repository import PlatformBlocked
from media_engine.platform.service import Platform


def prepare_test_grant(
    platform: Platform,
    *,
    user_id: str,
    credits: int,
    price_credits: int,
    idempotency_key: str,
    service: str = "image",
    model: str = "qwen-image",
) -> str:
    if os.environ.get("MEDIA_ENGINE_ALLOW_TEST_CREDITS") != "1":
        raise PlatformBlocked("TEST_CREDITS_DISABLED")
    if credits < 1 or price_credits < 1 or credits < price_credits:
        raise ValueError("test grant")
    if not idempotency_key or "/" in idempotency_key:
        raise ValueError("test grant")
    version = "test-" + idempotency_key
    platform.grant(
        user_id, "trial", credits, "admin_adjustment",
        idempotency_key="test-grant:" + idempotency_key,
    )
    current = platform.repo.price_at(service, model, platform.now_iso())
    if current is None or str(current["version"]) != version:
        platform.add_price(
            version=version, service=service, model=model, quality="test",
            unit="image", credit_price=price_credits, internal_cost_units=price_credits,
            effective_at=platform.now_iso(),
        )
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description="Grant one non-production test credit.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--credits", required=True, type=int)
    parser.add_argument("--price-credits", required=True, type=int)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    if str(args.db).startswith("postgres"):
        raise SystemExit("TEST_GRANT_REFUSES_PRODUCTION_DSN")
    platform = Platform(args.db)
    version = prepare_test_grant(
        platform, user_id=args.user, credits=args.credits, price_credits=args.price_credits,
        idempotency_key=args.key,
    )
    print({"granted": True, "pricing_version": version, "user": args.user})


if __name__ == "__main__":
    main()
