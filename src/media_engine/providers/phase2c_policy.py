"""Print the Phase 2C policy fingerprint. No network and no create."""
from __future__ import annotations

import sys

from media_engine.providers.phase2c import format_policy


def main() -> int:
    sys.stdout.write(format_policy())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
