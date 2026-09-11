#!/usr/bin/env python3
"""Print a deterministic read-only inventory of saved composite records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.composite_migration_inventory import (
    render_composite_migration_inventory,
    scan_composite_migration_inventory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="MoDiff data directory containing user-workflows/ and studio/blocks/ (default: repository data/).",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Print canonical one-line JSON instead of indented JSON."
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Return exit status 2 after printing when the report contains an error-level issue.",
    )
    args = parser.parse_args(argv)

    report = scan_composite_migration_inventory(args.data_dir)
    sys.stdout.write(render_composite_migration_inventory(report, pretty=not args.compact))
    if args.fail_on_errors and report["summary"]["errorCount"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
