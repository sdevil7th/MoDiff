#!/usr/bin/env python3
"""Reconcile explicitly removed review bytes with the pending-review index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.human_review_assets import record_removed_review_assets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", required=True, help="Explicit human or policy reason for removing the cards.")
    parser.add_argument("--recovery-path", help="Optional recoverable Trash/quarantine root containing the bytes.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    index = record_removed_review_assets(
        ROOT,
        reason=args.reason,
        recovery_path=args.recovery_path,
    )
    print(
        f"Review index now has {len(index.get('items') or [])} visible item(s) and "
        f"{len(index.get('removedReviewItems') or [])} removed item record(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
