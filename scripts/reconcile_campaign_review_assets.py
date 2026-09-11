#!/usr/bin/env python3
"""Reconcile exact Studio campaign outputs with human quality decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.human_review_assets import reconcile_campaign_review_assets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--studio-outputs",
        type=Path,
        default=ROOT / "data" / "studio" / "outputs.json",
        help="Persisted Studio output history JSON.",
    )
    parser.add_argument(
        "--user-review",
        type=Path,
        default=ROOT / "review-pending" / "user-quality-review-2026-08-21.v1.json",
        help="Machine-readable human quality decision ledger.",
    )
    parser.add_argument(
        "--frozen-receipts-root",
        type=Path,
        help="Optional immutable campaign checkout root containing exact quality-acceptance receipts.",
    )
    parser.add_argument(
        "--workflow",
        action="append",
        dest="workflows",
        help="Limit reconciliation to one canonical workflow ID; may be repeated.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = reconcile_campaign_review_assets(
        ROOT,
        studio_outputs_path=args.studio_outputs,
        user_review_path=args.user_review,
        workflow_ids=set(args.workflows) if args.workflows else None,
        frozen_receipts_root=args.frozen_receipts_root,
    )
    print(
        json.dumps(
            {
                "reconciled": result["reconciled"],
                "skipped": result["skipped"],
                "summary": result["index"]["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
