#!/usr/bin/env python3
"""Approve or reject a staged review asset and stitch only after explicit approval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.human_review_assets import approve_review_asset, reject_review_asset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, help="Canonical workflow ID from review-pending/index.json.")
    parser.add_argument("--reviewer", required=True, help="Name of the human reviewer.")
    parser.add_argument(
        "--approve-quality",
        action="store_true",
        help="Required with --approve-rights. Confirms visual/listening quality.",
    )
    parser.add_argument(
        "--approve-rights",
        action="store_true",
        help="Required with --approve-quality. Confirms rights to keep the output as a template example.",
    )
    parser.add_argument("--reject", action="store_true", help="Reject the staged asset instead of approving it.")
    parser.add_argument("--reason", help="Required when rejecting.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.reject:
        if args.approve_quality or args.approve_rights:
            raise SystemExit("Rejection cannot be combined with approval flags.")
        item = reject_review_asset(
            ROOT,
            workflow_id=args.workflow,
            reviewer=args.reviewer,
            reason=args.reason or "",
        )
        print(f"Rejected {item['workflowId']}. It was not registered or stitched.")
        return 0
    binding = approve_review_asset(
        ROOT,
        workflow_id=args.workflow,
        reviewer=args.reviewer,
        approve_quality=args.approve_quality,
        approve_rights=args.approve_rights,
    )
    print(
        f"Approved {binding['workflowId']} and bound it to {binding['candidateContractId']}. "
        f"Gallery Dataset registration remains false. Files are in review-approved/{binding['workflowId'].replace(':', '__')}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
