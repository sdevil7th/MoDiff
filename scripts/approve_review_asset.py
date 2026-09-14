#!/usr/bin/env python3
"""Approve or reject a staged review asset and stitch only after explicit approval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.human_review_assets import approve_review_asset, record_quality_review, reject_review_asset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, help="Canonical workflow ID from review-pending/index.json.")
    parser.add_argument("--reviewer", required=True, help="Name of the human reviewer.")
    parser.add_argument(
        "--approve-quality",
        action="store_true",
        help=(
            "Confirms visual/listening quality. Without --approve-rights this records quality only and "
            "does not copy, bind, register, or publish the asset."
        ),
    )
    parser.add_argument(
        "--approve-rights",
        action="store_true",
        help="With --approve-quality, confirms rights to keep and bind the output as a template example.",
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
    if args.approve_quality and not args.approve_rights:
        item = record_quality_review(
            ROOT,
            workflow_id=args.workflow,
            reviewer=args.reviewer,
            approved=True,
        )
        print(
            f"Quality-approved {item['workflowId']}; rights and publication remain pending. "
            "No review-approved copy or Gallery binding was created."
        )
        return 0
    if args.approve_rights and not args.approve_quality:
        raise SystemExit("Rights approval alone cannot bind an asset; record quality first or pass both flags.")
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
