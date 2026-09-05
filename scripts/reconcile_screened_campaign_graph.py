#!/usr/bin/env python3
"""Rebind a pending shortlist after a proven non-execution graph change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.human_review_assets import load_pending_index, reconcile_screened_campaign_graph  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--prior-ref", required=True)
    args = parser.parse_args(argv)
    index = load_pending_index(ROOT)
    matches = [item for item in index.get("items") or [] if item.get("workflowId") == args.workflow]
    if len(matches) != 1:
        raise SystemExit(f"Pending review workflow was not found exactly once: {args.workflow}")
    graph_path = matches[0]["graphPath"]
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", "--end-of-options", f"{args.prior_ref}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    prior_bytes = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:data/graphs/{graph_path}"],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    result = reconcile_screened_campaign_graph(
        ROOT,
        workflow_id=args.workflow,
        prior_graph=json.loads(prior_bytes),
    )
    reconciliation = result["receipt"]["canonicalGraphReconciliation"]
    print(
        f"Reconciled {args.workflow}: {reconciliation['priorSha256']} -> "
        f"{reconciliation['currentSha256']} with unchanged execution semantics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
