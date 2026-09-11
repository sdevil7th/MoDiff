#!/usr/bin/env python3
"""Stage one hash-bound campaign output for human review without approving it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.human_review_assets import stage_screened_campaign_asset  # noqa: E402
from modiff.local_review_receipts import loopback_base_url  # noqa: E402


def _run_document(server: str, task_id: str) -> dict:
    request = urllib.request.Request(f"{loopback_base_url(server)}/runs/{task_id}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            document = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"MoDiff returned HTTP {error.code} for task {task_id}: {detail}") from error
    if not isinstance(document, dict):
        raise RuntimeError("MoDiff returned a malformed run document.")
    return document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--comparison-card", type=Path, required=True)
    parser.add_argument("--fixture-provenance", type=Path, required=True)
    parser.add_argument("--codex-screen", type=Path, required=True)
    parser.add_argument("--studio-outputs", type=Path, default=ROOT / "data" / "studio" / "outputs.json")
    parser.add_argument("--server", default="http://127.0.0.1:8088")
    args = parser.parse_args(argv)
    result = stage_screened_campaign_asset(
        ROOT,
        workflow_id=args.workflow,
        task_id=args.task,
        media_path=args.media,
        quality_report_path=args.quality_report,
        comparison_card_path=args.comparison_card,
        fixture_provenance_path=args.fixture_provenance,
        codex_screen_path=args.codex_screen,
        studio_outputs_path=args.studio_outputs,
        run_document=_run_document(args.server, args.task),
    )
    item = result["item"]
    print(
        f"Staged {item['workflowId']} task {item['taskId']} for human review. "
        "Quality approval, rights approval, Gallery registration, and publication remain pending."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
