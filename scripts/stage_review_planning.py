#!/usr/bin/env python3
"""Stage review-planning ledgers without rerunning graphs or mutating sealed specs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.generic_task_gap_inventory import write_generic_task_gap_inventory
from modiff.original_input_fixtures import write_original_input_fixtures
from modiff.public_template_reconciliation import write_public_template_reconciliation


def _planning_html(reconciliation: dict, fixtures: dict, gaps: dict) -> str:
    rec = reconciliation["summary"]
    fix = fixtures["summary"]
    gap = gaps["summary"]
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\" />"
        "<title>MoDiff review planning</title>"
        "<style>body{font-family:sans-serif;max-width:880px;margin:24px auto;padding:0 16px}"
        "a{color:#124}section{margin:0 0 28px;padding:0 0 16px;border-bottom:1px solid #ccc}</style>"
        "</head><body>"
        "<h1>Review planning</h1>"
        "<p>No graph was submitted. The sealed authoring ledger was not mutated. "
        "Gallery and Dataset were not changed.</p>"
        "<section><h2>1. Public templates (77)</h2>"
        f"<p>Keep {rec['currentKeepCount']} current historical examples. "
        f"Plan canaries for {rec['staleCanaryCount']} stale templates and "
        f"{rec['neverHadExampleCount']} that never had an example.</p>"
        "<p><a href=\"public-templates/index.html\">Open reconciliation</a></p></section>"
        "<section><h2>2. Original input fixtures (92)</h2>"
        f"<p>{fix['workflowCount']} hidden candidates now have original procedural inputs "
        f"from a {fix['kitAssetCount']}-file kit. Rights review is still required. "
        "Nothing was selected into the authoring ledger.</p>"
        "<p><a href=\"input-fixtures/index.html\">Open fixtures</a></p></section>"
        "<section><h2>3. Remaining generic-task gaps</h2>"
        f"<p>{gap['newTaskBoundaryCount']} new Comfy-derived task boundaries still need a "
        "generic contract plus a model. Zero remaining tasks are authorable without weights. "
        "Image stitch is already implemented and waiting for human review.</p>"
        "<p><a href=\"generic-task-gaps/index.html\">Open inventory</a></p></section>"
        "<section><h2>Still waiting on you</h2>"
        "<ul>"
        "<li>Quality and rights review of the 38 staged local outputs</li>"
        "<li>Which stale/never-had public templates to canary on a qualification host</li>"
        "<li>Which of the 28 new generic tasks to author first</li>"
        "<li>Hub token, disk, and accelerator for P2.5 live proof</li>"
        "</ul></section>"
        "</body></html>\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    pending = ROOT / "review-pending"
    pending.mkdir(parents=True, exist_ok=True)
    reconciliation = write_public_template_reconciliation(ROOT)
    fixtures = write_original_input_fixtures(ROOT)
    gaps = write_generic_task_gap_inventory(ROOT)
    (pending / "planning.html").write_text(_planning_html(reconciliation, fixtures, gaps), encoding="utf-8")
    rec = reconciliation["summary"]
    print(
        f"Reconciled {rec['templateCount']} public templates "
        f"({rec['currentKeepCount']} keep, {rec['staleCanaryCount']} stale canary, "
        f"{rec['neverHadExampleCount']} never-had-example). "
        f"Staged original fixtures for {fixtures['summary']['workflowCount']} hidden candidates. "
        f"{gaps['summary']['newTaskBoundaryCount']} new generic tasks still need models. "
        f"Open {pending / 'planning.html'}. Nothing was Gallery-registered or re-run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
