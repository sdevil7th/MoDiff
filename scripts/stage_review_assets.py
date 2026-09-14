#!/usr/bin/env python3
"""Copy validated local-review outputs into review-pending/ for human approval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.human_review_assets import stage_review_assets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    document = stage_review_assets(ROOT)
    pending = ROOT / "review-pending"
    summary = document["summary"]
    print(
        f"Staged {summary['pendingCount']} candidates into {pending} "
        f"({summary['blockedGenerationCount']} execution-pending workflows were not generated). "
        f"Open {pending / 'index.html'} to review. Nothing was Gallery-registered."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
