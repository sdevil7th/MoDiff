#!/usr/bin/env python3
"""Print exact checked-in Cluster publication eligibility as JSON."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.huggingface_cluster_publication_audit import audit_current_cluster_publication_routes


def main() -> int:
    report = {
        "schemaVersion": 1,
        "kind": "huggingface_cluster_publication_audit",
        "routes": audit_current_cluster_publication_routes(),
    }
    json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
