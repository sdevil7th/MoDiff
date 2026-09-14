#!/usr/bin/env python3
"""Print the exact catalog-only and legacy-Cluster evidence backlog as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.composite_migration_recovery_audit import scan_registered_cluster_recovery_audit
from modiff.huggingface_catalog_only_gates import audit_huggingface_catalog_only_definitions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit evidence-blocked Hugging Face Cluster definitions and legacy manifests."
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data"),
        help="MoDiff data directory containing user-workflows (default: repository data directory).",
    )
    args = parser.parse_args()
    report = {
        "schemaVersion": 1,
        "kind": "huggingface_cluster_backlog_audit",
        "catalogOnly": audit_huggingface_catalog_only_definitions(),
        "legacyRecovery": scan_registered_cluster_recovery_audit(args.data_dir),
    }
    json.dump(report, sys.stdout, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

