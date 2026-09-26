#!/usr/bin/env python3
"""Extract checked-in partial Studio-spec evidence from exact gzip captures.

The command writes canonical JSON to stdout only.  It never modifies workflows,
the registered catalog, migration journals, or evidence ledgers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.composite_migration_inventory import scan_composite_migration_inventory
from modiff.huggingface_node_library import build_huggingface_node_library
from modiff.legacy_cluster_studio_spec_evidence import (
    REVIEWED_CAPTURE_SOURCES,
    LegacyStudioSpecEvidenceError,
    extract_studio_spec_evidence,
)


def _source(value: str) -> tuple[str, Path]:
    source_id, separator, path = value.partition("=")
    if not separator or source_id not in REVIEWED_CAPTURE_SOURCES or not path:
        available = ", ".join(sorted(REVIEWED_CAPTURE_SOURCES))
        raise argparse.ArgumentTypeError(f"source must be one of {available}, followed by =PATH")
    return source_id, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract evidence-only historical Studio specification bodies from exact pinned "
            "frontend captures. JSON is printed to stdout and never authorizes conversion."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data"),
        help="MoDiff data directory used for the read-only legacy inventory.",
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=_source,
        metavar="SOURCE_ID=PATH",
        help="Exact pinned capture; repeat for each retained source.",
    )
    parser.add_argument("--compact", action="store_true", help="Print canonical one-line JSON.")
    args = parser.parse_args()
    paths: dict[str, Path] = {}
    for source_id, path in args.source:
        if source_id in paths:
            parser.error(f"source {source_id} was supplied more than once")
        paths[source_id] = path
    try:
        ledger = extract_studio_spec_evidence(
            paths,
            scan_composite_migration_inventory(args.data_dir),
            build_huggingface_node_library(),
        )
    except (LegacyStudioSpecEvidenceError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    json.dump(
        ledger,
        sys.stdout,
        ensure_ascii=False,
        sort_keys=args.compact,
        separators=(",", ":") if args.compact else None,
        indent=None if args.compact else 2,
        allow_nan=False,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
