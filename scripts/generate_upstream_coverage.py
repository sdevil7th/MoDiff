#!/usr/bin/env python3
"""Generate or verify the deterministic no-weight upstream coverage ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.upstream_coverage import (  # noqa: E402
    UPSTREAM_COVERAGE_PATH,
    build_upstream_coverage,
    render_upstream_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify reviewed upstream source without importing models or downloading weights."
    )
    parser.add_argument("--check", action="store_true", help="Fail unless the checked-in ledger is current.")
    parser.add_argument(
        "--diffusers-source",
        type=Path,
        help="Exact pinned Diffusers Git checkout/package (defaults to the installed package).",
    )
    parser.add_argument(
        "--transformers-source",
        type=Path,
        required=True,
        help="Exact reviewed Transformers main Git checkout.",
    )
    parser.add_argument(
        "--transformers-wheel",
        type=Path,
        required=True,
        help="Exact locked production Transformers wheel; it is inspected as a zip and never installed.",
    )
    parser.add_argument("--output", type=Path, default=UPSTREAM_COVERAGE_PATH)
    args = parser.parse_args()
    rendered = render_upstream_coverage(
        build_upstream_coverage(
            ROOT,
            diffusers_source=args.diffusers_source,
            transformers_source=args.transformers_source,
            transformers_wheel=args.transformers_wheel,
        )
    )
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError:
            current = ""
        if current != rendered:
            raise SystemExit(f"{args.output} is stale; regenerate it with {Path(__file__).name}.")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
