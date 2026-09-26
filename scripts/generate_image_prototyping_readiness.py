#!/usr/bin/env python3
"""Generate or verify the deterministic image prototyping route ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.image_prototyping_readiness import (  # noqa: E402
    IMAGE_PROTOTYPING_READINESS_PATH,
    build_image_prototyping_readiness,
    render_image_prototyping_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail unless the checked-in ledger is current.")
    parser.add_argument("--output", type=Path, default=IMAGE_PROTOTYPING_READINESS_PATH)
    args = parser.parse_args()
    rendered = render_image_prototyping_readiness(build_image_prototyping_readiness(ROOT))
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError:
            current = ""
        if current != rendered:
            raise SystemExit(f"{args.output} is stale; regenerate it with {Path(__file__).name}.")
        return 0
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
