#!/usr/bin/env python3
"""Generate or validate the non-public template candidate contract ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.template_candidate_contracts import (  # noqa: E402
    TEMPLATE_CANDIDATE_CONTRACT_PATH,
    build_template_candidate_contract_ledger,
    load_template_candidate_contract_ledger,
    render_template_candidate_contract_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TEMPLATE_CANDIDATE_CONTRACT_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Fail if the checked-in ledger differs from current canonical sources.",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Validate ledger integrity and every current source binding.",
    )
    args = parser.parse_args()

    if args.validate:
        load_template_candidate_contract_ledger(args.output, root=ROOT)
        return 0

    rendered = render_template_candidate_contract_ledger(build_template_candidate_contract_ledger(ROOT))
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except OSError:
            current = ""
        if current != rendered:
            raise SystemExit(
                f"{args.output} is stale; regenerate it with {Path(__file__).name}."
            )
        return 0
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
