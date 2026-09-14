#!/usr/bin/env python3
"""Generate or validate the Comfy-to-MoDiff contract resolution ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.comfy_contract_resolution import (  # noqa: E402
    COMFY_CONTRACT_RESOLUTION_PATH,
    build_comfy_contract_resolution_ledger,
    load_comfy_contract_resolution_ledger,
    render_comfy_contract_resolution_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=COMFY_CONTRACT_RESOLUTION_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        load_comfy_contract_resolution_ledger(args.output, root=ROOT)
        return 0
    rendered = render_comfy_contract_resolution_ledger(build_comfy_contract_resolution_ledger(ROOT))
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
