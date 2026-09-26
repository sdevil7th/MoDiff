#!/usr/bin/env python3
"""Rebuild the pinned Diffusers operation inventory without model downloads."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modiff.operation_inventory import OPERATION_INVENTORY_PATH, build_operation_inventory  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build_operation_inventory(ROOT), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.check:
        if OPERATION_INVENTORY_PATH.read_text() != rendered:
            parser.error("The pinned operation inventory is stale; regenerate it.")
    else:
        OPERATION_INVENTORY_PATH.write_text(rendered)


if __name__ == "__main__":
    main()
