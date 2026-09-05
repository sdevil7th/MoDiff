#!/usr/bin/env python3
"""Regenerate the reviewed unpruned Modular Diffusers block snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modiff.modular_conditional_contracts import (
    MODULAR_CONDITIONAL_CONTRACT_SNAPSHOT,
    build_modular_conditional_contract,
    merge_modular_conditional_contracts,
)
from generate_modular_block_contracts import _reviewed_pipelines


def generate() -> dict:
    return merge_modular_conditional_contracts(
        [build_modular_conditional_contract(pipeline) for pipeline in _reviewed_pipelines()]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=MODULAR_CONDITIONAL_CONTRACT_SNAPSHOT)
    args = parser.parse_args()
    rendered = json.dumps(generate(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
