#!/usr/bin/env python3
"""Regenerate the reviewed unpruned Modular Diffusers block snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import diffusers

from modiff.modular_conditional_contracts import (
    MODULAR_CONDITIONAL_CONTRACT_SNAPSHOT,
    build_modular_conditional_contract,
    merge_modular_conditional_contracts,
)
from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH
from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot


def generate() -> dict:
    # This companion must cover the reviewed snapshot, independently of whether
    # a class has moved between executable, equivalent, or contract-only routes.
    parts = []
    for contract in load_reviewed_modular_workflow_snapshot()["contracts"]:
        name = contract["pipelineClass"]
        truth = PINNED_MODULAR_WORKFLOW_TRUTH.get(name)
        config = dict(truth.constructor_config) if truth is not None else {}
        constructor = getattr(diffusers, name)
        pipeline = constructor(config_dict=config) if config else constructor()
        parts.append(build_modular_conditional_contract(pipeline))
    return merge_modular_conditional_contracts(parts)


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
