#!/usr/bin/env python3
"""Regenerate the reviewed no-weight Modular Diffusers block snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import diffusers

from modiff.modular_block_contracts import (
    MODULAR_BLOCK_CONTRACT_SNAPSHOT,
    build_modular_block_contracts,
    merge_modular_block_contracts,
)
from modiff.modular_contract_only_registry import (
    CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES,
    CURRENT_PIN_EQUIVALENT_MODULAR_TARGETS,
    CURRENT_PIN_PROMOTED_MODULAR_DISCOVERY,
)
from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH


def _reviewed_pipelines():
    for pipeline_class_name in CURRENT_PIN_PROMOTED_MODULAR_DISCOVERY:
        yield getattr(diffusers, pipeline_class_name)()
    for pipeline_class_name, truth in PINNED_MODULAR_WORKFLOW_TRUTH.items():
        pipeline_class = getattr(diffusers, pipeline_class_name)
        constructor_config = dict(truth.constructor_config)
        yield pipeline_class(config_dict=constructor_config) if constructor_config else pipeline_class()
    for specification in CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES:
        yield getattr(diffusers, specification.class_name)()
    for pipeline_class_name in CURRENT_PIN_EQUIVALENT_MODULAR_TARGETS:
        yield getattr(diffusers, pipeline_class_name)()


def generate() -> dict:
    return merge_modular_block_contracts(
        [build_modular_block_contracts(pipeline) for pipeline in _reviewed_pipelines()]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=MODULAR_BLOCK_CONTRACT_SNAPSHOT)
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
