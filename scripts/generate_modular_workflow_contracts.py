#!/usr/bin/env python3
"""Regenerate the reviewed no-weight Modular workflow contract snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import diffusers
from diffusers.modular_pipelines.modular_pipeline import PipelineState

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION, PINNED_MODULAR_WORKFLOW_TRUTH
from modiff.modular_workflow_discovery import (
    MODULAR_WORKFLOW_SNAPSHOT,
    build_modular_workflow_contract,
    validate_modular_workflow_snapshot,
)


_GENERIC_TASK_ALIASES = {
    "text2image": "text_to_image",
    "image2image": "image_to_image",
    "inpainting": "inpaint",
    "image_conditioned": "edit_image",
    "image_conditioned_inpainting": "inpaint",
    "image2video": "image_to_video",
}


def _aliases(truth) -> dict[str, str]:
    aliases = {
        workflow.name: _GENERIC_TASK_ALIASES[workflow.name]
        for workflow in truth.workflows
        if workflow.name in _GENERIC_TASK_ALIASES
    }
    for mode, mode_truth in truth.modes:
        workflow = mode_truth.upstream_workflow
        if workflow is not None and workflow not in aliases:
            aliases[workflow] = mode
    if not truth.workflows and len(truth.modes) == 1:
        aliases["default"] = truth.modes[0][0]
    return aliases


def generate() -> dict:
    contracts = []
    for pipeline_class_name, truth in PINNED_MODULAR_WORKFLOW_TRUTH.items():
        pipeline_class = getattr(diffusers, pipeline_class_name)
        constructor_config = dict(truth.constructor_config)
        pipeline = pipeline_class(config_dict=constructor_config) if constructor_config else pipeline_class()
        contract = build_modular_workflow_contract(
            pipeline,
            aliases=_aliases(truth),
            pipeline_state_factory=PipelineState,
        )
        if contract["blocksClass"] != truth.blocks_class:
            raise RuntimeError(
                f"{pipeline_class_name} discovered {contract['blocksClass']}, expected {truth.blocks_class}."
            )
        contracts.append(contract)
    contracts.sort(key=lambda item: item["pipelineClass"])
    return validate_modular_workflow_snapshot(
        {
            "schemaVersion": 1,
            "diffusersRevision": PINNED_DIFFUSERS_REVISION,
            "contracts": contracts,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=MODULAR_WORKFLOW_SNAPSHOT)
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
