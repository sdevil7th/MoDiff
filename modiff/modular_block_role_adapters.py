"""Reviewed semantic roles for contract-only Modular Diffusers blocks.

The pinned upstream block snapshot retains exact classes, fields, components,
container kinds, and placement paths.  This module adds only MoDiff's reusable
semantic role for all 50 contract-only workflows and the five workflows that
previously had only an equivalent standard route. The nine custom loop classes
retain separate exact container-state adapters. This module deliberately does
not claim that a role adapter is executable, select an artifact, or flatten an
upstream loop/container into a different graph primitive.

The role vocabulary was reviewed against the official Modular Diffusers source
at :data:`PINNED_DIFFUSERS_REVISION`.  Runtime readers derive contracts from
the checked-in no-weight snapshots and never import Diffusers or Transformers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from modiff.modular_block_contracts import (
    reviewed_modular_block_snapshot,
    validate_modular_block_snapshot,
)
from modiff.modular_contract_only_registry import CURRENT_PIN_EQUIVALENT_MODULAR_TARGETS
from modiff.modular_workflow_discovery import (
    load_reviewed_modular_workflow_snapshot,
    validate_modular_workflow_snapshot,
)


MODULAR_BLOCK_ROLE_ADAPTER_SCHEMA_VERSION = 1
MODULAR_BLOCK_ROLE_ADAPTER_EXECUTION_CLAIM = "structural_adapter_only"

_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")

# These are the 38 contract-only workflows whose pinned leaf/container state is
# statically closed without a custom-container supplement.
PINNED_STATICALLY_CLOSED_BLOCK_ROLE_WORKFLOWS = frozenset(
    {
        ("AnimaModularPipeline", "img2img"),
        ("AnimaModularPipeline", "text2image"),
        ("Cosmos3DistilledModularPipeline", "image2video"),
        ("Cosmos3DistilledModularPipeline", "text2image"),
        ("Cosmos3DistilledModularPipeline", "text2video"),
        ("Cosmos3DistilledModularPipeline", "video2video"),
        ("Cosmos3OmniModularPipeline", "action_forward_dynamics"),
        ("Cosmos3OmniModularPipeline", "action_inverse_dynamics"),
        ("Cosmos3OmniModularPipeline", "action_policy"),
        ("Cosmos3OmniModularPipeline", "image2video"),
        ("Cosmos3OmniModularPipeline", "image2video_with_sound"),
        ("Cosmos3OmniModularPipeline", "text2image"),
        ("Cosmos3OmniModularPipeline", "text2video"),
        ("Cosmos3OmniModularPipeline", "text2video_with_sound"),
        ("Cosmos3OmniModularPipeline", "video2video"),
        ("Cosmos3OmniModularPipeline", "video2video_with_sound"),
        ("Flux2KleinBaseModularPipeline", "image_conditioned"),
        ("Flux2KleinBaseModularPipeline", "text2image"),
        ("Flux2ModularPipeline", "image_conditioned"),
        ("Flux2ModularPipeline", "text2image"),
        ("HunyuanVideo15ModularPipeline", "image2video"),
        ("HunyuanVideo15ModularPipeline", "text2video"),
        ("Ideogram4ModularPipeline", "text2image"),
        ("Krea2ModularPipeline", "text2image"),
        ("Krea2TurboModularPipeline", "text2image"),
        ("LTX25ModularPipeline", "condition"),
        ("LTX25ModularPipeline", "image2video"),
        ("LTX25ModularPipeline", "in_context"),
        ("LTX25ModularPipeline", "text2video"),
        ("LTX2ModularPipeline", "condition"),
        ("LTX2ModularPipeline", "image2video"),
        ("LTX2ModularPipeline", "in_context"),
        ("LTX2ModularPipeline", "text2video"),
        ("MiniMaxH3ModularPipeline", "fl2va"),
        ("MiniMaxH3ModularPipeline", "ref2va"),
        ("MiniMaxH3ModularPipeline", "t2va"),
        ("StableDiffusion3ModularPipeline", "image2image"),
        ("StableDiffusion3ModularPipeline", "text2image"),
    }
)

PINNED_CUSTOM_CONTAINER_BLOCK_ROLE_WORKFLOWS = frozenset(
    {
        *((pipeline, workflow) for pipeline in (
            "HeliosModularPipeline",
            "HeliosPyramidModularPipeline",
            "HeliosPyramidDistilledModularPipeline",
        ) for workflow in ("text2video", "image2video", "video2video")),
        ("MiniMaxMusic3ModularPipeline", "default"),
        ("WanAnimate2ModularPipeline", "default"),
        ("WanAnimate2DistilledModularPipeline", "default"),
    }
)

# The exact top-level names are upstream data.  Dotted top-level keys remain
# single placement path segments; only their semantic family prefix is mapped.
_TOP_LEVEL_BLOCK_ROLES = {
    "text_encoder": "text_encoder",
    "vae_encoder": "vae_encoder",
    "image_encoder": "image_encoder",
    "video_encoder": "video_encoder",
    "semantic_generator": "semantic_generator",
    "prompt_upsample": "prompt_transform",
    "duration": "duration",
    "condition_encoder": "condition_encoder",
    "reference_encoder": "reference_encoder",
    "before_encode": "before_encode",
    "denoise": "denoise",
    "decode": "decoder",
    "after_decode": "after_decode",
}


class ModularBlockRoleAdapterError(ValueError):
    """A pinned block-role mapping is missing, conflicting, or malformed."""


def _role_for_path(path: Any) -> str:
    if not isinstance(path, list) or not path or not isinstance(path[0], str):
        raise ModularBlockRoleAdapterError("A block-role placement path is malformed.")
    top_level_prefix = path[0].split(".", 1)[0]
    role = _TOP_LEVEL_BLOCK_ROLES.get(top_level_prefix)
    if role is None:
        raise ModularBlockRoleAdapterError(
            f"Pinned top-level block {path[0]!r} has no reviewed reusable role."
        )
    return role


def _adapter_contract(block_definition_id: str, role: str) -> dict[str, Any]:
    content_hash = ":".join(block_definition_id.rsplit(":", 2)[-2:])
    if _CONTENT_HASH.fullmatch(content_hash) is None:
        raise ModularBlockRoleAdapterError("A block-role adapter references an invalid block definition id.")
    return {
        "schemaVersion": MODULAR_BLOCK_ROLE_ADAPTER_SCHEMA_VERSION,
        "id": f"diffusers.modular-block-role:{role}:{content_hash}",
        "blockDefinitionId": block_definition_id,
        "role": role,
        "adapterKind": "workflow_root" if role == "workflow" else "pipeline_state_block",
        "executionClaim": MODULAR_BLOCK_ROLE_ADAPTER_EXECUTION_CLAIM,
    }


def build_reviewed_modular_block_role_adapters(
    block_snapshot: Mapping[str, Any] | None = None,
    workflow_snapshot: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build deduplicated roles for every contract-only workflow block."""

    normalized_workflows = validate_modular_workflow_snapshot(
        workflow_snapshot if workflow_snapshot is not None else load_reviewed_modular_workflow_snapshot()
    )
    normalized_blocks = validate_modular_block_snapshot(
        block_snapshot if block_snapshot is not None else reviewed_modular_block_snapshot(),
        workflow_snapshot=normalized_workflows,
    )
    workflows_by_key = {
        (workflow["pipelineClass"], workflow["workflowId"]): workflow
        for workflow in normalized_blocks["workflows"]
    }
    target_workflows = (
        PINNED_STATICALLY_CLOSED_BLOCK_ROLE_WORKFLOWS | PINNED_CUSTOM_CONTAINER_BLOCK_ROLE_WORKFLOWS
    )
    target_workflows |= {
        key for key in workflows_by_key if key[0] in CURRENT_PIN_EQUIVALENT_MODULAR_TARGETS
    }
    missing = target_workflows - set(workflows_by_key)
    if missing:
        raise ModularBlockRoleAdapterError(
            f"Pinned state-closed workflows are absent from the block snapshot: {sorted(missing)}."
        )

    roles_by_definition_id: dict[str, str] = {}

    def assign(block_definition_id: str, role: str) -> None:
        previous = roles_by_definition_id.setdefault(block_definition_id, role)
        if previous != role:
            raise ModularBlockRoleAdapterError(
                f"Block definition {block_definition_id!r} maps to conflicting roles {previous!r} and {role!r}."
            )

    for workflow_key in sorted(target_workflows):
        workflow = workflows_by_key[workflow_key]
        assign(str(workflow["rootBlockDefinitionId"]), "workflow")
        for placement in workflow["placements"]:
            assign(str(placement["blockDefinitionId"]), _role_for_path(placement["path"]))

    return sorted(
        (_adapter_contract(block_definition_id, role) for block_definition_id, role in roles_by_definition_id.items()),
        key=lambda item: item["id"],
    )


def reviewed_modular_block_role_adapters() -> list[dict[str, Any]]:
    """Return the exact reviewed role adapter catalog."""

    return build_reviewed_modular_block_role_adapters()
