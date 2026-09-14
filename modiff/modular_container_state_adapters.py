"""Exact state contracts for pinned custom Modular Diffusers loop containers.

Nine upstream loop subclasses initialize and publish state inside their custom
``__call__`` implementations.  The generic block snapshot cannot infer those
writes from ordinary ``inputs``/``outputs`` alone, so MoDiff records them here
instead of inventing edges or treating every loop as equivalent.

These contracts close structural planning only.  They do not select weights,
admit a Studio execution graph, or publish an executable Cluster Node.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from modiff.modular_block_contracts import (
    reviewed_modular_block_snapshot,
    validate_modular_block_snapshot,
)


MODULAR_CONTAINER_STATE_ADAPTER_SCHEMA_VERSION = 1
MODULAR_CONTAINER_STATE_ADAPTER_EXECUTION_CLAIM = "container_state_adapter_only"
_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ContainerStateAdapterTruth:
    class_name: str
    iteration_input: str
    iteration_cardinality: str
    state_initializers: tuple[tuple[str, str], ...]
    published_state: tuple[str, ...]
    progress_semantics: str


# Reviewed directly against the official implementations at the pinned
# Diffusers revision. Helios and Wan iterate an integer count; MiniMax derives
# its chunk count from ``len(chunk_starts)`` and owns a steps-per-chunk progress
# context around the loop.
PINNED_CONTAINER_STATE_ADAPTER_TRUTH = (
    *(
        ContainerStateAdapterTruth(
            class_name,
            "num_latent_chunk",
            "value",
            (("latent_chunks", "empty_list"), ("image_latents", "none_if_missing")),
            ("latent_chunks",),
            "container_default",
        )
        for class_name in (
            "HeliosChunkDenoiseStep",
            "HeliosI2VChunkDenoiseStep",
            "HeliosPyramidChunkDenoiseStep",
            "HeliosPyramidI2VChunkDenoiseStep",
            "HeliosPyramidDistilledChunkDenoiseStep",
            "HeliosPyramidDistilledI2VChunkDenoiseStep",
        )
    ),
    ContainerStateAdapterTruth(
        "MiniMaxMusic3ChunkDenoiseStep",
        "chunk_starts",
        "length",
        (
            ("latent_chunks", "empty_list"),
            ("previous_latent", "none"),
            ("previous_condition", "none"),
        ),
        ("latent_chunks",),
        "steps_per_iteration",
    ),
    ContainerStateAdapterTruth(
        "WanAnimate2DenoiseStep",
        "num_segments",
        "value",
        (("segment_frames", "empty_list"), ("out_frames", "none")),
        ("segment_frames",),
        "container_default",
    ),
    ContainerStateAdapterTruth(
        "WanAnimate2DistilledDenoiseStep",
        "num_segments",
        "value",
        (("segment_frames", "empty_list"), ("out_frames", "none")),
        ("segment_frames",),
        "container_default",
    ),
)


class ModularContainerStateAdapterError(ValueError):
    """A custom-container state contract is missing or inconsistent."""


def _adapter_contract(block_definition: Mapping[str, Any], truth: ContainerStateAdapterTruth) -> dict[str, Any]:
    block_definition_id = str(block_definition["id"])
    content_hash = ":".join(block_definition_id.rsplit(":", 2)[-2:])
    if _CONTENT_HASH.fullmatch(content_hash) is None:
        raise ModularContainerStateAdapterError("A container adapter references an invalid block definition id.")
    if block_definition.get("kind") != "loop" or block_definition.get("className") != truth.class_name:
        raise ModularContainerStateAdapterError("A container adapter references an incompatible upstream block.")
    return {
        "schemaVersion": MODULAR_CONTAINER_STATE_ADAPTER_SCHEMA_VERSION,
        "id": f"diffusers.modular-container-state:{truth.class_name}:{content_hash}",
        "blockDefinitionId": block_definition_id,
        "className": truth.class_name,
        "containerKind": "loop",
        "iteration": {
            "input": truth.iteration_input,
            "cardinality": truth.iteration_cardinality,
            "index": "k",
        },
        "stateInitializers": [
            {"name": name, "operation": operation} for name, operation in truth.state_initializers
        ],
        "publishedState": list(truth.published_state),
        "progressSemantics": truth.progress_semantics,
        "executionClaim": MODULAR_CONTAINER_STATE_ADAPTER_EXECUTION_CLAIM,
    }


def build_reviewed_modular_container_state_adapters(
    block_snapshot: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the nine exact custom-loop state adapter contracts."""

    normalized = validate_modular_block_snapshot(
        block_snapshot if block_snapshot is not None else reviewed_modular_block_snapshot()
    )
    definitions_by_class: dict[str, list[dict[str, Any]]] = {}
    for definition in normalized["blockDefinitions"]:
        definitions_by_class.setdefault(str(definition["className"]), []).append(definition)

    adapters = []
    for truth in PINNED_CONTAINER_STATE_ADAPTER_TRUTH:
        matches = definitions_by_class.get(truth.class_name, [])
        if len(matches) != 1:
            raise ModularContainerStateAdapterError(
                f"Pinned container class {truth.class_name!r} does not resolve to one exact block definition."
            )
        adapter = _adapter_contract(matches[0], truth)
        input_names = {field["name"] for field in matches[0]["inputs"]}
        if truth.iteration_input not in input_names:
            raise ModularContainerStateAdapterError(
                f"Pinned container {truth.class_name!r} does not declare iteration input {truth.iteration_input!r}."
            )
        adapters.append(adapter)
    return sorted(adapters, key=lambda item: item["id"])


def reviewed_modular_container_state_adapters() -> list[dict[str, Any]]:
    """Return the exact reviewed custom-container adapter catalog."""

    return build_reviewed_modular_container_state_adapters()
