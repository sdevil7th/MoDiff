"""Deterministic image-route ledger for the prototyping handoff.

This inventory joins existing public execution profiles, model capabilities,
artifact pins, the pinned Diffusers operation inventory, and the reviewed
Modular workflow snapshot.  It performs no registry import, custom-extension
load, network request, package installation, or model construction.

The ledger deliberately reports implementation gaps.  A native upstream
workflow does not become an allowed whole-pipeline exception merely because
MoDiff currently routes the same repository through a standard adapter.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from modiff.diffusers_profiles import public_execution_profiles
from modiff.image_native_variant_reviews import image_native_variant_review
from modiff.model_artifact_catalog import catalog_download_inventory, catalog_repository_pin
from modiff.modular_action_bindings import MODULAR_ACTION_BINDINGS
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.modular_workflow_contracts import (
    PINNED_DIFFUSERS_REVISION,
    PINNED_MODULAR_WORKFLOW_TRUTH,
    PINNED_MODULAR_WORKFLOW_REPOSITORY_VARIANTS,
)
from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot
from modiff.operation_inventory import OPERATION_INVENTORY_PATH, load_operation_inventory
from modiff.studio_execution_specs import (
    reviewed_repository_download_files, studio_capability_definitions, studio_execution_spec_for_pair,
)


IMAGE_PROTOTYPING_READINESS_SCHEMA_VERSION = 1
IMAGE_PROTOTYPING_READINESS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "image-prototyping-readiness.v1.json"
)

_IMAGE_TASKS = frozenset(
    {
        "control_edit_image",
        "control_image",
        "control_inpaint",
        "control_union_edit_image",
        "control_union_image",
        "control_union_inpaint",
        "depth_estimation",
        "edit_image",
        "image_adjustment",
        "image_channels",
        "image_crop",
        "image_filter",
        "image_stitch",
        "image_tile",
        "image_to_image",
        "image_to_text",
        "image_upscale",
        "inpaint",
        "inpainting",
        "ip_adapter_control_edit_image",
        "ip_adapter_control_image",
        "ip_adapter_control_inpaint",
        "ip_adapter_control_union_edit_image",
        "ip_adapter_control_union_image",
        "ip_adapter_control_union_inpaint",
        "ip_adapter_edit_image",
        "ip_adapter_image",
        "ip_adapter_inpaint",
        "layer_decomposition",
        "mask_composite",
        "modular_image_to_image",
        "modular_inpainting",
        "modular_text_to_image",
        "multi_image_reference_edit",
        "outpaint",
        "text_to_image",
        "unconditional_image",
    }
)

_TASK_ALIASES = {
    "modular_image_to_image": "image_to_image",
    "modular_inpainting": "inpaint",
    "modular_text_to_image": "text_to_image",
    "inpainting": "inpaint",
    "image2image": "image_to_image",
    "controlnet_image2image": "control_edit_image",
    "controlnet_inpainting": "control_inpaint",
    "image_conditioned_inpainting": "inpaint",
    "controlnet_union_text2image": "control_union_image",
    "controlnet_union_image2image": "control_union_edit_image",
    "controlnet_union_inpainting": "control_union_inpaint",
    "ip_adapter_text2image": "ip_adapter_image",
    "ip_adapter_image2image": "ip_adapter_edit_image",
    "ip_adapter_inpainting": "ip_adapter_inpaint",
    "ip_adapter_controlnet_text2image": "ip_adapter_control_image",
    "ip_adapter_controlnet_image2image": "ip_adapter_control_edit_image",
    "ip_adapter_controlnet_inpainting": "ip_adapter_control_inpaint",
    "ip_adapter_controlnet_union_text2image": "ip_adapter_control_union_image",
    "ip_adapter_controlnet_union_image2image": "ip_adapter_control_union_edit_image",
    "ip_adapter_controlnet_union_inpainting": "ip_adapter_control_union_inpaint",
}

_NON_DIFFUSION_PATHS = frozenset(
    {
        "builtin-image-operation",
        "direct-huggingface-transformers-any-to-any",
        "direct-huggingface-transformers-depth",
        "direct-huggingface-transformers-image-text",
        "spandrel-image-upscale",
    }
)

_TOP_LEVEL_STAGE_ROLES = {
    "after_decode": "postprocess_image",
    "decode": "decode_latents",
    "denoise": "denoise",
    "image_encoder": "encode_image",
    "prompt_enhancer": "rewrite_prompt",
    "prompt_upsample": "rewrite_prompt",
    "text_encoder": "encode_prompt",
    "vae_encoder": "encode_image",
}

_TASK_REQUIRED_NATIVE_ROLES = {
    "layer_decomposition": frozenset({"decode_latents"}),
}
_DEFAULT_REQUIRED_NATIVE_ROLES = frozenset({"encode_prompt", "denoise", "decode_latents"})

# Independent upstream family audit. These are candidates, NOT claims that a
# variant (PAG, Turbo, KV, ControlNet, etc.) is interchangeable with the base.
# An absent MoDiff artifact binding cannot establish an upstream exception.
_STANDARD_NATIVE_CANDIDATES = {
    "ErnieImagePipeline": "ErnieImageModularPipeline",
    "FluxPipeline": "FluxModularPipeline",
    "FluxImg2ImgPipeline": "FluxModularPipeline",
    "Flux2Pipeline": "Flux2ModularPipeline",
    "Flux2KleinPipeline": "Flux2KleinModularPipeline",
    "Flux2KleinKVPipeline": "Flux2KleinModularPipeline",
    "FluxKontextPipeline": "FluxKontextModularPipeline",
    "QwenImagePipeline": "QwenImageModularPipeline",
    "QwenImageEditPipeline": "QwenImageEditModularPipeline",
    "QwenImageEditPlusPipeline": "QwenImageEditPlusModularPipeline",
    "StableDiffusionXLPipeline": "StableDiffusionXLModularPipeline",
    "StableDiffusionXLImg2ImgPipeline": "StableDiffusionXLModularPipeline",
    "StableDiffusionXLInpaintPipeline": "StableDiffusionXLModularPipeline",
    "StableDiffusionXLTurboPipeline": "StableDiffusionXLModularPipeline",
    "StableDiffusionXLPAGPipeline": "StableDiffusionXLModularPipeline",
    "StableDiffusionXLPAGImg2ImgPipeline": "StableDiffusionXLModularPipeline",
    "StableDiffusionXLPAGInpaintPipeline": "StableDiffusionXLModularPipeline",
    "StableDiffusionXLControlNetPAGImg2ImgPipeline": "StableDiffusionXLModularPipeline",
    "StableDiffusion3Pipeline": "StableDiffusion3ModularPipeline",
    "ZImagePipeline": "ZImageModularPipeline",
}


class ImagePrototypingReadinessError(ValueError):
    """Raised when the deterministic image readiness ledger is malformed."""


def _content_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_task(task: str) -> str:
    return _TASK_ALIASES.get(task, task)


def _workflow_stage_roles(workflow: Mapping[str, Any]) -> list[str]:
    roles = set()
    for step in workflow.get("steps", []):
        path = str(step.get("path") or "")
        prefix = path.split(".", 1)[0]
        roles.add(_TOP_LEVEL_STAGE_ROLES.get(prefix, f"upstream:{prefix}"))
    return sorted(roles)


def _matching_workflow(
    workflows: Mapping[tuple[str, str], list[dict[str, Any]]],
    pipeline_class: str,
    task: str,
) -> dict[str, Any] | None:
    route = _native_mode(pipeline_class, task)
    # Public task names differ from upstream workflow identifiers (notably
    # SDXL Union/IP-Adapter). Join through the reviewed mode contract, not a
    # guessed spelling or the first vaguely related workflow.
    if route is not None:
        matches = [
            workflow
            for (candidate_class, _task), candidates in workflows.items()
            if candidate_class == pipeline_class
            for workflow in candidates
            if workflow["id"] == (route.upstream_workflow or "default")
        ]
    else:
        matches = workflows.get((pipeline_class, _canonical_task(task)), [])
    return matches[0] if len(matches) == 1 else None


def _native_mode(pipeline_class: str, task: str):
    truth = PINNED_MODULAR_WORKFLOW_TRUTH.get(pipeline_class)
    if truth is None:
        return None
    canonical_task = _canonical_task(task)
    if pipeline_class == "StableDiffusionXLModularPipeline" and task == "edit_image":
        canonical_task = "image_to_image"
    if pipeline_class in {"Flux2ModularPipeline", "Flux2KleinModularPipeline", "FluxKontextModularPipeline"} and task == "multi_image_reference_edit":
        canonical_task = "edit_image"
    matches = [route for mode, route in truth.modes if _canonical_task(mode) == canonical_task]
    if not matches:
        matches = [route for name, route in truth.state_flows if _canonical_task(name) == canonical_task]
    return matches[0] if len(matches) == 1 else None


def _integrated_native_route(pipeline_class: str, task: str, workflow: Mapping[str, Any] | None) -> bool:
    route = _native_mode(pipeline_class, task)
    if workflow is None:
        return False
    whole = reviewed_whole_workflow_graph_adapter(pipeline_class, workflow["id"])
    actions = whole["actionSequence"] if whole else route.action_sequence if route else ()
    edges = whole["stateEdges"] if whole else route.state_edges if route else ()
    if not actions or not edges or any(action not in MODULAR_ACTION_BINDINGS for action in actions):
        return False
    required = _TASK_REQUIRED_NATIVE_ROLES.get(_canonical_task(task), _DEFAULT_REQUIRED_NATIVE_ROLES)
    if not required.issubset(_workflow_stage_roles(workflow)):
        return False
    # Generic stage adapters predate the optional whole-workflow block list.
    # Its absence is not a missing implementation. When provided, verify every
    # named block against the independent upstream snapshot.
    prefixes = {str(step["path"]).split(".", 1)[0] for step in workflow.get("steps", [])}
    sequence = whole["upstreamBlockSequence"] if whole else route.upstream_block_sequence
    return all(block.split(".", 1)[0] in prefixes for block in sequence)


def _artifact_inventory(profile: Mapping[str, Any], capability: Mapping[str, Any] | None) -> dict[str, Any]:
    repository = str(profile.get("default_repo") or "")
    pin = catalog_repository_pin(repository, model_type=str(profile.get("model_type") or "")) if repository else None
    files = list((capability or {}).get("downloadFiles") or []) if (
        (capability or {}).get("defaultRepo") == repository
    ) else []
    if not files:
        selections = [item for item in (capability or {}).get("artifactSelections", [])
                      if item.get("repo") == repository]
        if len(selections) == 1:
            files = list(selections[0].get("downloadFiles") or [])
    if not files:
        files = reviewed_repository_download_files(repository)
    pinned_files = [item for item in (pin or {}).get("files", []) if isinstance(item, Mapping)]
    exact_bytes = None
    if pinned_files and all(type(item.get("byteSize")) is int and item["byteSize"] > 0 for item in pinned_files):
        exact_bytes = sum(item["byteSize"] for item in pinned_files)
    inventory = catalog_download_inventory(repository, (pin or {}).get("revision"), files)
    if inventory:
        exact_bytes = inventory["exactBytes"]
    return {
        "repository": repository or None,
        "revision": (pin or {}).get("revision"),
        "license": (pin or {}).get("license"),
        "gated": (pin or {}).get("gated", (inventory or {}).get("gated")),
        "declaredFileCount": len(files) if files else None,
        "exactBytes": exact_bytes,
        "inventoryState": "exact_bytes" if exact_bytes is not None else "paths_only" if files else "unknown",
    }


def _display_name(profile: Mapping[str, Any], capability: Mapping[str, Any] | None) -> str:
    return str(
        (capability or {}).get("displayName")
        or (capability or {}).get("label")
        or profile.get("model_type")
        or profile.get("id")
    )


def _evidence(profile: Mapping[str, Any], *, native_integrated: bool) -> dict[str, str]:
    return {
        "structural": "implemented_awaiting_qualification" if native_integrated else "not_applicable_or_missing",
        "browser": "pending_final_candidate",
        "nativeOutput": "historical_not_final_revision" if profile.get("live_proof") else "pending",
        "parameterConsumption": "pending",
        "reuse": "pending",
        "service": "pending",
        "platform": "pending",
    }


def build_image_prototyping_readiness(root: Path) -> dict[str, Any]:
    """Build the backend-owned image-route denominator without side effects."""

    root = root.resolve()
    operation_inventory = load_operation_inventory()
    modular_snapshot_path = root / "data" / "modular-workflow-contracts.json"
    modular_snapshot = load_reviewed_modular_workflow_snapshot(modular_snapshot_path)
    capabilities = studio_capability_definitions()
    profiles = public_execution_profiles(observe_optional_runtime=False)

    workflows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pipeline in modular_snapshot["contracts"]:
        for workflow in pipeline["workflows"]:
            workflows.setdefault((pipeline["pipelineClass"], _canonical_task(workflow["taskId"])), []).append(workflow)

    image_profiles = [
        profile
        for profile in profiles
        if any(mode in _IMAGE_TASKS for mode in profile.get("modes", []))
    ]
    # Ordinary operation recipes also advertise ordered-reference tasks. They
    # do not add legacy Cluster modes or silently change its compiled admission.
    image_profiles = [
        {**profile, "modes": [*profile["modes"], "multi_image_reference_edit"]}
        if profile["id"] in {"flux2:modular", "flux2-klein:modular", "flux-kontext:modular"}
        else profile for profile in image_profiles
    ]
    # Compatible/fallback artifacts are selectable routes, not aliases for the
    # default weights. Their architecture, dtype and quantization evidence may
    # differ even when they share the execution profile and stage graph.
    advertised_profiles = []
    for profile in image_profiles:
        advertised_profiles.append(profile)
        alternatives = {
            repo: set(profile["modes"])
            for repo in (profile.get("fallback_repo"), *(profile.get("compatible_repos") or []))
            if repo and repo != profile.get("default_repo")
        }
        if profile["execution_path"] == "modular-diffusers":
            for mode in profile["modes"]:
                native = _native_mode(profile["pipeline_class"], mode)
                if native is None:
                    continue
                for repo in PINNED_MODULAR_WORKFLOW_REPOSITORY_VARIANTS.get(
                    (profile["pipeline_class"], native.upstream_workflow), (),
                ):
                    if repo != profile.get("default_repo"):
                        alternatives.setdefault(repo, set()).add(mode)
        advertised_profiles.extend(
            {**profile, "default_repo": repo, "_artifact_variant": repo, "modes": sorted(modes)}
            for repo, modes in sorted(alternatives.items())
        )
    modular_by_repository_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for profile in advertised_profiles:
        if profile["execution_path"] != "modular-diffusers":
            continue
        for mode in profile["modes"]:
            if mode in _IMAGE_TASKS:
                modular_by_repository_task.setdefault(
                    (str(profile.get("default_repo") or "").lower(), _canonical_task(mode)), []
                ).append(profile)

    inventory_by_class = {item["pipelineClass"]: item for item in operation_inventory["pipelines"]}
    routes = []
    seen_route_ids = set()
    for profile in sorted(advertised_profiles, key=lambda item: (item["id"], item.get("_artifact_variant", ""))):
        capability = capabilities.get(profile["model_type"])
        for mode in sorted(item for item in profile["modes"] if item in _IMAGE_TASKS):
            task = _canonical_task(mode)
            route_id = f"{profile['id']}/{mode}"
            if profile.get("_artifact_variant"):
                route_id += f"@{profile['_artifact_variant']}"
            variant_review = image_native_variant_review(route_id)
            if route_id in seen_route_ids:
                raise ImagePrototypingReadinessError(f"Duplicate image route {route_id!r}.")
            seen_route_ids.add(route_id)

            own_workflow = _matching_workflow(workflows, profile["pipeline_class"], task)
            model_workflow = _matching_workflow(workflows, profile["model_type"], task)
            stage_roles = _workflow_stage_roles(own_workflow) if own_workflow else []
            required_roles = _TASK_REQUIRED_NATIVE_ROLES.get(task, _DEFAULT_REQUIRED_NATIVE_ROLES)
            required_roles_present = bool(own_workflow and required_roles.issubset(stage_roles))
            full_stage_chain = _integrated_native_route(profile["pipeline_class"], task, own_workflow)
            native_integrated = profile["execution_path"] == "modular-diffusers" and full_stage_chain

            same_artifact_native = modular_by_repository_task.get(
                (str(profile.get("default_repo") or "").lower(), task), []
            )
            integrated_alternatives = [
                peer["id"] for peer in same_artifact_native
                if peer["model_type"] == peer["pipeline_class"] and _integrated_native_route(
                    peer["pipeline_class"], task, _matching_workflow(workflows, peer["pipeline_class"], task)
                )
            ]
            native_candidate = _STANDARD_NATIVE_CANDIDATES.get(profile["pipeline_class"])
            if any(marker in profile["pipeline_class"] for marker in ("PAG", "Turbo", "KV")):
                integrated_alternatives = []  # variant equivalence requires its own reviewed adapter
            if variant_review and variant_review["decision"] == "native_recipe":
                reviewed_task = variant_review["nativeTask"]
                integrated_alternatives = [
                    peer["id"] for peer in advertised_profiles
                    if peer["id"] == variant_review["nativeProfileId"]
                    and peer["default_repo"] == profile["default_repo"]
                    and reviewed_task in peer["modes"]
                    and peer["execution_path"] == "modular-diffusers"
                    and _integrated_native_route(peer["pipeline_class"], reviewed_task,
                        _matching_workflow(workflows, peer["pipeline_class"], reviewed_task))
                ]
            elif variant_review:
                integrated_alternatives = []
            usable_native_alternative = bool(
                same_artifact_native
                or (
                    model_workflow
                    and _integrated_native_route(profile["model_type"], task, model_workflow)
                )
                # Keep a native export that MoDiff has not integrated visible as
                # a gap rather than laundering it into an upstream exception.
                or (model_workflow and profile["model_type"].endswith("ModularPipeline"))
            )
            artifact = _artifact_inventory(profile, capability)

            if variant_review and variant_review["decision"] in {"artifact_blocked", "upstream_exception"}:
                disposition = ("blocked" if variant_review["decision"] == "artifact_blocked"
                               else "documented_whole_pipeline_exception")
                reason_code = variant_review["reason"]
            elif variant_review and variant_review["decision"] == "native_recipe" and not integrated_alternatives:
                disposition = "native_integration_missing"
                reason_code = "reviewed_native_recipe_stage_chain_unavailable"
            elif profile["execution_path"] in _NON_DIFFUSION_PATHS:
                disposition = "non_diffusion_image_task"
                reason_code = "task_has_no_diffusion_stage_semantics"
            elif native_integrated:
                disposition = "native_integrated"
                reason_code = "complete_native_stage_chain_declared"
            elif profile["execution_path"] == "modular-diffusers":
                disposition = "native_integration_missing"
                reason_code = "public_modular_profile_lacks_complete_stage_chain"
            elif integrated_alternatives:
                disposition = "preserved_standard_alternative"
                reason_code = "exact_artifact_task_has_integrated_native_alternative"
            elif usable_native_alternative:
                disposition = "native_integration_missing"
                reason_code = "public_whole_pipeline_route_has_usable_native_alternative"
            elif native_candidate:
                disposition = "blocked"
                reason_code = "native_family_requires_exact_artifact_variant_task_review"
            elif profile["execution_path"].startswith("direct-"):
                disposition = "documented_whole_pipeline_exception"
                reason_code = "no_pinned_modular_workflow_for_exact_profile"
            else:
                disposition = "blocked"
                reason_code = "unclassified_execution_path"

            spec = studio_execution_spec_for_pair(profile["model_type"], mode)
            advertisement_entry_points = [f"execution_profile:{profile['id']}"]
            if profile.get("_artifact_variant"):
                advertisement_entry_points.append(f"execution_profile_artifact:{profile['_artifact_variant']}")
            if capability is not None:
                advertisement_entry_points.append(f"model_capability:{profile['model_type']}")
            if spec is not None and spec.get("executionProfileId") == profile["id"]:
                advertisement_entry_points.append(f"studio_execution_spec:{spec['id']}")
            advertisement_entry_points.append(
                f"node_model_dropdown:{profile['loader_module']}.{profile['loader_action']}"
            )

            upstream_record = inventory_by_class.get(profile["pipeline_class"])
            known_defects = []
            if disposition in {"native_integration_missing", "blocked"}:
                known_defects.append(reason_code)
            if artifact["revision"] is None and profile["execution_path"] not in _NON_DIFFUSION_PATHS:
                known_defects.append("missing_immutable_artifact_revision")
            if artifact["inventoryState"] != "exact_bytes" and profile["execution_path"] not in _NON_DIFFUSION_PATHS:
                known_defects.append("exact_artifact_byte_inventory_not_in_catalog")

            routes.append(
                {
                    "routeId": route_id,
                    "displayName": _display_name(profile, capability),
                    "modelType": profile["model_type"],
                    "task": mode,
                    "canonicalTask": task,
                    "artifactVariantOf": f"{profile['id']}/{mode}" if profile.get("_artifact_variant") else None,
                    "implementation": {
                        "profileId": profile["id"],
                        "executionPath": profile["execution_path"],
                        "pipelineClass": profile["pipeline_class"],
                        "loader": f"{profile['loader_module']}.{profile['loader_action']}",
                    },
                    "advertisementEntryPoints": sorted(advertisement_entry_points),
                    "artifact": artifact,
                    "optionalRuntime": profile["optionalRuntimeRequirement"],
                    "upstream": {
                        "diffusersRevision": PINNED_DIFFUSERS_REVISION,
                        "inventoryCoverage": (upstream_record or {}).get("coverage"),
                        "nativePipelineClass": (own_workflow or model_workflow or {}).get("executionPipelineClass"),
                        "nativeWorkflowId": (own_workflow or model_workflow or {}).get("id"),
                        "candidateNativePipelineClass": native_candidate,
                        "stageRoles": stage_roles
                        if own_workflow
                        else _workflow_stage_roles(model_workflow)
                        if model_workflow
                        else [],
                        "requiredCoreRolesPresent": required_roles_present,
                        "fullStageChain": full_stage_chain,
                    },
                    "disposition": disposition,
                    "dispositionReason": reason_code,
                    "standardAndNativeAlternativesCoexist": bool(
                        profile["execution_path"] != "modular-diffusers" and same_artifact_native
                    ),
                    "integratedNativeAlternativeProfileIds": sorted(integrated_alternatives),
                    "exactVariantReview": variant_review,
                    "exception": {
                        "checkedDiffusersRevision": PINNED_DIFFUSERS_REVISION,
                        "limitations": [
                            "no_independent_stage_editing",
                            "no_cross_stage_prompt_or_latent_cache",
                            "component_substitution_limited_to_standard_pipeline_contract",
                        ],
                        "revisitTrigger": "reviewed_diffusers_pin_or_native_profile_change",
                    }
                    if disposition == "documented_whole_pipeline_exception"
                    else None,
                    "evidence": _evidence(profile, native_integrated=native_integrated),
                    "knownDefects": sorted(set(known_defects)),
                }
            )

    dispositions = Counter(route["disposition"] for route in routes)
    semantic = {
        "schemaVersion": IMAGE_PROTOTYPING_READINESS_SCHEMA_VERSION,
        "kind": "image_prototyping_readiness",
        "diffusersRevision": PINNED_DIFFUSERS_REVISION,
        "coverageBoundary": {
            "backendExecutionProfiles": "snapshotted",
            "backendModelCapabilities": "snapshotted",
            "backendArtifactPins": "snapshotted",
            "upstreamModularWorkflows": "snapshotted",
            "clientTemplatesAndTaskChoosers": "requires_paired_client_gate",
            "customFirstPartyExamples": "requires_paired_custom_e2e",
        },
        "sources": {
            "operationInventory": _file_hash(OPERATION_INVENTORY_PATH),
            "modularWorkflowSnapshot": _file_hash(modular_snapshot_path),
            "modelArtifactCatalog": _file_hash(root / "data" / "model-artifact-catalog.json"),
            "publicExecutionProfiles": _content_hash({"profiles": image_profiles}),
            "exactVariantReviews": _file_hash(root / "modiff" / "image_native_variant_reviews.py"),
        },
        "imageTasks": sorted(_IMAGE_TASKS),
        "routes": routes,
        "summary": {
            "routeCount": len(routes),
            "profileCount": len(image_profiles),
            "dispositions": dict(sorted(dispositions.items())),
            "nativeIntegrationMissingCount": dispositions["native_integration_missing"],
            "knownDefectRouteCount": sum(bool(route["knownDefects"]) for route in routes),
            "denominatorFrozen": False,
            "freezeBlockers": [
                "paired_client_advertisement_gate_not_recorded",
                "custom_first_party_examples_not_recorded",
                *(
                    ["native_integration_gaps_present"]
                    if dispositions["native_integration_missing"]
                    else []
                ),
            ],
        },
    }
    return {**semantic, "contentHash": _content_hash(semantic)}


def render_image_prototyping_readiness(value: Mapping[str, Any]) -> str:
    validate_image_prototyping_readiness(value)
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def validate_image_prototyping_readiness(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ImagePrototypingReadinessError("Image readiness ledger must be an object.")
    normalized = json.loads(json.dumps(value, allow_nan=False))
    if normalized.get("schemaVersion") != IMAGE_PROTOTYPING_READINESS_SCHEMA_VERSION:
        raise ImagePrototypingReadinessError("Unsupported image readiness schema.")
    if normalized.get("diffusersRevision") != PINNED_DIFFUSERS_REVISION:
        raise ImagePrototypingReadinessError("Image readiness ledger targets the wrong Diffusers revision.")
    semantic = {key: item for key, item in normalized.items() if key != "contentHash"}
    if normalized.get("contentHash") != _content_hash(semantic):
        raise ImagePrototypingReadinessError("Image readiness content hash is invalid.")
    routes = normalized.get("routes")
    if not isinstance(routes, list) or not routes or len(routes) > 1024:
        raise ImagePrototypingReadinessError("Image readiness routes are missing or oversized.")
    route_ids = [route.get("routeId") for route in routes if isinstance(route, Mapping)]
    if len(route_ids) != len(routes) or len(route_ids) != len(set(route_ids)):
        raise ImagePrototypingReadinessError("Image readiness route IDs are invalid or duplicated.")
    allowed = {
        "native_integrated",
        "native_integration_missing",
        "preserved_standard_alternative",
        "documented_whole_pipeline_exception",
        "non_diffusion_image_task",
        "blocked",
    }
    if any(route.get("disposition") not in allowed for route in routes):
        raise ImagePrototypingReadinessError("Image readiness route has an invalid disposition.")
    return normalized


def load_image_prototyping_readiness(
    path: Path = IMAGE_PROTOTYPING_READINESS_PATH,
) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ImagePrototypingReadinessError("Image readiness ledger exceeds 4 MiB.")
    return validate_image_prototyping_readiness(json.loads(raw))
