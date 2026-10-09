"""Exact Super T2I selection adds no model class or resource authority."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from modules import MODULE_MAP
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, resolve_execution_profiles_for_loader
from modiff.huggingface_cluster_admission import audit_reviewed_cluster_execution_candidates
from modiff.huggingface_cluster_runtime import _effective_reviewed_artifact
from modiff.model_artifact_catalog import catalog_download_inventory, catalog_repository_pin
from modiff.modular_workflow_contracts import (
    PINNED_MODULAR_REPOSITORY_VARIANTS, PINNED_MODULAR_WORKFLOW_REPOSITORY_VARIANTS,
    require_reviewed_modular_repository_workflow,
)
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modiff.studio_execution_specs import (
    COSMOS3_SUPER_T2I_DIFFUSERS_FILES, STUDIO_EXECUTION_SPEC_DEFINITIONS,
    studio_capability_definition, studio_execution_spec_for_pair,
)
from modules.ModularDiffusers.loaders import ModelsLoader, _validate_reviewed_pipeline_index
from modules.ModularDiffusers.route_state import issue_pipeline_instance_token, require_component_binding
from modules.ModularDiffusers.loaders import annotate_modular_loader_outputs
from modules.ModularDiffusers import workflow_blocks as stages

PIPELINE = "Cosmos3OmniModularPipeline"
NANO = "nvidia/Cosmos3-Nano"
SUPER = "nvidia/Cosmos3-Super-Text2Image"
REVISION = "daf3d374804be4c512c2135568a7cb95d4341d79"
PROFILE = "cosmos3-super-text-to-image:official-modular-workflow"


@pytest.mark.parametrize("fault", [None, "pipeline", "blocks", "component", "foreign_repository",
                                  "foreign_revision", "foreign_subfolder", "variant", "extra_field",
                                  "missing_component", "extra_component"])
def test_publisher_super_index_binds_exact_installed_component_signature(fault):
    raw = (Path(__file__).parent / "fixtures/cosmos3_super_t2i_modular_index.v1.json").read_bytes()
    assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == "f7085926b0883cbb77a90a5d9ae95a6963d7ce3e"
    document = json.loads(raw)
    if fault == "pipeline":
        document["_class_name"] = "Cosmos3DistilledModularPipeline"
    elif fault == "blocks":
        document["_blocks_class_name"] = "Cosmos3DistilledBlocks"
    elif fault == "component":
        document["transformer"][0:2] = ["diffusers", "CosmosTransformer3DModel"]
    elif fault == "foreign_repository":
        document["transformer"][2]["pretrained_model_name_or_path"] = NANO
    elif fault == "foreign_revision":
        document["transformer"][2]["revision"] = "main"
    elif fault == "foreign_subfolder":
        document["transformer"][2]["subfolder"] = "../transformer"
    elif fault == "variant":
        document["transformer"][2]["variant"] = "fp16"
    elif fault == "extra_field":
        document["transformer"][2]["config"] = {"unexpected": True}
    elif fault == "missing_component":
        del document["sound_tokenizer"]
    elif fault == "extra_component":
        document["vision_encoder"] = deepcopy(document["transformer"])
    with patch("modules.ModularDiffusers.loaders._load_reviewed_pipeline_index", return_value=("modular_model_index.json", document)):
        if fault:
            with pytest.raises(ValueError):
                _validate_reviewed_pipeline_index(PIPELINE, SUPER, REVISION)
        else:
            name, result = _validate_reviewed_pipeline_index(PIPELINE, SUPER, REVISION)
            assert name == "modular_model_index.json"
            assert result["transformer"][2]["pretrained_model_name_or_path"] == SUPER


def test_super_exact_inventory_keeps_the_thirty_weight_closure_without_qualification():
    pin = catalog_repository_pin(SUPER, model_type=PIPELINE)
    assert pin["revision"] == REVISION
    assert pin["gated"] is False
    assert pin["weightByteSize"] == 132_489_727_416
    inventory = catalog_download_inventory(SUPER, REVISION, COSMOS3_SUPER_T2I_DIFFUSERS_FILES)
    assert inventory is not None
    assert len([entry for entry in inventory["files"] if entry["path"].endswith(".safetensors")]) == 30
    assert catalog_download_inventory(SUPER, "a" * 40, COSMOS3_SUPER_T2I_DIFFUSERS_FILES) is None
    assert PINNED_MODULAR_REPOSITORY_VARIANTS[PIPELINE] == (NANO, SUPER)
    assert PINNED_MODULAR_WORKFLOW_REPOSITORY_VARIANTS[(PIPELINE, "text2image")] == (NANO, SUPER)
    assert (PIPELINE, "text2video") not in PINNED_MODULAR_WORKFLOW_REPOSITORY_VARIANTS


def test_super_profile_leaves_unique_nano_spec_and_all_execution_authority_closed():
    profile = DIFFUSERS_EXECUTION_PROFILES[PROFILE]
    assert profile.pipeline_class == profile.model_type == PIPELINE
    assert profile.default_repo == SUPER and profile.modes == ("text_to_image",)
    assert profile.supported_offload_modes == ("none",)
    assert not profile.default_quantized_components and not profile.live_proof
    assert profile.optional_runtime_profiles == ("cosmos-guardrail-0.3.1",)
    assert studio_execution_spec_for_pair(PIPELINE, "text_to_image")["id"] == "cosmos3-nano:modular-text-to-image:v1"
    matches = [row for row in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
               if row["modelType"] == PIPELINE and row["mode"] == "text_to_image"]
    assert len(matches) == 1 and matches[0]["profile"]["default_repo"] == NANO
    assert studio_capability_definition(PIPELINE)["defaultRepo"] == NANO
    capability = studio_capability_definition(PIPELINE, repository=SUPER)
    assert capability["guidanceLabel"] == "Classifier-free guidance"
    assert not any("Nano" in value for value in (
        capability["label"], capability["displayName"], capability["artifactLabel"],
        capability["guidanceLabel"], *capability["notes"],
    ))
    assert not any(capability[key] for key in ("autoEligible", "templateEligible", "galleryEligible", "liveProof"))
    assert capability["qualifiedModes"] == []
    admission = next(row for row in audit_reviewed_cluster_execution_candidates()
                     if row["definitionId"] == f"diffusers.modular:{PIPELINE}:text2image")
    assert admission["artifact"]["repo"] == NANO
    assert not admission["executable"] and not admission["publication"]["autoEligible"]
    assert "modelVariant" in admission["executionParameterSources"]
    assert _effective_reviewed_artifact(admission, profile, SUPER, workflow_id="text2image") == {
        "repo": SUPER, "revision": REVISION,
    }


def test_super_catalog_profile_is_scoped_to_the_exact_artifact_workflow():
    from modiff.diffusers_profiles import public_execution_profiles

    contracts, support = build_operation_catalog(
        MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {},
    )
    pipeline = next(row for row in support if row["pipelineClass"] == PIPELINE)
    tasks = {row["task"]: row for row in pipeline["tasks"] if row["operationIds"]}
    assert len(tasks) == 10
    assert PROFILE in tasks["text_to_image"]["executionProfileIds"]
    for task, row in tasks.items():
        assert "cosmos3-nano:official-modular-workflow" in row["executionProfileIds"]
        if task == "text_to_image":
            continue
        assert PROFILE not in row["executionProfileIds"]
        with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed")):
            with pytest.raises(ValueError, match="workflow/revision"):
                resolve_operation_starter(MODULE_MAP, contracts, {
                    "pipelineClass": PIPELINE, "task": task, "executionProfileId": PROFILE,
                })


@pytest.mark.parametrize("workflow", ["", "text2video", "image2video", "video2video", "text2video_with_sound", "action_policy"])
@pytest.mark.parametrize("variant", [None, SUPER])
def test_super_direct_and_variant_kwargs_reject_unreviewed_workflow_before_load(workflow, variant):
    with patch("modules.ModularDiffusers.loaders._instantiate_reviewed_builtin_pipeline", side_effect=AssertionError("allocated")):
        with pytest.raises(ValueError, match="workflow|variant"):
            ModelsLoader._effective_builtin_selector(model_type=PIPELINE,
                repo_id={"source": "hub", "value": SUPER}, revision=REVISION,
                workflow_id=workflow, reviewed_variant=variant)


def test_super_exact_revision_and_pipeline_binding_are_not_labels():
    for revision in (None, "main", "a" * 40):
        with pytest.raises(ValueError, match="workflow/revision"):
            require_reviewed_modular_repository_workflow(PIPELINE, SUPER, revision, "text2image")
    selector, revision = ModelsLoader._effective_builtin_selector(model_type=PIPELINE,
        repo_id={"source": "hub", "value": NANO}, revision=None,
        workflow_id="text2image", reviewed_variant=SUPER)
    assert selector == {"source": "hub", "value": SUPER} and revision == REVISION
    assert ModelsLoader._reviewed_builtin_selection(model_type=PIPELINE, repo_id=selector, revision=revision) == ("hub", SUPER, REVISION)


def test_exact_super_starter_uses_publisher_recipe_without_constructing_models():
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed")):
        starter = resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": PIPELINE, "task": "text_to_image", "executionProfileId": PROFILE,
        })
    nodes = {row["action"]: row for row in starter["nodes"]}
    loader = nodes["ModelsLoader"]
    assert loader["values"]["repo_id"] == {"source": "hub", "value": SUPER}
    assert loader["values"]["revision"] == REVISION
    assert loader["values"]["reviewed_variant"] == SUPER
    assert loader["values"]["auto_offload"] is False and loader["values"]["offload_mode"] == "none"
    assert loader["values"]["dtype"] == "bfloat16"
    profiles, reason = resolve_execution_profiles_for_loader(loader["module"], loader["action"], loader["values"])
    assert reason is None and [row.id for row in profiles] == [PROFILE]
    denoise, prompt = nodes["WorkflowCosmos3OmniDenoise"], nodes["WorkflowCosmos3OmniTextEncode"]
    assert loader["operation"]["binding"]["values"].get("workflow_id") == "text2image"
    assert loader["params"]["workflow_id"]["value"] == "text2image"
    assert loader["values"]["workflow_id"] == "text2image"
    assert denoise["values"]["num_inference_steps"] == 50
    assert denoise["values"]["guidance_scale"] == 4.0 and denoise["values"]["seed"] == 1143
    assert prompt["values"]["width"] == prompt["values"]["height"] == 1024
    assert prompt["values"]["num_frames"] == 1 and prompt["values"]["negative_prompt"] == ""
    source = (Path(__file__).resolve().parents[1] / "data/cosmos3-super-t2i-publisher-caption.v1.json").read_bytes()
    assert hashlib.sha256(source).hexdigest() == "c068a8d430c87bc752c775567113463c8fa3c9370ac7047318852bdc124bd5e3"
    assert json.loads(prompt["values"]["prompt"]) == json.loads(source)
    with pytest.raises(ValueError, match="workflow/revision"):
        resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": PIPELINE, "task": "text_to_video", "executionProfileId": PROFILE,
        })


def test_super_sealed_bundle_cannot_run_video_stage_and_cross_variant_state_is_rejected():
    outputs = {"pipeline_components": {"transformer": {"model_id": "owned-transformer"}}}
    token = issue_pipeline_instance_token(model_type=PIPELINE, repo_id=SUPER, repo_source="hub", revision=REVISION)
    annotate_modular_loader_outputs(outputs, model_type=PIPELINE, repo_id=SUPER, repo_source="hub", revision=REVISION,
                                    trust_remote_code=False, pipeline_instance_token=token)
    bundle = outputs["pipeline_components"]
    assert require_component_binding(bundle, label="owned") is token
    node = stages.WorkflowCosmos3OmniTextEncode("no-video")
    with patch.object(stages, "pipeline_class_from_model_type", side_effect=AssertionError("constructed")):
        with pytest.raises(ValueError, match="workflow/revision"):
            node._prepare_pipeline(pipeline_components=bundle, pipeline_class=PIPELINE,
                                   workflow_id="text2video", block_path="text_encoder")
    foreign = issue_pipeline_instance_token(model_type=PIPELINE, repo_id=NANO, repo_source="hub", revision="a" * 40)
    state = stages._issue_workflow_state(token=foreign, pipeline_class=PIPELINE, workflow_id="text2image",
                                        completed_stage="text_encoder", state=object())
    with pytest.raises(ValueError, match="different Models Loader"):
        stages._require_workflow_state(state, token=token, pipeline_class=PIPELINE,
                                      workflow_id="text2image", completed_stage="text_encoder")
