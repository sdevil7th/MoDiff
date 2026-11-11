"""The H3 Image provider has its own immutable library and target admission."""
from modiff.optional_runtimes import (
    DIFFUSERS_H3_IMAGE_RUNTIME_PROFILE_ID,
    DIFFUSERS_VIDEO_RUNTIME_PROFILE_ID,
    OPTIONAL_RUNTIME_PROFILES,
)
from modiff.runtime_source_builds import validate_source_build_contract


def test_minimax_image_source_build_and_target_contract():
    profile = OPTIONAL_RUNTIME_PROFILES[DIFFUSERS_H3_IMAGE_RUNTIME_PROFILE_ID]
    build, = profile.source_builds
    validated = validate_source_build_contract(build)
    source = validated["sourceArtifact"]
    assert source["repository"] == "huggingface/diffusers"
    assert source["commit"] == "875373ac7f5856071e345a3e394c74efccc6b9e6"
    assert source["sha256"] == "9b2e58957ef3039c6dd3ee75aeb11aacfb7b156a5552f60ab424d9e3176111dc"
    assert validated["outputWheel"]["sha256"] == "f6f0af6b7dde686809b44bde23fa77228960e5d6d6f0430f2dc92d88cd861504"
    assert profile.provides_diffusers
    assert profile.pipeline_adapter_symbols == ("MiniMaxH3ImageModularPipeline",)
    assert {"MiniMaxH3ImageBlocks", "MiniMaxH3ImageTransformer2DModel", "AutoencoderKLMiniMaxH3Image"} <= set(
        profile.required_diffusers_symbols
    )
    video = OPTIONAL_RUNTIME_PROFILES[DIFFUSERS_VIDEO_RUNTIME_PROFILE_ID]
    assert video.spec_digest != profile.spec_digest
    for platform, machine in (("linux", "x86_64"), ("windows", "x86_64"), ("macos", "arm64"), ("linux", "aarch64")):
        target = profile.contract_for_target(platform_name=platform, machine=machine)
        qualified = (platform, machine) in {("linux", "x86_64"), ("windows", "x86_64")}
        assert target.contract_state == ("qualified" if qualified else "candidate_unqualified")
        assert target.cutover_ready is qualified
        assert target.install_action_available is qualified
        assert target.activation_available is qualified



def test_image_workflows_use_the_generic_library_with_exact_graph_admission():
    from modiff.huggingface_node_library import build_huggingface_node_library
    from modiff.modular_conditional_contracts import reviewed_modular_conditional_collection
    from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract

    pipeline = "MiniMaxH3ImageModularPipeline"
    contract = reviewed_modular_workflow_contract(pipeline)
    assert {w["id"]: w["taskId"] for w in contract["workflows"]} == {
        "text2image": "text_to_image", "image_conditioned": "edit_image",
    }
    assert all(
        not {"num_inference_steps", "guidance_scale", "negative_prompt"} & {f["name"] for f in w["inputs"]}
        for w in contract["workflows"]
    )
    library = build_huggingface_node_library()
    entries = [item for item in library["definitions"] if item.get("pipelineClass") == pipeline]
    assert len(entries) == 2
    assert all(item["integrationStatus"] == "reviewed_modular_workflow_route" and item["executionAdmissions"] for item in entries)
    source = next(s for s in library["sourceSupplements"] if s["runtimeProfileId"] == DIFFUSERS_H3_IMAGE_RUNTIME_PROFILE_ID)
    assert set(source["definitionIds"]) == {item["id"] for item in entries}
    snapshot = next(s for s in reviewed_modular_conditional_collection()["sourceSupplements"]
                    if s["diffusersRevision"] == source["libraryRevision"])
    assert [p["pipelineClass"] for p in snapshot["pipelines"]] == [pipeline]


def test_pre_image_v7_library_preserves_its_exact_source_membership(monkeypatch):
    import pytest
    from modiff import supplemental_modular_contracts as sources
    from modiff.huggingface_node_library import (
        HuggingFaceNodeLibraryError, build_huggingface_node_library, validate_huggingface_node_library,
    )

    previous_sources = [s for s in sources.reviewed_modular_source_supplements()
                        if s["libraryRevision"] == sources.PIKA_SOURCE_REVISION]
    with monkeypatch.context() as patch:
        patch.setattr(sources, "reviewed_modular_source_supplements", lambda: previous_sources)
        historical = build_huggingface_node_library()
    assert len(historical["definitions"]) == 142
    assert validate_huggingface_node_library(historical) == historical
    historical["sourceSupplements"][0]["schemaVersion"] = True
    with pytest.raises(HuggingFaceNodeLibraryError):
        validate_huggingface_node_library(historical)


def test_rebuilt_generic_stage_recovers_only_its_checkpoint_reference_config(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import pytest
    from modules.ModularDiffusers import reviewed_blocks
    from modules.ModularDiffusers.route_state import issue_pipeline_instance_token

    selected = "MiniMaxH3ImageModularPipeline"
    identity = dict(model_type=selected, repo_id="local-test-fixture", repo_source="local", revision=None)
    config = {"reference_min_short_edge": 32, "reference_max_short_edge": 64,
              "vision_min_short_edge": 32, "vision_max_short_edge": 32, "unrelated": "discard"}
    token = issue_pipeline_instance_token(**identity, model_index=config)
    config["reference_max_short_edge"] = 2048
    with pytest.raises(AttributeError):
        token._workflow_config = ()
    for key, value in (("reference_min_short_edge", True), ("vision_max_short_edge", 0),
                       ("reference_max_short_edge", 4096), ("reference_max_short_edge", 24)):
        with pytest.raises(ValueError, match="reference geometry"):
            issue_pipeline_instance_token(**identity, model_index={**config, key: value})
    pipeline = Mock(pretrained_component_names=())
    blocks = Mock(expected_configs=[SimpleNamespace(name=n) for n in ("reference_min_short_edge", "reference_max_short_edge")])
    blocks.init_pipeline.return_value = pipeline
    monkeypatch.setattr(reviewed_blocks, "_component_bundle_token", lambda *args, **kwargs: token)
    monkeypatch.setattr(reviewed_blocks, "pipeline_class_from_model_type",
                        lambda _: lambda: SimpleNamespace(blocks=blocks))
    reviewed_blocks._new_runtime(bundle={}, pipeline_class=selected, workflow_id="default", execution_scope="selected_workflow")
    assert pipeline.update_components.call_args_list[0].kwargs == {"reference_min_short_edge": 32, "reference_max_short_edge": 64}


def test_pinned_image_artifact_requires_exact_provider_and_workflow():
    import pytest
    from modiff.model_artifact_catalog import catalog_download_inventory, catalog_model, require_catalog_revision, read_model_artifact_catalog
    from modules.ModularDiffusers.loaders import ModelsLoader
    from modules.ModularDiffusers.modular_utils import get_model_type_metadata

    pipe = "MiniMaxH3ImageModularPipeline"
    repo = "MiniMaxAI/MiniMax-H3-Image"
    rev = "492fd8c4eca666f1085f57e30738c1e0d86b3439"
    assert require_catalog_revision(repo, model_type=pipe) == rev
    assert catalog_model(pipe)["baseLicense"] == "minimax-h3-image-community-license-agreement"
    selection, = [inv for inv in read_model_artifact_catalog()["downloadInventories"] if inv["repo"] == repo]
    paths = [entry["path"] for entry in selection["files"]]
    assert len(paths) == 52
    assert catalog_download_inventory(repo, rev, paths)["exactBytes"] == 135_375_974_816
    assert catalog_download_inventory(repo, rev, paths[:-1]) is None
    assert catalog_download_inventory(repo, "a" * 40, paths) is None
    assert get_model_type_metadata(pipe)["execution_status"] == "expert_only"
    ModelsLoader._reviewed_builtin_selection(model_type=pipe, repo_id={"source": "hub", "value": repo}, revision=rev)
    from modiff.modular_workflow_contracts import require_reviewed_modular_repository_workflow
    with pytest.raises(ValueError, match="workflow/revision"):
        require_reviewed_modular_repository_workflow(pipe, repo, rev, "unsupported")
