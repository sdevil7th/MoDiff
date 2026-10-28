"""Planning must inspect the same immutable variant that ModelsLoader executes."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from modiff import workflow_auto_resource as planner
from modiff.diffusers_profiles import resolve_execution_profiles_for_loader


MODEL = "QwenImageModularPipeline"
BASE = "Qwen/Qwen-Image-2512"
BASE_REVISION = "25468b98e3276ca6700de15c6628e51b7de54a26"
ORIGINAL = "Qwen/Qwen-Image"
ORIGINAL_REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
PACKED = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"
PACKED_REVISION = "f50b8c24fe21e9265509b15113b7cca82d0a4443"


def graph(variant=ORIGINAL):
    values = {
        "model_type": MODEL, "repo_id": {"source": "hub", "value": BASE},
        "revision": BASE_REVISION, "workflow_id": "text2image",
        "reviewed_variant": variant, "dtype": "bfloat16", "device": "cuda:0",
        "offload_mode": "none", "auto_offload": False,
    }
    return {"nodes": {
        "load": {"module": "modules.ModularDiffusers", "action": "ModelsLoader",
                 "params": {key: {"value": value} for key, value in values.items()}},
        "denoise": {"module": "modules.ModularDiffusers", "action": "Denoise", "params": {
            "pipeline_components": {"sourceId": "load", "sourceKey": "pipeline_components"},
            "mode": {"value": "modular_text_to_image"}, "width": {"value": 1024},
            "height": {"value": 1024}, "num_inference_steps": {"value": 50}}},
        "preview": {"module": "modules.Image", "action": "Preview", "params": {
            "image": {"sourceId": "denoise", "sourceKey": "latents"}}},
    }, "paths": [["load", "denoise", "preview"]]}


def inspect(source, tmp_path, repository, revision):
    profiles, reason = resolve_execution_profiles_for_loader(
        "modules.ModularDiffusers", "ModelsLoader",
        {key: value.get("value") for key, value in source["nodes"]["load"]["params"].items()},
    )
    assert reason is None and len(profiles) == 1
    profile = profiles[0]
    requests = []

    def recipe(payload, **_kwargs):
        requests.append(deepcopy(payload))
        return {"candidates": [{
            "id": "controlled-resource-contract", "executionProfileId": profile.id,
            "modelRepo": repository, "artifactRevision": revision, "modelType": MODEL,
            "mode": "modular_text_to_image", "loaderModule": profile.loader_module,
            "loaderAction": profile.loader_action, "executionPath": profile.execution_path,
            "dtype": "bfloat16", "quantizationMode": "none", "offloadMode": "none",
            "autoOffload": False, "canAutoRun": True, "proof": {"status": "declared_safe"},
            "generation": {"width": 1024, "height": 1024, "steps": 50},
            "requirements": {"systemRamBytes": 300, "vramBytes": 200, "diskFreeBytes": 0},
        }]}

    hardware = {"accelerator": {"freeBytes": 1000, "memoryKind": "dedicated"},
                "systemMemory": {"availableBytes": 1000}, "offloadDisk": {"freeBytes": 1000}}
    result = planner.build_workflow_auto_plan(
        source, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path),
        plan_recipe=recipe, hardware=hardware,
    )
    return result, requests


@pytest.mark.parametrize("repository,revision", [
    (ORIGINAL, ORIGINAL_REVISION), (PACKED, PACKED_REVISION),
])
def test_selected_variant_is_planned_with_its_own_immutable_repository_and_revision(
        tmp_path, repository, revision):
    source = graph(repository)
    before = deepcopy(source)
    result, requests = inspect(source, tmp_path, repository, revision)
    assert requests[0]["form"]["modelRepo"] == repository
    assert requests[0]["form"]["revision"] == revision
    assert result["canAutoRun"], result["issues"]
    assert result["loaders"][0]["repository"] == repository
    assert source == before


@pytest.mark.parametrize("variant", [None, "", "   "])
def test_empty_variant_keeps_the_existing_repository_and_revision(tmp_path, variant):
    result, requests = inspect(graph(variant), tmp_path, BASE, BASE_REVISION)
    assert requests[0]["form"]["modelRepo"] == BASE
    assert requests[0]["form"]["revision"] == BASE_REVISION
    assert result["canAutoRun"], result["issues"]


def test_absent_variant_keeps_the_existing_repository_and_revision(tmp_path):
    source = graph()
    del source["nodes"]["load"]["params"]["reviewed_variant"]
    result, requests = inspect(source, tmp_path, BASE, BASE_REVISION)
    assert requests[0]["form"]["modelRepo"] == BASE
    assert requests[0]["form"]["revision"] == BASE_REVISION
    assert result["canAutoRun"], result["issues"]


@pytest.mark.parametrize("variant", ["unreviewed/model", "Qwen/Qwen-Image-Edit-2511", 4, False])
def test_invalid_variant_never_reaches_recipe_inspection(tmp_path, variant):
    result, requests = inspect(graph(variant), tmp_path, BASE, BASE_REVISION)
    assert not result["canAutoRun"]
    assert not requests
    assert any("variant" in issue or "selection" in issue for issue in result["issues"])


def test_connected_variant_is_inspected_without_executing_the_producer(tmp_path):
    source = graph()
    source["nodes"]["variant"] = {"module": "modules.Primitive", "action": "TextValue",
                                   "params": {"text": {"value": ORIGINAL}}}
    source["nodes"]["load"]["params"]["reviewed_variant"] = {
        "sourceId": "variant", "sourceKey": "output"}
    source["paths"][0].insert(0, "variant")
    result, requests = inspect(source, tmp_path, ORIGINAL, ORIGINAL_REVISION)
    assert requests[0]["form"]["modelRepo"] == ORIGINAL
    assert requests[0]["form"]["revision"] == ORIGINAL_REVISION
    assert result["resolvedFields"]["load"]["reviewed_variant"] == ORIGINAL
    assert result["canAutoRun"], result["issues"]


def test_unknown_connected_variant_requires_normal_preparation(tmp_path):
    source = graph()
    source["nodes"]["variant"] = {"module": "modules.Primitive", "action": "TextValue",
                                   "params": {"text": {"value": ORIGINAL}}}
    source["nodes"]["load"]["params"]["reviewed_variant"] = {
        "sourceId": "variant", "sourceKey": "unreviewed_output"}
    source["paths"][0].insert(0, "variant")
    result, requests = inspect(source, tmp_path, BASE, BASE_REVISION)
    assert not requests
    assert result["requiresPreparation"]
    assert result["preparationNodeIds"] == ["variant"]
    assert not result["loaders"] and not result["patches"]
    assert result["deferredFields"] == [{"nodeId": "load", "field": "reviewed_variant",
                                         "sourceId": "variant", "sourceKey": "unreviewed_output"}]


def test_variant_cannot_borrow_another_workflows_checkpoint_choices(tmp_path):
    source = graph(ORIGINAL)
    source["nodes"]["load"]["params"]["workflow_id"]["value"] = "image2image"
    result, requests = inspect(source, tmp_path, BASE, BASE_REVISION)
    assert not result["canAutoRun"] and not requests
    assert any("does not admit model variant" in issue for issue in result["issues"])


def test_selected_packed_artifact_keeps_its_windows_auto_restriction(tmp_path):
    from modiff.auto_resource import _apply_catalog_hardware_support

    result, requests = inspect(graph(PACKED), tmp_path, PACKED, PACKED_REVISION)
    assert result["canAutoRun"]  # Synthetic recipe isolates checkpoint selection only.
    assert requests[0]["form"]["modelRepo"] == PACKED
    candidate = {"id": "controlled", "modelType": MODEL, "artifact": PACKED,
                 "canAutoRun": True, "proof": {"status": "declared_safe"}}
    guarded = _apply_catalog_hardware_support([candidate], {
        "platform": "windows", "architecture": "x86_64", "backend": "cuda",
        "accelerator": {"kind": "cuda", "capability": [8, 9]},
    })[0]
    assert not guarded["canAutoRun"]
    assert "Artifact does not support windows" in guarded["requirementsMissing"]


def test_shared_selector_does_not_import_model_libraries():
    root = str(Path(__file__).resolve().parents[1])
    script = (
        "import sys; "
        f"sys.path.insert(0, {root!r}); "
        "from modiff.modular_workflow_contracts import effective_reviewed_modular_selector; "
        "actual = effective_reviewed_modular_selector("
        f"model_type={MODEL!r}, repo_id={{'source': 'hub', 'value': {BASE!r}}}, "
        f"revision={BASE_REVISION!r}, workflow_id='text2image', reviewed_variant={ORIGINAL!r}, "
        f"default_repository={BASE!r}); "
        f"assert actual == ({{'source': 'hub', 'value': {ORIGINAL!r}}}, {ORIGINAL_REVISION!r}); "
        "assert not {'torch', 'diffusers', 'transformers', 'modules.ModularDiffusers'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
