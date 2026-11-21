"""The reviewed Qwen text Control auxiliary keeps its own placement policy."""

import copy

from modiff.huggingface_cluster_admission import audit_reviewed_cluster_execution_candidates
from modiff.huggingface_node_library import build_huggingface_node_library
from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS, studio_execution_spec_for_pair


def _control_admission():
    identity = "diffusers.cluster-admission:QwenImageModularPipeline:controlnet_text2image:mode:control_image"
    return next(
        result
        for result in audit_reviewed_cluster_execution_candidates(build_huggingface_node_library())
        if result["id"] == identity
    )


def test_qwen_text_control_materializes_separate_auxiliary_and_base_placement():
    """Resolve actual spec/admission bindings with the ordinary Auto base policy."""
    spec = studio_execution_spec_for_pair("QwenImageModularPipeline", "control_image")
    admission = _control_admission()
    assert admission["status"] == "admitted"
    assert admission["studioExecutionSpec"]["id"] == spec["id"]
    sources = {
        **admission["sealedBindingValues"],
        "autoOffload": True,
        "offloadMode": "model_cpu",
        "dtype": "bfloat16",
        "device": "cuda:0",
    }
    fields = {
        (role, field): sources[source]
        for role, field, source in spec["bindings"]
        if field in {"auto_offload", "offload_mode", "dtype", "device"}
    }
    assert fields["models", "auto_offload"] is True
    assert fields["models", "offload_mode"] == "model_cpu"
    assert fields["controlnetModel", "auto_offload"] is False
    assert fields["controlnetModel", "offload_mode"] == "none"
    for field in ("dtype", "device"):
        assert fields["models", field] == fields["controlnetModel", field]
    assert "controlnetOffloadMode" not in admission["executionParameterSources"]
    assert admission["sealedBindingValues"]["controlnetOffloadMode"] == "none"


def test_only_qwen_text_control_seals_auxiliary_placement():
    text = STUDIO_EXECUTION_SPEC_DEFINITIONS["qwen-image-2512:control-image:v1"]
    assert ("controlnetModel", "auto_offload", "false") in text["bindings"]
    assert ("controlnetModel", "offload_mode", "controlnetOffloadMode") in text["bindings"]
    for spec in STUDIO_EXECUTION_SPEC_DEFINITIONS.values():
        if spec is text:
            continue
        assert not any(source == "controlnetOffloadMode" for _role, _field, source in spec.get("bindings", ()))
        if spec["modelType"] == "QwenImageModularPipeline" and spec["mode"] in {
            "control_edit_image", "control_inpaint",
        }:
            assert ("controlnetModel", "auto_offload", "autoOffload") in spec["bindings"]
            assert ("controlnetModel", "offload_mode", "offloadMode") in spec["bindings"]


def test_sealed_auxiliary_values_are_detached_from_caller_edits():
    first = _control_admission()
    original = copy.deepcopy(first)
    first["sealedBindingValues"]["false"] = True
    first["sealedBindingValues"]["controlnetOffloadMode"] = "model_cpu"
    second = _control_admission()
    assert second == original
    assert second["sealedBindingValues"]["controlnetOffloadMode"] == "none"
