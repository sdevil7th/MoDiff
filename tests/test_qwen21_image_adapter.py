"""Generic image contracts for Qwen 2.1; no weights or optional imports required."""
from types import SimpleNamespace

import pytest
from PIL import Image

from modules.DiffusersImage.call_inputs import apply_call_inputs, normalize_call_inputs, record_image_call_inputs
from modules.DiffusersImage.main import (
    IMAGE_PIPELINE_ADAPTERS, _tag_image_pipeline, image_pipeline_contract,
    preflight_image_action, prepare_reference_images,
)


PIPELINE = "QwenImage21Pipeline"
REPO = "Qwen/Qwen-Image-2.1"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def runtime(mode):
    pipeline = type(PIPELINE, (), {})()
    _tag_image_pipeline(pipeline, IMAGE_PIPELINE_ADAPTERS[PIPELINE], mode, REPO, "hub", REVISION)
    return pipeline


@pytest.mark.parametrize("mode,action,count", [
    ("text_to_image", "Generate", 0),
    ("edit_image", "Edit", 1),
    ("multi_image_reference_edit", "Edit", 10),
])
def test_same_adapter_handles_generation_and_reference_editing(mode, action, count):
    images = [Image.new("RGBA", (32, 32), (10, 20, 30, 80)) for _ in range(count)]
    adapter, values = preflight_image_action(runtime(mode), action, {
        "prompt": "Keep the translucent blue glass; change the background.",
        "image": images or None, "width": 1024, "height": 1024,
    })
    contract = image_pipeline_contract(adapter, mode)
    assert contract["fieldParams"]["max_sequence_length"]["hidden"]
    assert contract["fieldParams"]["strength"]["hidden"]
    assert not contract["fieldParams"]["use_kv_cache"]["hidden"]
    assert values["use_kv_cache"] is True
    if images:
        refs = prepare_reference_images(images, adapter)
        refs = refs if isinstance(refs, list) else [refs]
        assert all(a is b for a, b in zip(images, refs, strict=True))
        assert all(image.mode == "RGBA" for image in refs)


@pytest.mark.parametrize("value", [True, False])
def test_cache_flag_reaches_native_call_and_consumed_receipt(value):
    selected = normalize_call_inputs(PIPELINE, {"use_kv_cache": value})
    target = {}
    apply_call_inputs(selected, target)
    assert target["use_kv_cache"] is value
    records = []
    node = SimpleNamespace(record_generation_inputs=records.append)
    record_image_call_inputs(node, selected, target, IMAGE_PIPELINE_ADAPTERS[PIPELINE])
    assert records[0]["use_kv_cache"] is value


@pytest.mark.parametrize("value", [0, 1, "false", "true", {}, []])
def test_cache_control_rejects_coercion_and_mutable_cache_objects(value):
    with pytest.raises(ValueError, match="use_kv_cache must be a boolean"):
        normalize_call_inputs(PIPELINE, {"use_kv_cache": value})


def test_cache_control_does_not_leak_to_other_model_families():
    with pytest.raises(ValueError, match="does not support optional input"):
        normalize_call_inputs("FluxPipeline", {"use_kv_cache": True})
    assert "use_kv_cache" not in normalize_call_inputs("FluxPipeline", {})


def test_reference_and_output_limits_remain_bounded():
    with pytest.raises(ValueError, match="at most 10"):
        preflight_image_action(runtime("multi_image_reference_edit"), "Edit", {
            "image": [Image.new("RGB", (16, 16))] * 11,
        })
    preflight_image_action(runtime("text_to_image"), "Generate", {"width": 2752, "height": 1536})
    with pytest.raises(ValueError, match="output cannot exceed"):
        preflight_image_action(runtime("text_to_image"), "Generate", {"width": 4096, "height": 4096})
    assert IMAGE_PIPELINE_ADAPTERS["QwenImagePipeline"].max_output_side == 2048


def test_qwen21_starters_share_existing_nodes_and_exact_runtime_requirement():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    from modiff.optional_runtimes import TRANSFORMERS_517_PEFT_RUNTIME_PROFILE_ID

    contracts = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})[0]
    profile = DIFFUSERS_EXECUTION_PROFILES["qwen-image-21:direct"]
    assert profile.optional_runtime_profiles == (TRANSFORMERS_517_PEFT_RUNTIME_PROFILE_ID,)
    for task in ("text_to_image", "edit_image", "multi_image_reference_edit"):
        graph = resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": PIPELINE, "task": task, "executionProfileId": profile.id,
        })
        loader = next(n for n in graph["nodes"] if n["action"] == "LoadPipeline")
        action = "Generate" if task == "text_to_image" else "Edit"
        consumer = next(n for n in graph["nodes"] if n["action"] == action)
        assert loader["values"]["revision"] == REVISION
        assert consumer["params"]["use_kv_cache"]["default"] is True
        assert consumer["params"]["width"]["value"] == 2048
        assert consumer["params"]["guidance_scale"]["value"] == 1
        assert all("Qwen" not in n["action"] for n in graph["nodes"])
