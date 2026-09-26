"""Non-diffusion image tasks use their real existing actions and typed wires."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from modules import MODULE_MAP
from modiff.diffusers_profiles import public_execution_profiles
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter


@pytest.fixture(scope="module")
def catalog():
    return build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})[0]


@pytest.mark.parametrize("pipeline,task,profile,mode,dtype", [
    ("AutoModelForImageTextToText", "image_to_text", "smolvlm-256m-instruct:direct", None, "float32"),
    ("JanusForConditionalGeneration", "image_to_text", "janus-pro-1b:direct", "text", "bfloat16"),
    ("JanusForConditionalGeneration", "text_to_image", "janus-pro-1b:direct", "image", "bfloat16"),
])
def test_caption_and_image_starters_preserve_real_model_handles(catalog, pipeline, task, profile, mode, dtype):
    original = deepcopy(MODULE_MAP["modules.HuggingFaceTransformers"])
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed")):
        draft = resolve_operation_starter(MODULE_MAP, catalog, {
            "pipelineClass": pipeline, "task": task, "executionProfileId": profile,
        })
    loader, generate = draft["nodes"]
    assert loader["params"]["dtype"]["value"] == dtype
    assert len(loader["params"]["revision"]["value"]) == 40
    assert draft["edges"] == [{"source": "model.load", "sourceHandle": "model", "target": generate["operation"]["operationId"], "targetHandle": "model"}]
    assert draft["upstreamBlocks"] == []
    assert generate["params"]["images"]["required"] == (task == "image_to_text")
    if mode:
        assert generate["values"]["generation_mode"] == mode
        assert generate["params"]["image"]["hidden"] == (mode != "image")
        if mode == "image":
            assert generate["values"]["do_sample"] is True
    assert MODULE_MAP["modules.HuggingFaceTransformers"] == original


@pytest.mark.parametrize("task", ["image_adjustment", "image_filter", "image_crop", "image_upscale", "image_stitch", "image_tile", "image_channels", "mask_composite"])
def test_builtin_starters_have_no_fictitious_weights_or_diffusion_stages(catalog, task):
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed")):
        draft = resolve_operation_starter(MODULE_MAP, catalog, {
            "pipelineClass": "BuiltinImageOperationV1", "task": task,
            "executionProfileId": "builtin-image-operations:direct",
        })
    assert len(draft["nodes"]) == 1
    node = draft["nodes"][0]
    assert node["values"] == {"pipeline_class": "BuiltinImageOperationV1", "operation": task}
    assert node["params"]["width"]["hidden"] == (task != "image_crop")
    assert node["params"]["seed"]["hidden"] == (task != "image_filter")
    required = {item["field"] for item in draft["requiredInputs"]}
    assert required == ({"image", "mask"} if task == "mask_composite" else {"image"})
    assert draft["edges"] == [] and draft["upstreamBlocks"] == []


def test_image_text_binding_rejects_unrelated_profile(catalog):
    with pytest.raises(ValueError):
        resolve_operation_starter(MODULE_MAP, catalog, {
            "pipelineClass": "AutoModelForImageTextToText", "task": "image_to_text",
            "executionProfileId": "janus-pro-1b:direct",
        })
