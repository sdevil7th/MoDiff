import pytest

from modules.HuggingFaceTransformers.main import (
    GenerateAnyToAny,
    GenerateImageVideoText,
    _bounded_string,
    image_text_operation_schema,
)


@pytest.mark.parametrize("pipeline,action,node", [
    ("AutoModelForImageTextToText", "GenerateImageVideoText", GenerateImageVideoText),
    ("JanusForConditionalGeneration", "GenerateAnyToAny", GenerateAnyToAny),
])
def test_fresh_caption_authoring_supplies_a_visible_prompt_accepted_by_its_runtime(pipeline, action, node):
    before = node.params["prompt"].copy()
    fields, values = image_text_operation_schema(pipeline, "image_to_text", action)
    prompt = values.get("prompt", node.params["prompt"]["default"])
    assert _bounded_string(prompt, field="prompt") == "Describe the image."
    assert fields["images"]["required"] is True
    assert node.params["prompt"] == before


@pytest.mark.parametrize("pipeline,action", [
    ("AutoModelForImageTextToText", "LoadImageTextToTextModel"),
    ("JanusForConditionalGeneration", "LoadAnyToAnyModel"),
])
def test_caption_authoring_default_does_not_change_loader_controls(pipeline, action):
    fields, values = image_text_operation_schema(pipeline, "image_to_text", action)
    assert fields == {} and values == {"pipeline_class": pipeline}


def test_image_generation_keeps_its_explicit_operator_prompt_policy():
    _, values = image_text_operation_schema("JanusForConditionalGeneration", "text_to_image", "GenerateAnyToAny")
    assert "prompt" not in values and values["generation_mode"] == "image"
