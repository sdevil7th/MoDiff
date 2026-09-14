"""Exercise the actual expanded-workflow runtime constructor without weights."""
from unittest.mock import patch, PropertyMock

import pytest

pytest.importorskip("transformers")
from diffusers import ClassifierFreeGuidance, ClassifierFreeZeroStarGuidance
from diffusers.modular_pipelines import ModularPipeline
from modules.ModularDiffusers import reviewed_blocks


@pytest.mark.parametrize("workflow", ["text2video", "image2video", "video2video"])
@pytest.mark.parametrize("pipeline_class,expected", [
    ("HeliosPyramidModularPipeline", ClassifierFreeZeroStarGuidance),
    ("HeliosModularPipeline", ClassifierFreeGuidance),
    ("HeliosPyramidDistilledModularPipeline", ClassifierFreeGuidance),
])
def test_expanded_runtime_uses_native_guider_before_and_after_control_recreation(workflow, pipeline_class, expected):
    # Suppress only authenticated bundle/model transfer; construct real installed
    # selected blocks and their real from-config components through _new_runtime.
    token = object()
    with (
        patch.object(reviewed_blocks, "_component_bundle_token", return_value=token),
        patch.object(reviewed_blocks, "collect_model_ids", return_value=[]),
        patch.object(ModularPipeline, "pretrained_component_names", new_callable=PropertyMock, return_value=[]),
        patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("no download allowed")),
    ):
        actual_token, pipeline, state = reviewed_blocks._new_runtime(
            bundle={}, pipeline_class=pipeline_class, workflow_id=workflow, execution_scope="selected_workflow")
    assert actual_token is token
    assert type(pipeline.guider) is expected
    assert all(getattr(pipeline, name) is None for name in ("transformer", "vae", "text_encoder", "tokenizer"))
    # This is the official component API used by ReviewedModularWorkflowStep
    # when the graph supplies its unchanged guidance_scale control.
    spec = pipeline.get_component_spec("guider")
    pipeline.update_components(guider=spec.create(guidance_scale=5.0))
    assert type(pipeline.guider) is expected
    assert pipeline.guider.config.guidance_scale == 5.0
    if expected is ClassifierFreeZeroStarGuidance:
        assert pipeline.guider.config.zero_init_steps == 2
        assert pipeline.guider.zero_init_steps == 2
