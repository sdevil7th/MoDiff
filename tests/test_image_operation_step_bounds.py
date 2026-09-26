"""Ordinary image forms must publish the step limits enforced at execution."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from modules.DiffusersImage.main import (
    Generate, IMAGE_PIPELINE_ADAPTERS, IMAGE_ACTION_MODES,
    image_pipeline_contract, image_action_field_params,
)


def test_prx_form_does_not_offer_steps_its_executor_rejects():
    adapter = IMAGE_PIPELINE_ADAPTERS["PRXPipeline"]
    fields = image_action_field_params(image_pipeline_contract(adapter, "text_to_image"), "Generate")
    assert fields["num_inference_steps"]["min"] == 1
    assert fields["num_inference_steps"]["max"] == adapter.max_inference_steps == 28


@pytest.mark.parametrize("pipeline", sorted(IMAGE_PIPELINE_ADAPTERS))
def test_prompt_conditioned_actions_publish_the_selected_adapter_bound(pipeline):
    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline]
    for mode in adapter.mode_options:
        contract = image_pipeline_contract(adapter, mode)
        for action, modes in IMAGE_ACTION_MODES.items():
            if mode not in modes or action in {"UnconditionalGenerate", "PredictMap"}:
                continue
            original = deepcopy(contract)
            field = image_action_field_params(contract, action)["num_inference_steps"]
            assert field == {"min": 1, "max": adapter.max_inference_steps}
            assert contract == original


def test_dynamic_model_switch_refreshes_bound_without_rewriting_authored_steps():
    node = object.__new__(Generate)
    node.class_name = "Generate"
    node.set_field_params = Mock()
    for pipeline in ("PRXPipeline", "FluxPipeline", "PRXPipeline"):
        adapter = IMAGE_PIPELINE_ADAPTERS[pipeline]
        node.set_field_params.reset_mock()
        Generate.update_image_contract(node, {
            "image_contract": image_pipeline_contract(adapter, "text_to_image"),
            "num_inference_steps": 17,
        }, None)
        node.set_field_params.assert_any_call("num_inference_steps", {"min": 1, "max": adapter.max_inference_steps})
        for args, _ in node.set_field_params.call_args_list:
            assert "value" not in args[1]


def test_unconditional_and_perception_keep_their_own_step_contracts():
    for pipeline, mode, action in (
        ("DDPMPipeline", "unconditional_image", "UnconditionalGenerate"),
        ("MarigoldDepthPipeline", "depth_estimation", "PredictMap"),
    ):
        fields = image_action_field_params(image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS[pipeline], mode), action)
        assert "num_inference_steps" not in fields
