"""Persisted component choices must replay without a browser RPC."""
from unittest.mock import patch

import pytest
import diffusers

from modules import MODULE_MAP
from modules.ModularDiffusers.guiders import Guider, Layers
from modules.ModularDiffusers.schedulers import Scheduler
from modiff.operation_catalog import build_operation_catalog, resolve_operation


def test_guider_replays_persisted_identity_without_a_websocket():
    node = object.__new__(Guider)
    node.node_id = "headless-guider"
    with patch.object(Guider, "get_signal_value", side_effect=AssertionError("browser RPC")):
        output = node.execute("ClassifierFreeGuidance", model_type="StableDiffusionXLModularPipeline", guidance_scale=5)
        assert isinstance(output["guider_out"], diffusers.ClassifierFreeGuidance)
        with pytest.raises(ValueError, match="allowed"):
            node.execute("PerturbedAttentionGuidance", model_type="Flux2KleinModularPipeline")
        with pytest.raises(ValueError, match="allowed"):
            node.execute("ClassifierFreeGuidance", model_type="UnreviewedPipeline")


def test_layers_replay_persisted_identity_without_a_websocket():
    node = object.__new__(Layers)
    with patch.object(Layers, "get_signal_value", side_effect=AssertionError("browser RPC")):
        output = node.execute(model_type="QwenImageModularPipeline", blocks_select=["transformer_blocks"],
                              transformer_blocks={"indices": "0,1"})
        assert output["layers_config"][0]["indices"] == [0, 1]
        with pytest.raises(ValueError, match="allowed"):
            node.execute(model_type="QwenImageModularPipeline", blocks_select=["single_transformer_blocks"],
                         single_transformer_blocks={"indices": "0"})


def test_scheduler_selection_can_use_the_connected_runtime_model_identity():
    node = object.__new__(Scheduler)
    with patch.object(Scheduler, "get_signal_value", side_effect=AssertionError("browser RPC")):
        assert node._selected_scheduler("DDIMScheduler", "StableDiffusionXLModularPipeline") == "DDIMScheduler"
        with pytest.raises(ValueError, match="allowed"):
            node._selected_scheduler("DDIMScheduler", "UnreviewedPipeline")


def test_controlnet_starter_persists_exact_identity_for_recreated_instances():
    from modules.ModularDiffusers.modular_utils import pipeline_class_from_runtime_inputs

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    checked = 0
    for contract in contracts:
        if contract["nodeKey"] != "modules.ModularDiffusers.Controlnet":
            continue
        draft = resolve_operation(MODULE_MAP, contracts, {
            key: contract[key] for key in ("pipelineClass", "task", "operationId")
        })
        assert draft["params"]["model_type"]["value"] == contract["pipelineClass"]
        assert pipeline_class_from_runtime_inputs(None, draft["values"]).__name__ == contract["pipelineClass"]
        checked += 1
    assert checked >= 2
