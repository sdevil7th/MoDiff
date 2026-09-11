"""Legacy misplaced seeds require explicit repair, never silent RNG changes."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("transformers")
from diffusers.modular_pipelines.helios.before_denoise import HeliosPrepareHistoryStep
from modules.ModularDiffusers import reviewed_blocks


class State:
    def __init__(self):
        self.values = {}

    def set(self, key, value, kwargs_type=None):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)


@pytest.mark.parametrize("pipeline_class", [
    "HeliosModularPipeline", "HeliosPyramidModularPipeline", "HeliosPyramidDistilledModularPipeline",
])
@pytest.mark.parametrize("workflow", ["text2video", "image2video", "video2video"])
@pytest.mark.parametrize("existing_generator", [False, True])
def test_history_seed_preserves_legacy_behavior_until_explicit_repair(pipeline_class, workflow, existing_generator):
    definition = reviewed_blocks._reviewed_definition(pipeline_class, workflow)
    placement = next(p for p in definition["blockPlacements"] if p["path"] == ["denoise.prepare_history"])
    block = next(b for b in reviewed_blocks.reviewed_huggingface_node_library()["blockDefinitions"]
                 if b["id"] == placement["blockDefinitionId"])
    inputs = HeliosPrepareHistoryStep().inputs
    assert "generator" not in {i.name for i in inputs}
    # Preserve the real installed block's input contract, but avoid transformer
    # loading and tensor preparation. Exercise the actual graph-node executor.
    fake_type = type("HeliosPrepareHistoryStep", (), {"__call__": lambda self, p, s: (p, s)})
    leaf = fake_type()
    leaf.inputs = inputs
    pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={"denoise.prepare_history": leaf}),
                               component_names=(), _execution_device="cpu")
    for seed in [42, 42, 43]:
        state = State()
        generator = torch.Generator().manual_seed(81) if existing_generator else None
        if generator is not None:
            torch.randn((32,), generator=generator)
            state.set("generator", generator)
            rng_state = generator.get_state().clone()
        node = reviewed_blocks.ReviewedModularWorkflowStep()
        with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
            node.execute(
                pipeline_class=pipeline_class, workflow_id=workflow,
                placement_path=placement["path"], block_definition_id=block["id"],
                block_class=block["className"], block_contract_hash=block["contentHash"],
                execution_kind="step", pipeline_components={}, seed=seed)
        assert state.get("generator") is generator
        if generator is not None:
            assert torch.equal(generator.get_state(), rng_state)
        assert "seed" not in node._execution_input_record["fields"]
