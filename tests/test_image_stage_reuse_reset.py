"""Actual resident loader adoption must invalidate removed LoRA material."""

import pytest
import torch
from peft.tuners.tuners_utils import BaseTunerLayer

from test_image_stage_graph import runtime as graph_runtime
from test_image_stage_lora_epoch import loaded, weight_file, apply_adapter
from modules.DiffusersImage import stage_execution as core

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

# Re-export the shared pytest fixture under its injected name.
runtime = graph_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


@pytest.mark.parametrize("kind", ["fill", "control"])
@pytest.mark.parametrize("check", ["epoch", "scale_metadata"])
def test_actual_loader_workflow_reuse_resets_official_adapter_material(runtime, tmp_path, kind, check):
    owner, strategy, given = loaded(runtime, kind)
    base = core.encode(owner, strategy, given)
    adapter = tmp_path / "style.safetensors"
    digest = weight_file(owner, adapter, 1.0)
    apply_adapter(owner, adapter, digest, identifier="actual-style")
    encoded = core.encode(owner, strategy, given)
    assert any(isinstance(layer, BaseTunerLayer) for layer in owner.transformer.modules())
    assert owner._modiff_adapter_scales == {"style": 0.35}
    loader = runtime[0].node_cache["loader"]
    assert loader.output["pipeline"] is owner
    loader.prepare_for_workflow_reuse()
    assert loader.output["pipeline"] is owner
    assert not any(isinstance(layer, BaseTunerLayer) for layer in owner.transformer.modules())
    if check == "epoch":
        with pytest.raises(ValueError):
            core.require_encoding(owner, strategy, encoded)
    else:
        assert owner._modiff_adapter_scales == {}
        reset = core.encode(owner, strategy, given)
        assert torch.equal(reset.material["pooled_prompt_embeds"], base.material["pooled_prompt_embeds"])


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_actual_no_adapter_whole_loader_reuse_retains_valid_stage_material(runtime, kind):
    owner, strategy, given = loaded(runtime, kind)
    before = core.encode(owner, strategy, given)
    assert not any(isinstance(layer, BaseTunerLayer) for layer in owner.transformer.modules())
    runtime[0].node_cache["loader"].prepare_for_workflow_reuse()
    assert core.require_encoding(owner, strategy, before) is before
