"""Qwen control strength is a multiplier, distinct from its [0, 1] window."""

from types import SimpleNamespace

import diffusers
import pytest
import torch
from diffusers.modular_pipelines import BlockState
from diffusers.modular_pipelines.qwenimage.denoise import QwenImageLoopBeforeDenoiserControlNet

from modules.ModularDiffusers import controlnet as control_module
from modules.ModularDiffusers.loaders import annotate_modular_loader_outputs
from modules.ModularDiffusers.modular_utils import normalize_modular_runtime_params, require_modiff_node_contract
from modules.ModularDiffusers.route_state import (
    bind_standalone_component_output, issue_pipeline_instance_token, issue_standalone_component_issuer,
)


def qwen_contract():
    return require_modiff_node_contract(diffusers.QwenImageModularPipeline, "controlnet")


@pytest.mark.parametrize("scale", [1.1, 1.2, 2.0])
def test_original_qwen_strengths_normalize_without_clamping(scale):
    _, config = qwen_contract()
    assert normalize_modular_runtime_params({"controlnet_conditioning_scale": scale}, config)[
        "controlnet_conditioning_scale"] == scale
    assert config["params"]["controlnet_conditioning_scale"]["default"] == .5


@pytest.mark.parametrize("value", [-.01, 2.01, float("nan"), float("inf"), True])
def test_qwen_strength_still_rejects_invalid_values(value):
    _, config = qwen_contract()
    with pytest.raises(ValueError):
        normalize_modular_runtime_params({"controlnet_conditioning_scale": value}, config)


@pytest.mark.parametrize("field", ["control_guidance_start", "control_guidance_end"])
def test_window_remains_bounded_at_one(field):
    _, config = qwen_contract()
    with pytest.raises(ValueError):
        normalize_modular_runtime_params({field: 1.01}, config)


def test_sdxl_strength_schema_is_unchanged():
    _, config = require_modiff_node_contract(diffusers.StableDiffusionXLModularPipeline, "controlnet",
                                           require_blocks=False)
    assert config["params"]["controlnet_conditioning_scale"]["max"] == 1.0


@pytest.mark.parametrize("scale", [1.1, 1.2])
def test_actual_qwen_control_node_forwards_original_scale_in_issued_bundle(monkeypatch, scale):
    # Real backend schema, normalization and route publication. Only the VAE
    # image computation is substituted; no pretrained weights are needed.
    blocks, config = qwen_contract()
    latents = torch.zeros((1, 4, 4))
    calls = []
    pipeline = SimpleNamespace(_execution_device=torch.device("cpu"),
                               update_components=lambda **kwargs: None)
    class CapturingPipeline:
        _execution_device = pipeline._execution_device
        update_components = staticmethod(pipeline.update_components)
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(values={"control_image_latents": latents})
    monkeypatch.setattr(blocks, "init_pipeline", lambda **kwargs: CapturingPipeline())
    original_contract = control_module.require_modiff_node_contract
    monkeypatch.setattr(control_module, "require_modiff_node_contract",
                        lambda cls, action, **kwargs: (blocks, config) if action == "controlnet"
                        else original_contract(cls, action, **kwargs))
    token = issue_pipeline_instance_token(model_type="QwenImageModularPipeline", repo_id="test/qwen",
                                         repo_source="hub", revision="a" * 40)
    outputs = {"vae_out": {"model_id": "test-vae"}}
    annotate_modular_loader_outputs(outputs, repo_id="test/qwen", repo_source="hub",
                                    model_type="QwenImageModularPipeline", revision="a" * 40,
                                    trust_remote_code=False, pipeline_instance_token=token)
    control = {"model_id": "test-control", "class_name": "QwenImageControlNetModel",
               "repo_id": "test/control", "repo_source": "hub", "revision": "b" * 40,
               "trust_remote_code": False}
    bind_standalone_component_output(control, issuer=issue_standalone_component_issuer(), component_kind="controlnet",
                                    reviewed_identity=("hub", "test/control", "b" * 40, None,
                                                       "QwenImageControlNetModel", "c" * 64))
    monkeypatch.setattr(control_module.components, "get_components_by_ids", lambda **kwargs: {})
    result = control_module.Controlnet("qwen-scale").execute(
        model_type="QwenImageModularPipeline", controlnet=control, vae=outputs["vae_out"],
        control_image=object(), controlnet_conditioning_scale=scale, control_guidance_start=0.,
        control_guidance_end=1., width=768, height=768, seed=5201,
    )
    assert len(calls) == 1
    assert result["controlnet_bundle"]["controlnet_conditioning_scale"] == scale
    assert result["controlnet_bundle"]["control_image_latents"] is latents
    assert result["route_state_out"] is not None


@pytest.mark.parametrize("scale", [1.1, 1.2])
def test_pinned_control_loop_applies_strength_and_window_without_clamping(scale):
    seen = []
    def model(**kwargs):
        seen.append(kwargs["conditioning_scale"])
        return [torch.ones((1, 2, 4)) * kwargs["conditioning_scale"]]
    owner = SimpleNamespace(controlnet=model)
    state = BlockState(controlnet_keep=[1., 0.], controlnet_conditioning_scale=scale,
                       latent_model_input=torch.zeros((1, 2, 4)), control_image_latents=torch.zeros((1, 2, 4)),
                       timestep=torch.tensor([500.]), img_shapes=[[(1, 1, 2)]],
                       prompt_embeds=torch.zeros((1, 2, 4)), prompt_embeds_mask=None, additional_cond_kwargs={})
    loop = QwenImageLoopBeforeDenoiserControlNet()
    returned, state = loop(owner, state, 0, torch.tensor(500.))
    assert returned is owner
    assert seen == [scale]
    torch.testing.assert_close(state.additional_cond_kwargs["controlnet_block_samples"][0],
                               torch.full((1, 2, 4), scale))
    loop(owner, state, 1, torch.tensor(250.))
    assert seen == [scale, 0.]
