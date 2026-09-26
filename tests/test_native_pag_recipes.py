"""Native PAG recipes exercise official guidance; no model downloads required."""
from unittest.mock import patch

import pytest
import importlib.util
import torch

from modules import MODULE_MAP
from modules.ModularDiffusers.guiders import Guider, Layers
from modiff.diffusers_profiles import public_execution_profiles, resolve_execution_profiles_for_loader
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter


@pytest.fixture(scope="module")
def catalog():
    return build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})


@pytest.mark.parametrize("task", [
    "text_to_image", "image_to_image", "inpaint", "control_image", "control_edit_image",
])
def test_pag_recipes_have_one_explicit_guider_and_preserve_native_stage_wires(catalog, task):
    contracts, support = catalog
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed model")):
        result = resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": "StableDiffusionXLModularPipeline", "task": task,
            "executionProfileId": "sdxl-pag:modular",
        })
    nodes = {node["operation"]["operationId"]: node for node in result["nodes"]}
    assert {"diffusion.encode_prompt", "diffusion.denoise", "diffusion.decode_latents"} <= nodes.keys()
    loader = nodes["diffusion.load_models"]
    profiles, reason = resolve_execution_profiles_for_loader(loader["module"], loader["action"], loader["values"])
    assert reason is None and [p.id for p in profiles] == ["sdxl-base:modular"]
    assert loader["values"]["model_type"] == "StableDiffusionXLModularPipeline"
    guidance = nodes["diffusion.guidance"]["values"]
    assert guidance["guider"] == "PerturbedAttentionGuidance"
    assert guidance["perturbed_guidance_scale"] == 3
    assert guidance["perturbed_guidance_start"] == 0 and guidance["perturbed_guidance_stop"] == 1
    layers = nodes["diffusion.guidance_layers"]["values"]
    assert layers["blocks_select"] == ["mid_block.attentions.0.transformer_blocks"]
    assert layers["mid_block.attentions.0.transformer_blocks"]["indices"] == "0,1,2,3,4,5,6,7,8,9"
    consumers = {e["target"] for e in result["edges"] if e["source"] == "diffusion.guidance"}
    expected = {key for key, node in nodes.items() if "guider" in node["params"] and key != "diffusion.guidance"}
    assert consumers == expected
    assert not any(item["field"] == "guider" for item in result["requiredInputs"])
    row = next(p for p in support if p["pipelineClass"] == result["pipelineClass"])
    assert "sdxl-pag:modular" in next(t for t in row["tasks"] if t["task"] == task)["executionProfileIds"]


def test_pag_profile_does_not_leak_to_unreviewed_tasks_or_change_old_standard_profile(catalog):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    contracts, _ = catalog
    with pytest.raises(ValueError, match="pipeline/task"):
        resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": "StableDiffusionXLModularPipeline", "task": "control_inpaint",
            "executionProfileId": "sdxl-pag:modular",
        })
    assert DIFFUSERS_EXECUTION_PROFILES["sdxl-pag:direct"].pipeline_class == "StableDiffusionXLPAGPipeline"


def _guider(**changes):
    node = object.__new__(Guider)
    node.node_id = "pag-test"
    return node.execute(**{
        "guider": "PerturbedAttentionGuidance", "model_type": "StableDiffusionXLModularPipeline",
        "layers_config": [{"fqn": "mid_block.attentions.0.transformer_blocks", "indices": list(range(10))}],
        "guidance_scale": 5.0, "perturbed_guidance_scale": 3.0,
        "perturbed_guidance_start": 0.0, "perturbed_guidance_stop": 1.0, **changes,
    })["guider_out"]


def test_pag_recipe_uses_all_mid_layers_and_layers_reject_other_pipeline():
    guider = _guider()
    config, = guider.skip_layer_config
    assert config.indices == list(range(10))
    assert config.fqn == "mid_block.attentions.0.transformer_blocks"
    assert config.skip_attention_scores and not config.skip_attention and not config.skip_ff
    node = object.__new__(Layers)
    block = "mid_block.attentions.0.transformer_blocks"
    with pytest.raises(ValueError, match="allowed"):
        node.execute(model_type="QwenImageModularPipeline", blocks_select=[block], **{block: {"indices": "0"}})


def test_actual_upstream_attention_changes_only_during_perturbed_pass_and_restores():
    from diffusers.models.attention import BasicTransformerBlock
    from diffusers.modular_pipelines.modular_pipeline import BlockState

    # The real upstream hook traverses precisely the same FQN as SDXL. Tiny
    # random CPU layers test attention semantics, not model/image quality.
    torch.manual_seed(17)
    model = torch.nn.Module()
    model.mid_block = torch.nn.Module()
    attention = torch.nn.Module()
    attention.transformer_blocks = torch.nn.ModuleList([
        BasicTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4, cross_attention_dim=8)
        for _ in range(10)
    ])
    model.mid_block.attentions = torch.nn.ModuleList([attention])
    layer = attention.transformer_blocks[0]
    inputs, context = torch.randn(1, 4, 8), torch.randn(1, 3, 8)
    baseline = layer(inputs, encoder_hidden_states=context)
    guider = _guider()
    guider.set_state(step=0, num_inference_steps=2, timestep=torch.tensor(999))
    # At this pin the native guider's start boundary is exclusive. Do not
    # misrepresent it as bitwise-equivalent to the standard PAG pipeline.
    assert len(guider.prepare_inputs({"context": (context, context)})) == 2
    guider.set_state(step=1, num_inference_steps=2, timestep=torch.tensor(500))
    batches = guider.prepare_inputs({"context": (context, context)})
    assert len(batches) == 3
    predictions = []
    for batch in batches:
        guider.prepare_models(model)
        try:
            batch.noise_pred = layer(inputs, encoder_hidden_states=batch.context)
            predictions.append(batch.noise_pred)
        finally:
            guider.cleanup_models(model)
    assert torch.equal(predictions[0], baseline)
    assert torch.equal(predictions[1], baseline)
    assert not torch.allclose(predictions[2], baseline)
    assert torch.equal(layer(inputs, encoder_hidden_states=context), baseline)
    combined = guider(batches)[0]
    assert torch.allclose(combined, baseline + 3 * (baseline - predictions[2]))
    assert isinstance(batches[0], BlockState)


@pytest.mark.parametrize("profile_id,pipeline,steps,guidance", [
    ("flux-schnell:modular", "FluxModularPipeline", 4, 0.0),
    ("flux-krea:modular", "FluxModularPipeline", 28, 3.5),
    ("sdxl-turbo:modular", "StableDiffusionXLModularPipeline", 1, 0.0),
    ("flux2-klein-kv:t2i-modular", "Flux2KleinModularPipeline", 4, 0.0),
])
def test_exact_native_artifact_defaults_and_loader_admission(catalog, profile_id, pipeline, steps, guidance):
    from modules.ModularDiffusers.loaders import ModelsLoader
    contracts, _ = catalog
    result = resolve_operation_starter(MODULE_MAP, contracts, {
        "pipelineClass": pipeline, "task": "text_to_image", "executionProfileId": profile_id,
    })
    nodes = {n["operation"]["operationId"]: n for n in result["nodes"]}
    loader = nodes["diffusion.load_models"]["values"]
    resolved, reason = resolve_execution_profiles_for_loader("modules.ModularDiffusers", "ModelsLoader", loader)
    assert reason is None and [p.id for p in resolved] == [profile_id]
    _, repository, revision = ModelsLoader._reviewed_builtin_selection(
        model_type=loader["model_type"], repo_id=loader["repo_id"], revision=loader["revision"],
    )
    assert repository == loader["repo_id"]["value"] and revision == loader["revision"]
    denoise = nodes["diffusion.denoise"]["values"]
    assert denoise["num_inference_steps"] == steps and denoise["guidance_scale"] == guidance
    if profile_id == "flux-schnell:modular":
        assert nodes["diffusion.encode_prompt"]["values"]["max_sequence_length"] == 256
    if profile_id == "sdxl-turbo:modular":
        assert denoise["width"] == denoise["height"] == 512
        assert nodes["diffusion.guidance"]["values"]["guidance_scale"] == 0


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="requires the staged optional Transformers runtime")
def test_schnell_prompt_limit_is_enforced_before_pipeline_construction():
    import diffusers
    from modules.ModularDiffusers.embeddings import EncodePrompt
    node = object.__new__(EncodePrompt)
    node._pipeline_class = diffusers.FluxModularPipeline
    with pytest.raises(ValueError, match="schnell.*256"):
        node.execute(text_encoders={"model_type": "FluxModularPipeline",
                                   "repo_id": "black-forest-labs/FLUX.1-schnell"},
                     prompt="test", max_sequence_length=300)


def test_operation_recipes_do_not_replace_legacy_auto_owners():
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, execution_profiles_for_execution
    for model_type, expected in (("FluxSchnellPipeline", "flux-schnell:direct"),
                                 ("FluxKreaPipeline", "flux-krea:direct")):
        assert [p.id for p in execution_profiles_for_execution(model_type, "text_to_image")] == [expected]
    recipe = DIFFUSERS_EXECUTION_PROFILES["sdxl-pag:modular"]
    assert recipe.to_public_dict()["operation_recipe"] is True
    assert recipe.public and recipe.pipeline_class == "StableDiffusionXLModularPipeline"
