"""Actual pinned FLUX loops, tiny real transformers, no weights or quality claims."""

import importlib.util

import pytest

from modiff.huggingface_node_library import reviewed_huggingface_node_library


def admissions():
    return [item for item in reviewed_huggingface_node_library()["definitions"]
            if item.get("pipelineClass", "").startswith("Flux")
            and item.get("definitionKind") == "modular_pipeline_workflow"]


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Optional model runtime not active")
@pytest.mark.parametrize("definition", admissions(), ids=lambda item: item["id"])
@pytest.mark.parametrize('masked_attention', [False, True])
def test_flux_iteration_bindings_match_actual_upstream_loop_and_adapter(definition, masked_attention):
    import torch
    from diffusers import FluxTransformer2DModel, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
    from diffusers.guiders import ClassifierFreeGuidance
    from diffusers.modular_pipelines import PipelineState
    from modiff.modular_composition import _default_blocks_resolver, _block_at_path
    from modiff.modular_loop_bindings import bind_upstream_loop_inputs, validate_loop_bindings
    from modules.ModularDiffusers import reviewed_blocks
    from modules.Tensor.main import AttentionArguments

    library = reviewed_huggingface_node_library()
    contracts = {item["id"]: item for item in library["blockDefinitions"]}
    owner = next(item for item in definition["blockPlacements"]
                 if contracts[item["blockDefinitionId"]]["className"].endswith("DenoiseStep"))
    contract = contracts[owner["blockDefinitionId"]]
    tree = _default_blocks_resolver(definition["pipelineClass"], definition["workflowId"])
    pipeline = tree.init_pipeline()
    loop = _block_at_path(pipeline.blocks, tuple(owner["path"]))
    assert type(loop).__name__ == contract["className"]
    torch.manual_seed(42)
    flux2 = definition["pipelineClass"].startswith("Flux2")
    klein = "Klein" in definition["pipelineClass"]
    base = "Base" in definition["pipelineClass"]
    conditioned = definition["workflowId"] == "image_conditioned"
    common = dict(in_channels=8, out_channels=8, num_layers=1, num_single_layers=1,
                  attention_head_dim=16, num_attention_heads=1, joint_attention_dim=8)
    transformer = (Flux2Transformer2DModel(**common, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=not klein)
                   if flux2 else FluxTransformer2DModel(**common, pooled_projection_dim=8,
                                                       axes_dims_rope=(4, 6, 6), guidance_embeds=True))
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False)
    pipeline.update_components(transformer=transformer, scheduler=scheduler)
    if base:
        pipeline.update_components(guider=ClassifierFreeGuidance(guidance_scale=4.0))
    original = dict(loop.sub_blocks)
    members = [{"path": [*owner["path"], name], "blockClass": type(block).__name__}
               for name, block in loop.sub_blocks.items()]
    members[0]["iterationBindings"] = {"latents": {"kind": "state", "sourcePath": members[-1]["path"],
        "output": "latents", "timing": "previous"}}
    initial = torch.randn((1, 4, 8))
    prompts = torch.randn((1, 3, 8))
    negative = torch.randn_like(prompts)
    image_latents = torch.randn((1, 2, 8)) if conditioned else None
    dims = 4 if flux2 else 3
    txt_ids = torch.zeros((1, 3, dims)) if flux2 else torch.zeros((3, dims))
    latent_ids = torch.zeros((1, 4, dims)) if flux2 else torch.zeros((6 if conditioned else 4, dims))

    def run(edited, adapter=False, use_mask=masked_attention):
        scheduler.set_timesteps(3)
        state = PipelineState()
        for name, value in {"latents": initial.clone(), "prompt_embeds": prompts,
            "negative_prompt_embeds": negative, "pooled_prompt_embeds": prompts.mean(dim=1),
            "guidance": torch.tensor([4.0]), "timesteps": scheduler.timesteps, "num_inference_steps": 3,
            "txt_ids": txt_ids, "negative_txt_ids": txt_ids, "img_ids": latent_ids,
            "latent_ids": latent_ids, "image_latents": image_latents,
            "image_latent_ids": torch.zeros((1, 2, dims)) if conditioned else None}.items():
            state.set(name, value)
        options = AttentionArguments('modular-attention-options').execute(
            attention_mask=torch.zeros(1, 1, 1, 1, dtype=torch.bool) if use_mask else None)['options']
        if not adapter:
            state.set('joint_attention_kwargs', options)
        if adapter:
            issued = reviewed_blocks._issue_state(token=object(), pipeline_class=definition["pipelineClass"],
                workflow_id=definition["workflowId"], execution_scope="selected_workflow", pipeline=pipeline,
                state=state, completed_path=tuple(owner["path"][:-1]))
            result = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                pipeline_class=definition["pipelineClass"], workflow_id=definition["workflowId"],
                execution_scope="selected_workflow", placement_path=owner["path"],
                block_definition_id=contract["id"], block_class=contract["className"],
                block_contract_hash=contract["contentHash"], execution_kind="loop_owner",
                state_in=issued, loop_members_in=tuple(members), joint_attention_kwargs=options)
            return result["state_output__latents"]
        with bind_upstream_loop_inputs(loop, members if edited else [{"path": member["path"]} for member in members],
                                      parent_path=owner["path"]):
            _, state = loop(pipeline, state)
        return state.get("latents")

    baseline = run(False)
    torch.testing.assert_close(run(True), baseline, rtol=0, atol=0)
    torch.testing.assert_close(run(True, adapter=True), baseline, rtol=0, atol=0)
    if masked_attention:
        assert not torch.equal(baseline, run(False, use_mask=False))
    assert torch.isfinite(baseline).all()
    assert not torch.equal(baseline, initial)
    assert loop.sub_blocks == original
    # The same edge cannot silently mean a forward reference in this iteration.
    # Keep the actual path/input in the diagnostic and leave the owned loop intact.
    members[0]["iterationBindings"]["latents"]["timing"] = "current"
    with pytest.raises(ValueError, match=r"Loop member .*input latents: current-iteration producer must run before"):
        validate_loop_bindings(loop, members, parent_path=owner["path"])
    assert loop.sub_blocks == original
