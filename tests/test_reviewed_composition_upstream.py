"""Weightless execution against the installed, reviewed Diffusers implementation.

Run in the explicitly activated optional runtime for Qwen; the base CPU profile
does not implicitly install or activate Transformers just to collect this test.
"""
import importlib.util
import gzip
import json
from pathlib import Path

import pytest


def _qwen_admissions():
    catalog = json.loads(gzip.decompress((Path(__file__).resolve().parents[1] / "modiff/registered_block_v2_catalog.v1.json.gz").read_bytes()))
    return [entry["definition"] for entry in catalog["entries"]
            if entry["definition"]["source"].get("pipelineClass", "").startswith("Qwen")]


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="optional runtime not active")
@pytest.mark.parametrize("pipeline_class", ["FluxModularPipeline", "FluxKontextModularPipeline"])
def test_moved_conditional_text_inputs_initializes_and_executes_without_reselecting_old_branch(pipeline_class):
    """The graph, not the now-edited conditional selector, chooses each step."""
    from unittest.mock import patch
    import torch
    from modiff.modular_composition import _block_at_path
    from modiff.modular_requirements import validate_runtime_component_requirements
    from modules.ModularDiffusers import reviewed_blocks

    definition = reviewed_blocks._reviewed_definition(pipeline_class, "text2image")
    snapshot = reviewed_blocks.reviewed_modular_conditional_snapshot()
    tree = next(p for p in snapshot["pipelines"] if p["pipelineClass"] == pipeline_class)
    contracts = {b["id"]: b for b in snapshot["blockDefinitions"]}
    source = next(p for p in tree["placements"]
                  if contracts[p["blockDefinitionId"]]["className"] == "FluxTextInputStep"
                  and p["path"][-1] == "text2image")
    path = source["path"]
    destination = [*path[:-2], path[-1]]
    recipe = {
        "schemaVersion": 1, "diffusersRevision": snapshot["diffusersRevision"],
        "pipelineClass": pipeline_class, "workflowId": "text2image",
        "definitionId": definition["id"], "blockContractHash": definition["blockContractHash"],
        "operations": [{"kind": "move", "path": path, "parentPath": destination[:-1],
                        "name": destination[-1], "index": 3}],
    }
    with (patch.object(reviewed_blocks, "_component_bundle_token", return_value="issued"),
          patch.object(reviewed_blocks, "collect_model_ids", return_value=[])):
        _token, pipeline, state = reviewed_blocks._new_runtime(
            bundle={}, pipeline_class=pipeline_class, workflow_id="text2image",
            execution_scope="unpruned_pipeline", composition_recipe=recipe)
    block = _block_at_path(pipeline.blocks, tuple(destination))
    assert type(block).__name__ == "FluxTextInputStep"
    state.set("prompt_embeds", torch.ones((1, 4, 8)))
    state.set("pooled_prompt_embeds", torch.ones((1, 8)))
    state.set("num_images_per_prompt", 2)
    _, state = block(pipeline, state)
    assert state.get("prompt_embeds").shape == (2, 4, 8)
    assert state.get("pooled_prompt_embeds").shape == (2, 8)
    # Deferring aggregate loading must not permit an actual executable block
    # to run without its own required model components.
    prompt = _block_at_path(pipeline.blocks, ("text_encoder",))
    with pytest.raises(ValueError, match="required component is missing"):
        validate_runtime_component_requirements(prompt, pipeline, path=("text_encoder",))


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="optional runtime not active")
@pytest.mark.parametrize("pipeline_class", ["FluxModularPipeline", "FluxKontextModularPipeline"])
def test_moved_conditional_child_preview_inspects_edited_tree_without_reselecting_old_branch(pipeline_class):
    from modiff.huggingface_node_library import reviewed_huggingface_node_library
    from modiff.modular_composition import rebuild_reviewed_modular_composition

    library = reviewed_huggingface_node_library()
    definition = next(d for d in library["definitions"]
                      if d.get("pipelineClass") == pipeline_class and d.get("workflowId") == "text2image")
    recipe = {
        "schemaVersion": 1, "diffusersRevision": library["diffusersRevision"],
        "pipelineClass": pipeline_class, "workflowId": "text2image",
        "definitionId": definition["id"], "blockContractHash": definition["blockContractHash"],
        "operations": [{"kind": "move", "path": ["denoise", "input", "text2image"],
                        "parentPath": ["denoise"], "name": "text2image", "index": 3}],
    }
    receipt = rebuild_reviewed_modular_composition(recipe)
    assert receipt["inspectionScope"] == "edited_unpruned_tree"
    assert receipt["executable"] is False
    assert ["denoise", "text2image"] in receipt["composedPaths"]
    assert ["denoise", "input", "text2image"] not in receipt["composedPaths"]
    assert "transformer" in receipt["components"]


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Qwen optional runtime not active")
@pytest.mark.parametrize("admission", _qwen_admissions(), ids=lambda item: item["definitionId"])
def test_explicit_binding_runs_inside_actual_qwen_loop_with_tiny_real_transformer(admission):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, QwenImageControlNetModel
    from diffusers.guiders import ClassifierFreeGuidance
    from diffusers.modular_pipelines import PipelineState
    from modiff.modular_loop_bindings import bind_upstream_loop_inputs
    from modiff.modular_composition import _default_blocks_resolver, _block_at_path
    from modules.ModularDiffusers import reviewed_blocks

    torch.manual_seed(42)
    transformer = QwenImageTransformer2DModel(patch_size=2, in_channels=4, out_channels=1,
        num_layers=1, attention_head_dim=16, num_attention_heads=1, joint_attention_dim=8,
        axes_dims_rope=(4, 6, 6))
    owner = next(node for node in admission["graph"]["nodes"]
                 if node.get("data", {}).get("params", {}).get("execution_kind", {}).get("value") == "loop_owner")
    tree = _default_blocks_resolver(admission["source"]["pipelineClass"], "__unpruned__")
    loop = _block_at_path(tree, tuple(owner["modularDiffusers"]["placementPath"]))
    assert type(loop).__name__ == owner["modularDiffusers"]["blockClass"]
    pipeline = tree.init_pipeline()
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False)
    pipeline.update_components(transformer=transformer, scheduler=scheduler, guider=ClassifierFreeGuidance(guidance_scale=1.0))
    if "controlnet" in pipeline.component_names:
        controlnet = QwenImageControlNetModel.from_transformer(transformer,num_layers=1,attention_head_dim=16,num_attention_heads=1)
        pipeline.update_components(controlnet=controlnet)
    loop = _block_at_path(pipeline.blocks, tuple(owner["modularDiffusers"]["placementPath"]))
    original = dict(loop.sub_blocks)
    parent_path = owner["modularDiffusers"]["placementPath"]
    members = [{"path": [*parent_path, name], "blockClass": type(member).__name__} for name, member in loop.sub_blocks.items()]
    members[0]["iterationBindings"] = {"latents": {"kind": "state", "sourcePath": [*parent_path, members[-1]["path"][-1]],
        "output": "latents", "timing": "previous"}}
    initial = torch.randn((1, 4, 4))
    prompts = torch.randn((1, 3, 8))
    image_latents = torch.randn_like(initial)
    mask = torch.ones_like(initial)
    mask[:, :2] = 0
    image_conditioning = type(loop.sub_blocks["before_denoiser"]).__name__ == "QwenImageEditLoopBeforeDenoiser"
    # One batch entry containing both generated and conditioning images.
    shapes = [[(1,2,2),(1,2,2)]] if image_conditioning else [(1,2,2)]

    def run(edited, through_adapter=False):
        scheduler.set_timesteps(3)
        state = PipelineState()
        for key, value in {"timesteps": scheduler.timesteps, "num_inference_steps": 3,
                           "latents": initial.clone(), "img_shapes": shapes,
                           "image_latents": image_latents, "mask": mask, "initial_noise": initial,
                           "control_image_latents": image_latents, "controlnet_keep": [1.0]*3,
                           "controlnet_conditioning_scale": 0.7}.items():
            state.set(key, value)
        state.set("prompt_embeds", prompts, kwargs_type="denoiser_input_fields")
        state.set("prompt_embeds_mask", torch.ones((1, 3)), kwargs_type="denoiser_input_fields")
        state.set("img_shapes", shapes, kwargs_type="denoiser_input_fields")
        if through_adapter:
            metadata = owner["modularDiffusers"]
            issued = reviewed_blocks._issue_state(token=object(), pipeline_class=admission["source"]["pipelineClass"],
                workflow_id=admission["source"]["workflow"], execution_scope="unpruned_pipeline", pipeline=pipeline,
                state=state, completed_path=tuple(parent_path[:-1]))
            output = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                pipeline_class=admission["source"]["pipelineClass"], workflow_id=admission["source"]["workflow"],
                execution_scope="unpruned_pipeline", placement_path=parent_path,
                block_definition_id=metadata["blockDefinitionId"], block_class=metadata["blockClass"],
                block_contract_hash=metadata["blockContractHash"], execution_kind="loop_owner", state_in=issued,
                loop_members_in=tuple(members))
            return output["state_output__latents"]
        with bind_upstream_loop_inputs(loop, members if edited else [{"path": [*parent_path, name]} for name in loop.sub_blocks], parent_path=parent_path):
            _, state = loop(pipeline, state)
        return state.get("latents")

    baseline = run(False)
    edited = run(True)
    adapted = run(True, through_adapter=True)
    torch.testing.assert_close(edited, baseline, rtol=0, atol=0)
    torch.testing.assert_close(adapted, baseline, rtol=0, atol=0)
    assert torch.isfinite(edited).all()
    assert not torch.equal(edited, initial)
    if "after_denoiser_inpaint" in original:
        torch.testing.assert_close(edited[mask == 0],image_latents[mask == 0],rtol=0,atol=0)
    assert loop.sub_blocks == original


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Qwen optional runtime not active")
def test_qwen_executes_foreign_timestep_block_absent_from_its_original_hierarchy():
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.modular_pipelines import PipelineState
    from modiff.huggingface_node_library import reviewed_huggingface_node_library
    from modiff.modular_composition import build_reviewed_modular_composition_blocks
    from modules.ModularDiffusers import reviewed_blocks

    library = reviewed_huggingface_node_library()
    target = next(d for d in library["definitions"] if d["id"] == "diffusers.modular:QwenImageModularPipeline:text2image")
    source = next(d for d in library["definitions"] if d.get("pipelineClass") == "ErnieImageModularPipeline")
    contracts = {b["id"]: b for b in library["blockDefinitions"]}
    placement = next(p for p in source["blockPlacements"] if contracts[p["blockDefinitionId"]]["className"] == "ErnieImageSetTimestepsStep")
    contract = contracts[placement["blockDefinitionId"]]
    path = ["denoise", "text2image", "foreign_timesteps"]
    recipe = {"schemaVersion": 1, "diffusersRevision": library["diffusersRevision"],
              "pipelineClass": target["pipelineClass"], "workflowId": target["workflowId"],
              "definitionId": target["id"], "blockContractHash": target["blockContractHash"],
              "operations": [{"kind": "insert", "sourceDefinitionId": source["id"],
                              "sourceBlockDefinitionId": contract["id"], "sourcePath": placement["path"],
                              "sourceExecutionScope": "selected_workflow", "parentPath": path[:-1],
                              "name": path[-1], "index": 0}]}
    validated, blocks = build_reviewed_modular_composition_blocks(recipe)
    pipeline = blocks.init_pipeline()
    # Explicit test component config, not a change to any creator model/default.
    pipeline.update_components(scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False))
    pipeline._modiff_composition_hash = validated["recipeHash"]
    state = PipelineState()
    issued = reviewed_blocks._issue_state(token=object(), pipeline_class=target["pipelineClass"],
        workflow_id=target["workflowId"], execution_scope="unpruned_pipeline", pipeline=pipeline,
        state=state, completed_path=("text_encoder",))
    output = reviewed_blocks.ReviewedModularWorkflowStep().execute(
        pipeline_class=target["pipelineClass"], workflow_id=target["workflowId"],
        execution_scope="unpruned_pipeline", composition_recipe=recipe, placement_path=path,
        block_definition_id=contract["id"], block_class=contract["className"],
        block_contract_hash=contract["contentHash"], execution_kind="step", state_in=issued,
        num_inference_steps=3)
    torch.testing.assert_close(output["state_output__timesteps"], torch.tensor([1000., 2000./3, 1000./3]))
    assert output["state_out"]._completed_path == tuple(path)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Qwen optional runtime not active")
@pytest.mark.parametrize("admission", _qwen_admissions(), ids=lambda item: item["definitionId"])
def test_each_qwen_admission_executes_shared_step_after_insert_move_replace_and_remove(admission):
    """Actual upstream tensor execution, not eleven full model generations."""
    import torch
    from diffusers.modular_pipelines import PipelineState
    from modiff.huggingface_node_library import reviewed_huggingface_node_library
    from modiff.modular_composition import build_reviewed_modular_composition_blocks
    from modules.ModularDiffusers import reviewed_blocks

    library = reviewed_huggingface_node_library()
    definition = next(item for item in library["definitions"] if item["id"] == admission["source"]["manifestDefinitionId"])
    node = next(item for item in admission["graph"]["nodes"] if item.get("modularDiffusers", {}).get("blockClass") == "QwenImageTextInputsStep")
    metadata = node["modularDiffusers"]
    path = metadata["placementPath"]
    parent = path[:-1]
    source = next(item for item in library["definitions"] if item["id"] == "diffusers.modular:QwenImageEditModularPipeline:image_conditioned")
    source_placement = next(item for item in source["blockPlacements"] if item["blockDefinitionId"] == metadata["blockDefinitionId"])
    identity = {"sourceDefinitionId": source["id"], "sourceBlockDefinitionId": metadata["blockDefinitionId"],
                "sourcePath": source_placement["path"], "sourceExecutionScope": "selected_workflow"}
    destination = [*parent, "completion_reused"]
    recipe = {
        "schemaVersion": 1, "diffusersRevision": library["diffusersRevision"],
        "pipelineClass": definition["pipelineClass"], "workflowId": definition["workflowId"],
        "definitionId": definition["id"], "blockContractHash": definition["blockContractHash"],
        "operations": [
            {"kind": "insert", **identity, "parentPath": parent, "name": "completion_added", "index": 0},
            {"kind": "move", "path": [*parent, "completion_added"], "parentPath": parent, "name": "completion_reused", "index": 1},
            {"kind": "replace", **identity, "path": destination},
            {"kind": "remove", "path": path},
        ],
    }
    validated, blocks = build_reviewed_modular_composition_blocks(recipe)
    assert not any(item["path"] == path for item in validated["composedPlacements"])
    pipeline = blocks.init_pipeline()
    pipeline._modiff_composition_hash = validated["recipeHash"]
    state = PipelineState()
    state.set("prompt_embeds", torch.zeros((1, 4, 8)))
    state.set("prompt_embeds_mask", torch.ones((1, 4)))
    issued = reviewed_blocks._issue_state(token=object(), pipeline_class=definition["pipelineClass"],
        workflow_id=definition["workflowId"], execution_scope="unpruned_pipeline", pipeline=pipeline,
        state=state, completed_path=("text_encoder",))
    explicit = torch.ones((1, 4, 8))
    output = reviewed_blocks.ReviewedModularWorkflowStep().execute(
        pipeline_class=definition["pipelineClass"], workflow_id=definition["workflowId"],
        execution_scope="unpruned_pipeline", composition_recipe=recipe, placement_path=destination,
        block_definition_id=metadata["blockDefinitionId"], block_class=metadata["blockClass"],
        block_contract_hash=metadata["blockContractHash"], execution_kind="step", state_in=issued,
        prompt_embeds=explicit, num_images_per_prompt="2")
    assert torch.equal(output["state_output__prompt_embeds"], explicit.repeat(2, 1, 1))
    assert output["state_output__prompt_embeds_mask"].shape == (2, 4)
    assert output["state_out"]._completed_path == tuple(destination)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Qwen optional runtime not active")
def test_composed_controlnet_inpaint_retains_selected_required_latents_contract():
    from modiff.huggingface_node_library import reviewed_huggingface_node_library
    from modiff.modular_composition import build_reviewed_modular_composition_blocks, _block_at_path

    library = reviewed_huggingface_node_library()
    definition = next(d for d in library["definitions"] if d["id"] == "diffusers.modular:QwenImageModularPipeline:controlnet_inpainting")
    definitions = {b["id"]: b for b in library["blockDefinitions"]}
    placement = next(p for p in definition["blockPlacements"]
                     if definitions[p["blockDefinitionId"]]["className"] == "QwenImageAdditionalInputsStep"
                     and "image_latents" in definitions[p["blockDefinitionId"]]["requiredInputs"])
    full_path = ["denoise", "controlnet_inpaint", "input", "additional_inputs"]
    recipe = {
        "schemaVersion": 1, "diffusersRevision": library["diffusersRevision"],
        "pipelineClass": definition["pipelineClass"], "workflowId": definition["workflowId"],
        "definitionId": definition["id"], "blockContractHash": definition["blockContractHash"],
        "operations": [{"kind": "replace", "path": full_path, "sourceDefinitionId": definition["id"],
                        "sourceBlockDefinitionId": placement["blockDefinitionId"], "sourcePath": placement["path"],
                        "sourceExecutionScope": "selected_workflow"}],
    }
    validated, blocks = build_reviewed_modular_composition_blocks(recipe)
    contract = next(p for p in validated["composedPlacements"] if p["path"] == full_path)
    assert contract["blockDefinitionId"] == placement["blockDefinitionId"]
    actual = _block_at_path(blocks, tuple(full_path))
    assert next(spec for spec in actual.inputs if spec.name == "image_latents").required


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Qwen optional runtime not active")
@pytest.mark.parametrize("cross_context", [False, True])
def test_inserted_qwen_text_inputs_uses_real_upstream_implementation(cross_context):
    import torch
    from diffusers.modular_pipelines import PipelineState
    from modiff.huggingface_node_library import reviewed_huggingface_node_library
    from modiff.modular_composition import build_reviewed_modular_composition_blocks
    from modules.ModularDiffusers import reviewed_blocks

    library = reviewed_huggingface_node_library()
    definition = next(d for d in library["definitions"] if d["id"] == "diffusers.modular:QwenImageModularPipeline:text2image")
    recipe = {
        "schemaVersion": 1, "diffusersRevision": library["diffusersRevision"],
        "pipelineClass": definition["pipelineClass"], "workflowId": definition["workflowId"],
        "definitionId": definition["id"], "blockContractHash": definition["blockContractHash"],
        "operations": [{"kind": "duplicate", "path": ["denoise", "text2image", "input"],
                        "parentPath": ["denoise", "text2image"], "name": "demo_inputs", "index": 1}],
    }
    if cross_context:
        source = next(d for d in library["definitions"] if d["id"] == "diffusers.modular:QwenImageEditModularPipeline:image_conditioned")
        contracts = {b["id"]: b for b in library["blockDefinitions"]}
        placement = next(p for p in source["blockPlacements"] if contracts[p["blockDefinitionId"]]["className"] == "QwenImageTextInputsStep")
        recipe["operations"] = [{"kind": "insert", "sourceDefinitionId": source["id"],
                                 "sourceBlockDefinitionId": placement["blockDefinitionId"],
                                 "sourcePath": placement["path"], "sourceExecutionScope": "selected_workflow",
                                 "parentPath": ["denoise", "text2image"], "name": "demo_inputs", "index": 1}]
    validated, blocks = build_reviewed_modular_composition_blocks(recipe)
    pipeline = blocks.init_pipeline()
    assert "controlnet" in pipeline.pretrained_component_names
    assert "controlnet" not in {spec.name for spec in blocks.get_workflow("text2image").expected_components}
    pipeline._modiff_composition_hash = validated["recipeHash"]
    path = ("denoise", "text2image", "demo_inputs")
    actual = reviewed_blocks._block_at_path(pipeline.blocks, path)
    assert type(actual).__name__ == "QwenImageTextInputsStep"
    state = PipelineState()
    state.set("prompt_embeds", torch.ones((1, 4, 8)))
    state.set("prompt_embeds_mask", torch.ones((1, 4)))
    issued = reviewed_blocks._issue_state(
        token=object(), pipeline_class=definition["pipelineClass"], workflow_id=definition["workflowId"],
        execution_scope="unpruned_pipeline", pipeline=pipeline, state=state, completed_path=("text_encoder",),
    )
    snapshot = reviewed_blocks.reviewed_modular_conditional_snapshot()
    contract = next(b for b in snapshot["blockDefinitions"] if b["className"] == type(actual).__name__)
    output = reviewed_blocks.ReviewedModularWorkflowStep().execute(
        pipeline_class=definition["pipelineClass"], workflow_id=definition["workflowId"],
        execution_scope="unpruned_pipeline", composition_recipe=recipe, placement_path=list(path),
        block_definition_id=contract["id"], block_class=contract["className"],
        block_contract_hash=contract["contentHash"], execution_kind="step", state_in=issued,
        num_images_per_prompt="2",
    )
    assert output["state_out"]._state.get("prompt_embeds").shape == (2, 4, 8)
    assert output["state_output__prompt_embeds"] is output["state_out"]._state.get("prompt_embeds")
    assert output["state_output__prompt_embeds_mask"].shape == (2, 4)
    assert output["state_out"]._state.get("prompt_embeds_mask").shape == (2, 4)
    assert output["state_out"]._completed_path == path
